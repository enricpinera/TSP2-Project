import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
import copy

import time

from sklearn.utils import shuffle
from scipy.spatial.distance import pdist, squareform


#@title Dataloader definitions
class DotDict(dict):
    """Wrapper around in-built dict class to access members through the dot operation.
    """

    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self


class SequentialTSPReader(object):
    """
    Iterator that reads TSP dataset files (Vinyals format) and yields SEQUENTIAL
    mini-batches for autoregressive next-city prediction.

    For each TSP graph of N nodes, generates N training samples:
        - coords: (N, 2)
        - dist_matrix: (N, N)
        - current_city: int
        - visited_mask: (N,)
        - target_next_city: int
    """

    def __init__(self, num_nodes, num_neighbors, batch_size, filepath, shuffle_data=True):
        """
        Args:
            num_nodes: Number of nodes in TSP tours
            num_neighbors: Reserved for compatibility with the Transformer pipeline
            batch_size: Batch size
            filepath: Path to dataset file (.txt file)
        """
        self.num_nodes = num_nodes
        self.num_neighbors = num_neighbors
        self.batch_size = batch_size
        self.filepath = filepath
        with open(filepath, "r") as f:
            self.filedata = f.readlines()
        if shuffle_data:
            self.filedata = shuffle(self.filedata)
        self.max_iter = len(self.filedata)  # one graph per iteration

    def __iter__(self):
        """
        Yields batches of sequential samples.
        Each graph produces N samples, so batch_size refers to sequential samples.
        """
        batch = []

        for line in self.filedata:
            seq_examples = self.process_line(line)

            for ex in seq_examples:
                batch.append(ex)

                if len(batch) == self.batch_size:
                    yield self.collate(batch)
                    batch = []

        # Yield last incomplete batch
        if batch:
            yield self.collate(batch)

    def process_line(self, line):
        """
        Convert a single TSP instance into N sequential examples.
        """
        parts = line.split()

        # Extract coordinates
        coords = []
        for i in range(0, 2 * self.num_nodes, 2):
            coords.append([float(parts[i]), float(parts[i + 1])])
        coords = np.array(coords)

        # Compute distance matrix
        dist_matrix = squareform(pdist(coords, metric='euclidean'))

        # Extract tour (1-based indexing → convert to 0-based)
        idx_output = parts.index("output")
        tour = [int(x) - 1 for x in parts[idx_output + 1:]][:-1]  # remove repeated last node

        # Generate sequential examples
        examples = []
        visited = np.zeros(self.num_nodes, dtype=np.float32)

        for step in range(self.num_nodes - 1):
            current = tour[step]
            next_city = tour[step + 1]

            example = DotDict(
                coords=coords,
                dist_matrix=dist_matrix,
                current_city=current,
                visited_mask=visited.copy(),
                target_next_city=next_city
            )

            examples.append(example)
            visited[current] = 1

        return examples

    def collate(self, batch):
        """
        Convert list of DotDicts into a single DotDict batch.
        """
        return DotDict(
            coords=np.stack([b.coords for b in batch]),
            dist_matrix=np.stack([b.dist_matrix for b in batch]),
            current_city=np.array([b.current_city for b in batch]),
            visited_mask=np.stack([b.visited_mask for b in batch]),
            target_next_city=np.array([b.target_next_city for b in batch])
        )
    

#@title MLP layer
class MLP(nn.Module):
    """Multi-layer Perceptron for output prediction.
    """

    def __init__(self, hidden_dim, output_dim, L=2):
        super(MLP, self).__init__()
        self.L = L
        U = []
        for layer in range(self.L - 1):
            U.append(nn.Linear(hidden_dim, hidden_dim, True))
        self.U = nn.ModuleList(U)
        self.V = nn.Linear(hidden_dim, output_dim, True)

    def forward(self, x):
        """
        Args:
            x: Input features (batch_size, hidden_dim)

        Returns:
            y: Output predictions (batch_size, output_dim)
        """
        Ux = x
        for U_i in self.U:
            Ux = U_i(Ux)  # B x H
            Ux = F.relu(Ux)  # B x H
        y = self.V(Ux)  # B x O
        return y
    

#@title Node Embedding Layer
class NodeEmbedding(nn.Module):
    """
    Initial node embedding from:
      - coordinates (x, y)
      - visited_mask (0/1)
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),   # x, y, visited
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, coords, visited_mask):
        """
        coords: (B, N, 2)
        visited_mask: (B, N)
        """
        x = torch.cat([coords, visited_mask.unsqueeze(-1)], dim=-1)  # (B, N, 3)
        return self.mlp(x)  # (B, N, hidden_dim)


#@title GCN Encoder
class GCNEncoder(nn.Module):
    """
    Multi-layer GCN with residual connections and distance-based edge weights.
    """

    def __init__(self, hidden_dim, num_layers=2, num_neighbors=-1, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.num_neighbors = num_neighbors

        # Each layer projects H -> H.
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_emb, dist_matrix):
        """
        node_emb: (B, N, H)
        dist_matrix: (B, N, N)
        """
        B, N, _ = dist_matrix.shape
        scores = -dist_matrix

        # Optional k-NN sparsification (keeps nearest neighbors per node).
        if 0 < self.num_neighbors < N:
            k = max(1, self.num_neighbors)
            knn_idx = torch.topk(dist_matrix, k=k, dim=-1, largest=False).indices  # (B, N, k)
            knn_mask = torch.zeros_like(dist_matrix, dtype=torch.bool)
            knn_mask.scatter_(dim=-1, index=knn_idx, value=True)
            scores = scores.masked_fill(~knn_mask, float("-inf"))

        # Normalize adjacency weights to keep message scale stable across nodes.
        weights = torch.softmax(scores, dim=-1)  # (B, N, N)

        h = node_emb  # (B, N, H)

        for layer_idx, layer in enumerate(self.layers):
            # Message passing
            agg = torch.bmm(weights, h)  # (B, N, H)

            # Residual block with normalization for more stable optimization.
            delta = F.relu(layer(agg))
            delta = self.dropout(delta)
            h = self.norms[layer_idx](h + delta)

        return h  # (B, N, H)


#@title Policy Head
class PolicyHead(nn.Module):
    """
    Given node embeddings and current node index, outputs logits over next node.
    """

    def __init__(self, hidden_dim, num_nodes):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, node_emb, current_city, visited_mask):
        """
        node_emb: (B, N, H)
        current_city: (B,) -- current node index
        visited_mask: (B, N) -- 1 if visited, 0 otherwise
        """
        B, N, H = node_emb.shape

        # Gather current node embedding for each item in the batch.
        batch_idx = torch.arange(B, device=node_emb.device)
        h_current = node_emb[batch_idx, current_city]  # (B, H)

        # Score each candidate node from current-node query vs node keys.
        q = self.query(h_current)  # (B, H)
        k = self.key(node_emb)  # (B, N, H)
        logits = torch.einsum("bh,bnh->bn", q, k) / (H ** 0.5)  # (B, N)

        # Mask visited nodes and also disallow staying in current node.
        invalid_mask = visited_mask.bool().clone()
        invalid_mask[batch_idx, current_city] = True
        logits = logits.masked_fill(invalid_mask, -1e9)

        return logits  # (B, N)
    

#@title Sequential TSP Model
class SequentialTSPModel(nn.Module):
    """
    Sequential model to predict the next node in TSP.
    Combines:
      - NodeEmbedding
      - GCNEncoder
      - PolicyHead
    """

    def __init__(self, hidden_dim, num_nodes, num_neighbors=-1, gcn_num_layers=2, gcn_dropout=0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim

        # 1) Initial node embeddings
        self.node_embedding = NodeEmbedding(hidden_dim)

        # 2) GCN encoder
        self.encoder = GCNEncoder(
            hidden_dim,
            num_layers=gcn_num_layers,
            num_neighbors=num_neighbors,
            dropout=gcn_dropout
        )

        # 3) Policy head (logits over N nodes)
        self.policy_head = PolicyHead(hidden_dim, num_nodes)

    def forward(self, coords, dist_matrix, visited_mask, current_city):
        """
        coords: (B, N, 2)
        dist_matrix: (B, N, N)
        visited_mask: (B, N)
        current_city: (B,)
        """
        # 1) Initial node embeddings
        node_emb = self.node_embedding(coords, visited_mask)  # (B, N, H)

        # 2) Encoder
        node_emb = self.encoder(node_emb, dist_matrix)     # (B, N, H)

        # 3) Policy head -> logits over next node
        logits = self.policy_head(node_emb, current_city, visited_mask)  # (B, N)

        return logits


#@title Hyperparameters
num_nodes = 10 #@param # Could also be 10, 20, or 30!
num_neighbors = 5
batch_size = 128
hidden_dim = 50 #@param
gcn_num_layers = 2
gcn_dropout = 0.1
learning_rate = 1e-3 #@param
max_epochs = 50 #@param
batches_per_epoch = 10000
accumulation_steps = 1
grad_clip_norm = 1.0
lr_factor = 0.5
lr_patience = 3
min_lr = 1e-6
early_stop_patience = 8
train_filepath = f"tsp-data/tsp{num_nodes}_train_concorde.txt"
val_filepath   = f"tsp-data/tsp{num_nodes}_val_concorde.txt"
test_filepath  = f"tsp-data/tsp{num_nodes}_test_concorde.txt"

config = {
    'num_nodes': num_nodes,
    'num_neighbors': num_neighbors,
    'batch_size': batch_size,
    'hidden_dim': hidden_dim,
    'gcn_num_layers': gcn_num_layers,
    'gcn_dropout': gcn_dropout,
    'learning_rate': learning_rate,
    'max_epochs': max_epochs,
    'batches_per_epoch': batches_per_epoch,
    'accumulation_steps': accumulation_steps,
    'grad_clip_norm': grad_clip_norm,
    'lr_factor': lr_factor,
    'lr_patience': lr_patience,
    'min_lr': min_lr,
    'early_stop_patience': early_stop_patience,
    'train_filepath': train_filepath,
    'val_filepath': val_filepath,
    'test_filepath': test_filepath
}


#@title Train one epoch
def train_one_epoch(model, optimizer, config):
    model.train()

    num_nodes = config['num_nodes']
    num_neighbors = config['num_neighbors']
    batch_size = config['batch_size']
    batches_per_epoch = config['batches_per_epoch']
    train_filepath = config['train_filepath']

    # Prepare dataloader
    dataset = SequentialTSPReader(
        num_nodes,
        num_neighbors,
        batch_size,
        train_filepath,
        shuffle_data=True
    )
    dataset = iter(dataset)

    running_loss = 0.0
    running_batches = 0

    start_time = time.time()

    for batch_idx in range(batches_per_epoch):
        try:
            batch = next(dataset)
        except StopIteration:
            break

        # Move batch to GPU
        coords = torch.tensor(batch.coords, dtype=torch.float32, device=device)
        dist_matrix = torch.tensor(batch.dist_matrix, dtype=torch.float32, device=device)
        visited_mask = torch.tensor(batch.visited_mask, dtype=torch.float32, device=device)
        current_city = torch.tensor(batch.current_city, dtype=torch.long, device=device)
        target_next_city = torch.tensor(batch.target_next_city, dtype=torch.long, device=device)

        # Forward
        logits = model(coords, dist_matrix, visited_mask, current_city)

        # Loss
        loss = F.cross_entropy(logits, target_next_city)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['grad_clip_norm'])
        optimizer.step()

        # Stats
        running_loss += loss.item()
        running_batches += 1

    epoch_loss = running_loss / running_batches
    epoch_time = time.time() - start_time

    return epoch_time, epoch_loss


#@title Test
def test(model, config, mode='test'):
    model.eval()

    num_nodes = config['num_nodes']
    num_neighbors = config['num_neighbors']
    batch_size = config['batch_size']
    val_filepath = config['val_filepath']
    test_filepath = config['test_filepath']

    # Select dataset
    filepath = val_filepath if mode == 'val' else test_filepath

    dataset = SequentialTSPReader(
        num_nodes,
        num_neighbors,
        batch_size,
        filepath,
        shuffle_data=False
    )
    dataset = iter(dataset)

    running_loss = 0.0
    running_batches = 0

    with torch.no_grad():
        while True:
            try:
                batch = next(dataset)
            except StopIteration:
                break

            coords = torch.tensor(batch.coords, dtype=torch.float32, device=device)
            dist_matrix = torch.tensor(batch.dist_matrix, dtype=torch.float32, device=device)
            visited_mask = torch.tensor(batch.visited_mask, dtype=torch.float32, device=device)
            current_city = torch.tensor(batch.current_city, dtype=torch.long, device=device)
            target_next_city = torch.tensor(batch.target_next_city, dtype=torch.long, device=device)

            logits = model(coords, dist_matrix, visited_mask, current_city)
            loss = F.cross_entropy(logits, target_next_city)

            running_loss += loss.item()
            running_batches += 1

    return running_loss / running_batches


#@title Model instantiation + parameter count
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)
model = SequentialTSPModel(
    config['hidden_dim'],
    config['num_nodes'],
    num_neighbors=config['num_neighbors'],
    gcn_num_layers=config['gcn_num_layers'],
    gcn_dropout=config['gcn_dropout']
).to(device)
nb_param = sum(p.numel() for p in model.parameters())
print("Number of parameters:", nb_param)


#@title Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=config['lr_factor'],
    patience=config['lr_patience'],
    min_lr=config['min_lr']
)
train_losses = []
val_losses = []
best_val_loss = float('inf')
best_state_dict = None
epochs_without_improvement = 0

for epoch in range(config['max_epochs']):
    # Train
    train_time, train_loss = train_one_epoch(model, optimizer, config)
    train_losses.append(train_loss)
    current_lr = optimizer.param_groups[0]['lr']
    print(f"\nEpoch {epoch+1} | Train Loss: {train_loss:.4f} | LR: {current_lr:.2e}")
    # Validation
    val_loss = test(model, config, mode='val')
    val_losses.append(val_loss)
    print(f"Epoch {epoch+1} | Val Loss: {val_loss:.4f}")
    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state_dict = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= config['early_stop_patience']:
        print(f"Early stopping at epoch {epoch+1} (best val: {best_val_loss:.4f})")
        break

if best_state_dict is not None:
    model.load_state_dict(best_state_dict)

test_loss = test(model, config, mode='test')
print(f"\nFinal Test Loss (best val checkpoint): {test_loss:.4f}")
