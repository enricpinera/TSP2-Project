import os
import csv
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from scipy.spatial.distance import pdist, squareform


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


#@title GCN Model
class GCNModel(nn.Module):
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


def parse_instance(line, num_nodes):
    parts = line.split()
    coords = []
    for i in range(0, 2 * num_nodes, 2):
        coords.append([float(parts[i]), float(parts[i + 1])])
    coords = np.array(coords, dtype=np.float32)

    idx_output = parts.index("output")
    optimal_tour = [int(x) - 1 for x in parts[idx_output + 1:]]  # keeps final return node
    return coords, optimal_tour


def tour_length(coords, tour):
    return float(
        sum(np.linalg.norm(coords[tour[i]] - coords[tour[i + 1]]) for i in range(len(tour) - 1))
    )


CSV_DECIMALS = 6


def format_csv_float(value):
    rounded = round(float(value), CSV_DECIMALS)
    if rounded == 0.0:
        rounded = 0.0
    return format(rounded, f".{CSV_DECIMALS}f")


def predict_tour(model, coords, start_node_one_based, device):
    num_nodes = coords.shape[0]
    start = start_node_one_based - 1
    tour = [start]
    visited = np.zeros(num_nodes, dtype=np.float32)
    current = start
    dist_matrix = squareform(pdist(coords, metric='euclidean')).astype(np.float32)

    for _ in range(num_nodes - 2):
        coords_t = torch.tensor(coords, dtype=torch.float32, device=device).unsqueeze(0)
        dist_t = torch.tensor(dist_matrix, dtype=torch.float32, device=device).unsqueeze(0)
        visited_t = torch.tensor(visited, dtype=torch.float32, device=device).unsqueeze(0)
        current_t = torch.tensor([current], dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(coords_t, dist_t, visited_t, current_t)
        next_city = int(torch.argmax(logits, dim=-1).item())

        tour.append(next_city)
        visited[current] = 1.0
        current = next_city

    remaining = [idx for idx in range(num_nodes) if visited[idx] == 0.0 and idx != current]
    if len(remaining) != 1:
        raise RuntimeError(f"Expected 1 remaining node, got {len(remaining)}")

    tour.append(remaining[0])
    tour.append(start)
    return tour


def evaluate(model, config, device):
    lines = open(config['test_filepath'], "r").readlines()
    if config['max_instances'] is not None:
        lines = lines[:config['max_instances']]

    results = []
    scores = []
    times = []
    model_tour_col = f"{config['save_prefix']}_tour"
    model_tour_length_col = f"{config['save_prefix']}_tour_length"

    for line in lines:
        coords, optimal_tour = parse_instance(line, config['num_nodes'])
        t0 = time.perf_counter()
        pred_tour = predict_tour(model, coords, config['start_node'], device)
        tour_time = time.perf_counter() - t0

        pred_len = tour_length(coords, pred_tour)
        opt_len = tour_length(coords, optimal_tour)
        score = pred_len / opt_len - 1.0

        optimal_tour_str = "{" + ", ".join(str(node + 1) for node in optimal_tour) + "}"
        pred_tour_str = "{" + ", ".join(str(node + 1) for node in pred_tour) + "}"
        results.append((
            optimal_tour_str,
            format_csv_float(opt_len),
            pred_tour_str,
            format_csv_float(pred_len),
            format_csv_float(score),
            format_csv_float(tour_time),
        ))
        scores.append(score)
        times.append(tour_time)

    os.makedirs(config['results_dir'], exist_ok=True)
    csv_path = os.path.join(config['results_dir'], f"{config['save_prefix']}_tsp{config['num_nodes']}.csv")
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["optimal_tour", "optimal_tour_length", model_tour_col, model_tour_length_col, "score", "time"])
        writer.writerows(results)

    return float(np.mean(scores)), float(np.mean(times))


#@title Hyperparameters
num_nodes = 10 #@param # Could also be 10, 20, or 30!
num_neighbors = 7 #@param
batch_size = 256 #@param
hidden_dim = 256 #@param
gcn_num_layers = 7 #@param
gcn_dropout = 0.15 #@param
save_dir = "Models"
results_dir = "Results"
save_prefix = "gcn"
test_filepath = f"tsp-data/tsp{num_nodes}_test_concorde.txt"
start_node = 1
max_instances = None  # Set an int to evaluate a subset

config = {
    'num_nodes': num_nodes,
    'num_neighbors': num_neighbors,
    'batch_size': batch_size,
    'hidden_dim': hidden_dim,
    'gcn_num_layers': gcn_num_layers,
    'gcn_dropout': gcn_dropout,
    'save_dir': save_dir,
    'results_dir': results_dir,
    'save_prefix': save_prefix,
    'test_filepath': test_filepath,
    'start_node': start_node,
    'max_instances': max_instances,
}


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

checkpoint_path = os.path.join(config['save_dir'], f"{config['save_prefix']}_tsp{config['num_nodes']}.pt")
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
ckpt_cfg = checkpoint.get('config', {})
model = GCNModel(
    hidden_dim=ckpt_cfg.get('hidden_dim', config['hidden_dim']),
    num_nodes=ckpt_cfg.get('num_nodes', config['num_nodes']),
    num_neighbors=ckpt_cfg.get('num_neighbors', config['num_neighbors']),
    gcn_num_layers=ckpt_cfg.get('gcn_num_layers', config['gcn_num_layers']),
    gcn_dropout=ckpt_cfg.get('gcn_dropout', config['gcn_dropout'])
).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

mean_score, mean_time = evaluate(model, config, device)
print(f"Checkpoint: {checkpoint_path}")
print(f"Mean score: {mean_score:.6f}")
print(f"Mean time: {mean_time:.6f}s")
