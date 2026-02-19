import argparse
import glob
import math
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NodeEmbedding(nn.Module):
    """Initial node embedding from coordinates and visited mask."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, coords: torch.Tensor, visited_mask: torch.Tensor) -> torch.Tensor:
        x = torch.cat([coords, visited_mask.unsqueeze(-1)], dim=-1)
        return self.mlp(x)


class GCNEncoder(nn.Module):
    """GCN encoder matching the training architecture."""

    def __init__(self, hidden_dim: int, num_layers: int = 2, num_neighbors: int = -1, dropout: float = 0.1):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_emb: torch.Tensor, dist_matrix: torch.Tensor) -> torch.Tensor:
        _, num_nodes, _ = dist_matrix.shape
        scores = -dist_matrix

        if 0 < self.num_neighbors < num_nodes:
            knn_idx = torch.topk(dist_matrix, k=max(1, self.num_neighbors), dim=-1, largest=False).indices
            knn_mask = torch.zeros_like(dist_matrix, dtype=torch.bool)
            knn_mask.scatter_(dim=-1, index=knn_idx, value=True)
            scores = scores.masked_fill(~knn_mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        h = node_emb
        for layer_idx, layer in enumerate(self.layers):
            agg = torch.bmm(weights, h)
            delta = F.relu(layer(agg))
            delta = self.dropout(delta)
            h = self.norms[layer_idx](h + delta)
        return h


class TransformerEncoder(nn.Module):
    """Transformer encoder matching the training architecture."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 5,
        num_layers: int = 2,
        dropout: float = 0.1,
        ff_multiplier: int = 4,
        enable_nested_tensor: bool = False,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * ff_multiplier,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=enable_nested_tensor,
        )

    def forward(self, node_emb: torch.Tensor, dist_matrix: torch.Tensor | None = None) -> torch.Tensor:
        return self.encoder(node_emb)


class PolicyHead(nn.Module):
    """Policy head matching the training architecture."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, node_emb: torch.Tensor, current_city: torch.Tensor, visited_mask: torch.Tensor) -> torch.Tensor:
        batch_size, _, hidden_dim = node_emb.shape
        batch_idx = torch.arange(batch_size, device=node_emb.device)
        h_current = node_emb[batch_idx, current_city]

        q = self.query(h_current)
        k = self.key(node_emb)
        logits = torch.einsum("bh,bnh->bn", q, k) / math.sqrt(hidden_dim)

        invalid_mask = visited_mask.bool().clone()
        invalid_mask[batch_idx, current_city] = True
        logits = logits.masked_fill(invalid_mask, -1e9)
        return logits


class GCNModel(nn.Module):
    def __init__(self, hidden_dim: int, num_nodes: int, num_neighbors: int, gcn_num_layers: int, gcn_dropout: float):
        super().__init__()
        self.node_embedding = NodeEmbedding(hidden_dim)
        self.encoder = GCNEncoder(
            hidden_dim,
            num_layers=gcn_num_layers,
            num_neighbors=num_neighbors,
            dropout=gcn_dropout,
        )
        self.policy_head = PolicyHead(hidden_dim)

    def forward(self, coords: torch.Tensor, dist_matrix: torch.Tensor, visited_mask: torch.Tensor, current_city: torch.Tensor) -> torch.Tensor:
        node_emb = self.node_embedding(coords, visited_mask)
        node_emb = self.encoder(node_emb, dist_matrix)
        return self.policy_head(node_emb, current_city, visited_mask)


class TransformerModel(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_nodes: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        ff_multiplier: int,
        enable_nested_tensor: bool,
    ):
        super().__init__()
        self.node_embedding = NodeEmbedding(hidden_dim)
        self.encoder = TransformerEncoder(
            hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            ff_multiplier=ff_multiplier,
            enable_nested_tensor=enable_nested_tensor,
        )
        self.policy_head = PolicyHead(hidden_dim)

    def forward(self, coords: torch.Tensor, dist_matrix: torch.Tensor, visited_mask: torch.Tensor, current_city: torch.Tensor) -> torch.Tensor:
        node_emb = self.node_embedding(coords, visited_mask)
        node_emb = self.encoder(node_emb, dist_matrix)
        return self.policy_head(node_emb, current_city, visited_mask)


@dataclass
class EvalStats:
    instances: int = 0
    avg_pred_len: float = 0.0
    avg_opt_len: float = 0.0
    avg_abs_gap: float = 0.0
    avg_pct_gap: float = 0.0


def parse_instance(line: str, num_nodes: int) -> tuple[np.ndarray, list[int]]:
    parts = line.split()
    coords = np.array([float(parts[i]) for i in range(2 * num_nodes)], dtype=np.float32).reshape(num_nodes, 2)
    idx_output = parts.index("output")
    optimal_tour = [int(x) - 1 for x in parts[idx_output + 1 :]]
    return coords, optimal_tour


def tour_length(coords: np.ndarray, tour: list[int]) -> float:
    return float(
        sum(
            np.linalg.norm(coords[tour[i]] - coords[tour[i + 1]])
            for i in range(len(tour) - 1)
        )
    )


def build_tour(
    model: nn.Module,
    coords: np.ndarray,
    start_node_zero_based: int,
    device: torch.device,
) -> list[int]:
    num_nodes = coords.shape[0]
    coords_t = torch.tensor(coords, dtype=torch.float32, device=device).unsqueeze(0)
    diff = coords_t[:, :, None, :] - coords_t[:, None, :, :]
    dist_matrix_t = torch.linalg.norm(diff, dim=-1)

    visited = np.zeros(num_nodes, dtype=np.float32)
    current = start_node_zero_based
    tour = [current]

    # Model decisions: num_nodes - 2.
    for _ in range(num_nodes - 2):
        visited_t = torch.tensor(visited, dtype=torch.float32, device=device).unsqueeze(0)
        current_t = torch.tensor([current], dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(coords_t, dist_matrix_t, visited_t, current_t)
        nxt = int(torch.argmax(logits, dim=-1).item())

        tour.append(nxt)
        visited[current] = 1.0
        current = nxt

    # Deterministic close: go to only other unvisited node, then back to start.
    remaining = [idx for idx in range(num_nodes) if visited[idx] == 0.0 and idx != current]
    if len(remaining) != 1:
        raise RuntimeError(f"Expected exactly one remaining candidate, found {len(remaining)}")
    tour.append(remaining[0])
    tour.append(start_node_zero_based)
    return tour


def build_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> tuple[nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model_type = checkpoint["model_type"]
    hidden_dim = config["hidden_dim"]
    num_nodes = config["num_nodes"]

    if model_type == "gcn":
        model = GCNModel(
            hidden_dim=hidden_dim,
            num_nodes=num_nodes,
            num_neighbors=config.get("num_neighbors", -1),
            gcn_num_layers=config.get("gcn_num_layers", 2),
            gcn_dropout=config.get("gcn_dropout", 0.1),
        )
    elif model_type == "transformer":
        model = TransformerModel(
            hidden_dim=hidden_dim,
            num_nodes=num_nodes,
            num_heads=config.get("transformer_num_heads", 5),
            num_layers=config.get("transformer_num_layers", 2),
            dropout=config.get("transformer_dropout", 0.1),
            ff_multiplier=config.get("transformer_ff_multiplier", 4),
            enable_nested_tensor=config.get("transformer_enable_nested_tensor", False),
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def evaluate_model(
    model: nn.Module,
    lines: list[str],
    num_nodes: int,
    device: torch.device,
    start_node_one_based: int,
    max_instances: int | None,
) -> EvalStats:
    start_node_zero_based = start_node_one_based - 1
    if not (0 <= start_node_zero_based < num_nodes):
        raise ValueError(f"start_node must be in [1, {num_nodes}]")

    pred_lengths = []
    opt_lengths = []
    num_eval = len(lines) if max_instances is None else min(len(lines), max_instances)

    for i in range(num_eval):
        coords, optimal_tour = parse_instance(lines[i], num_nodes)
        pred_tour = build_tour(model, coords, start_node_zero_based, device)

        pred_len = tour_length(coords, pred_tour)
        opt_len = tour_length(coords, optimal_tour)
        pred_lengths.append(pred_len)
        opt_lengths.append(opt_len)

    pred_arr = np.array(pred_lengths, dtype=np.float64)
    opt_arr = np.array(opt_lengths, dtype=np.float64)
    gap = pred_arr - opt_arr
    pct_gap = np.where(opt_arr > 0.0, (gap / opt_arr) * 100.0, 0.0)

    return EvalStats(
        instances=num_eval,
        avg_pred_len=float(pred_arr.mean()),
        avg_opt_len=float(opt_arr.mean()),
        avg_abs_gap=float(gap.mean()),
        avg_pct_gap=float(pct_gap.mean()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved TSP models by building full tours on test instances.")
    parser.add_argument("--num-nodes", type=int, required=True, help="Number of nodes for the TSP dataset/checkpoints.")
    parser.add_argument("--test-filepath", type=str, default=None, help="Path to test .txt file (defaults to tsp-data/tsp{num_nodes}_test_concorde.txt).")
    parser.add_argument("--models-dir", type=str, default="Models", help="Directory containing saved checkpoints.")
    parser.add_argument("--gcn-prefix", type=str, default="gcn", help="GCN checkpoint prefix.")
    parser.add_argument("--transformer-prefix", type=str, default="transformer", help="Transformer checkpoint prefix.")
    parser.add_argument("--start-node", type=int, default=1, help="Fixed start node (1-based indexing).")
    parser.add_argument("--max-instances", type=int, default=None, help="Max number of test instances to evaluate.")
    parser.add_argument("--device", type=str, default=None, help="cuda/cpu (default: cuda if available).")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_filepath = args.test_filepath or f"tsp-data/tsp{args.num_nodes}_test_concorde.txt"
    gcn_path = os.path.join(args.models_dir, f"{args.gcn_prefix}_tsp{args.num_nodes}.pt")
    transformer_path = os.path.join(args.models_dir, f"{args.transformer_prefix}_tsp{args.num_nodes}.pt")

    def resolve_checkpoint(preferred_path: str, model_keyword: str) -> str | None:
        if os.path.exists(preferred_path):
            return preferred_path
        pattern = os.path.join(args.models_dir, "*.pt")
        candidates = [p for p in glob.glob(pattern) if f"tsp{args.num_nodes}" in os.path.basename(p)]
        keyword_matches = [p for p in candidates if model_keyword in os.path.basename(p).lower()]
        if keyword_matches:
            return sorted(keyword_matches)[0]
        return None

    if not os.path.exists(test_filepath):
        raise FileNotFoundError(f"Test file not found: {test_filepath}")

    with open(test_filepath, "r") as f:
        lines = f.readlines()

    print(f"Device: {device}")
    print(f"Test file: {test_filepath}")
    print(f"Instances available: {len(lines)}")
    if args.max_instances is not None:
        print(f"Instances evaluated: {min(len(lines), args.max_instances)}")
    else:
        print("Instances evaluated: all")

    model_specs = [
        ("GCN", resolve_checkpoint(gcn_path, "gcn")),
        ("Transformer", resolve_checkpoint(transformer_path, "transformer")),
    ]
    for model_name, checkpoint_path in model_specs:
        if checkpoint_path is None:
            print(f"\n[{model_name}] Skipped (checkpoint not found for tsp{args.num_nodes} in {args.models_dir})")
            continue

        model, checkpoint = build_model_from_checkpoint(checkpoint_path, device)
        checkpoint_test = checkpoint.get("final_test_loss", None)
        stats = evaluate_model(
            model=model,
            lines=lines,
            num_nodes=args.num_nodes,
            device=device,
            start_node_one_based=args.start_node,
            max_instances=args.max_instances,
        )

        print(f"\n[{model_name}]")
        print(f"Checkpoint: {checkpoint_path}")
        if checkpoint_test is not None:
            print(f"Stored checkpoint test loss (training script): {checkpoint_test:.4f}")
        print(f"Avg predicted tour length: {stats.avg_pred_len:.6f}")
        print(f"Avg optimal tour length:   {stats.avg_opt_len:.6f}")
        print(f"Avg absolute gap:          {stats.avg_abs_gap:.6f}")
        print(f"Avg percentage gap:        {stats.avg_pct_gap:.4f}%")


if __name__ == "__main__":
    main()
