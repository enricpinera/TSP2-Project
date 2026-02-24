import os
import csv
import math
import random
import time

import numpy as np
from scipy.spatial.distance import pdist, squareform


CSV_DECIMALS = 6


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


def route_length(dist_matrix, route):
    n = len(route)
    return float(sum(dist_matrix[route[i], route[(i + 1) % n]] for i in range(n)))


def format_csv_float(value):
    rounded = round(float(value), CSV_DECIMALS)
    if rounded == 0.0:
        rounded = 0.0
    return format(rounded, f".{CSV_DECIMALS}f")


def rotate_route_to_start(route, start):
    idx = route.index(start)
    return route[idx:] + route[:idx]


def nearest_neighbor_route(dist_matrix, start):
    n = dist_matrix.shape[0]
    unvisited = set(range(n))
    unvisited.remove(start)
    route = [start]
    current = start

    while unvisited:
        nxt = min(unvisited, key=lambda node: dist_matrix[current, node])
        route.append(nxt)
        unvisited.remove(nxt)
        current = nxt

    return route


def greedy_route(dist_matrix, start):
    n = dist_matrix.shape[0]
    degree = [0] * n
    parent = list(range(n))
    rank = [0] * n
    selected_edges = []

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return
        if rank[rx] < rank[ry]:
            parent[rx] = ry
        elif rank[rx] > rank[ry]:
            parent[ry] = rx
        else:
            parent[ry] = rx
            rank[rx] += 1

    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((dist_matrix[i, j], i, j))
    edges.sort(key=lambda x: x[0])

    for _, u, v in edges:
        if degree[u] == 2 or degree[v] == 2:
            continue

        same_component = find(u) == find(v)
        if same_component and len(selected_edges) != n - 1:
            continue

        selected_edges.append((u, v))
        degree[u] += 1
        degree[v] += 1
        if not same_component:
            union(u, v)

        if len(selected_edges) == n:
            break

    if len(selected_edges) != n:
        return nearest_neighbor_route(dist_matrix, start)

    adj = [[] for _ in range(n)]
    for u, v in selected_edges:
        adj[u].append(v)
        adj[v].append(u)

    route = [start]
    prev = -1
    current = start
    for _ in range(n - 1):
        nxt_candidates = [x for x in adj[current] if x != prev]
        if not nxt_candidates:
            return nearest_neighbor_route(dist_matrix, start)
        nxt = nxt_candidates[0]
        route.append(nxt)
        prev, current = current, nxt

    return route


def two_opt_route(dist_matrix, start):
    route = nearest_neighbor_route(dist_matrix, start)
    n = len(route)

    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for k in range(i + 1, n):
                a, b = route[i - 1], route[i]
                c, d = route[k], route[(k + 1) % n]
                delta = (dist_matrix[a, c] + dist_matrix[b, d]) - (dist_matrix[a, b] + dist_matrix[c, d])
                if delta < -1e-12:
                    route[i:k + 1] = reversed(route[i:k + 1])
                    improved = True
                    break
            if improved:
                break

    return route


def three_opt_route(dist_matrix, start):
    route = two_opt_route(dist_matrix, start)
    n = len(route)

    improved = True
    while improved:
        improved = False
        best_delta = 0.0
        best_candidate = None
        current_len = route_length(dist_matrix, route)

        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                for k in range(j + 1, n):
                    a = route[:i]
                    b = route[i:j]
                    c = route[j:k]
                    d = route[k:]
                    candidates = [
                        a + b[::-1] + c + d,
                        a + b + c[::-1] + d,
                        a + b[::-1] + c[::-1] + d,
                        a + c + b + d,
                        a + c[::-1] + b + d,
                        a + c + b[::-1] + d,
                        a + c[::-1] + b[::-1] + d,
                    ]

                    for candidate in candidates:
                        cand_len = route_length(dist_matrix, candidate)
                        delta = cand_len - current_len
                        if delta < best_delta - 1e-12:
                            best_delta = delta
                            best_candidate = candidate

        if best_candidate is not None:
            route = best_candidate
            improved = True

    return route


def christofides_route(dist_matrix, start):
    n = dist_matrix.shape[0]

    in_mst = [False] * n
    key = [float("inf")] * n
    parent = [-1] * n
    key[start] = 0.0

    for _ in range(n):
        u = min((idx for idx in range(n) if not in_mst[idx]), key=lambda idx: key[idx])
        in_mst[u] = True
        for v in range(n):
            w = dist_matrix[u, v]
            if not in_mst[v] and w < key[v]:
                key[v] = w
                parent[v] = u

    mst_edges = []
    degree = [0] * n
    for v in range(n):
        if parent[v] != -1:
            u = parent[v]
            mst_edges.append((u, v))
            degree[u] += 1
            degree[v] += 1

    odd_vertices = [v for v in range(n) if degree[v] % 2 == 1]
    odd_set = set(odd_vertices)
    matching = []
    while odd_set:
        u = odd_set.pop()
        v = min(odd_set, key=lambda node: dist_matrix[u, node])
        odd_set.remove(v)
        matching.append((u, v))

    multiedges = mst_edges + matching
    adj = [[] for _ in range(n)]
    for edge_id, (u, v) in enumerate(multiedges):
        adj[u].append((v, edge_id))
        adj[v].append((u, edge_id))

    used = [False] * len(multiedges)
    stack = [start]
    euler = []

    while stack:
        u = stack[-1]
        while adj[u] and used[adj[u][-1][1]]:
            adj[u].pop()
        if not adj[u]:
            euler.append(stack.pop())
        else:
            v, edge_id = adj[u].pop()
            if used[edge_id]:
                continue
            used[edge_id] = True
            stack.append(v)

    visited = set()
    route = []
    for node in reversed(euler):
        if node not in visited:
            visited.add(node)
            route.append(node)

    if len(route) < n:
        missing = [node for node in range(n) if node not in visited]
        route.extend(missing)

    route = rotate_route_to_start(route, start)
    return route


def random_two_opt_neighbor(route, rng):
    n = len(route)
    i, j = sorted(rng.sample(range(1, n), 2))
    candidate = route[:]
    candidate[i:j + 1] = reversed(candidate[i:j + 1])
    return candidate


def simulated_annealing_route(dist_matrix, start, rng, iterations=5000, cooling=0.995):
    current = nearest_neighbor_route(dist_matrix, start)
    current_len = route_length(dist_matrix, current)
    best = current[:]
    best_len = current_len

    avg_edge = float(np.mean(dist_matrix))
    temperature = max(1e-9, avg_edge * len(current))

    for _ in range(iterations):
        candidate = random_two_opt_neighbor(current, rng)
        candidate_len = route_length(dist_matrix, candidate)
        delta = candidate_len - current_len

        if delta < 0 or rng.random() < math.exp(-delta / max(temperature, 1e-12)):
            current = candidate
            current_len = candidate_len
            if current_len < best_len:
                best = current[:]
                best_len = current_len

        temperature *= cooling

    return best


def threshold_accepting_route(dist_matrix, start, rng, iterations=5000, cooling=0.995):
    current = nearest_neighbor_route(dist_matrix, start)
    current_len = route_length(dist_matrix, current)
    best = current[:]
    best_len = current_len

    threshold = max(1e-9, float(np.mean(dist_matrix)) * 0.5)

    for _ in range(iterations):
        candidate = random_two_opt_neighbor(current, rng)
        candidate_len = route_length(dist_matrix, candidate)
        delta = candidate_len - current_len

        if delta <= threshold:
            current = candidate
            current_len = candidate_len
            if current_len < best_len:
                best = current[:]
                best_len = current_len

        threshold *= cooling

    return best


def build_tour_from_route(route, start):
    route = rotate_route_to_start(route, start)
    return route + [start]


def evaluate_heuristic(heuristic_name, heuristic_fn, config, rng):
    lines = open(config["test_filepath"], "r").readlines()
    if config["max_instances"] is not None:
        lines = lines[:config["max_instances"]]

    start = config["start_node"] - 1
    results = []
    scores = []
    times = []
    tour_col = f"{heuristic_name}_tour"
    tour_length_col = f"{heuristic_name}_tour_length"

    for line in lines:
        coords, optimal_tour = parse_instance(line, config["num_nodes"])
        dist_matrix = squareform(pdist(coords, metric="euclidean")).astype(np.float32)

        t0 = time.perf_counter()
        route = heuristic_fn(dist_matrix, start, rng)
        pred_tour = build_tour_from_route(route, start)
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

    os.makedirs(config["results_dir"], exist_ok=True)
    csv_path = os.path.join(config["results_dir"], f"{heuristic_name}_tsp{config['num_nodes']}.csv")
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["optimal_tour", "optimal_tour_length", tour_col, tour_length_col, "score", "time"])
        writer.writerows(results)

    return csv_path, float(np.mean(scores)), float(np.mean(times))


def run_all_heuristics(config):
    rng = random.Random(config["seed"])
    heuristics = {
        "nearest_neighbor": lambda dist, s, _rng: nearest_neighbor_route(dist, s),
        "greedy": lambda dist, s, _rng: greedy_route(dist, s),
        "2opt": lambda dist, s, _rng: two_opt_route(dist, s),
        "3opt": lambda dist, s, _rng: three_opt_route(dist, s),
        "christofides": lambda dist, s, _rng: christofides_route(dist, s),
        "simulated_annealing": lambda dist, s, _rng: simulated_annealing_route(
            dist,
            s,
            _rng,
            iterations=config["sa_iterations"],
            cooling=config["sa_cooling"],
        ),
        "threshold_accepting": lambda dist, s, _rng: threshold_accepting_route(
            dist,
            s,
            _rng,
            iterations=config["ta_iterations"],
            cooling=config["ta_cooling"],
        ),
    }

    for heuristic_name in config["heuristics"]:
        if heuristic_name not in heuristics:
            raise ValueError(f"Unknown heuristic: {heuristic_name}")

        csv_path, mean_score, mean_time = evaluate_heuristic(
            heuristic_name,
            heuristics[heuristic_name],
            config,
            rng,
        )
        print(f"Heuristic: {heuristic_name}")
        print(f"CSV: {csv_path}")
        print(f"Mean score: {mean_score:.6f}")
        print(f"Mean time: {mean_time:.6f} s")
        print("")


#@title Hyperparameters
num_nodes = 20  # Could also be 10, 20, or 30
results_dir = "Results"
test_filepath = f"tsp-data/tsp{num_nodes}_test_concorde.txt"
start_node = 1
max_instances = None  # Set an int to evaluate a subset
seed = 42

heuristics = [
    "nearest_neighbor",
    "greedy",
    "2opt",
    "3opt",
    "christofides",
    "simulated_annealing",
    "threshold_accepting",
]

sa_iterations = 5000
sa_cooling = 0.995
ta_iterations = 5000
ta_cooling = 0.995

config = {
    "num_nodes": num_nodes,
    "results_dir": results_dir,
    "test_filepath": test_filepath,
    "start_node": start_node,
    "max_instances": max_instances,
    "seed": seed,
    "heuristics": heuristics,
    "sa_iterations": sa_iterations,
    "sa_cooling": sa_cooling,
    "ta_iterations": ta_iterations,
    "ta_cooling": ta_cooling,
}


run_all_heuristics(config)
