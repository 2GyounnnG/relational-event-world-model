# data/generate_graph_event_data.py

import os
import copy
import pickle
import random
from typing import Dict, List, Optional, Tuple, Any

import numpy as np


# =========================================================
# Global config
# =========================================================

DEFAULT_CONFIG = {
    "num_train": 10000,
    "num_val": 2000,
    "num_test": 2000,
    "num_nodes_range": (8, 12),
    "edge_prob": 0.25,
    "state_dim": 4,
    "num_types": 3,
    "max_events_per_step": 2,
    "prob_two_events": 0.3,
    "random_seed": 42,
    "output_dir": "data",
    "train_file": "graph_event_train.pkl",
    "val_file": "graph_event_val.pkl",
    "test_file": "graph_event_test.pkl",
}


# =========================================================
# Utilities
# =========================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def edge_tuple(i: int, j: int) -> Tuple[int, int]:
    """Canonical undirected edge tuple."""
    return (i, j) if i < j else (j, i)


def edges_from_adj(adj: np.ndarray) -> List[Tuple[int, int]]:
    n = adj.shape[0]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 1:
                edges.append((i, j))
    return edges


def adj_from_edges(num_nodes: int, edges: List[Tuple[int, int]]) -> np.ndarray:
    adj = np.zeros((num_nodes, num_nodes), dtype=np.int64)
    for i, j in edges:
        adj[i, j] = 1
        adj[j, i] = 1
    return adj


def get_neighbors(adj: np.ndarray, node_idx: int) -> List[int]:
    return np.where(adj[node_idx] == 1)[0].tolist()


def graph_to_edge_index(adj: np.ndarray) -> np.ndarray:
    """
    Convert undirected adjacency matrix to edge_index with both directions.
    Shape: [2, E_dir]
    """
    src = []
    dst = []
    n = adj.shape[0]
    for i in range(n):
        for j in range(n):
            if adj[i, j] == 1:
                src.append(i)
                dst.append(j)
    if len(src) == 0:
        return np.zeros((2, 0), dtype=np.int64)
    return np.array([src, dst], dtype=np.int64)


def two_hop_reachable(adj: np.ndarray, i: int, j: int) -> bool:
    """
    Return True if j is within 2-hop neighborhood of i (excluding direct equality).
    """
    if i == j:
        return False
    neighbors_i = get_neighbors(adj, i)
    if j in neighbors_i:
        return True
    for k in neighbors_i:
        if adj[k, j] == 1:
            return True
    return False


# =========================================================
# Graph state
# =========================================================

def generate_initial_graph(
    num_nodes_range: Tuple[int, int],
    edge_prob: float,
    state_dim: int,
    num_types: int,
) -> Dict[str, Any]:
    """
    Generate one initial graph state.

    Returns:
        {
            "node_features": np.ndarray [N, 1 + state_dim],
            "adj": np.ndarray [N, N],
        }
    """
    n = random.randint(num_nodes_range[0], num_nodes_range[1])

    # Node features: [type_id, state_vector...]
    node_features = np.zeros((n, 1 + state_dim), dtype=np.float32)
    for i in range(n):
        type_id = np.random.randint(0, num_types)
        state = np.random.randn(state_dim).astype(np.float32)
        node_features[i, 0] = float(type_id)
        node_features[i, 1:] = state

    # Undirected random graph
    adj = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(i + 1, n):
            if np.random.rand() < edge_prob:
                adj[i, j] = 1
                adj[j, i] = 1

    return {
        "node_features": node_features,
        "adj": adj,
    }


# =========================================================
# Event sampling rules
# =========================================================

def sample_node_state_update(graph: Dict[str, Any], num_types: int) -> Optional[Dict[str, Any]]:
    adj = graph["adj"]
    node_features = graph["node_features"]
    n = node_features.shape[0]
    candidates = []

    for i in range(n):
        neighbors = get_neighbors(adj, i)
        if len(neighbors) > 0:
            candidates.append(i)

    if len(candidates) == 0:
        return None

    i = random.choice(candidates)
    neighbors = get_neighbors(adj, i)

    return {
        "event_type": "node_state_update",
        "node_scope": [i] + neighbors,
        "edge_scope": [edge_tuple(i, j) for j in neighbors],
        "meta": {},
    }


def sample_edge_add(graph: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    adj = graph["adj"]
    node_features = graph["node_features"]
    n = node_features.shape[0]
    candidates = []

    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 0 and two_hop_reachable(adj, i, j):
                xi = node_features[i, 1:]
                xj = node_features[j, 1:]
                dist = np.linalg.norm(xi - xj)
                if dist < 2.5:
                    candidates.append((i, j))

    if len(candidates) == 0:
        return None

    i, j = random.choice(candidates)

    return {
        "event_type": "edge_add",
        "node_scope": [i, j],
        "edge_scope": [edge_tuple(i, j)],
        "meta": {},
    }


def sample_edge_delete(graph: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    adj = graph["adj"]
    node_features = graph["node_features"]
    edges = edges_from_adj(adj)
    candidates = []

    for i, j in edges:
        xi = node_features[i, 1:]
        xj = node_features[j, 1:]
        dist = np.linalg.norm(xi - xj)
        if dist > 2.0:
            candidates.append((i, j))

    if len(candidates) == 0:
        return None

    i, j = random.choice(candidates)

    return {
        "event_type": "edge_delete",
        "node_scope": [i, j],
        "edge_scope": [edge_tuple(i, j)],
        "meta": {},
    }


def sample_motif_type_flip(graph: Dict[str, Any], num_types: int) -> Optional[Dict[str, Any]]:
    adj = graph["adj"]
    node_features = graph["node_features"]
    n = node_features.shape[0]
    candidates = []

    for i in range(n):
        neighbors = get_neighbors(adj, i)
        if len(neighbors) < 2:
            continue
        for idx_a in range(len(neighbors)):
            for idx_b in range(idx_a + 1, len(neighbors)):
                j = neighbors[idx_a]
                k = neighbors[idx_b]
                type_i = int(node_features[i, 0])
                type_j = int(node_features[j, 0])
                type_k = int(node_features[k, 0])

                if type_j == type_k and type_i != type_j:
                    candidates.append((i, j, k))

    if len(candidates) == 0:
        return None

    i, j, k = random.choice(candidates)

    return {
        "event_type": "motif_type_flip",
        "node_scope": [i, j, k],
        "edge_scope": [edge_tuple(i, j), edge_tuple(i, k)],
        "meta": {},
    }


def sample_valid_event(
    graph: Dict[str, Any],
    event_type: str,
    num_types: int,
) -> Optional[Dict[str, Any]]:
    if event_type == "node_state_update":
        return sample_node_state_update(graph, num_types)
    elif event_type == "edge_add":
        return sample_edge_add(graph)
    elif event_type == "edge_delete":
        return sample_edge_delete(graph)
    elif event_type == "motif_type_flip":
        return sample_motif_type_flip(graph, num_types)
    else:
        raise ValueError(f"Unknown event_type: {event_type}")


def scopes_overlap(event_a: Dict[str, Any], event_b: Dict[str, Any]) -> bool:
    nodes_a = set(event_a["node_scope"])
    nodes_b = set(event_b["node_scope"])
    return len(nodes_a.intersection(nodes_b)) > 0


def sample_events(
    graph: Dict[str, Any],
    max_events: int,
    prob_two_events: float,
    num_types: int,
) -> List[Dict[str, Any]]:
    event_types = [
        "node_state_update",
        "edge_add",
        "edge_delete",
        "motif_type_flip",
    ]

    num_events = 2 if (max_events >= 2 and np.random.rand() < prob_two_events) else 1

    # First event
    random.shuffle(event_types)
    first_event = None
    for et in event_types:
        e = sample_valid_event(graph, et, num_types)
        if e is not None:
            first_event = e
            break

    if first_event is None:
        return []

    if num_events == 1:
        return [first_event]

    # Second event: prefer non-overlapping event if possible
    second_candidates = []
    for et in event_types:
        e2 = sample_valid_event(graph, et, num_types)
        if e2 is not None:
            second_candidates.append(e2)

    if len(second_candidates) == 0:
        return [first_event]

    non_overlapping = [e for e in second_candidates if not scopes_overlap(first_event, e)]
    if len(non_overlapping) > 0:
        second_event = random.choice(non_overlapping)
    else:
        second_event = random.choice(second_candidates)

    return [first_event, second_event]


# =========================================================
# Event application rules
# =========================================================

def apply_node_state_update(graph: Dict[str, Any], event: Dict[str, Any]) -> None:
    node_features = graph["node_features"]
    adj = graph["adj"]

    center = event["node_scope"][0]
    neighbors = get_neighbors(adj, center)
    if len(neighbors) == 0:
        return

    xi = node_features[center, 1:]
    mean_neighbor = node_features[neighbors, 1:].mean(axis=0)

    type_id = int(node_features[center, 0])
    # Small deterministic type-dependent bias
    bias_table = {
        0: np.array([0.10, 0.00, 0.00, 0.00], dtype=np.float32),
        1: np.array([0.00, 0.10, 0.00, 0.00], dtype=np.float32),
        2: np.array([0.00, 0.00, 0.10, 0.00], dtype=np.float32),
    }
    bias = bias_table.get(type_id, np.zeros_like(xi))

    alpha = 0.3
    new_x = xi + alpha * (mean_neighbor - xi) + bias[: xi.shape[0]]

    event["meta"]["old_state"] = xi.copy().tolist()
    event["meta"]["new_state"] = new_x.copy().tolist()

    node_features[center, 1:] = new_x


def apply_edge_add(graph: Dict[str, Any], event: Dict[str, Any]) -> None:
    adj = graph["adj"]
    i, j = event["node_scope"]
    adj[i, j] = 1
    adj[j, i] = 1


def apply_edge_delete(graph: Dict[str, Any], event: Dict[str, Any]) -> None:
    adj = graph["adj"]
    i, j = event["node_scope"]
    adj[i, j] = 0
    adj[j, i] = 0


def apply_motif_type_flip(graph: Dict[str, Any], event: Dict[str, Any], num_types: int) -> None:
    node_features = graph["node_features"]
    i, j, k = event["node_scope"]

    old_type = int(node_features[i, 0])
    target_type = int(node_features[j, 0])  # since j and k should match by construction

    event["meta"]["old_type"] = old_type
    event["meta"]["new_type"] = target_type

    node_features[i, 0] = float(target_type)


def apply_single_event(graph: Dict[str, Any], event: Dict[str, Any], num_types: int) -> Dict[str, Any]:
    if event["event_type"] == "node_state_update":
        apply_node_state_update(graph, event)
    elif event["event_type"] == "edge_add":
        apply_edge_add(graph, event)
    elif event["event_type"] == "edge_delete":
        apply_edge_delete(graph, event)
    elif event["event_type"] == "motif_type_flip":
        apply_motif_type_flip(graph, event, num_types)
    else:
        raise ValueError(f"Unknown event_type: {event['event_type']}")
    return graph


def apply_events(
    graph: Dict[str, Any],
    events: List[Dict[str, Any]],
    num_types: int,
) -> Dict[str, Any]:
    graph_next = copy.deepcopy(graph)
    for event in events:
        graph_next = apply_single_event(graph_next, event, num_types)
    return graph_next


# =========================================================
# Change masks / independence / event scope unions
# =========================================================

def compute_changed_nodes(graph_t: Dict[str, Any], graph_t1: Dict[str, Any]) -> np.ndarray:
    x_t = graph_t["node_features"]
    x_t1 = graph_t1["node_features"]

    assert x_t.shape == x_t1.shape
    changed = np.any(np.abs(x_t - x_t1) > 1e-6, axis=1)
    return changed.astype(np.bool_)


def compute_changed_edges(graph_t: Dict[str, Any], graph_t1: Dict[str, Any]) -> List[Tuple[int, int]]:
    adj_t = graph_t["adj"]
    adj_t1 = graph_t1["adj"]
    n = adj_t.shape[0]
    changed_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj_t[i, j] != adj_t1[i, j]:
                changed_edges.append((i, j))
    return changed_edges


def compute_event_scope_union_nodes(events: List[Dict[str, Any]]) -> List[int]:
    nodes = set()
    for event in events:
        nodes.update(event["node_scope"])
    return sorted(nodes)


def compute_event_scope_union_edges(events: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    edges = set()
    for event in events:
        for e in event["edge_scope"]:
            edges.add(edge_tuple(e[0], e[1]))
    return sorted(edges)


def find_independent_pairs(events: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    independent = []
    for i in range(len(events)):
        for j in range(i + 1, len(events)):
            if not scopes_overlap(events[i], events[j]):
                independent.append((i, j))
    return independent


def make_sample(
    graph_t: Dict[str, Any],
    graph_t1: Dict[str, Any],
    events: List[Dict[str, Any]],
) -> Dict[str, Any]:
    changed_nodes = compute_changed_nodes(graph_t, graph_t1)
    changed_edges = compute_changed_edges(graph_t, graph_t1)
    independent_pairs = find_independent_pairs(events)

    event_scope_union_nodes = compute_event_scope_union_nodes(events)
    event_scope_union_edges = compute_event_scope_union_edges(events)

    return {
        "graph_t": {
            "node_features": graph_t["node_features"].copy(),
            "adj": graph_t["adj"].copy(),
            "edge_index": graph_to_edge_index(graph_t["adj"]),
        },
        "graph_t1": {
            "node_features": graph_t1["node_features"].copy(),
            "adj": graph_t1["adj"].copy(),
            "edge_index": graph_to_edge_index(graph_t1["adj"]),
        },
        "events": copy.deepcopy(events),
        "independent_pairs": independent_pairs,
        "changed_nodes": changed_nodes,
        "changed_edges": changed_edges,
        "event_scope_union_nodes": event_scope_union_nodes,
        "event_scope_union_edges": event_scope_union_edges,
    }


# =========================================================
# Dataset generation
# =========================================================

def generate_one_sample(config: Dict[str, Any]) -> Dict[str, Any]:
    graph_t = generate_initial_graph(
        num_nodes_range=config["num_nodes_range"],
        edge_prob=config["edge_prob"],
        state_dim=config["state_dim"],
        num_types=config["num_types"],
    )

    events = sample_events(
        graph=graph_t,
        max_events=config["max_events_per_step"],
        prob_two_events=config["prob_two_events"],
        num_types=config["num_types"],
    )

    # If no event found, keep resampling initial graph
    while len(events) == 0:
        graph_t = generate_initial_graph(
            num_nodes_range=config["num_nodes_range"],
            edge_prob=config["edge_prob"],
            state_dim=config["state_dim"],
            num_types=config["num_types"],
        )
        events = sample_events(
            graph=graph_t,
            max_events=config["max_events_per_step"],
            prob_two_events=config["prob_two_events"],
            num_types=config["num_types"],
        )

    graph_t1 = apply_events(
        graph=graph_t,
        events=events,
        num_types=config["num_types"],
    )

    sample = make_sample(graph_t, graph_t1, events)
    return sample


def generate_dataset(num_samples: int, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    dataset = []
    for idx in range(num_samples):
        sample = generate_one_sample(config)
        dataset.append(sample)

        if (idx + 1) % 500 == 0:
            print(f"Generated {idx + 1}/{num_samples} samples")
    return dataset


def save_dataset(dataset: List[Dict[str, Any]], path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(dataset, f)


# =========================================================
# Stats / Debug
# =========================================================

def summarize_dataset(dataset: List[Dict[str, Any]], split_name: str) -> None:
    num_samples = len(dataset)
    one_event = 0
    two_event = 0
    independent_two_event = 0
    event_type_counts = {}

    for sample in dataset:
        events = sample["events"]
        if len(events) == 1:
            one_event += 1
        elif len(events) == 2:
            two_event += 1
            if len(sample["independent_pairs"]) > 0:
                independent_two_event += 1

        for e in events:
            et = e["event_type"]
            event_type_counts[et] = event_type_counts.get(et, 0) + 1

    print("=" * 60)
    print(f"Summary for {split_name}")
    print(f"Num samples: {num_samples}")
    print(f"One-event samples: {one_event}")
    print(f"Two-event samples: {two_event}")
    print(f"Independent two-event samples: {independent_two_event}")
    print("Event type counts:")
    for k, v in sorted(event_type_counts.items()):
        print(f"  {k}: {v}")
    print("=" * 60)


def inspect_sample(sample: Dict[str, Any], idx: int = 0) -> None:
    print(f"\n===== Inspect Sample {idx} =====")
    print("graph_t node_features shape:", sample["graph_t"]["node_features"].shape)
    print("graph_t adj shape:", sample["graph_t"]["adj"].shape)
    print("graph_t1 node_features shape:", sample["graph_t1"]["node_features"].shape)
    print("events:")
    for i, e in enumerate(sample["events"]):
        print(f"  Event {i}: {e}")
    print("independent_pairs:", sample["independent_pairs"])
    print("changed_nodes:", sample["changed_nodes"].astype(int).tolist())
    print("changed_edges:", sample["changed_edges"])
    print("event_scope_union_nodes:", sample["event_scope_union_nodes"])
    print("event_scope_union_edges:", sample["event_scope_union_edges"])
    print("===============================\n")


# =========================================================
# Main
# =========================================================

def main(config: Dict[str, Any]) -> None:
    os.makedirs(config["output_dir"], exist_ok=True)
    set_seed(config["random_seed"])

    print("Generating train dataset...")
    train_dataset = generate_dataset(config["num_train"], config)
    train_path = os.path.join(config["output_dir"], config["train_file"])
    save_dataset(train_dataset, train_path)
    summarize_dataset(train_dataset, "train")
    inspect_sample(train_dataset[0], idx=0)

    print("Generating val dataset...")
    val_dataset = generate_dataset(config["num_val"], config)
    val_path = os.path.join(config["output_dir"], config["val_file"])
    save_dataset(val_dataset, val_path)
    summarize_dataset(val_dataset, "val")
    inspect_sample(val_dataset[0], idx=0)

    print("Generating test dataset...")
    test_dataset = generate_dataset(config["num_test"], config)
    test_path = os.path.join(config["output_dir"], config["test_file"])
    save_dataset(test_dataset, test_path)
    summarize_dataset(test_dataset, "test")
    inspect_sample(test_dataset[0], idx=0)

    print("Done.")
    print(f"Saved train to: {train_path}")
    print(f"Saved val   to: {val_path}")
    print(f"Saved test  to: {test_path}")


if __name__ == "__main__":
    main(DEFAULT_CONFIG)