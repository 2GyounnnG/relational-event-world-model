# data/generate_graph_event_data.py

import argparse
import hashlib
import json
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
    "step3_matched_test_pairs": 1000,
    "step3_matched_test_file": "graph_event_step3_matched_test.pkl",
    "step3_sequential_test_pairs": 500,
    "step3_sequential_test_file": "graph_event_step3_sequential_test.pkl",
    "rollout_test_file": "graph_event_rollout_test.pkl",
    "rollout_num_samples_per_horizon": {
        2: 200,
        3: 200,
        5: 200,
    },
    "step5_test_file": "graph_event_step5_test.pkl",
    "step5_val_file": "graph_event_step5_val.pkl",
    "step5_train_file": "graph_event_step5_train.pkl",
    "step5_sequence_length": 3,
    "step5_train_num_samples_per_bucket": {
        "fully_independent": 300,
        "partially_dependent": 300,
        "strongly_interacting": 300,
    },
    "step5_test_num_samples_per_bucket": {
        "fully_independent": 200,
        "partially_dependent": 200,
        "strongly_interacting": 200,
    },
    "step5_val_num_samples_per_bucket": {
        "fully_independent": 100,
        "partially_dependent": 100,
        "strongly_interacting": 100,
    },
    "step6a_test_file": "graph_event_step6a_test.pkl",
    "step6a_val_file": "graph_event_step6a_val.pkl",
    "step6a_train_file": "graph_event_step6a_train.pkl",
    "step6a_corruption_settings": {
        "N1": {
            "node_state_noise_std": 0.05,
            "edge_dropout_prob": 0.05,
            "edge_false_positive_prob": 0.02,
        },
        "N2": {
            "node_state_noise_std": 0.10,
            "edge_dropout_prob": 0.10,
            "edge_false_positive_prob": 0.05,
        },
        "N3": {
            "node_state_noise_std": 0.20,
            "edge_dropout_prob": 0.15,
            "edge_false_positive_prob": 0.08,
        },
    },
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


def graph_state_hash(graph: Dict[str, Any]) -> str:
    h = hashlib.sha1()
    h.update(np.asarray(graph["node_features"]).tobytes())
    h.update(np.asarray(graph["adj"]).tobytes())
    return h.hexdigest()


def event_type_list(events: List[Dict[str, Any]]) -> List[str]:
    return [str(e["event_type"]) for e in events]


def ordered_signature(events: List[Dict[str, Any]]) -> str:
    return "->".join(event_type_list(events))


def unordered_signature(events: List[Dict[str, Any]]) -> str:
    return "+".join(sorted(event_type_list(events)))


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


def event_is_valid_for_graph(
    graph: Dict[str, Any],
    event: Dict[str, Any],
    num_types: int,
) -> bool:
    adj = graph["adj"]
    node_features = graph["node_features"]
    n = node_features.shape[0]
    node_scope = list(event.get("node_scope", []))
    event_type = str(event["event_type"])

    if event_type == "node_state_update":
        if len(node_scope) < 2:
            return False
        center = int(node_scope[0])
        if center < 0 or center >= n:
            return False
        neighbors = get_neighbors(adj, center)
        return len(neighbors) > 0 and sorted(node_scope[1:]) == sorted(neighbors)

    if event_type == "edge_add":
        if len(node_scope) != 2:
            return False
        i, j = int(node_scope[0]), int(node_scope[1])
        if i == j or min(i, j) < 0 or max(i, j) >= n:
            return False
        if adj[i, j] != 0:
            return False
        if not two_hop_reachable(adj, i, j):
            return False
        dist = np.linalg.norm(node_features[i, 1:] - node_features[j, 1:])
        return bool(dist < 2.5)

    if event_type == "edge_delete":
        if len(node_scope) != 2:
            return False
        i, j = int(node_scope[0]), int(node_scope[1])
        if i == j or min(i, j) < 0 or max(i, j) >= n:
            return False
        if adj[i, j] != 1:
            return False
        dist = np.linalg.norm(node_features[i, 1:] - node_features[j, 1:])
        return bool(dist > 2.0)

    if event_type == "motif_type_flip":
        if len(node_scope) != 3:
            return False
        i, j, k = [int(x) for x in node_scope]
        if min(i, j, k) < 0 or max(i, j, k) >= n or len({i, j, k}) < 3:
            return False
        if not (adj[i, j] == 1 and adj[i, k] == 1):
            return False
        type_i = int(node_features[i, 0])
        type_j = int(node_features[j, 0])
        type_k = int(node_features[k, 0])
        return type_j == type_k and type_i != type_j

    raise ValueError(f"Unknown event_type for validity check: {event_type}")


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
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    changed_nodes = compute_changed_nodes(graph_t, graph_t1)
    changed_edges = compute_changed_edges(graph_t, graph_t1)
    independent_pairs = find_independent_pairs(events)

    event_scope_union_nodes = compute_event_scope_union_nodes(events)
    event_scope_union_edges = compute_event_scope_union_edges(events)

    sample = {
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
    if extra_metadata:
        sample.update(copy.deepcopy(extra_metadata))
    return sample


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


def generate_one_independent_two_event_base(config: Dict[str, Any]) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    while True:
        graph_t = generate_initial_graph(
            num_nodes_range=config["num_nodes_range"],
            edge_prob=config["edge_prob"],
            state_dim=config["state_dim"],
            num_types=config["num_types"],
        )
        events = sample_events(
            graph=graph_t,
            max_events=2,
            prob_two_events=1.0,
            num_types=config["num_types"],
        )
        if len(events) != 2:
            continue
        if (0, 1) not in find_independent_pairs(events):
            continue
        return graph_t, events


def make_step3_matched_pair_samples(
    graph_t: Dict[str, Any],
    events: List[Dict[str, Any]],
    pair_idx: int,
    num_types: int,
    pair_id_prefix: str = "step3_pair",
) -> List[Dict[str, Any]]:
    if len(events) != 2:
        raise ValueError("Matched Step 3 pair generation requires exactly two events")
    if (0, 1) not in find_independent_pairs(events):
        raise ValueError("Matched Step 3 pair generation requires an independent two-event pair")

    base_graph_id = graph_state_hash(graph_t)
    pair_id = f"{pair_id_prefix}_{pair_idx:06d}"
    event_specs = copy.deepcopy(events)
    variants = [
        ("A_then_B", copy.deepcopy(events)),
        ("B_then_A", copy.deepcopy(list(reversed(events)))),
    ]

    samples: List[Dict[str, Any]] = []
    for ordered_variant, ordered_events in variants:
        graph_t1 = apply_events(
            graph=graph_t,
            events=ordered_events,
            num_types=num_types,
        )
        samples.append(
            make_sample(
                graph_t=graph_t,
                graph_t1=graph_t1,
                events=ordered_events,
                extra_metadata={
                    "step3_pair_id": pair_id,
                    "step3_ordered_variant": ordered_variant,
                    "step3_ordered_signature": ordered_signature(ordered_events),
                    "step3_unordered_signature": unordered_signature(ordered_events),
                    "step3_base_graph_id": base_graph_id,
                    "step3_event_specs": event_specs,
                },
            )
        )
    return samples


def generate_step3_matched_dataset(num_pairs: int, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    dataset: List[Dict[str, Any]] = []
    for pair_idx in range(num_pairs):
        graph_t, events = generate_one_independent_two_event_base(config)
        dataset.extend(
            make_step3_matched_pair_samples(
                graph_t,
                events,
                pair_idx=pair_idx,
                num_types=config["num_types"],
            )
        )
        if (pair_idx + 1) % 100 == 0:
            print(f"Generated {pair_idx + 1}/{num_pairs} exact matched Step 3 pairs")
    return dataset


def generate_step3_sequential_dataset(num_pairs: int, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    dataset: List[Dict[str, Any]] = []
    for pair_idx in range(num_pairs):
        graph_t, events = generate_one_independent_two_event_base(config)
        event_a = copy.deepcopy(events[0])
        event_b = copy.deepcopy(events[1])

        graph_a = apply_events(graph=graph_t, events=[copy.deepcopy(event_a)], num_types=config["num_types"])
        graph_b = apply_events(graph=graph_t, events=[copy.deepcopy(event_b)], num_types=config["num_types"])
        graph_ab = apply_events(
            graph=graph_t,
            events=[copy.deepcopy(event_a), copy.deepcopy(event_b)],
            num_types=config["num_types"],
        )
        graph_ba = apply_events(
            graph=graph_t,
            events=[copy.deepcopy(event_b), copy.deepcopy(event_a)],
            num_types=config["num_types"],
        )

        if not np.array_equal(graph_ab["adj"], graph_ba["adj"]):
            raise ValueError("Independent two-event sequential dataset expected commutative final adjacency")
        if not np.allclose(graph_ab["node_features"], graph_ba["node_features"]):
            raise ValueError("Independent two-event sequential dataset expected commutative final node features")

        pair_id = f"step3_seq_pair_{pair_idx:06d}"
        base_graph_id = graph_state_hash(graph_t)
        pair_event_specs = [copy.deepcopy(event_a), copy.deepcopy(event_b)]
        shared_meta = {
            "step3_pair_id": pair_id,
            "step3_base_graph_id": base_graph_id,
            "step3_unordered_signature": unordered_signature(pair_event_specs),
            "step3_pair_event_specs": copy.deepcopy(pair_event_specs),
        }

        transitions = [
            (
                "base_to_A",
                graph_t,
                graph_a,
                [copy.deepcopy(event_a)],
                {"step3_primary_event_index": 0, "step3_primary_event_type": event_a["event_type"]},
            ),
            (
                "base_to_B",
                graph_t,
                graph_b,
                [copy.deepcopy(event_b)],
                {"step3_primary_event_index": 1, "step3_primary_event_type": event_b["event_type"]},
            ),
            (
                "A_to_AB",
                graph_a,
                graph_ab,
                [copy.deepcopy(event_b)],
                {"step3_primary_event_index": 1, "step3_primary_event_type": event_b["event_type"]},
            ),
            (
                "B_to_AB",
                graph_b,
                graph_ab,
                [copy.deepcopy(event_a)],
                {"step3_primary_event_index": 0, "step3_primary_event_type": event_a["event_type"]},
            ),
        ]

        for role, graph_src, graph_dst, transition_events, role_meta in transitions:
            dataset.append(
                make_sample(
                    graph_t=graph_src,
                    graph_t1=graph_dst,
                    events=transition_events,
                    extra_metadata={
                        **shared_meta,
                        **role_meta,
                        "step3_transition_role": role,
                    },
                )
            )

        if (pair_idx + 1) % 100 == 0:
            print(f"Generated {pair_idx + 1}/{num_pairs} sequential Step 3 pairs")

    return dataset


def generate_one_rollout_sample(horizon: int, config: Dict[str, Any], rollout_idx: int) -> Dict[str, Any]:
    while True:
        graph_0 = generate_initial_graph(
            num_nodes_range=config["num_nodes_range"],
            edge_prob=config["edge_prob"],
            state_dim=config["state_dim"],
            num_types=config["num_types"],
        )

        graph_cur = copy.deepcopy(graph_0)
        events: List[Dict[str, Any]] = []
        graph_steps: List[Dict[str, Any]] = []
        transition_samples: List[Dict[str, Any]] = []
        generation_ok = True

        for _ in range(horizon):
            sampled = sample_events(
                graph=graph_cur,
                max_events=1,
                prob_two_events=0.0,
                num_types=config["num_types"],
            )
            if len(sampled) != 1:
                generation_ok = False
                break

            event = copy.deepcopy(sampled[0])
            graph_next = apply_events(
                graph=graph_cur,
                events=[event],
                num_types=config["num_types"],
            )
            events.append(copy.deepcopy(event))
            graph_steps.append(
                {
                    "node_features": graph_next["node_features"].copy(),
                    "adj": graph_next["adj"].copy(),
                    "edge_index": graph_to_edge_index(graph_next["adj"]),
                }
            )
            transition_samples.append(
                make_sample(
                    graph_t=graph_cur,
                    graph_t1=graph_next,
                    events=[copy.deepcopy(event)],
                )
            )
            graph_cur = graph_next

        if not generation_ok:
            continue

        return {
            "rollout_id": f"rollout_{rollout_idx:06d}",
            "horizon": horizon,
            "graph_0": {
                "node_features": graph_0["node_features"].copy(),
                "adj": graph_0["adj"].copy(),
                "edge_index": graph_to_edge_index(graph_0["adj"]),
            },
            "graph_steps": graph_steps,
            "transition_samples": transition_samples,
            "events": copy.deepcopy(events),
            "event_type_sequence": event_type_list(events),
            "rollout_base_graph_id": graph_state_hash(graph_0),
        }


def generate_rollout_dataset(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    dataset: List[Dict[str, Any]] = []
    rollout_idx = 0
    rollout_counts = config["rollout_num_samples_per_horizon"]
    for horizon in sorted(rollout_counts.keys()):
        num_samples = int(rollout_counts[horizon])
        for sample_idx in range(num_samples):
            dataset.append(generate_one_rollout_sample(horizon, config, rollout_idx))
            rollout_idx += 1
            if (sample_idx + 1) % 100 == 0:
                print(
                    f"Generated {sample_idx + 1}/{num_samples} rollout samples for horizon {horizon}"
                )
    return dataset


def sample_single_event(graph: Dict[str, Any], num_types: int) -> Optional[Dict[str, Any]]:
    events = sample_events(
        graph=graph,
        max_events=1,
        prob_two_events=0.0,
        num_types=num_types,
    )
    if len(events) != 1:
        return None
    return copy.deepcopy(events[0])


def sample_non_overlapping_event(
    graph: Dict[str, Any],
    existing_events: List[Dict[str, Any]],
    num_types: int,
    max_tries: int = 64,
) -> Optional[Dict[str, Any]]:
    for _ in range(max_tries):
        candidate = sample_single_event(graph, num_types)
        if candidate is None:
            continue
        if all(not scopes_overlap(candidate, existing) for existing in existing_events):
            return candidate
    return None


def sample_event_matching(
    graph: Dict[str, Any],
    num_types: int,
    predicate,
    max_tries: int = 128,
) -> Optional[Dict[str, Any]]:
    for _ in range(max_tries):
        candidate = sample_single_event(graph, num_types)
        if candidate is None:
            continue
        if predicate(candidate):
            return candidate
    return None


def scope_overlap_pairs(events: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    for i in range(len(events)):
        for j in range(i + 1, len(events)):
            if scopes_overlap(events[i], events[j]):
                pairs.append((i, j))
    return pairs


def build_step5_sample(
    graph_0: Dict[str, Any],
    events: List[Dict[str, Any]],
    dependency_bucket: str,
    dependency_reason: str,
    num_types: int,
    sample_idx: int,
) -> Dict[str, Any]:
    graph_steps: List[Dict[str, Any]] = []
    transition_samples: List[Dict[str, Any]] = []
    current_graph = copy.deepcopy(graph_0)

    for event in events:
        next_graph = apply_events(
            graph=current_graph,
            events=[copy.deepcopy(event)],
            num_types=num_types,
        )
        graph_steps.append(
            {
                "node_features": next_graph["node_features"].copy(),
                "adj": next_graph["adj"].copy(),
                "edge_index": graph_to_edge_index(next_graph["adj"]),
            }
        )
        transition_samples.append(
            make_sample(
                graph_t=current_graph,
                graph_t1=next_graph,
                events=[copy.deepcopy(event)],
            )
        )
        current_graph = next_graph

    valid_on_base = [event_is_valid_for_graph(graph_0, event, num_types) for event in events]
    overlap_pairs = scope_overlap_pairs(events)
    return {
        "step5_sample_id": f"step5_{sample_idx:06d}",
        "horizon": len(events),
        "graph_0": {
            "node_features": graph_0["node_features"].copy(),
            "adj": graph_0["adj"].copy(),
            "edge_index": graph_to_edge_index(graph_0["adj"]),
        },
        "graph_steps": graph_steps,
        "transition_samples": transition_samples,
        "events": copy.deepcopy(events),
        "event_type_sequence": event_type_list(events),
        "rollout_base_graph_id": graph_state_hash(graph_0),
        "step5_ordered_signature": ordered_signature(events),
        "step5_unordered_signature": unordered_signature(events),
        "step5_dependency_bucket": dependency_bucket,
        "step5_dependency_reason": dependency_reason,
        "step5_pairwise_scope_overlaps": overlap_pairs,
        "step5_event_valid_on_base": valid_on_base,
    }


def generate_step5_fully_independent_sequence(config: Dict[str, Any], sample_idx: int) -> Dict[str, Any]:
    while True:
        graph_0 = generate_initial_graph(
            num_nodes_range=config["num_nodes_range"],
            edge_prob=config["edge_prob"],
            state_dim=config["state_dim"],
            num_types=config["num_types"],
        )
        event_a = sample_single_event(graph_0, config["num_types"])
        if event_a is None:
            continue
        event_b = sample_non_overlapping_event(graph_0, [event_a], config["num_types"])
        if event_b is None:
            continue
        event_c = sample_non_overlapping_event(graph_0, [event_a, event_b], config["num_types"])
        if event_c is None:
            continue
        events = [event_a, event_b, event_c]
        return build_step5_sample(
            graph_0=graph_0,
            events=events,
            dependency_bucket="fully_independent",
            dependency_reason="all_three_events_sampled_from_base_with_pairwise_disjoint_scopes",
            num_types=config["num_types"],
            sample_idx=sample_idx,
        )


def generate_step5_partially_dependent_sequence(config: Dict[str, Any], sample_idx: int) -> Dict[str, Any]:
    while True:
        graph_0 = generate_initial_graph(
            num_nodes_range=config["num_nodes_range"],
            edge_prob=config["edge_prob"],
            state_dim=config["state_dim"],
            num_types=config["num_types"],
        )
        event_a = sample_single_event(graph_0, config["num_types"])
        if event_a is None:
            continue
        event_b = sample_non_overlapping_event(graph_0, [event_a], config["num_types"])
        if event_b is None:
            continue
        graph_after_a = apply_events(graph_0, [copy.deepcopy(event_a)], config["num_types"])
        if not event_is_valid_for_graph(graph_after_a, event_b, config["num_types"]):
            continue

        def partial_predicate(event_c: Dict[str, Any]) -> bool:
            overlaps_a = scopes_overlap(event_c, event_a)
            overlaps_b = scopes_overlap(event_c, event_b)
            valid_on_base = event_is_valid_for_graph(graph_0, event_c, config["num_types"])
            return (overlaps_a ^ overlaps_b) or ((overlaps_a or overlaps_b) and not valid_on_base)

        event_c = sample_event_matching(graph_after_a, config["num_types"], partial_predicate)
        if event_c is None:
            continue
        graph_after_ab = apply_events(graph_after_a, [copy.deepcopy(event_b)], config["num_types"])
        if not event_is_valid_for_graph(graph_after_ab, event_c, config["num_types"]):
            continue
        events = [event_a, event_b, event_c]
        return build_step5_sample(
            graph_0=graph_0,
            events=events,
            dependency_bucket="partially_dependent",
            dependency_reason="third_event_sampled_after_first_event_and coupled_to_one prior event",
            num_types=config["num_types"],
            sample_idx=sample_idx,
        )


def generate_step5_strongly_interacting_sequence(config: Dict[str, Any], sample_idx: int) -> Dict[str, Any]:
    while True:
        graph_0 = generate_initial_graph(
            num_nodes_range=config["num_nodes_range"],
            edge_prob=config["edge_prob"],
            state_dim=config["state_dim"],
            num_types=config["num_types"],
        )
        event_a = sample_single_event(graph_0, config["num_types"])
        if event_a is None:
            continue
        graph_after_a = apply_events(graph_0, [copy.deepcopy(event_a)], config["num_types"])

        def strong_second_predicate(event_b: Dict[str, Any]) -> bool:
            return scopes_overlap(event_b, event_a) or (not event_is_valid_for_graph(graph_0, event_b, config["num_types"]))

        event_b = sample_event_matching(graph_after_a, config["num_types"], strong_second_predicate)
        if event_b is None:
            continue
        graph_after_ab = apply_events(graph_after_a, [copy.deepcopy(event_b)], config["num_types"])

        def strong_third_predicate(event_c: Dict[str, Any]) -> bool:
            overlap_count = int(scopes_overlap(event_c, event_a)) + int(scopes_overlap(event_c, event_b))
            invalid_on_base = not event_is_valid_for_graph(graph_0, event_c, config["num_types"])
            return overlap_count >= 1 and invalid_on_base

        event_c = sample_event_matching(graph_after_ab, config["num_types"], strong_third_predicate)
        if event_c is None:
            continue
        events = [event_a, event_b, event_c]
        return build_step5_sample(
            graph_0=graph_0,
            events=events,
            dependency_bucket="strongly_interacting",
            dependency_reason="later_events depend on evolving graph and are not valid on base state",
            num_types=config["num_types"],
            sample_idx=sample_idx,
        )


def generate_step5_dataset(
    num_samples_per_bucket: Dict[str, int],
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    generators = {
        "fully_independent": generate_step5_fully_independent_sequence,
        "partially_dependent": generate_step5_partially_dependent_sequence,
        "strongly_interacting": generate_step5_strongly_interacting_sequence,
    }
    dataset: List[Dict[str, Any]] = []
    sample_idx = 0
    for bucket_name in ["fully_independent", "partially_dependent", "strongly_interacting"]:
        num_samples = int(num_samples_per_bucket[bucket_name])
        generator_fn = generators[bucket_name]
        for bucket_idx in range(num_samples):
            dataset.append(generator_fn(config, sample_idx))
            sample_idx += 1
            if (bucket_idx + 1) % 50 == 0:
                print(f"Generated {bucket_idx + 1}/{num_samples} Step 5 samples for bucket {bucket_name}")
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


def summarize_rollout_dataset(dataset: List[Dict[str, Any]], split_name: str) -> None:
    horizon_counts: Dict[int, int] = {}
    for sample in dataset:
        horizon = int(sample["horizon"])
        horizon_counts[horizon] = horizon_counts.get(horizon, 0) + 1

    print("=" * 60)
    print(f"Summary for {split_name}")
    print(f"Num rollout samples: {len(dataset)}")
    print("Horizon distribution:")
    for horizon in sorted(horizon_counts.keys()):
        print(f"  T={horizon}: {horizon_counts[horizon]}")
    print("=" * 60)


def inspect_rollout_sample(sample: Dict[str, Any], idx: int = 0) -> None:
    print(f"\n===== Inspect Rollout Sample {idx} =====")
    print("rollout_id:", sample["rollout_id"])
    print("horizon:", sample["horizon"])
    print("graph_0 node_features shape:", sample["graph_0"]["node_features"].shape)
    print("graph_0 adj shape:", sample["graph_0"]["adj"].shape)
    print("event_type_sequence:", sample.get("event_type_sequence"))
    print("events:")
    for i, event in enumerate(sample["events"]):
        print(f"  Step {i + 1}: {event}")
    print("num graph_steps:", len(sample["graph_steps"]))
    print("rollout_base_graph_id:", sample.get("rollout_base_graph_id"))
    print("===============================\n")


def summarize_step5_dataset(dataset: List[Dict[str, Any]], split_name: str) -> None:
    bucket_counts: Dict[str, int] = {}
    for sample in dataset:
        bucket = str(sample["step5_dependency_bucket"])
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    print("=" * 60)
    print(f"Summary for {split_name}")
    print(f"Num Step 5 samples: {len(dataset)}")
    print(f"Sequence length: {dataset[0]['horizon'] if dataset else 'NA'}")
    print("Dependency bucket counts:")
    for bucket in ["fully_independent", "partially_dependent", "strongly_interacting"]:
        print(f"  {bucket}: {bucket_counts.get(bucket, 0)}")
    print("=" * 60)


def inspect_step5_sample(sample: Dict[str, Any], idx: int = 0) -> None:
    print(f"\n===== Inspect Step5 Sample {idx} =====")
    print("step5_sample_id:", sample["step5_sample_id"])
    print("horizon:", sample["horizon"])
    print("ordered signature:", sample["step5_ordered_signature"])
    print("unordered signature:", sample["step5_unordered_signature"])
    print("dependency bucket:", sample["step5_dependency_bucket"])
    print("dependency reason:", sample["step5_dependency_reason"])
    print("pairwise scope overlaps:", sample["step5_pairwise_scope_overlaps"])
    print("event valid on base:", sample["step5_event_valid_on_base"])
    print("event_type_sequence:", sample["event_type_sequence"])
    print("===============================\n")


def load_dataset(path: str) -> List[Dict[str, Any]]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"Dataset is empty or malformed: {path}")
    return data


def corrupt_graph_observation(
    graph: Dict[str, Any],
    corruption_cfg: Dict[str, float],
    rng: np.random.Generator,
) -> Dict[str, Any]:
    node_features = np.asarray(graph["node_features"], dtype=np.float32).copy()
    adj = np.asarray(graph["adj"], dtype=np.int64).copy()
    num_nodes = node_features.shape[0]

    noise_std = float(corruption_cfg["node_state_noise_std"])
    if node_features.shape[1] > 1 and noise_std > 0:
        node_features[:, 1:] += rng.normal(
            loc=0.0,
            scale=noise_std,
            size=node_features[:, 1:].shape,
        ).astype(np.float32)

    dropout_prob = float(corruption_cfg["edge_dropout_prob"])
    false_positive_prob = float(corruption_cfg["edge_false_positive_prob"])
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adj[i, j] == 1:
                if rng.random() < dropout_prob:
                    adj[i, j] = 0
                    adj[j, i] = 0
            else:
                if rng.random() < false_positive_prob:
                    adj[i, j] = 1
                    adj[j, i] = 1

    np.fill_diagonal(adj, 0)
    return {
        "node_features": node_features,
        "adj": adj,
    }


def generate_step6a_dataset_from_clean(
    clean_dataset: List[Dict[str, Any]],
    corruption_settings: Dict[str, Dict[str, float]],
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    dataset: List[Dict[str, Any]] = []
    base_seed = int(config["random_seed"])

    for sample_idx, raw_sample in enumerate(clean_dataset):
        for setting_idx, setting_name in enumerate(sorted(corruption_settings.keys())):
            rng = np.random.default_rng(base_seed + sample_idx * 101 + setting_idx * 10007)
            noisy_sample = copy.deepcopy(raw_sample)
            noisy_sample["obs_graph_t"] = corrupt_graph_observation(
                raw_sample["graph_t"],
                corruption_settings[setting_name],
                rng,
            )
            noisy_sample["step6a_corruption_setting"] = setting_name
            noisy_sample["step6a_corruption_config"] = dict(corruption_settings[setting_name])
            noisy_sample["step6a_source_sample_index"] = sample_idx
            dataset.append(noisy_sample)

    return dataset


def summarize_step6a_dataset(dataset: List[Dict[str, Any]], split_name: str) -> None:
    setting_counts: Dict[str, int] = {}
    for sample in dataset:
        setting_name = str(sample.get("step6a_corruption_setting", "unknown"))
        setting_counts[setting_name] = setting_counts.get(setting_name, 0) + 1

    print("=" * 60)
    print(f"Summary for {split_name}")
    print(f"Num Step 6a samples: {len(dataset)}")
    print("Corruption setting counts:")
    for setting_name in sorted(setting_counts.keys()):
        print(f"  {setting_name}: {setting_counts[setting_name]}")
    print("=" * 60)


def inspect_step6a_sample(sample: Dict[str, Any], idx: int = 0) -> None:
    print(f"\n===== Inspect Step6a Sample {idx} =====")
    print("corruption setting:", sample.get("step6a_corruption_setting"))
    print("corruption config:", sample.get("step6a_corruption_config"))
    print("source sample index:", sample.get("step6a_source_sample_index"))
    print("events:", sample.get("events"))
    print("obs_graph_t keys:", sorted(sample.get("obs_graph_t", {}).keys()))
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate_step3_matched_test", action="store_true")
    parser.add_argument("--generate_step3_sequential_test", action="store_true")
    parser.add_argument("--generate_rollout_test", action="store_true")
    parser.add_argument("--generate_step6a_train", action="store_true")
    parser.add_argument("--generate_step6a_test", action="store_true")
    parser.add_argument("--generate_step6a_val", action="store_true")
    parser.add_argument("--generate_step5_train", action="store_true")
    parser.add_argument("--generate_step5_test", action="store_true")
    parser.add_argument("--generate_step5_val", action="store_true")
    parser.add_argument("--step3_pairs", type=int, default=DEFAULT_CONFIG["step3_matched_test_pairs"])
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--random_seed", type=int, default=DEFAULT_CONFIG["random_seed"])
    args = parser.parse_args()

    config = copy.deepcopy(DEFAULT_CONFIG)
    config["random_seed"] = args.random_seed

    if args.generate_step3_matched_test:
        os.makedirs(config["output_dir"], exist_ok=True)
        set_seed(config["random_seed"])
        dataset = generate_step3_matched_dataset(args.step3_pairs, config)
        output_path = args.output_path or os.path.join(config["output_dir"], config["step3_matched_test_file"])
        save_dataset(dataset, output_path)
        summarize_dataset(dataset, "step3_matched_test")
        inspect_sample(dataset[0], idx=0)
        print(f"Saved matched Step 3 test set to: {output_path}")
    elif args.generate_step3_sequential_test:
        os.makedirs(config["output_dir"], exist_ok=True)
        set_seed(config["random_seed"])
        dataset = generate_step3_sequential_dataset(args.step3_pairs, config)
        output_path = args.output_path or os.path.join(config["output_dir"], config["step3_sequential_test_file"])
        save_dataset(dataset, output_path)
        summarize_dataset(dataset, "step3_sequential_test")
        inspect_sample(dataset[0], idx=0)
        print(f"Saved sequential Step 3 test set to: {output_path}")
    elif args.generate_rollout_test:
        os.makedirs(config["output_dir"], exist_ok=True)
        set_seed(config["random_seed"])
        dataset = generate_rollout_dataset(config)
        output_path = args.output_path or os.path.join(config["output_dir"], config["rollout_test_file"])
        save_dataset(dataset, output_path)
        summarize_rollout_dataset(dataset, "rollout_test")
        inspect_rollout_sample(dataset[0], idx=0)
        print(f"Saved rollout test set to: {output_path}")
    elif args.generate_step6a_test:
        os.makedirs(config["output_dir"], exist_ok=True)
        clean_path = os.path.join(config["output_dir"], config["test_file"])
        clean_dataset = load_dataset(clean_path)
        dataset = generate_step6a_dataset_from_clean(
            clean_dataset,
            config["step6a_corruption_settings"],
            config,
        )
        output_path = args.output_path or os.path.join(config["output_dir"], config["step6a_test_file"])
        save_dataset(dataset, output_path)
        summarize_step6a_dataset(dataset, "step6a_test")
        inspect_step6a_sample(dataset[0], idx=0)
        print(f"Saved Step 6a test set to: {output_path}")
    elif args.generate_step6a_train:
        os.makedirs(config["output_dir"], exist_ok=True)
        clean_path = os.path.join(config["output_dir"], config["train_file"])
        clean_dataset = load_dataset(clean_path)
        dataset = generate_step6a_dataset_from_clean(
            clean_dataset,
            config["step6a_corruption_settings"],
            config,
        )
        output_path = args.output_path or os.path.join(config["output_dir"], config["step6a_train_file"])
        save_dataset(dataset, output_path)
        summarize_step6a_dataset(dataset, "step6a_train")
        inspect_step6a_sample(dataset[0], idx=0)
        print(f"Saved Step 6a train set to: {output_path}")
    elif args.generate_step6a_val:
        os.makedirs(config["output_dir"], exist_ok=True)
        clean_path = os.path.join(config["output_dir"], config["val_file"])
        clean_dataset = load_dataset(clean_path)
        dataset = generate_step6a_dataset_from_clean(
            clean_dataset,
            config["step6a_corruption_settings"],
            config,
        )
        output_path = args.output_path or os.path.join(config["output_dir"], config["step6a_val_file"])
        save_dataset(dataset, output_path)
        summarize_step6a_dataset(dataset, "step6a_val")
        inspect_step6a_sample(dataset[0], idx=0)
        print(f"Saved Step 6a val set to: {output_path}")
    elif args.generate_step5_test:
        os.makedirs(config["output_dir"], exist_ok=True)
        set_seed(config["random_seed"])
        dataset = generate_step5_dataset(config["step5_test_num_samples_per_bucket"], config)
        output_path = args.output_path or os.path.join(config["output_dir"], config["step5_test_file"])
        save_dataset(dataset, output_path)
        summarize_step5_dataset(dataset, "step5_test")
        inspect_step5_sample(dataset[0], idx=0)
        print(f"Saved Step 5 test set to: {output_path}")
    elif args.generate_step5_train:
        os.makedirs(config["output_dir"], exist_ok=True)
        set_seed(config["random_seed"])
        dataset = generate_step5_dataset(config["step5_train_num_samples_per_bucket"], config)
        output_path = args.output_path or os.path.join(config["output_dir"], config["step5_train_file"])
        save_dataset(dataset, output_path)
        summarize_step5_dataset(dataset, "step5_train")
        inspect_step5_sample(dataset[0], idx=0)
        print(f"Saved Step 5 train set to: {output_path}")
    elif args.generate_step5_val:
        os.makedirs(config["output_dir"], exist_ok=True)
        set_seed(config["random_seed"])
        dataset = generate_step5_dataset(config["step5_val_num_samples_per_bucket"], config)
        output_path = args.output_path or os.path.join(config["output_dir"], config["step5_val_file"])
        save_dataset(dataset, output_path)
        summarize_step5_dataset(dataset, "step5_val")
        inspect_step5_sample(dataset[0], idx=0)
        print(f"Saved Step 5 val set to: {output_path}")
    else:
        main(config)
