from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.generate_sandbox_local_event_mvp import (  # noqa: E402
    DEFAULT_CHANGED_TOLERANCE,
    EDGE_FEATURE_NAMES,
    EVENT_TYPES,
    NODE_FEATURE_NAMES,
    changed_masks,
    edge_distances,
    make_edge_features,
    make_event_scope,
    make_node_features,
    rollout_one_step,
    sample_positions,
)


def resolve_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def save_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def make_chain_edge_index(num_nodes: int) -> np.ndarray:
    edges = [(node, node + 1) for node in range(num_nodes - 1)]
    return np.asarray(edges, dtype=np.int64).T


def make_base_world(num_nodes: int, rng: np.random.Generator) -> Dict[str, Any]:
    edge_index = make_chain_edge_index(num_nodes)
    num_edges = int(edge_index.shape[1])
    position = sample_positions(num_nodes, rng)
    velocity = np.zeros((num_nodes, 2), dtype=np.float32)
    mass = rng.uniform(0.8, 1.4, size=num_nodes).astype(np.float32)
    radius = rng.uniform(0.025, 0.045, size=num_nodes).astype(np.float32)
    pinned = np.zeros(num_nodes, dtype=np.float32)
    if rng.random() < 0.2:
        pinned[int(rng.integers(0, num_nodes))] = 1.0

    current_distance = edge_distances(position, edge_index)
    rest_length = (current_distance * rng.uniform(0.92, 1.08, size=num_edges)).astype(np.float32)
    stiffness = rng.uniform(0.8, 2.2, size=num_edges).astype(np.float32)
    return {
        "position": position,
        "velocity": velocity,
        "mass": mass,
        "radius": radius,
        "pinned": pinned,
        "edge_index": edge_index,
        "spring_active": np.ones(num_edges, dtype=np.float32),
        "rest_length": rest_length,
        "stiffness": stiffness,
    }


def copy_world(world: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in world.items()}


def world_features(world: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    node_features = make_node_features(
        position=world["position"],
        velocity=world["velocity"],
        mass=world["mass"],
        radius=world["radius"],
        pinned=world["pinned"],
    )
    edge_features = make_edge_features(
        spring_active=world["spring_active"],
        rest_length=world["rest_length"],
        stiffness=world["stiffness"],
        position=world["position"],
        edge_index=world["edge_index"],
    )
    return node_features.astype(np.float32), edge_features.astype(np.float32)


def pair_type(event_a: Dict[str, Any], event_b: Dict[str, Any]) -> str:
    names = {event_a["event_type"], event_b["event_type"]}
    if names == {"node_impulse"}:
        return "impulse+impulse"
    if names == {"spring_break"}:
        return "break+break"
    return "impulse+break"


def make_event(world: Dict[str, Any], event_type: str, target: int, rng: np.random.Generator, args: argparse.Namespace) -> Dict[str, Any]:
    edge_index = world["edge_index"]
    num_nodes = int(world["position"].shape[0])
    num_edges = int(edge_index.shape[1])
    event_node_mask = np.zeros(num_nodes, dtype=bool)
    event_edge_mask = np.zeros(num_edges, dtype=bool)

    if event_type == "node_impulse":
        event_node_mask[target] = True
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        magnitude = float(rng.uniform(args.impulse_min, args.impulse_max))
        impulse = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32) * magnitude
        event_params = np.array([impulse[0], impulse[1], -1.0, -1.0], dtype=np.float32)
    elif event_type == "spring_break":
        event_edge_mask[target] = True
        event_node_mask[edge_index[:, target]] = True
        u, v = edge_index[:, target]
        event_params = np.array([0.0, 0.0, float(u), float(v)], dtype=np.float32)
    else:
        raise ValueError(f"unknown event_type: {event_type}")

    event_scope_node_mask, event_scope_edge_mask = make_event_scope(
        event_type=event_type,
        event_node_mask=event_node_mask,
        event_edge_mask=event_edge_mask,
        edge_index=edge_index,
    )
    return {
        "event_type": event_type,
        "event_type_id": int(EVENT_TYPES.index(event_type)),
        "event_node_mask": event_node_mask,
        "event_edge_mask": event_edge_mask,
        "event_params": event_params,
        "event_scope_node_mask": event_scope_node_mask,
        "event_scope_edge_mask": event_scope_edge_mask,
    }


def candidate_events(world: Dict[str, Any], rng: np.random.Generator, args: argparse.Namespace) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for node in np.flatnonzero(world["pinned"] < 0.5):
        events.append(make_event(world, "node_impulse", int(node), rng, args))
    for edge in np.flatnonzero(world["spring_active"] > 0.5):
        events.append(make_event(world, "spring_break", int(edge), rng, args))
    rng.shuffle(events)
    return events


def apply_event(world: Dict[str, Any], event: Dict[str, Any], args: argparse.Namespace) -> Tuple[Dict[str, Any] | None, Dict[str, Any] | None, str]:
    if event["event_type"] == "node_impulse" and np.any(world["pinned"][event["event_node_mask"]] > 0.5):
        return None, None, "pinned_impulse_target"
    if event["event_type"] == "spring_break" and np.any(world["spring_active"][event["event_edge_mask"]] < 0.5):
        return None, None, "inactive_break_target"

    node_features_t, edge_features_t = world_features(world)
    next_world = copy_world(world)
    if event["event_type"] == "node_impulse":
        next_world["velocity"][event["event_node_mask"]] += event["event_params"][:2]
    else:
        next_world["spring_active"][event["event_edge_mask"]] = 0.0

    next_position, next_velocity = rollout_one_step(
        position=next_world["position"],
        velocity=next_world["velocity"],
        mass=next_world["mass"],
        pinned=next_world["pinned"],
        edge_index=next_world["edge_index"],
        spring_active=next_world["spring_active"],
        rest_length=next_world["rest_length"],
        stiffness=next_world["stiffness"],
        node_scope_mask=event["event_scope_node_mask"],
        edge_scope_mask=event["event_scope_edge_mask"],
        dt=args.dt,
    )
    next_world["position"] = next_position
    next_world["velocity"] = next_velocity
    node_features_next, edge_features_next = world_features(next_world)
    changed_node_mask, changed_edge_mask = changed_masks(
        node_features_t=node_features_t,
        edge_features_t=edge_features_t,
        node_features_next=node_features_next,
        edge_features_next=edge_features_next,
        tolerance=args.changed_tolerance,
    )
    if np.any(changed_node_mask & ~event["event_scope_node_mask"]):
        return None, None, "node_changed_outside_scope"
    if np.any(changed_edge_mask & ~event["event_scope_edge_mask"]):
        return None, None, "edge_changed_outside_scope"
    if not np.any(changed_node_mask) and not np.any(changed_edge_mask):
        return None, None, "no_changed_region"

    result = {
        "changed_node_mask": changed_node_mask,
        "changed_edge_mask": changed_edge_mask,
        "node_features_next": node_features_next,
        "edge_features_next": edge_features_next,
    }
    return next_world, result, "accepted"


def disjoint(mask_a: np.ndarray, mask_b: np.ndarray) -> bool:
    return not bool(np.any(mask_a & mask_b))


def pack_event(event: Dict[str, Any], single: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "event_type": event["event_type"],
        "event_type_id": int(event["event_type_id"]),
        "event_node_mask": event["event_node_mask"].astype(np.float32),
        "event_edge_mask": event["event_edge_mask"].astype(np.float32),
        "event_params": event["event_params"].astype(np.float32),
        "event_scope_node_mask": event["event_scope_node_mask"].astype(np.float32),
        "event_scope_edge_mask": event["event_scope_edge_mask"].astype(np.float32),
        "changed_node_mask": single["changed_node_mask"].astype(np.float32),
        "changed_edge_mask": single["changed_edge_mask"].astype(np.float32),
    }


def try_make_pair(world: Dict[str, Any], rng: np.random.Generator, args: argparse.Namespace, rejections: Counter) -> Dict[str, Any] | None:
    events = candidate_events(world, rng, args)
    singles: List[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]] = []
    for event in events:
        next_world, single, reason = apply_event(world, event, args)
        if single is None or next_world is None:
            rejections[f"single_{reason}"] += 1
            continue
        singles.append((event, single, next_world))

    order = [(i, j) for i in range(len(singles)) for j in range(len(singles)) if i != j]
    rng.shuffle(order)
    for i, j in order:
        event_a, single_a, world_a = singles[i]
        event_b, single_b, world_b = singles[j]
        if not disjoint(event_a["event_scope_node_mask"], event_b["event_scope_node_mask"]):
            rejections["scope_node_overlap"] += 1
            continue
        if not disjoint(event_a["event_scope_edge_mask"], event_b["event_scope_edge_mask"]):
            rejections["scope_edge_overlap"] += 1
            continue
        if not disjoint(single_a["changed_node_mask"], single_b["changed_node_mask"]):
            rejections["changed_node_overlap"] += 1
            continue
        if not disjoint(single_a["changed_edge_mask"], single_b["changed_edge_mask"]):
            rejections["changed_edge_overlap"] += 1
            continue

        world_ab, _, reason_ab = apply_event(world_a, event_b, args)
        world_ba, _, reason_ba = apply_event(world_b, event_a, args)
        if world_ab is None:
            rejections[f"ab_{reason_ab}"] += 1
            continue
        if world_ba is None:
            rejections[f"ba_{reason_ba}"] += 1
            continue

        ab_node, ab_edge = world_features(world_ab)
        ba_node, ba_edge = world_features(world_ba)
        discrepancy = max(float(np.max(np.abs(ab_node - ba_node))), float(np.max(np.abs(ab_edge - ba_edge))))
        if discrepancy > args.order_tolerance:
            rejections["ab_ba_mismatch"] += 1
            continue

        node_features_t, edge_features_t = world_features(world)
        return {
            "node_features_t": node_features_t.astype(np.float32),
            "node_feature_names": list(NODE_FEATURE_NAMES),
            "edge_index": world["edge_index"].astype(np.int64),
            "edge_features_t": edge_features_t.astype(np.float32),
            "edge_feature_names": list(EDGE_FEATURE_NAMES),
            "event_a": pack_event(event_a, single_a),
            "event_b": pack_event(event_b, single_b),
            "independent_pair_flag": True,
            "pair_type": pair_type(event_a, event_b),
            "ab_node_features_next": ab_node.astype(np.float32),
            "ab_edge_features_next": ab_edge.astype(np.float32),
            "ba_node_features_next": ba_node.astype(np.float32),
            "ba_edge_features_next": ba_edge.astype(np.float32),
            "oracle_ab_ba_discrepancy": discrepancy,
        }
    return None


def generate_pairs(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Counter]:
    rng = np.random.default_rng(args.seed)
    samples: List[Dict[str, Any]] = []
    rejections: Counter = Counter()
    max_attempts = max(1000, int(args.num_pairs * args.max_attempts_multiplier))
    attempts = 0
    while len(samples) < args.num_pairs and attempts < max_attempts:
        attempts += 1
        num_nodes = int(rng.integers(args.min_nodes, args.max_nodes + 1))
        world = make_base_world(num_nodes, rng)
        sample = try_make_pair(world, rng, args, rejections)
        if sample is None:
            rejections["no_valid_pair_in_world"] += 1
            continue
        samples.append(sample)
    rejections["attempts"] = attempts
    if len(samples) < args.num_pairs:
        raise RuntimeError(f"only generated {len(samples)} / {args.num_pairs} pairs after {attempts} attempts")
    return samples, rejections


def summarize(samples: List[Dict[str, Any]], rejections: Counter, args: argparse.Namespace) -> Dict[str, Any]:
    pair_counts = Counter(sample["pair_type"] for sample in samples)
    return {
        "num_pairs": len(samples),
        "pair_type_counts": dict(pair_counts),
        "rejections": dict(rejections),
        "accepted": len(samples),
        "min_nodes": args.min_nodes,
        "max_nodes": args.max_nodes,
        "changed_tolerance": args.changed_tolerance,
        "order_tolerance": args.order_tolerance,
        "scope_policy": "same conservative expanded scope as sandbox local-event MVP",
        "graph_policy": "chain spring graph to make disjoint expanded scopes feasible",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate independent sandbox local-event pairs.")
    parser.add_argument("--num_pairs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_path", default="artifacts/sandbox_local_event_pairs/pairs.pkl")
    parser.add_argument("--summary_json", default="artifacts/sandbox_local_event_pairs/summary.json")
    parser.add_argument("--min_nodes", type=int, default=8)
    parser.add_argument("--max_nodes", type=int, default=10)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--impulse_min", type=float, default=0.08)
    parser.add_argument("--impulse_max", type=float, default=0.18)
    parser.add_argument("--changed_tolerance", type=float, default=DEFAULT_CHANGED_TOLERANCE)
    parser.add_argument("--order_tolerance", type=float, default=1e-4)
    parser.add_argument("--max_attempts_multiplier", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples, rejections = generate_pairs(args)
    output_path = resolve_path(args.output_path)
    summary_path = resolve_path(args.summary_json)
    save_pickle(output_path, samples)
    summary = summarize(samples, rejections, args)
    summary["output_path"] = str(output_path)
    save_json(summary_path, summary)

    print("sandbox local-event pair generation complete")
    print(f"pairs={len(samples)} pair_types={summary['pair_type_counts']}")
    print(f"accepted={summary['accepted']} attempts={summary['rejections'].get('attempts', 0)}")
    print(f"output_path={output_path}")
    print(f"summary_json={summary_path}")


if __name__ == "__main__":
    main()
