from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


NODE_FEATURE_NAMES = ["x", "y", "vx", "vy", "mass", "radius", "pinned"]
EDGE_FEATURE_NAMES = ["spring_active", "rest_length", "stiffness", "current_distance"]
EVENT_TYPES = ("node_impulse", "spring_break")
DEFAULT_CHANGED_TOLERANCE = 1e-4


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[1] / path


def save_pickle(path: Path, samples: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(samples, f)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def make_edge_index(num_nodes: int, rng: np.random.Generator) -> np.ndarray:
    """Build a tiny connected undirected spring graph with fixed edge slots."""
    edges: List[Tuple[int, int]] = []
    for node in range(1, num_nodes):
        parent = int(rng.integers(0, node))
        edges.append((parent, node))

    existing = set(edges)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if (i, j) in existing:
                continue
            if rng.random() < 0.25:
                edges.append((i, j))

    edges = sorted(set((min(i, j), max(i, j)) for i, j in edges))
    return np.asarray(edges, dtype=np.int64).T


def sample_positions(num_nodes: int, rng: np.random.Generator) -> np.ndarray:
    for _ in range(200):
        position = rng.uniform(0.15, 0.85, size=(num_nodes, 2)).astype(np.float32)
        distances = [
            float(np.linalg.norm(position[j] - position[i]))
            for i in range(num_nodes)
            for j in range(i + 1, num_nodes)
        ]
        if min(distances, default=1.0) > 0.12:
            return position
    return position


def edge_distances(position: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    src, dst = edge_index
    return np.linalg.norm(position[dst] - position[src], axis=1).astype(np.float32)


def make_node_features(
    position: np.ndarray,
    velocity: np.ndarray,
    mass: np.ndarray,
    radius: np.ndarray,
    pinned: np.ndarray,
) -> np.ndarray:
    return np.concatenate(
        [
            position.astype(np.float32),
            velocity.astype(np.float32),
            mass[:, None].astype(np.float32),
            radius[:, None].astype(np.float32),
            pinned[:, None].astype(np.float32),
        ],
        axis=1,
    )


def make_edge_features(
    spring_active: np.ndarray,
    rest_length: np.ndarray,
    stiffness: np.ndarray,
    position: np.ndarray,
    edge_index: np.ndarray,
) -> np.ndarray:
    current_distance = edge_distances(position, edge_index)
    return np.stack(
        [
            spring_active.astype(np.float32),
            rest_length.astype(np.float32),
            stiffness.astype(np.float32),
            current_distance.astype(np.float32),
        ],
        axis=1,
    )


def incident_edges(edge_index: np.ndarray, node_mask: np.ndarray) -> np.ndarray:
    src, dst = edge_index
    return np.logical_or(node_mask[src], node_mask[dst])


def close_scope(edge_index: np.ndarray, node_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Close edge scope around scoped nodes, then include endpoints of scoped edges.

    Calling this once gives a one-hop conservative closure around the input
    nodes. The MVP event scopes below intentionally call it twice. That makes
    scope slightly broader than the prose "event node plus one-hop" sketch:
    it includes a conservative second closure so the hard contract
    changed_region subset event_scope remains feasible while rollout is local.
    """
    edge_mask = incident_edges(edge_index, node_mask)
    closed_node_mask = node_mask.copy()
    if edge_mask.any():
        endpoints = edge_index[:, edge_mask].reshape(-1)
        closed_node_mask[endpoints] = True
    edge_mask = incident_edges(edge_index, closed_node_mask)
    return closed_node_mask, edge_mask


def make_event_scope(
    event_type: str,
    event_node_mask: np.ndarray,
    event_edge_mask: np.ndarray,
    edge_index: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    num_nodes = event_node_mask.shape[0]
    scope_nodes = event_node_mask.copy()

    if event_type == "node_impulse":
        # This is deliberately broader than one-hop. It applies two conservative
        # closures from the event node so any tiny rollout effects that survive
        # the changed-mask tolerance remain inside the declared event scope.
        scope_nodes, _ = close_scope(edge_index, scope_nodes)
        return close_scope(edge_index, scope_nodes)

    if event_type == "spring_break":
        if event_edge_mask.any():
            endpoints = edge_index[:, event_edge_mask].reshape(-1)
            scope_nodes[endpoints] = True
        # Spring breaks seed scope from the broken edge endpoints, then use the
        # same expanded conservative closure policy as node impulses.
        scope_nodes, _ = close_scope(edge_index, scope_nodes)
        return close_scope(edge_index, scope_nodes)

    raise ValueError(f"unknown event type: {event_type}")


def spring_forces(
    position: np.ndarray,
    edge_index: np.ndarray,
    spring_active: np.ndarray,
    rest_length: np.ndarray,
    stiffness: np.ndarray,
    edge_scope_mask: np.ndarray,
) -> np.ndarray:
    forces = np.zeros_like(position, dtype=np.float32)
    src, dst = edge_index
    for edge_id in np.flatnonzero(np.logical_and(spring_active > 0.5, edge_scope_mask)):
        i = int(src[edge_id])
        j = int(dst[edge_id])
        delta = position[j] - position[i]
        distance = float(np.linalg.norm(delta))
        if distance < 1e-6:
            continue
        unit = delta / distance
        magnitude = float(stiffness[edge_id] * (distance - rest_length[edge_id]))
        force = (magnitude * unit).astype(np.float32)
        forces[i] += force
        forces[j] -= force
    return forces


def rollout_one_step(
    position: np.ndarray,
    velocity: np.ndarray,
    mass: np.ndarray,
    pinned: np.ndarray,
    edge_index: np.ndarray,
    spring_active: np.ndarray,
    rest_length: np.ndarray,
    stiffness: np.ndarray,
    node_scope_mask: np.ndarray,
    edge_scope_mask: np.ndarray,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray]:
    next_position = position.copy()
    next_velocity = velocity.copy()

    forces = spring_forces(
        position=position,
        edge_index=edge_index,
        spring_active=spring_active,
        rest_length=rest_length,
        stiffness=stiffness,
        edge_scope_mask=edge_scope_mask,
    )

    movable = np.logical_and(node_scope_mask, pinned < 0.5)
    next_velocity[movable] += dt * forces[movable] / np.maximum(mass[movable, None], 1e-6)
    next_position[movable] += dt * next_velocity[movable]
    next_position[movable] = np.clip(next_position[movable], 0.05, 0.95)
    next_velocity[pinned > 0.5] = 0.0
    return next_position.astype(np.float32), next_velocity.astype(np.float32)


def make_base_world(rng: np.random.Generator) -> Dict[str, Any]:
    num_nodes = int(rng.integers(3, 6))
    edge_index = make_edge_index(num_nodes, rng)
    num_edges = int(edge_index.shape[1])

    position = sample_positions(num_nodes, rng)
    velocity = np.zeros((num_nodes, 2), dtype=np.float32)
    mass = rng.uniform(0.8, 1.4, size=num_nodes).astype(np.float32)
    radius = rng.uniform(0.025, 0.045, size=num_nodes).astype(np.float32)
    pinned = np.zeros(num_nodes, dtype=np.float32)
    if num_nodes >= 4 and rng.random() < 0.25:
        pinned[int(rng.integers(0, num_nodes))] = 1.0

    current_distance = edge_distances(position, edge_index)
    rest_length = (current_distance * rng.uniform(0.92, 1.08, size=num_edges)).astype(np.float32)
    stiffness = rng.uniform(0.8, 2.2, size=num_edges).astype(np.float32)
    spring_active = np.ones(num_edges, dtype=np.float32)

    return {
        "position": position,
        "velocity": velocity,
        "mass": mass,
        "radius": radius,
        "pinned": pinned,
        "edge_index": edge_index,
        "spring_active": spring_active,
        "rest_length": rest_length,
        "stiffness": stiffness,
    }


def changed_masks(
    node_features_t: np.ndarray,
    edge_features_t: np.ndarray,
    node_features_next: np.ndarray,
    edge_features_next: np.ndarray,
    tolerance: float,
) -> Tuple[np.ndarray, np.ndarray]:
    # Use a stable tolerance rather than float-noise sensitivity. The MVP
    # rollout is deterministic, but current_distance is recomputed from float32
    # positions; 1e-4 is small relative to sampled positions/velocities while
    # avoiding changed labels caused only by numerical dust.
    node_delta = np.max(np.abs(node_features_next[:, :4] - node_features_t[:, :4]), axis=1)
    edge_delta = np.max(np.abs(edge_features_next - edge_features_t), axis=1)
    return node_delta > tolerance, edge_delta > tolerance


def make_sample(rng: np.random.Generator, args: argparse.Namespace) -> Tuple[Dict[str, Any] | None, str]:
    world = make_base_world(rng)
    edge_index = world["edge_index"]
    num_nodes = int(world["position"].shape[0])
    num_edges = int(edge_index.shape[1])

    if rng.random() < 0.5:
        event_type = "node_impulse"
        candidates = np.flatnonzero(world["pinned"] < 0.5)
        if len(candidates) == 0:
            return None, "no_unpinned_node"
        event_node = int(rng.choice(candidates))
        event_node_mask = np.zeros(num_nodes, dtype=bool)
        event_node_mask[event_node] = True
        event_edge_mask = np.zeros(num_edges, dtype=bool)
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        magnitude = float(rng.uniform(args.impulse_min, args.impulse_max))
        impulse = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32) * magnitude
        event_params = np.array([impulse[0], impulse[1], -1.0, -1.0], dtype=np.float32)
    else:
        event_type = "spring_break"
        active_edges = np.flatnonzero(world["spring_active"] > 0.5)
        if len(active_edges) == 0:
            return None, "no_active_edge"
        event_edge = int(rng.choice(active_edges))
        event_node_mask = np.zeros(num_nodes, dtype=bool)
        event_node_mask[edge_index[:, event_edge]] = True
        event_edge_mask = np.zeros(num_edges, dtype=bool)
        event_edge_mask[event_edge] = True
        u, v = edge_index[:, event_edge]
        event_params = np.array([0.0, 0.0, float(u), float(v)], dtype=np.float32)

    event_scope_node_mask, event_scope_edge_mask = make_event_scope(
        event_type=event_type,
        event_node_mask=event_node_mask,
        event_edge_mask=event_edge_mask,
        edge_index=edge_index,
    )

    position = world["position"].copy()
    velocity = world["velocity"].copy()
    spring_active = world["spring_active"].copy()

    if event_type == "node_impulse":
        velocity[event_node_mask] += event_params[:2]
    elif event_type == "spring_break":
        spring_active[event_edge_mask] = 0.0

    next_position, next_velocity = rollout_one_step(
        position=position,
        velocity=velocity,
        mass=world["mass"],
        pinned=world["pinned"],
        edge_index=edge_index,
        spring_active=spring_active,
        rest_length=world["rest_length"],
        stiffness=world["stiffness"],
        node_scope_mask=event_scope_node_mask,
        edge_scope_mask=event_scope_edge_mask,
        dt=args.dt,
    )

    node_features_t = make_node_features(
        position=world["position"],
        velocity=world["velocity"],
        mass=world["mass"],
        radius=world["radius"],
        pinned=world["pinned"],
    )
    edge_features_t = make_edge_features(
        spring_active=world["spring_active"],
        rest_length=world["rest_length"],
        stiffness=world["stiffness"],
        position=world["position"],
        edge_index=edge_index,
    )
    node_features_next = make_node_features(
        position=next_position,
        velocity=next_velocity,
        mass=world["mass"],
        radius=world["radius"],
        pinned=world["pinned"],
    )
    edge_features_next = make_edge_features(
        spring_active=spring_active,
        rest_length=world["rest_length"],
        stiffness=world["stiffness"],
        position=next_position,
        edge_index=edge_index,
    )

    changed_node_mask, changed_edge_mask = changed_masks(
        node_features_t=node_features_t,
        edge_features_t=edge_features_t,
        node_features_next=node_features_next,
        edge_features_next=edge_features_next,
        tolerance=args.changed_tolerance,
    )

    if np.any(np.logical_and(changed_node_mask, ~event_scope_node_mask)):
        return None, "node_changed_outside_scope"
    if np.any(np.logical_and(changed_edge_mask, ~event_scope_edge_mask)):
        return None, "edge_changed_outside_scope"
    if not np.any(changed_node_mask) and not np.any(changed_edge_mask):
        return None, "no_changed_region"

    sample = {
        "node_features_t": node_features_t.astype(np.float32),
        "node_feature_names": list(NODE_FEATURE_NAMES),
        "edge_index": edge_index.astype(np.int64),
        "edge_features_t": edge_features_t.astype(np.float32),
        "edge_feature_names": list(EDGE_FEATURE_NAMES),
        "event_type": event_type,
        "event_type_id": int(EVENT_TYPES.index(event_type)),
        "event_node_mask": event_node_mask.astype(np.float32),
        "event_edge_mask": event_edge_mask.astype(np.float32),
        "event_params": event_params.astype(np.float32),
        "event_scope_node_mask": event_scope_node_mask.astype(np.float32),
        "event_scope_edge_mask": event_scope_edge_mask.astype(np.float32),
        "changed_node_mask": changed_node_mask.astype(np.float32),
        "changed_edge_mask": changed_edge_mask.astype(np.float32),
        "node_features_next": node_features_next.astype(np.float32),
        "edge_features_next": edge_features_next.astype(np.float32),
        "copy_node_features_next": node_features_t.astype(np.float32).copy(),
        "copy_edge_features_next": edge_features_t.astype(np.float32).copy(),
    }
    return sample, "accepted"


def generate_split(
    count: int,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], Counter]:
    samples: List[Dict[str, Any]] = []
    rejections: Counter = Counter()
    max_attempts = max(1000, int(count * args.max_attempts_multiplier))
    attempts = 0
    while len(samples) < count and attempts < max_attempts:
        attempts += 1
        sample, reason = make_sample(rng, args)
        if sample is None:
            rejections[reason] += 1
            continue
        samples.append(sample)
    if len(samples) < count:
        raise RuntimeError(f"only generated {len(samples)} / {count} samples after {attempts} attempts")
    rejections["attempts"] = attempts
    return samples, rejections


def summarize_split(samples: List[Dict[str, Any]], rejections: Counter) -> Dict[str, Any]:
    event_counts = Counter(sample["event_type"] for sample in samples)
    node_counts = [int(sample["node_features_t"].shape[0]) for sample in samples]
    edge_counts = [int(sample["edge_features_t"].shape[0]) for sample in samples]
    changed_node_counts = [float(sample["changed_node_mask"].sum()) for sample in samples]
    changed_edge_counts = [float(sample["changed_edge_mask"].sum()) for sample in samples]
    scope_violation_rejections = int(
        rejections.get("node_changed_outside_scope", 0) + rejections.get("edge_changed_outside_scope", 0)
    )
    return {
        "num_samples": len(samples),
        "event_type_counts": dict(event_counts),
        "avg_node_count": float(np.mean(node_counts)),
        "avg_edge_count": float(np.mean(edge_counts)),
        "avg_changed_node_count": float(np.mean(changed_node_counts)),
        "avg_changed_edge_count": float(np.mean(changed_edge_counts)),
        "rejections": dict(rejections),
        "scope_violation_rejections": scope_violation_rejections,
    }


def default_split_paths(output_prefix: str) -> Tuple[Path, Path, Path]:
    prefix = resolve_path(output_prefix)
    return (
        prefix.with_name(prefix.name + "_train.pkl"),
        prefix.with_name(prefix.name + "_val.pkl"),
        prefix.with_name(prefix.name + "_test.pkl"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_samples", type=int, default=1000)
    parser.add_argument("--val_samples", type=int, default=200)
    parser.add_argument("--test_samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_prefix", type=str, default="data/sandbox_local_event_mvp")
    parser.add_argument("--train_path", type=str, default="")
    parser.add_argument("--val_path", type=str, default="")
    parser.add_argument("--test_path", type=str, default="")
    parser.add_argument("--summary_json", type=str, default="artifacts/sandbox_local_event_mvp_data/summary.json")
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--impulse_min", type=float, default=0.08)
    parser.add_argument("--impulse_max", type=float, default=0.18)
    parser.add_argument("--changed_tolerance", type=float, default=DEFAULT_CHANGED_TOLERANCE)
    parser.add_argument("--max_attempts_multiplier", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_path, val_path, test_path = default_split_paths(args.output_prefix)
    if args.train_path:
        train_path = resolve_path(args.train_path)
    if args.val_path:
        val_path = resolve_path(args.val_path)
    if args.test_path:
        test_path = resolve_path(args.test_path)
    summary_path = resolve_path(args.summary_json)

    rng = np.random.default_rng(args.seed)
    split_specs = [
        ("train", args.train_samples, train_path),
        ("val", args.val_samples, val_path),
        ("test", args.test_samples, test_path),
    ]

    summary: Dict[str, Any] = {
        "benchmark": "sandbox_local_event_mvp",
        "seed": args.seed,
        "node_feature_names": list(NODE_FEATURE_NAMES),
        "edge_feature_names": list(EDGE_FEATURE_NAMES),
        "event_types": list(EVENT_TYPES),
        "changed_region_subset_event_scope_enforced": True,
        "changed_tolerance": args.changed_tolerance,
        "event_scope_policy": "two conservative graph closures around event seed nodes/edges",
        "splits": {},
    }

    for split_name, count, path in split_specs:
        samples, rejections = generate_split(count=count, rng=rng, args=args)
        save_pickle(path, samples)
        split_summary = summarize_split(samples, rejections)
        split_summary["path"] = str(path)
        summary["splits"][split_name] = split_summary

    save_json(summary_path, summary)
    summary["summary_json"] = str(summary_path)

    print("sandbox_local_event_mvp generation complete")
    for split_name in ("train", "val", "test"):
        split = summary["splits"][split_name]
        print(
            f"{split_name}: size={split['num_samples']} events={split['event_type_counts']} "
            f"avg_nodes={split['avg_node_count']:.2f} avg_edges={split['avg_edge_count']:.2f} "
            f"avg_changed_nodes={split['avg_changed_node_count']:.2f} "
            f"avg_changed_edges={split['avg_changed_edge_count']:.2f} "
            f"scope_rejections={split['scope_violation_rejections']}"
        )
    print(f"summary_json={summary_path}")


if __name__ == "__main__":
    main()
