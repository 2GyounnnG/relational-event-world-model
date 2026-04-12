from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.generate_graph_event_data import DEFAULT_CONFIG, graph_to_edge_index
from data.generate_step30_weak_observation_data import (
    STEP30_WEAK_OBS_CONFIGS,
    make_weak_observation,
    variant_list,
)
from data.generate_step31_multi_view_observation_data import make_view_config


STEP32_BENCHMARK_VERSION = "step32_synthetic_rendered_bridge"
STEP32_RENDER_VIEW_PROFILES = ("render_relation_view", "render_support_view")


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_pickle(path: Path) -> List[Dict[str, Any]]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"Dataset is empty or malformed: {path}")
    return data


def save_pickle(path: Path, data: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def _draw_disk(image: np.ndarray, channel: int, cx: float, cy: float, radius: int, value: float) -> None:
    _, height, width = image.shape
    x0 = max(0, int(round(cx)) - radius)
    x1 = min(width - 1, int(round(cx)) + radius)
    y0 = max(0, int(round(cy)) - radius)
    y1 = min(height - 1, int(round(cy)) + radius)
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            if (x - cx) ** 2 + (y - cy) ** 2 <= (radius + 0.35) ** 2:
                image[channel, y, x] = max(float(image[channel, y, x]), float(value))


def _draw_line(
    image: np.ndarray,
    channel: int,
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    value: float,
    width: int = 1,
) -> None:
    _, height, canvas_width = image.shape
    steps = int(max(abs(p0[0] - p1[0]), abs(p0[1] - p1[1]), 1.0) * 1.5) + 1
    for t in np.linspace(0.0, 1.0, steps):
        x = p0[0] * (1.0 - t) + p1[0] * t
        y = p0[1] * (1.0 - t) + p1[1] * t
        xi = int(round(x))
        yi = int(round(y))
        for yy in range(max(0, yi - width), min(height - 1, yi + width) + 1):
            for xx in range(max(0, xi - width), min(canvas_width - 1, xi + width) + 1):
                image[channel, yy, xx] = max(float(image[channel, yy, xx]), float(value))


def _line_sample(image: np.ndarray, channel: int, p0: np.ndarray, p1: np.ndarray, points: int = 9) -> float:
    values = []
    height, width = image.shape[-2:]
    for t in np.linspace(0.0, 1.0, points):
        xy = p0 * (1.0 - t) + p1 * t
        x = int(np.clip(round(float(xy[0])), 0, width - 1))
        y = int(np.clip(round(float(xy[1])), 0, height - 1))
        values.append(float(image[channel, y, x]))
    return float(np.mean(values)) if values else 0.0


def make_layout(
    num_nodes: int,
    canvas_size: int,
    rng: np.random.Generator,
    view_idx: int,
) -> np.ndarray:
    radius = canvas_size * (0.31 + 0.02 * view_idx)
    center = np.array([canvas_size * 0.50, canvas_size * 0.50], dtype=np.float32)
    phase = float(rng.uniform(-0.35, 0.35) + view_idx * 0.31)
    positions = []
    for node_idx in range(num_nodes):
        angle = phase + 2.0 * np.pi * node_idx / max(num_nodes, 1)
        jitter = rng.normal(0.0, canvas_size * 0.025, size=2)
        pos = center + radius * np.array([np.cos(angle), np.sin(angle)], dtype=np.float32) + jitter
        pos = np.clip(pos, 3.0, canvas_size - 4.0)
        positions.append(pos.astype(np.float32))
    return np.stack(positions, axis=0)


def profile_to_step31_profile(profile: str) -> str:
    if profile == "render_relation_view":
        return "relation_focus"
    if profile == "render_support_view":
        return "support_focus"
    raise ValueError(f"Unknown Step32 render profile: {profile}")


def render_weak_view(
    weak_obs: Dict[str, Any],
    positions: np.ndarray,
    variant: str,
    rng: np.random.Generator,
    canvas_size: int,
    num_types: int,
) -> Dict[str, Any]:
    slot_features = np.asarray(weak_obs["slot_features"], dtype=np.float32)
    relation = np.asarray(weak_obs["relation_hints"], dtype=np.float32)
    support = np.asarray(weak_obs["pair_support_hints"], dtype=np.float32)
    signed = np.asarray(weak_obs["signed_pair_witness"], dtype=np.float32)
    bundle = np.asarray(weak_obs["pair_evidence_bundle"], dtype=np.float32)

    num_nodes = slot_features.shape[0]
    image = rng.normal(0.03, 0.018 if variant == "clean" else 0.032, size=(4, canvas_size, canvas_size))
    image = np.clip(image, 0.0, 1.0).astype(np.float32)

    node_dropout = 0.08 if variant == "clean" else 0.18
    line_dropout = 0.08 if variant == "clean" else 0.18
    false_line_prob = 0.025 if variant == "clean" else 0.065
    visible_nodes = (rng.random(num_nodes) >= node_dropout).astype(np.float32)

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            p0 = tuple(float(x) for x in positions[i])
            p1 = tuple(float(x) for x in positions[j])
            if rng.random() >= line_dropout:
                rel_value = float(np.clip(0.10 + 0.82 * relation[i, j] + rng.normal(0.0, 0.035), 0.0, 1.0))
                if rel_value > 0.18:
                    _draw_line(image, 1, p0, p1, rel_value, width=0)
            if rng.random() < false_line_prob:
                _draw_line(image, 1, p0, p1, float(rng.uniform(0.38, 0.62)), width=0)

            support_value = float(np.clip(0.08 + 0.72 * support[i, j] + 0.18 * bundle[i, j, 0], 0.0, 1.0))
            witness_value = float(np.clip(0.50 + 0.32 * signed[i, j] + rng.normal(0.0, 0.04), 0.0, 1.0))
            if rng.random() >= line_dropout:
                _draw_line(image, 2, p0, p1, support_value, width=0)
                _draw_line(image, 3, p0, p1, witness_value, width=0)

    type_hints = slot_features[:, :num_types]
    type_observed = slot_features[:, num_types]
    state_start = num_types + 1
    state_width = (slot_features.shape[1] - state_start) // 2
    state_hints = slot_features[:, state_start : state_start + state_width]
    for node_idx in range(num_nodes):
        if visible_nodes[node_idx] <= 0.0:
            continue
        type_score = float(type_hints[node_idx].argmax()) / max(num_types - 1, 1)
        state_score = float(1.0 / (1.0 + np.exp(-np.mean(state_hints[node_idx]))))
        observed_scale = 0.35 + 0.65 * float(type_observed[node_idx])
        node_value = float(np.clip((0.25 + 0.45 * type_score + 0.25 * state_score) * observed_scale, 0.0, 1.0))
        _draw_disk(image, 0, float(positions[node_idx, 0]), float(positions[node_idx, 1]), 2, node_value)
        _draw_disk(image, 3, float(positions[node_idx, 0]), float(positions[node_idx, 1]), 1, 0.80)

    if rng.random() < (0.20 if variant == "clean" else 0.35):
        occ_w = int(rng.integers(max(4, canvas_size // 8), max(5, canvas_size // 4)))
        occ_h = int(rng.integers(max(4, canvas_size // 8), max(5, canvas_size // 4)))
        x0 = int(rng.integers(0, max(1, canvas_size - occ_w)))
        y0 = int(rng.integers(0, max(1, canvas_size - occ_h)))
        image[:, y0 : y0 + occ_h, x0 : x0 + occ_w] *= 0.25
        image[3, y0 : y0 + occ_h, x0 : x0 + occ_w] = np.maximum(
            image[3, y0 : y0 + occ_h, x0 : x0 + occ_w],
            0.18,
        )

    image = np.clip(image, 0.0, 1.0).astype(np.float32)
    trivial_relation = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    trivial_support = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            rel = _line_sample(image, 1, positions[i], positions[j])
            sup = _line_sample(image, 2, positions[i], positions[j])
            trivial_relation[i, j] = trivial_relation[j, i] = rel
            trivial_support[i, j] = trivial_support[j, i] = sup

    return {
        "image": image,
        "node_positions": (positions / float(canvas_size - 1)).astype(np.float32),
        "visible_node_mask": visible_nodes.astype(np.float32),
        "trivial_relation_scores": trivial_relation,
        "trivial_support_scores": trivial_support,
    }


def make_step32_sample(
    raw_sample: Dict[str, Any],
    source_sample_index: int,
    variant: str,
    variant_index: int,
    seed: int,
    num_types: int,
    canvas_size: int,
    view_profiles: Iterable[str],
) -> Dict[str, Any]:
    sample = copy.deepcopy(raw_sample)
    views = []
    for view_idx, profile in enumerate(view_profiles):
        step31_profile = profile_to_step31_profile(profile)
        cfg = make_view_config(variant, step31_profile)
        rng = np.random.default_rng(
            seed
            + source_sample_index * 1013
            + variant_index * 100_019
            + view_idx * 1_000_033
        )
        weak_obs = make_weak_observation(
            sample["graph_t"],
            variant=variant,
            cfg=cfg,
            rng=rng,
            num_types=num_types,
            include_signed_pair_witness=True,
            include_pair_evidence_bundle=True,
            include_positive_ambiguity_safety_hint=False,
        )
        positions = make_layout(
            num_nodes=np.asarray(sample["graph_t"]["node_features"]).shape[0],
            canvas_size=canvas_size,
            rng=rng,
            view_idx=view_idx,
        )
        rendered = render_weak_view(
            weak_obs=weak_obs,
            positions=positions,
            variant=variant,
            rng=rng,
            canvas_size=canvas_size,
            num_types=num_types,
        )
        rendered["view_index"] = view_idx
        rendered["view_profile"] = profile
        views.append(rendered)

    sample["rendered_observation"] = {
        "views": views,
        "view_profiles": list(view_profiles),
        "num_views": len(views),
        "canvas_size": canvas_size,
        "channels": [
            "node_mark_intensity",
            "weak_relation_line_intensity",
            "weak_support_line_intensity",
            "signed_witness_and_visibility",
        ],
        "note": (
            "Rendered views are rasterized from weak corrupted observations with "
            "view-dependent layout, dropout, false lines, noise, and occlusion; "
            "clean adjacency is never rendered directly."
        ),
    }
    sample["step32_observation_variant"] = variant
    sample["step32_benchmark_version"] = STEP32_BENCHMARK_VERSION
    sample["step32_source_sample_index"] = source_sample_index
    if "edge_index" not in sample["graph_t"]:
        sample["graph_t"]["edge_index"] = graph_to_edge_index(np.asarray(sample["graph_t"]["adj"]))
    if "edge_index" not in sample["graph_t1"]:
        sample["graph_t1"]["edge_index"] = graph_to_edge_index(np.asarray(sample["graph_t1"]["adj"]))
    return sample


def generate_step32_dataset(
    source_samples: List[Dict[str, Any]],
    variants: Iterable[str],
    seed: int,
    num_types: int,
    max_samples: int | None,
    canvas_size: int,
    num_views: int,
) -> List[Dict[str, Any]]:
    if num_views < 1 or num_views > len(STEP32_RENDER_VIEW_PROFILES):
        raise ValueError(f"num_views must be between 1 and {len(STEP32_RENDER_VIEW_PROFILES)}")
    selected = source_samples if max_samples is None else source_samples[:max_samples]
    out: list[Dict[str, Any]] = []
    view_profiles = STEP32_RENDER_VIEW_PROFILES[:num_views]
    for source_idx, raw_sample in enumerate(selected):
        for variant_idx, variant in enumerate(list(variants)):
            out.append(
                make_step32_sample(
                    raw_sample=raw_sample,
                    source_sample_index=source_idx,
                    variant=variant,
                    variant_index=variant_idx,
                    seed=seed,
                    num_types=num_types,
                    canvas_size=canvas_size,
                    view_profiles=view_profiles,
                )
            )
    return out


def summarize(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    variant_counts = Counter(str(sample.get("step32_observation_variant", "unknown")) for sample in dataset)
    event_counts: Counter[str] = Counter()
    node_counts: Counter[int] = Counter()
    for sample in dataset:
        node_counts[int(np.asarray(sample["graph_t"]["node_features"]).shape[0])] += 1
        for event in sample.get("events", []):
            event_counts[str(event.get("event_type", "unknown"))] += 1
    first = dataset[0] if dataset else {}
    rendered = first.get("rendered_observation", {}) if first else {}
    views = rendered.get("views", [])
    return {
        "sample_count": len(dataset),
        "observation_variant_counts": dict(sorted(variant_counts.items())),
        "num_nodes_distribution": {str(k): int(v) for k, v in sorted(node_counts.items())},
        "event_type_counts": dict(sorted(event_counts.items())),
        "num_views": len(views),
        "canvas_size": rendered.get("canvas_size"),
        "image_shape": list(np.asarray(views[0]["image"]).shape) if views else None,
        "view_profiles": rendered.get("view_profiles", []),
        "benchmark_version": first.get("step32_benchmark_version") if first else None,
    }


def generate_one_file(
    source_path: Path,
    output_path: Path,
    variants: List[str],
    seed: int,
    num_types: int,
    max_samples: int | None,
    canvas_size: int,
    num_views: int,
) -> Dict[str, Any]:
    source_samples = load_pickle(source_path)
    dataset = generate_step32_dataset(
        source_samples=source_samples,
        variants=variants,
        seed=seed,
        num_types=num_types,
        max_samples=max_samples,
        canvas_size=canvas_size,
        num_views=num_views,
    )
    save_pickle(output_path, dataset)
    summary = summarize(dataset)
    print(f"source path: {source_path}")
    print(f"output path: {output_path}")
    print(f"summary: {summary}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--variants", choices=["clean", "noisy", "both"], default="both")
    parser.add_argument("--seed", type=int, default=int(DEFAULT_CONFIG["random_seed"]) + 32)
    parser.add_argument("--num_types", type=int, default=int(DEFAULT_CONFIG["num_types"]))
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--canvas_size", type=int, default=32)
    parser.add_argument("--num_views", type=int, default=2)
    parser.add_argument("--generate_default_splits", action="store_true")
    parser.add_argument("--source_train_path", type=str, default="data/graph_event_train.pkl")
    parser.add_argument("--source_val_path", type=str, default="data/graph_event_val.pkl")
    parser.add_argument("--source_test_path", type=str, default="data/graph_event_test.pkl")
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--output_prefix", type=str, default="graph_event_step32_rendered")
    parser.add_argument("--summary_json", type=str, default=None)
    args = parser.parse_args()

    variants = variant_list(args.variants)
    if args.generate_default_splits:
        output_dir = resolve_path(args.output_dir)
        split_specs = [
            ("train", resolve_path(args.source_train_path), output_dir / f"{args.output_prefix}_train.pkl"),
            ("val", resolve_path(args.source_val_path), output_dir / f"{args.output_prefix}_val.pkl"),
            ("test", resolve_path(args.source_test_path), output_dir / f"{args.output_prefix}_test.pkl"),
        ]
        summaries = {}
        for split_name, source_path, output_path in split_specs:
            print(f"Generating Step32 {split_name} split")
            summaries[split_name] = generate_one_file(
                source_path=source_path,
                output_path=output_path,
                variants=variants,
                seed=args.seed,
                num_types=args.num_types,
                max_samples=args.max_samples,
                canvas_size=args.canvas_size,
                num_views=args.num_views,
            )
        if args.summary_json is not None:
            summary_path = resolve_path(args.summary_json)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summaries, f, indent=2)
        print(f"default split summaries: {summaries}")
        return

    if args.source_path is None or args.output_path is None:
        raise SystemExit("Provide --source_path and --output_path, or use --generate_default_splits.")
    summary = generate_one_file(
        source_path=resolve_path(args.source_path),
        output_path=resolve_path(args.output_path),
        variants=variants,
        seed=args.seed,
        num_types=args.num_types,
        max_samples=args.max_samples,
        canvas_size=args.canvas_size,
        num_views=args.num_views,
    )
    if args.summary_json is not None:
        summary_path = resolve_path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
