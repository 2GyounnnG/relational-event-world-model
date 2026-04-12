from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

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


STEP31_BENCHMARK_VERSION = "step31_multi_view_observation_bridge"
STEP31_VIEW_PROFILES = ("relation_focus", "support_focus", "evidence_focus")


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


def _scale_prob(value: float, scale: float) -> float:
    return float(np.clip(value * scale, 0.0, 0.95))


def _scale_std(value: float, scale: float) -> float:
    return float(max(value * scale, 0.0))


def make_view_config(variant: str, profile: str) -> Dict[str, float]:
    """Create one weak view config with a different reliability profile.

    The view profiles intentionally alter corruption style, not ground-truth
    access. Each view remains weak/noisy, but its errors are not exact copies of
    the other views, allowing cross-view agreement and disagreement to matter.
    """

    cfg = dict(STEP30_WEAK_OBS_CONFIGS[variant])
    if profile == "relation_focus":
        cfg["relation_dropout_prob"] = _scale_prob(cfg["relation_dropout_prob"], 0.75)
        cfg["relation_false_positive_prob"] = _scale_prob(cfg["relation_false_positive_prob"], 0.85)
        cfg["relation_jitter_std"] = _scale_std(cfg["relation_jitter_std"], 0.85)
        cfg["pair_support_dropout_prob"] = _scale_prob(cfg["pair_support_dropout_prob"], 1.20)
        cfg["pair_support_false_positive_prob"] = _scale_prob(
            cfg["pair_support_false_positive_prob"], 1.15
        )
        cfg["pair_bundle_dropout_prob"] = _scale_prob(cfg["pair_bundle_dropout_prob"], 1.15)
        cfg["pair_bundle_jitter_std"] = _scale_std(cfg["pair_bundle_jitter_std"], 1.10)
    elif profile == "support_focus":
        cfg["relation_dropout_prob"] = _scale_prob(cfg["relation_dropout_prob"], 1.20)
        cfg["relation_false_positive_prob"] = _scale_prob(cfg["relation_false_positive_prob"], 1.15)
        cfg["pair_support_dropout_prob"] = _scale_prob(cfg["pair_support_dropout_prob"], 0.70)
        cfg["pair_support_false_positive_prob"] = _scale_prob(
            cfg["pair_support_false_positive_prob"], 0.85
        )
        cfg["pair_support_jitter_std"] = _scale_std(cfg["pair_support_jitter_std"], 0.85)
        cfg["signed_witness_dropout_prob"] = _scale_prob(cfg["signed_witness_dropout_prob"], 1.10)
        cfg["signed_witness_jitter_std"] = _scale_std(cfg["signed_witness_jitter_std"], 1.10)
    elif profile == "evidence_focus":
        cfg["relation_dropout_prob"] = _scale_prob(cfg["relation_dropout_prob"], 1.10)
        cfg["pair_support_dropout_prob"] = _scale_prob(cfg["pair_support_dropout_prob"], 1.10)
        cfg["signed_witness_dropout_prob"] = _scale_prob(cfg["signed_witness_dropout_prob"], 0.80)
        cfg["signed_witness_flip_prob"] = _scale_prob(cfg["signed_witness_flip_prob"], 0.85)
        cfg["pair_bundle_dropout_prob"] = _scale_prob(cfg["pair_bundle_dropout_prob"], 0.75)
        cfg["pair_bundle_flip_prob"] = _scale_prob(cfg["pair_bundle_flip_prob"], 0.85)
        cfg["pair_bundle_jitter_std"] = _scale_std(cfg["pair_bundle_jitter_std"], 0.85)
    else:
        raise ValueError(f"Unknown Step31 view profile: {profile}")
    return cfg


def make_step31_sample(
    raw_sample: Dict[str, Any],
    source_sample_index: int,
    variant: str,
    variant_index: int,
    seed: int,
    num_types: int,
    view_profiles: Iterable[str],
    benchmark_version: str = STEP31_BENCHMARK_VERSION,
) -> Dict[str, Any]:
    sample = copy.deepcopy(raw_sample)
    views = []
    profile_list = list(view_profiles)
    for view_idx, profile in enumerate(profile_list):
        cfg = make_view_config(variant, profile)
        rng = np.random.default_rng(
            seed
            + source_sample_index * 1009
            + variant_index * 100_003
            + view_idx * 1_000_003
        )
        view = make_weak_observation(
            sample["graph_t"],
            variant=variant,
            cfg=cfg,
            rng=rng,
            num_types=num_types,
            include_signed_pair_witness=True,
            include_pair_evidence_bundle=True,
            include_positive_ambiguity_safety_hint=False,
        )
        view["view_index"] = view_idx
        view["view_profile"] = profile
        views.append(view)

    # Keep the first view in the Step30-compatible slot so existing single-view
    # tooling can train a fair baseline on the new Step31 data.
    sample["weak_observation"] = views[0]
    sample["multi_view_observation"] = {
        "views": views,
        "view_profiles": profile_list,
        "num_views": len(views),
        "note": (
            "Views are independent weak structured observations of graph_t with "
            "different corruption/reliability profiles; none expose clean adjacency."
        ),
    }
    sample["step30_observation_variant"] = variant
    sample["step31_observation_variant"] = variant
    sample["step31_benchmark_version"] = benchmark_version
    sample["step31_view_profiles"] = profile_list
    sample["step31_source_sample_index"] = source_sample_index
    sample["step31_note"] = (
        "multi_view_observation views are slot-aligned to graph_t nodes; graph_t "
        "remains the clean current structured recovery target."
    )
    if "edge_index" not in sample["graph_t"]:
        sample["graph_t"]["edge_index"] = graph_to_edge_index(np.asarray(sample["graph_t"]["adj"]))
    if "edge_index" not in sample["graph_t1"]:
        sample["graph_t1"]["edge_index"] = graph_to_edge_index(np.asarray(sample["graph_t1"]["adj"]))
    return sample


def generate_step31_dataset(
    source_samples: List[Dict[str, Any]],
    variants: Iterable[str],
    seed: int,
    num_types: int,
    max_samples: int | None = None,
    num_views: int = 3,
    benchmark_version: str = STEP31_BENCHMARK_VERSION,
) -> List[Dict[str, Any]]:
    if num_views < 2 or num_views > len(STEP31_VIEW_PROFILES):
        raise ValueError(f"num_views must be between 2 and {len(STEP31_VIEW_PROFILES)}")
    selected = source_samples if max_samples is None else source_samples[:max_samples]
    out: list[Dict[str, Any]] = []
    variants_list = list(variants)
    view_profiles = STEP31_VIEW_PROFILES[:num_views]
    for source_idx, raw_sample in enumerate(selected):
        for variant_idx, variant in enumerate(variants_list):
            out.append(
                make_step31_sample(
                    raw_sample=raw_sample,
                    source_sample_index=source_idx,
                    variant=variant,
                    variant_index=variant_idx,
                    seed=seed,
                    num_types=num_types,
                    view_profiles=view_profiles,
                    benchmark_version=benchmark_version,
                )
            )
    return out


def summarize(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    variant_counts = Counter(str(sample.get("step31_observation_variant", "unknown")) for sample in dataset)
    event_counts: Counter[str] = Counter()
    node_counts: Counter[int] = Counter()
    for sample in dataset:
        node_counts[int(np.asarray(sample["graph_t"]["node_features"]).shape[0])] += 1
        for event in sample.get("events", []):
            event_counts[str(event.get("event_type", "unknown"))] += 1
    first = dataset[0] if dataset else {}
    views = first.get("multi_view_observation", {}).get("views", []) if first else []
    return {
        "sample_count": len(dataset),
        "observation_variant_counts": dict(sorted(variant_counts.items())),
        "num_nodes_distribution": {str(k): int(v) for k, v in sorted(node_counts.items())},
        "event_type_counts": dict(sorted(event_counts.items())),
        "dataset_keys": sorted(first.keys()) if first else [],
        "view_profiles": first.get("step31_view_profiles", []) if first else [],
        "num_views": len(views),
        "weak_observation_keys": sorted(views[0].keys()) if views else [],
        "weak_slot_feature_dim": int(views[0]["slot_features"].shape[1]) if views else None,
        "weak_pair_evidence_bundle_dim": (
            int(views[0]["pair_evidence_bundle"].shape[-1])
            if views and "pair_evidence_bundle" in views[0]
            else None
        ),
    }


def generate_one_file(
    source_path: Path,
    output_path: Path,
    variants: List[str],
    seed: int,
    num_types: int,
    max_samples: int | None,
    num_views: int,
    benchmark_version: str,
) -> Dict[str, Any]:
    source_samples = load_pickle(source_path)
    dataset = generate_step31_dataset(
        source_samples=source_samples,
        variants=variants,
        seed=seed,
        num_types=num_types,
        max_samples=max_samples,
        num_views=num_views,
        benchmark_version=benchmark_version,
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
    parser.add_argument("--seed", type=int, default=int(DEFAULT_CONFIG["random_seed"]) + 31)
    parser.add_argument("--num_types", type=int, default=int(DEFAULT_CONFIG["num_types"]))
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--num_views", type=int, default=3)
    parser.add_argument("--generate_default_splits", action="store_true")
    parser.add_argument("--source_train_path", type=str, default="data/graph_event_train.pkl")
    parser.add_argument("--source_val_path", type=str, default="data/graph_event_val.pkl")
    parser.add_argument("--source_test_path", type=str, default="data/graph_event_test.pkl")
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--output_prefix", type=str, default="graph_event_step31_multi_view")
    parser.add_argument("--benchmark_version", type=str, default=STEP31_BENCHMARK_VERSION)
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
            print(f"Generating Step31 {split_name} split")
            summaries[split_name] = generate_one_file(
                source_path=source_path,
                output_path=output_path,
                variants=variants,
                seed=args.seed,
                num_types=args.num_types,
                max_samples=args.max_samples,
                num_views=args.num_views,
                benchmark_version=args.benchmark_version,
            )
        if args.summary_json is not None:
            summary_path = resolve_path(args.summary_json)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with open(summary_path, "w") as f:
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
        num_views=args.num_views,
        benchmark_version=args.benchmark_version,
    )
    if args.summary_json is not None:
        summary_path = resolve_path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
