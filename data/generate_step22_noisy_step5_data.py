from __future__ import annotations

import argparse
import copy
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.generate_graph_event_data import DEFAULT_CONFIG, corrupt_graph_observation, graph_to_edge_index


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


def add_edge_index(graph: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(graph)
    out["edge_index"] = graph_to_edge_index(np.asarray(out["adj"]))
    return out


def clean_current_graphs_for_step5(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return S0, S1, S2 current-state graphs for a 3-event Step 5 sample."""
    horizon = int(sample.get("horizon", len(sample.get("graph_steps", []))))
    current_graphs = [sample["graph_0"]]
    for step_idx in range(1, horizon):
        current_graphs.append(sample["graph_steps"][step_idx - 1])
    return current_graphs


def make_noisy_step5_sample(
    raw_sample: Dict[str, Any],
    source_sample_index: int,
    setting_name: str,
    corruption_cfg: Dict[str, float],
    base_seed: int,
    setting_index: int,
) -> Dict[str, Any]:
    noisy_sample = copy.deepcopy(raw_sample)
    noisy_inputs: list[Dict[str, Any]] = []
    for step_idx, clean_graph in enumerate(clean_current_graphs_for_step5(raw_sample)):
        rng = np.random.default_rng(
            base_seed
            + source_sample_index * 1009
            + setting_index * 100_003
            + step_idx * 9176
        )
        noisy_inputs.append(add_edge_index(corrupt_graph_observation(clean_graph, corruption_cfg, rng)))

    noisy_sample["obs_graph_inputs"] = noisy_inputs
    noisy_sample["step22_observation_regime"] = "noisy"
    noisy_sample["step22_corruption_setting"] = setting_name
    noisy_sample["step22_corruption_config"] = dict(corruption_cfg)
    noisy_sample["step22_source_sample_index"] = source_sample_index
    noisy_sample["step22_note"] = (
        "obs_graph_inputs[k] is the corrupted observation of the clean current graph "
        "for transition k; graph_steps remain clean targets."
    )
    return noisy_sample


def generate_noisy_step5_dataset(
    clean_samples: List[Dict[str, Any]],
    corruption_settings: Dict[str, Dict[str, float]],
    seed: int,
) -> List[Dict[str, Any]]:
    dataset: list[Dict[str, Any]] = []
    for sample_idx, raw_sample in enumerate(clean_samples):
        for setting_index, setting_name in enumerate(sorted(corruption_settings.keys())):
            dataset.append(
                make_noisy_step5_sample(
                    raw_sample=raw_sample,
                    source_sample_index=sample_idx,
                    setting_name=setting_name,
                    corruption_cfg=corruption_settings[setting_name],
                    base_seed=seed,
                    setting_index=setting_index,
                )
            )
    return dataset


def summarize(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    bucket_counts = Counter(str(sample.get("step5_dependency_bucket", "unknown")) for sample in dataset)
    setting_counts = Counter(str(sample.get("step22_corruption_setting", "unknown")) for sample in dataset)
    horizon_counts = Counter(int(sample.get("horizon", len(sample.get("graph_steps", [])))) for sample in dataset)
    return {
        "sample_count": len(dataset),
        "horizon_distribution": {str(k): int(v) for k, v in sorted(horizon_counts.items())},
        "dependency_bucket_counts": dict(sorted(bucket_counts.items())),
        "corruption_setting_counts": dict(sorted(setting_counts.items())),
        "dataset_keys": sorted(dataset[0].keys()) if dataset else [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_step5_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=int(DEFAULT_CONFIG["random_seed"]))
    args = parser.parse_args()

    clean_path = resolve_path(args.clean_step5_path)
    output_path = resolve_path(args.output_path)
    clean_samples = load_pickle(clean_path)
    dataset = generate_noisy_step5_dataset(
        clean_samples=clean_samples,
        corruption_settings=DEFAULT_CONFIG["step6a_corruption_settings"],
        seed=args.seed,
    )
    save_pickle(output_path, dataset)
    summary = summarize(dataset)
    print(f"clean Step 5 path: {clean_path}")
    print(f"output path: {output_path}")
    print(f"summary: {summary}")
    if dataset:
        example = dataset[0]
        print(
            "example metadata:",
            {
                "step5_sample_id": example.get("step5_sample_id"),
                "step5_dependency_bucket": example.get("step5_dependency_bucket"),
                "step22_corruption_setting": example.get("step22_corruption_setting"),
                "obs_graph_inputs": len(example.get("obs_graph_inputs", [])),
                "event_type_sequence": example.get("event_type_sequence"),
            },
        )


if __name__ == "__main__":
    main()
