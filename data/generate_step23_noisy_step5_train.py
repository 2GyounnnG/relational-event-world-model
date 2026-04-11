from __future__ import annotations

import argparse
import copy
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.generate_graph_event_data import DEFAULT_CONFIG
from data.generate_step22_noisy_step5_data import generate_noisy_step5_dataset, load_pickle, resolve_path, save_pickle


def flatten_noisy_step5_transitions(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert Step 22 rollout-style noisy Step 5 samples into single-transition samples.

    The proposal trainer can then reuse the existing GraphEventDataset path:
    graph_t is the clean current latent state, obs_graph_t is the corrupted
    structured observation, and graph_t1 is the clean target next state.
    中文说明：训练输入用 obs_graph_t，监督标签仍来自 clean oracle scope。
    """
    transitions: list[Dict[str, Any]] = []
    for sample in samples:
        obs_inputs = sample.get("obs_graph_inputs")
        transition_samples = sample.get("transition_samples")
        if obs_inputs is None or transition_samples is None:
            raise KeyError("Step 23 flattening requires Step 22 samples with obs_graph_inputs and transition_samples.")
        for step_idx, transition in enumerate(transition_samples):
            item = copy.deepcopy(transition)
            item["obs_graph_t"] = copy.deepcopy(obs_inputs[step_idx])
            item["step5_sample_id"] = sample.get("step5_sample_id")
            item["step5_dependency_bucket"] = sample.get("step5_dependency_bucket")
            item["step5_dependency_reason"] = sample.get("step5_dependency_reason")
            item["step5_ordered_signature"] = sample.get("step5_ordered_signature")
            item["step5_unordered_signature"] = sample.get("step5_unordered_signature")
            item["step23_step_index"] = step_idx + 1
            item["step23_source_sample_index"] = sample.get("step22_source_sample_index")
            item["step23_observation_regime"] = "noisy"
            item["step23_training_regime"] = "noisy_interaction_aware_P2"
            # Reuse existing collate metadata keys for corruption grouping.
            item["step6a_corruption_setting"] = sample.get("step22_corruption_setting")
            item["step6a_corruption_config"] = sample.get("step22_corruption_config")
            item["step6a_source_sample_index"] = sample.get("step22_source_sample_index")
            transitions.append(item)
    return transitions


def summarize(samples: List[Dict[str, Any]], transitions: List[Dict[str, Any]]) -> Dict[str, Any]:
    sample_buckets = Counter(str(sample.get("step5_dependency_bucket", "unknown")) for sample in samples)
    sample_settings = Counter(str(sample.get("step22_corruption_setting", "unknown")) for sample in samples)
    transition_buckets = Counter(str(item.get("step5_dependency_bucket", "unknown")) for item in transitions)
    transition_settings = Counter(str(item.get("step6a_corruption_setting", "unknown")) for item in transitions)
    event_types = Counter(str(item["events"][0].get("event_type", "unknown")) for item in transitions if item.get("events"))
    return {
        "sequence_sample_count": len(samples),
        "transition_sample_count": len(transitions),
        "sequence_dependency_bucket_counts": dict(sorted(sample_buckets.items())),
        "sequence_corruption_setting_counts": dict(sorted(sample_settings.items())),
        "transition_dependency_bucket_counts": dict(sorted(transition_buckets.items())),
        "transition_corruption_setting_counts": dict(sorted(transition_settings.items())),
        "transition_event_type_counts": dict(sorted(event_types.items())),
    }


def ensure_noisy_step22_split(clean_path: Path, noisy_output_path: Path, seed: int) -> List[Dict[str, Any]]:
    if noisy_output_path.exists():
        return load_pickle(noisy_output_path)
    clean_samples = load_pickle(clean_path)
    noisy_samples = generate_noisy_step5_dataset(
        clean_samples=clean_samples,
        corruption_settings=DEFAULT_CONFIG["step6a_corruption_settings"],
        seed=seed,
    )
    save_pickle(noisy_output_path, noisy_samples)
    return noisy_samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_step5_train_path", type=str, default="data/graph_event_step5_train.pkl")
    parser.add_argument("--clean_step5_val_path", type=str, default="data/graph_event_step5_val.pkl")
    parser.add_argument("--step22_train_output_path", type=str, default="data/graph_event_step22_noisy_step5_train.pkl")
    parser.add_argument("--step22_val_output_path", type=str, default="data/graph_event_step22_noisy_step5_val.pkl")
    parser.add_argument("--transition_train_output_path", type=str, default="data/graph_event_step23_noisy_step5_train_transitions.pkl")
    parser.add_argument("--transition_val_output_path", type=str, default="data/graph_event_step23_noisy_step5_val_transitions.pkl")
    parser.add_argument("--seed", type=int, default=int(DEFAULT_CONFIG["random_seed"]))
    args = parser.parse_args()

    clean_train_path = resolve_path(args.clean_step5_train_path)
    clean_val_path = resolve_path(args.clean_step5_val_path)
    step22_train_path = resolve_path(args.step22_train_output_path)
    step22_val_path = resolve_path(args.step22_val_output_path)
    transition_train_path = resolve_path(args.transition_train_output_path)
    transition_val_path = resolve_path(args.transition_val_output_path)

    train_sequences = ensure_noisy_step22_split(clean_train_path, step22_train_path, args.seed)
    val_sequences = ensure_noisy_step22_split(clean_val_path, step22_val_path, args.seed)
    train_transitions = flatten_noisy_step5_transitions(train_sequences)
    val_transitions = flatten_noisy_step5_transitions(val_sequences)
    save_pickle(transition_train_path, train_transitions)
    save_pickle(transition_val_path, val_transitions)

    train_summary = summarize(train_sequences, train_transitions)
    val_summary = summarize(val_sequences, val_transitions)
    print(f"step22 train sequence path: {step22_train_path}")
    print(f"step22 val sequence path: {step22_val_path}")
    print(f"transition train path: {transition_train_path}")
    print(f"transition val path: {transition_val_path}")
    print(f"train summary: {train_summary}")
    print(f"val summary: {val_summary}")
    if train_transitions:
        first = train_transitions[0]
        print(
            "example transition metadata:",
            {
                "step5_dependency_bucket": first.get("step5_dependency_bucket"),
                "step6a_corruption_setting": first.get("step6a_corruption_setting"),
                "step23_step_index": first.get("step23_step_index"),
                "event_type": first["events"][0].get("event_type") if first.get("events") else None,
                "has_obs_graph_t": "obs_graph_t" in first,
            },
        )


if __name__ == "__main__":
    main()
