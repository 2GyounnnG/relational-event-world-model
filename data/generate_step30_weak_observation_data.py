from __future__ import annotations

import argparse
import copy
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


STEP30_BENCHMARK_VERSION = "step30_weak_obs_rev6"
STEP30_REV13_BENCHMARK_VERSION = "step30_weak_obs_rev13_signed_pair_witness"
STEP30_REV17_BENCHMARK_VERSION = "step30_weak_obs_rev17_pair_evidence_bundle"
STEP30_REV30_BENCHMARK_VERSION = "step30_weak_obs_rev30_positive_ambiguity_safety"


STEP30_WEAK_OBS_CONFIGS: Dict[str, Dict[str, float]] = {
    "clean": {
        "type_dropout_prob": 0.05,
        "type_flip_prob": 0.02,
        "state_noise_std": 0.05,
        "state_dropout_prob": 0.05,
        "state_quantization_step": 0.50,
        "relation_dropout_prob": 0.08,
        "relation_false_positive_prob": 0.04,
        "relation_jitter_std": 0.12,
        "relation_present_hint": 0.64,
        "relation_absent_hint": 0.36,
        "pair_support_present_hint": 0.58,
        "pair_support_absent_hint": 0.42,
        "pair_support_dropout_prob": 0.06,
        "pair_support_false_positive_prob": 0.06,
        "pair_support_jitter_std": 0.14,
        "signed_witness_missed_true_mean": 0.28,
        "signed_witness_true_edge_mean": 0.10,
        "signed_witness_unsafe_false_mean": -0.22,
        "signed_witness_other_false_mean": -0.06,
        "signed_witness_dropout_prob": 0.20,
        "signed_witness_flip_prob": 0.12,
        "signed_witness_jitter_std": 0.30,
        "signed_witness_quantization_step": 0.25,
        "signed_witness_type_compat_scale": 0.05,
        "signed_witness_state_compat_scale": 0.08,
        "pair_bundle_positive_missed_true_mean": 0.62,
        "pair_bundle_positive_true_edge_mean": 0.56,
        "pair_bundle_positive_false_edge_mean": 0.44,
        "pair_bundle_positive_unsafe_false_mean": 0.42,
        "pair_bundle_warning_unsafe_false_mean": 0.62,
        "pair_bundle_warning_other_false_mean": 0.52,
        "pair_bundle_warning_true_edge_mean": 0.40,
        "pair_bundle_corroboration_safe_mean": 0.58,
        "pair_bundle_corroboration_unsafe_mean": 0.42,
        "pair_bundle_corroboration_other_mean": 0.50,
        "pair_bundle_endpoint_type_scale": 0.08,
        "pair_bundle_endpoint_state_scale": 0.12,
        "pair_bundle_dropout_prob": 0.16,
        "pair_bundle_flip_prob": 0.10,
        "pair_bundle_jitter_std": 0.22,
        "pair_bundle_quantization_step": 0.20,
        "positive_ambiguity_safety_safe_mean": 0.66,
        "positive_ambiguity_safety_false_mean": 0.34,
        "positive_ambiguity_safety_other_mean": 0.50,
        "positive_ambiguity_safety_dropout_prob": 0.20,
        "positive_ambiguity_safety_flip_prob": 0.12,
        "positive_ambiguity_safety_jitter_std": 0.18,
        "positive_ambiguity_safety_quantization_step": 0.20,
        "positive_ambiguity_safety_local_coherence_scale": 0.10,
    },
    "noisy": {
        "type_dropout_prob": 0.20,
        "type_flip_prob": 0.08,
        "state_noise_std": 0.20,
        "state_dropout_prob": 0.20,
        "state_quantization_step": 0.75,
        "relation_dropout_prob": 0.18,
        "relation_false_positive_prob": 0.10,
        "relation_jitter_std": 0.14,
        "relation_present_hint": 0.62,
        "relation_absent_hint": 0.38,
        "pair_support_present_hint": 0.57,
        "pair_support_absent_hint": 0.43,
        "pair_support_dropout_prob": 0.10,
        "pair_support_false_positive_prob": 0.10,
        "pair_support_jitter_std": 0.18,
        "signed_witness_missed_true_mean": 0.24,
        "signed_witness_true_edge_mean": 0.08,
        "signed_witness_unsafe_false_mean": -0.18,
        "signed_witness_other_false_mean": -0.05,
        "signed_witness_dropout_prob": 0.30,
        "signed_witness_flip_prob": 0.16,
        "signed_witness_jitter_std": 0.36,
        "signed_witness_quantization_step": 0.25,
        "signed_witness_type_compat_scale": 0.04,
        "signed_witness_state_compat_scale": 0.06,
        "pair_bundle_positive_missed_true_mean": 0.60,
        "pair_bundle_positive_true_edge_mean": 0.55,
        "pair_bundle_positive_false_edge_mean": 0.45,
        "pair_bundle_positive_unsafe_false_mean": 0.43,
        "pair_bundle_warning_unsafe_false_mean": 0.60,
        "pair_bundle_warning_other_false_mean": 0.52,
        "pair_bundle_warning_true_edge_mean": 0.42,
        "pair_bundle_corroboration_safe_mean": 0.56,
        "pair_bundle_corroboration_unsafe_mean": 0.44,
        "pair_bundle_corroboration_other_mean": 0.50,
        "pair_bundle_endpoint_type_scale": 0.06,
        "pair_bundle_endpoint_state_scale": 0.10,
        "pair_bundle_dropout_prob": 0.25,
        "pair_bundle_flip_prob": 0.15,
        "pair_bundle_jitter_std": 0.28,
        "pair_bundle_quantization_step": 0.20,
        "positive_ambiguity_safety_safe_mean": 0.62,
        "positive_ambiguity_safety_false_mean": 0.38,
        "positive_ambiguity_safety_other_mean": 0.50,
        "positive_ambiguity_safety_dropout_prob": 0.30,
        "positive_ambiguity_safety_flip_prob": 0.17,
        "positive_ambiguity_safety_jitter_std": 0.24,
        "positive_ambiguity_safety_quantization_step": 0.20,
        "positive_ambiguity_safety_local_coherence_scale": 0.08,
    },
}


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


def variant_list(variants_arg: str) -> List[str]:
    if variants_arg == "both":
        return ["clean", "noisy"]
    if variants_arg not in STEP30_WEAK_OBS_CONFIGS:
        raise ValueError(f"Unknown Step30 variant: {variants_arg}")
    return [variants_arg]


def feature_names(num_types: int, state_dim: int) -> List[str]:
    return (
        [f"type_hint_{i}" for i in range(num_types)]
        + ["type_observed_mask"]
        + [f"state_hint_{i}" for i in range(state_dim)]
        + [f"state_observed_mask_{i}" for i in range(state_dim)]
    )


def quantize(x: np.ndarray, step: float) -> np.ndarray:
    if step <= 0:
        return x
    return (np.round(x / step) * step).astype(np.float32)


def make_slot_features(
    node_features: np.ndarray,
    variant: str,
    cfg: Dict[str, float],
    rng: np.random.Generator,
    num_types: int,
) -> np.ndarray:
    node_features = np.asarray(node_features, dtype=np.float32)
    num_nodes = node_features.shape[0]
    state_dim = node_features.shape[1] - 1

    type_hints = np.zeros((num_nodes, num_types), dtype=np.float32)
    type_observed = np.ones((num_nodes, 1), dtype=np.float32)
    state_hints = node_features[:, 1:].copy()
    state_observed = np.ones((num_nodes, state_dim), dtype=np.float32)

    for node_idx in range(num_nodes):
        true_type = int(node_features[node_idx, 0])
        observed_type = true_type
        if rng.random() < float(cfg["type_flip_prob"]):
            choices = [t for t in range(num_types) if t != true_type]
            observed_type = int(rng.choice(choices))
        if rng.random() < float(cfg["type_dropout_prob"]):
            type_observed[node_idx, 0] = 0.0
        else:
            type_hints[node_idx, observed_type] = 1.0

    noise_std = float(cfg["state_noise_std"])
    if noise_std > 0:
        state_hints += rng.normal(0.0, noise_std, size=state_hints.shape).astype(np.float32)
    state_hints = quantize(state_hints, float(cfg["state_quantization_step"]))

    dropout_prob = float(cfg["state_dropout_prob"])
    if dropout_prob > 0:
        keep_mask = (rng.random(size=state_hints.shape) >= dropout_prob).astype(np.float32)
        state_observed *= keep_mask
        state_hints *= keep_mask

    # Keep a variant argument in the signature for easy future extension and clear call sites.
    _ = variant
    return np.concatenate([type_hints, type_observed, state_hints, state_observed], axis=1).astype(np.float32)


def make_relation_hints(
    adj: np.ndarray,
    cfg: Dict[str, float],
    rng: np.random.Generator,
) -> np.ndarray:
    adj = np.asarray(adj, dtype=np.int64)
    num_nodes = adj.shape[0]
    present_hint = float(cfg["relation_present_hint"])
    absent_hint = float(cfg["relation_absent_hint"])
    relation_hints = np.where(adj > 0, present_hint, absent_hint).astype(np.float32)

    dropout_prob = float(cfg["relation_dropout_prob"])
    false_positive_prob = float(cfg["relation_false_positive_prob"])
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adj[i, j] == 1 and rng.random() < dropout_prob:
                relation_hints[i, j] = absent_hint
                relation_hints[j, i] = absent_hint
            elif adj[i, j] == 0 and rng.random() < false_positive_prob:
                relation_hints[i, j] = present_hint
                relation_hints[j, i] = present_hint

    jitter_std = float(cfg["relation_jitter_std"])
    if jitter_std > 0:
        jitter = rng.normal(0.0, jitter_std, size=relation_hints.shape).astype(np.float32)
        jitter = 0.5 * (jitter + jitter.T)
        relation_hints += jitter

    relation_hints = np.clip(relation_hints, 0.0, 1.0).astype(np.float32)
    np.fill_diagonal(relation_hints, 0.0)
    return relation_hints


def make_pair_support_hints(
    adj: np.ndarray,
    cfg: Dict[str, float],
    rng: np.random.Generator,
) -> np.ndarray:
    adj = np.asarray(adj, dtype=np.int64)
    num_nodes = adj.shape[0]
    present_hint = float(cfg["pair_support_present_hint"])
    absent_hint = float(cfg["pair_support_absent_hint"])
    support_hints = np.where(adj > 0, present_hint, absent_hint).astype(np.float32)

    dropout_prob = float(cfg["pair_support_dropout_prob"])
    false_positive_prob = float(cfg["pair_support_false_positive_prob"])
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adj[i, j] == 1 and rng.random() < dropout_prob:
                # Drop toward uncertainty rather than clean absence; this keeps the
                # support cue weak while still providing independent evidence.
                support_hints[i, j] = 0.5
                support_hints[j, i] = 0.5
            elif adj[i, j] == 0 and rng.random() < false_positive_prob:
                support_hints[i, j] = present_hint
                support_hints[j, i] = present_hint

    jitter_std = float(cfg["pair_support_jitter_std"])
    if jitter_std > 0:
        jitter = rng.normal(0.0, jitter_std, size=support_hints.shape).astype(np.float32)
        jitter = 0.5 * (jitter + jitter.T)
        support_hints += jitter

    support_hints = np.clip(support_hints, 0.0, 1.0).astype(np.float32)
    np.fill_diagonal(support_hints, 0.0)
    return support_hints


def make_signed_pair_witness(
    adj: np.ndarray,
    relation_hints: np.ndarray,
    pair_support_hints: np.ndarray,
    node_features: np.ndarray,
    cfg: Dict[str, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Weak signed pair cue for rev13 rescue-safety evidence.

    This is intentionally not a clean adjacency copy. It is a noisy, low-amplitude
    witness channel centered near zero, with overlapping positive/negative
    distributions and dropout/flip corruption. Positive values weakly support
    low-hint true-edge rescue; negative values weakly warn against unsupported
    false admissions.
    """

    adj = np.asarray(adj, dtype=np.int64)
    relation_hints = np.asarray(relation_hints, dtype=np.float32)
    pair_support_hints = np.asarray(pair_support_hints, dtype=np.float32)
    node_features = np.asarray(node_features, dtype=np.float32)
    num_nodes = adj.shape[0]
    witness = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    node_types = node_features[:, 0].astype(np.int64)
    states = node_features[:, 1:].astype(np.float32)
    if states.size:
        state_dist = np.abs(states[:, None, :] - states[None, :, :]).mean(axis=-1)
        state_similarity = np.exp(-state_dist).astype(np.float32)
    else:
        state_similarity = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    type_compat_scale = float(cfg["signed_witness_type_compat_scale"])
    state_compat_scale = float(cfg["signed_witness_state_compat_scale"])
    type_compat = (node_types[:, None] == node_types[None, :]).astype(np.float32) - 0.5
    state_compat = state_similarity - float(state_similarity.mean())

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            low_relation = relation_hints[i, j] < 0.5
            support_candidate = pair_support_hints[i, j] >= 0.55
            if adj[i, j] == 1 and low_relation:
                mean = float(cfg["signed_witness_missed_true_mean"])
            elif adj[i, j] == 0 and low_relation and support_candidate:
                mean = float(cfg["signed_witness_unsafe_false_mean"])
            elif adj[i, j] == 1:
                mean = float(cfg["signed_witness_true_edge_mean"])
            else:
                mean = float(cfg["signed_witness_other_false_mean"])

            mean += type_compat_scale * float(type_compat[i, j])
            mean += state_compat_scale * float(state_compat[i, j])

            if rng.random() < float(cfg["signed_witness_dropout_prob"]):
                mean = 0.0
            if rng.random() < float(cfg["signed_witness_flip_prob"]):
                mean = -mean

            value = mean + float(rng.normal(0.0, float(cfg["signed_witness_jitter_std"])))
            step = float(cfg["signed_witness_quantization_step"])
            if step > 0:
                value = round(value / step) * step
            value = float(np.clip(value, -1.0, 1.0))
            witness[i, j] = value
            witness[j, i] = value

    np.fill_diagonal(witness, 0.0)
    return witness.astype(np.float32)


def _sample_weak_bundle_value(mean: float, cfg: Dict[str, float], rng: np.random.Generator) -> float:
    """Sample one weak 0..1 evidence value with overlap and corruption."""

    if rng.random() < float(cfg["pair_bundle_dropout_prob"]):
        mean = 0.5
    if rng.random() < float(cfg["pair_bundle_flip_prob"]):
        mean = 1.0 - mean
    value = mean + float(rng.normal(0.0, float(cfg["pair_bundle_jitter_std"])))
    step = float(cfg["pair_bundle_quantization_step"])
    if step > 0:
        value = round(value / step) * step
    return float(np.clip(value, 0.0, 1.0))


def _sample_positive_ambiguity_safety_value(
    mean: float,
    cfg: Dict[str, float],
    rng: np.random.Generator,
) -> float:
    """Sample the rev30 ambiguity-safety cue with heavy overlap and corruption."""

    if rng.random() < float(cfg["positive_ambiguity_safety_dropout_prob"]):
        mean = 0.5
    if rng.random() < float(cfg["positive_ambiguity_safety_flip_prob"]):
        mean = 1.0 - mean
    value = mean + float(rng.normal(0.0, float(cfg["positive_ambiguity_safety_jitter_std"])))
    step = float(cfg["positive_ambiguity_safety_quantization_step"])
    if step > 0:
        value = round(value / step) * step
    return float(np.clip(value, 0.0, 1.0))


def make_pair_evidence_bundle(
    adj: np.ndarray,
    relation_hints: np.ndarray,
    pair_support_hints: np.ndarray,
    node_features: np.ndarray,
    cfg: Dict[str, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Small multi-source weak pair evidence packet for rev17.

    Channel semantics:
    0. weak positive-support cue
    1. weak false-admission warning cue
    2. weak corroboration cue for low-relation rescue cases
    3. noisy endpoint-compatibility cue

    These cues are intentionally low-amplitude and corrupted. No single channel is
    meant to expose clean adjacency; the probe asks whether a tiny packet of
    complementary weak reasons helps more than one scalar witness.
    """

    adj = np.asarray(adj, dtype=np.int64)
    relation_hints = np.asarray(relation_hints, dtype=np.float32)
    pair_support_hints = np.asarray(pair_support_hints, dtype=np.float32)
    node_features = np.asarray(node_features, dtype=np.float32)
    num_nodes = adj.shape[0]
    bundle = np.zeros((num_nodes, num_nodes, 4), dtype=np.float32)

    node_types = node_features[:, 0].astype(np.int64)
    states = node_features[:, 1:].astype(np.float32)
    if states.size:
        state_dist = np.abs(states[:, None, :] - states[None, :, :]).mean(axis=-1)
        state_similarity = np.exp(-state_dist).astype(np.float32)
    else:
        state_similarity = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    state_center = float(state_similarity.mean()) if state_similarity.size else 0.0
    type_compat = (node_types[:, None] == node_types[None, :]).astype(np.float32) - 0.5
    state_compat = state_similarity - state_center

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            low_relation = relation_hints[i, j] < 0.5
            support_candidate = pair_support_hints[i, j] >= 0.55
            target_edge = adj[i, j] == 1
            unsafe_false = (not target_edge) and low_relation and support_candidate
            missed_true = target_edge and low_relation

            if missed_true:
                positive_mean = float(cfg["pair_bundle_positive_missed_true_mean"])
            elif target_edge:
                positive_mean = float(cfg["pair_bundle_positive_true_edge_mean"])
            elif unsafe_false:
                positive_mean = float(cfg["pair_bundle_positive_unsafe_false_mean"])
            else:
                positive_mean = float(cfg["pair_bundle_positive_false_edge_mean"])

            if unsafe_false:
                warning_mean = float(cfg["pair_bundle_warning_unsafe_false_mean"])
            elif target_edge:
                warning_mean = float(cfg["pair_bundle_warning_true_edge_mean"])
            else:
                warning_mean = float(cfg["pair_bundle_warning_other_false_mean"])

            if missed_true and support_candidate:
                corroboration_mean = float(cfg["pair_bundle_corroboration_safe_mean"])
            elif unsafe_false:
                corroboration_mean = float(cfg["pair_bundle_corroboration_unsafe_mean"])
            else:
                corroboration_mean = float(cfg["pair_bundle_corroboration_other_mean"])

            endpoint_mean = 0.5
            endpoint_mean += float(cfg["pair_bundle_endpoint_type_scale"]) * float(type_compat[i, j])
            endpoint_mean += float(cfg["pair_bundle_endpoint_state_scale"]) * float(state_compat[i, j])

            values = [
                _sample_weak_bundle_value(positive_mean, cfg, rng),
                _sample_weak_bundle_value(warning_mean, cfg, rng),
                _sample_weak_bundle_value(corroboration_mean, cfg, rng),
                _sample_weak_bundle_value(endpoint_mean, cfg, rng),
            ]
            bundle[i, j, :] = np.asarray(values, dtype=np.float32)
            bundle[j, i, :] = bundle[i, j, :]

    return bundle.astype(np.float32)


def make_positive_ambiguity_safety_hint(
    adj: np.ndarray,
    relation_hints: np.ndarray,
    pair_support_hints: np.ndarray,
    pair_evidence_bundle: np.ndarray,
    node_features: np.ndarray,
    cfg: Dict[str, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Weak rev30 safety cue for positive-looking ambiguous rescue candidates.

    High values weakly suggest that positive-looking ambiguity is internally
    coherent; low values weakly warn that the same positive-looking evidence is
    suspicious. The cue is targeted: outside low-relation, support-backed,
    positive-looking ambiguity it is centered near 0.5. It is not a clean
    adjacency copy: distributions overlap strongly and every value is corrupted
    by dropout, sign flips, jitter, and quantization.
    """

    adj = np.asarray(adj, dtype=np.int64)
    relation_hints = np.asarray(relation_hints, dtype=np.float32)
    pair_support_hints = np.asarray(pair_support_hints, dtype=np.float32)
    pair_evidence_bundle = np.asarray(pair_evidence_bundle, dtype=np.float32)
    node_features = np.asarray(node_features, dtype=np.float32)
    num_nodes = adj.shape[0]
    hint = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    node_types = node_features[:, 0].astype(np.int64)
    states = node_features[:, 1:].astype(np.float32)
    if states.size:
        state_dist = np.abs(states[:, None, :] - states[None, :, :]).mean(axis=-1)
        state_similarity = np.exp(-state_dist).astype(np.float32)
    else:
        state_similarity = np.full((num_nodes, num_nodes), 0.5, dtype=np.float32)

    adj_float = adj.astype(np.float32)
    common_neighbors = adj_float @ adj_float
    max_common = float(common_neighbors.max()) if common_neighbors.size else 0.0
    if max_common > 0:
        common_neighbors = common_neighbors / max_common
    type_match = (node_types[:, None] == node_types[None, :]).astype(np.float32)
    local_coherence = (
        0.45 * common_neighbors
        + 0.30 * state_similarity
        + 0.25 * type_match
    ).astype(np.float32)

    positive = pair_evidence_bundle[..., 0]
    warning = pair_evidence_bundle[..., 1]
    corroboration = pair_evidence_bundle[..., 2]
    endpoint_compat = pair_evidence_bundle[..., 3]

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            low_relation = relation_hints[i, j] < 0.5
            support_candidate = pair_support_hints[i, j] >= 0.55
            positive_looking = (
                (positive[i, j] - warning[i, j]) >= 0.12
                and (
                    corroboration[i, j] >= 0.45
                    or endpoint_compat[i, j] >= 0.50
                )
            )
            ambiguous_like = (
                relation_hints[i, j] >= 0.45
                or pair_support_hints[i, j] < 0.65
                or abs(float(positive[i, j] - warning[i, j])) < 0.22
            )
            targeted = low_relation and support_candidate and positive_looking and ambiguous_like
            if targeted and adj[i, j] == 1:
                mean = float(cfg["positive_ambiguity_safety_safe_mean"])
            elif targeted and adj[i, j] == 0:
                mean = float(cfg["positive_ambiguity_safety_false_mean"])
            else:
                mean = float(cfg["positive_ambiguity_safety_other_mean"])

            coherence_delta = float(local_coherence[i, j] - 0.5)
            mean += float(cfg["positive_ambiguity_safety_local_coherence_scale"]) * coherence_delta
            value = _sample_positive_ambiguity_safety_value(mean, cfg, rng)
            hint[i, j] = value
            hint[j, i] = value

    np.fill_diagonal(hint, 0.0)
    return hint.astype(np.float32)


def make_weak_observation(
    graph: Dict[str, Any],
    variant: str,
    cfg: Dict[str, float],
    rng: np.random.Generator,
    num_types: int,
    include_signed_pair_witness: bool = False,
    include_pair_evidence_bundle: bool = False,
    include_positive_ambiguity_safety_hint: bool = False,
) -> Dict[str, Any]:
    node_features = np.asarray(graph["node_features"], dtype=np.float32)
    adj = np.asarray(graph["adj"], dtype=np.int64)
    state_dim = node_features.shape[1] - 1
    slot_features = make_slot_features(node_features, variant, cfg, rng, num_types=num_types)
    relation_hints = make_relation_hints(adj, cfg, rng)
    pair_support_hints = make_pair_support_hints(adj, cfg, rng)
    weak_observation = {
        "slot_features": slot_features,
        "relation_hints": relation_hints,
        "pair_support_hints": pair_support_hints,
        "slot_mask": np.ones((node_features.shape[0],), dtype=np.float32),
        "feature_names": feature_names(num_types=num_types, state_dim=state_dim),
        "variant": variant,
        "config": dict(cfg),
    }
    if include_signed_pair_witness:
        weak_observation["signed_pair_witness"] = make_signed_pair_witness(
            adj=adj,
            relation_hints=relation_hints,
            pair_support_hints=pair_support_hints,
            node_features=node_features,
            cfg=cfg,
            rng=rng,
        )
    pair_evidence_bundle = None
    if include_pair_evidence_bundle:
        pair_evidence_bundle = make_pair_evidence_bundle(
            adj=adj,
            relation_hints=relation_hints,
            pair_support_hints=pair_support_hints,
            node_features=node_features,
            cfg=cfg,
            rng=rng,
        )
        weak_observation["pair_evidence_bundle"] = pair_evidence_bundle
    if include_positive_ambiguity_safety_hint:
        if pair_evidence_bundle is None:
            raise ValueError(
                "positive_ambiguity_safety_hint requires include_pair_evidence_bundle=True"
            )
        weak_observation["positive_ambiguity_safety_hint"] = make_positive_ambiguity_safety_hint(
            adj=adj,
            relation_hints=relation_hints,
            pair_support_hints=pair_support_hints,
            pair_evidence_bundle=pair_evidence_bundle,
            node_features=node_features,
            cfg=cfg,
            rng=rng,
        )
    return weak_observation


def make_step30_sample(
    raw_sample: Dict[str, Any],
    source_sample_index: int,
    variant: str,
    variant_index: int,
    seed: int,
    num_types: int,
    include_signed_pair_witness: bool = False,
    include_pair_evidence_bundle: bool = False,
    include_positive_ambiguity_safety_hint: bool = False,
    benchmark_version: str = STEP30_BENCHMARK_VERSION,
) -> Dict[str, Any]:
    sample = copy.deepcopy(raw_sample)
    cfg = STEP30_WEAK_OBS_CONFIGS[variant]
    rng = np.random.default_rng(seed + source_sample_index * 1009 + variant_index * 100_003)

    sample["weak_observation"] = make_weak_observation(
        sample["graph_t"],
        variant=variant,
        cfg=cfg,
        rng=rng,
        num_types=num_types,
        include_signed_pair_witness=include_signed_pair_witness,
        include_pair_evidence_bundle=include_pair_evidence_bundle,
        include_positive_ambiguity_safety_hint=include_positive_ambiguity_safety_hint,
    )
    sample["step30_observation_variant"] = variant
    sample["step30_benchmark_version"] = benchmark_version
    sample["step30_observation_config"] = dict(cfg)
    sample["step30_source_sample_index"] = source_sample_index
    sample["step30_note"] = (
        "weak_observation is slot-aligned to graph_t nodes; graph_t is the clean "
        "current structured recovery target and graph_t1 remains the clean next-state target."
    )
    if "edge_index" not in sample["graph_t"]:
        sample["graph_t"]["edge_index"] = graph_to_edge_index(np.asarray(sample["graph_t"]["adj"]))
    if "edge_index" not in sample["graph_t1"]:
        sample["graph_t1"]["edge_index"] = graph_to_edge_index(np.asarray(sample["graph_t1"]["adj"]))
    return sample


def generate_step30_dataset(
    source_samples: List[Dict[str, Any]],
    variants: Iterable[str],
    seed: int,
    num_types: int,
    max_samples: int | None = None,
    include_signed_pair_witness: bool = False,
    include_pair_evidence_bundle: bool = False,
    include_positive_ambiguity_safety_hint: bool = False,
    benchmark_version: str = STEP30_BENCHMARK_VERSION,
) -> List[Dict[str, Any]]:
    selected = source_samples if max_samples is None else source_samples[:max_samples]
    out: list[Dict[str, Any]] = []
    variants_list = list(variants)
    for source_idx, raw_sample in enumerate(selected):
        for variant_idx, variant in enumerate(variants_list):
            out.append(
                make_step30_sample(
                    raw_sample=raw_sample,
                    source_sample_index=source_idx,
                    variant=variant,
                    variant_index=variant_idx,
                    seed=seed,
                    num_types=num_types,
                    include_signed_pair_witness=include_signed_pair_witness,
                    include_pair_evidence_bundle=include_pair_evidence_bundle,
                    include_positive_ambiguity_safety_hint=include_positive_ambiguity_safety_hint,
                    benchmark_version=benchmark_version,
                )
            )
    return out


def summarize(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    variant_counts = Counter(str(sample.get("step30_observation_variant", "unknown")) for sample in dataset)
    event_counts: Counter[str] = Counter()
    node_counts: Counter[int] = Counter()
    for sample in dataset:
        node_counts[int(np.asarray(sample["graph_t"]["node_features"]).shape[0])] += 1
        for event in sample.get("events", []):
            event_counts[str(event.get("event_type", "unknown"))] += 1
    return {
        "sample_count": len(dataset),
        "observation_variant_counts": dict(sorted(variant_counts.items())),
        "num_nodes_distribution": {str(k): int(v) for k, v in sorted(node_counts.items())},
        "event_type_counts": dict(sorted(event_counts.items())),
        "dataset_keys": sorted(dataset[0].keys()) if dataset else [],
        "weak_observation_keys": sorted(dataset[0]["weak_observation"].keys()) if dataset else [],
        "weak_slot_feature_dim": int(dataset[0]["weak_observation"]["slot_features"].shape[1]) if dataset else None,
        "weak_pair_evidence_bundle_dim": (
            int(dataset[0]["weak_observation"]["pair_evidence_bundle"].shape[-1])
            if dataset and "pair_evidence_bundle" in dataset[0]["weak_observation"]
            else None
        ),
        "has_positive_ambiguity_safety_hint": (
            bool(dataset and "positive_ambiguity_safety_hint" in dataset[0]["weak_observation"])
        ),
    }


def generate_one_file(
    source_path: Path,
    output_path: Path,
    variants: List[str],
    seed: int,
    num_types: int,
    max_samples: int | None,
    include_signed_pair_witness: bool,
    include_pair_evidence_bundle: bool,
    include_positive_ambiguity_safety_hint: bool,
    benchmark_version: str,
) -> Dict[str, Any]:
    source_samples = load_pickle(source_path)
    dataset = generate_step30_dataset(
        source_samples=source_samples,
        variants=variants,
        seed=seed,
        num_types=num_types,
        max_samples=max_samples,
        include_signed_pair_witness=include_signed_pair_witness,
        include_pair_evidence_bundle=include_pair_evidence_bundle,
        include_positive_ambiguity_safety_hint=include_positive_ambiguity_safety_hint,
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
    parser.add_argument("--seed", type=int, default=int(DEFAULT_CONFIG["random_seed"]))
    parser.add_argument("--num_types", type=int, default=int(DEFAULT_CONFIG["num_types"]))
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--generate_default_splits", action="store_true")
    parser.add_argument("--source_train_path", type=str, default="data/graph_event_train.pkl")
    parser.add_argument("--source_val_path", type=str, default="data/graph_event_val.pkl")
    parser.add_argument("--source_test_path", type=str, default="data/graph_event_test.pkl")
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--output_prefix", type=str, default="graph_event_step30_weak_obs")
    parser.add_argument("--include_signed_pair_witness", action="store_true")
    parser.add_argument("--include_pair_evidence_bundle", action="store_true")
    parser.add_argument("--include_positive_ambiguity_safety_hint", action="store_true")
    parser.add_argument("--benchmark_version", type=str, default=None)
    args = parser.parse_args()

    if args.include_positive_ambiguity_safety_hint and not args.include_pair_evidence_bundle:
        raise SystemExit(
            "--include_positive_ambiguity_safety_hint requires --include_pair_evidence_bundle"
        )

    variants = variant_list(args.variants)
    benchmark_version = args.benchmark_version
    if benchmark_version is None:
        benchmark_version = (
            STEP30_REV30_BENCHMARK_VERSION
            if args.include_positive_ambiguity_safety_hint
            else STEP30_REV17_BENCHMARK_VERSION
            if args.include_pair_evidence_bundle
            else
            STEP30_REV13_BENCHMARK_VERSION
            if args.include_signed_pair_witness
            else STEP30_BENCHMARK_VERSION
        )

    if args.generate_default_splits:
        output_dir = resolve_path(args.output_dir)
        split_specs = [
            ("train", resolve_path(args.source_train_path), output_dir / f"{args.output_prefix}_train.pkl"),
            ("val", resolve_path(args.source_val_path), output_dir / f"{args.output_prefix}_val.pkl"),
            ("test", resolve_path(args.source_test_path), output_dir / f"{args.output_prefix}_test.pkl"),
        ]
        summaries = {}
        for split_name, source_path, output_path in split_specs:
            print(f"Generating Step30 {split_name} split")
            summaries[split_name] = generate_one_file(
                source_path=source_path,
                output_path=output_path,
                variants=variants,
                seed=args.seed,
                num_types=args.num_types,
                max_samples=args.max_samples,
                include_signed_pair_witness=args.include_signed_pair_witness,
                include_pair_evidence_bundle=args.include_pair_evidence_bundle,
                include_positive_ambiguity_safety_hint=args.include_positive_ambiguity_safety_hint,
                benchmark_version=benchmark_version,
            )
        print(f"default split summaries: {summaries}")
        return

    if args.source_path is None or args.output_path is None:
        raise SystemExit("Provide --source_path and --output_path, or use --generate_default_splits.")

    generate_one_file(
        source_path=resolve_path(args.source_path),
        output_path=resolve_path(args.output_path),
        variants=variants,
        seed=args.seed,
        num_types=args.num_types,
        max_samples=args.max_samples,
        include_signed_pair_witness=args.include_signed_pair_witness,
        include_pair_evidence_bundle=args.include_pair_evidence_bundle,
        include_positive_ambiguity_safety_hint=args.include_positive_ambiguity_safety_hint,
        benchmark_version=benchmark_version,
    )


if __name__ == "__main__":
    main()
