from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.step30_dataset import Step30WeakObservationDataset, step30_weak_observation_collate_fn
from models.encoder_step30 import build_pair_mask
from train.eval_step30_encoder_recovery import (
    load_model,
    move_batch_to_device,
    trivial_baseline_outputs,
)
from train.eval_step30_rescue_ambiguity_subtype_probe_rev28 import (
    edge_metrics_from_arrays,
    parse_thresholds,
    subtype_for_candidate,
    threshold_array,
)
from train.eval_step30_rescue_candidate_latent_probe_rev23 import (
    average_precision,
    auroc,
    binary_metrics,
    clean_json_numbers,
    rescue_class_label,
    sigmoid_np,
)
from train.eval_step30_weak_positive_ambiguity_safety_probe_rev29 import (
    ProbeFit,
    fit_logistic_probe,
    probe_scores,
    rev25_score,
)


BASE_FEATURE_NAMES = [
    "relation_hint",
    "pair_support_hint",
    "support_minus_relation",
    "signed_pair_witness",
    "bundle_positive",
    "bundle_warning",
    "bundle_corroboration",
    "bundle_endpoint_compat",
    "bundle_margin_positive_minus_warning",
    "bundle_total_evidence",
    "bundle_abs_margin",
    "rev6_score",
    "rev6_logit",
    "rev24_safe_prob",
    "rev24_false_prob",
    "rev24_ambiguous_prob",
    "rev24_safe_minus_reject",
    "rev26_binary_prob",
    "rev26_ambiguous_prob",
    "rev26_binary_minus_ambiguous",
]

REV30_FEATURE_NAMES = BASE_FEATURE_NAMES + [
    "positive_ambiguity_safety_hint",
    "positive_ambiguity_safety_centered",
    "safety_times_bundle_margin",
]


@dataclass
class Rev30Rows:
    target: np.ndarray
    variant: np.ndarray
    relation: np.ndarray
    support: np.ndarray
    signed_witness: np.ndarray
    bundle_positive: np.ndarray
    bundle_warning: np.ndarray
    bundle_corroboration: np.ndarray
    bundle_endpoint_compat: np.ndarray
    positive_ambiguity_safety_hint: np.ndarray
    is_candidate: np.ndarray
    candidate_label: np.ndarray
    is_ambiguous_signal: np.ndarray
    subtype: np.ndarray
    rev6_score: np.ndarray
    rev17_score: np.ndarray
    rev24_safe_prob: np.ndarray
    rev24_false_prob: np.ndarray
    rev24_ambiguous_prob: np.ndarray
    rev26_binary_prob: np.ndarray
    rev26_ambiguous_prob: np.ndarray
    rev30_safe_prob: np.ndarray
    rev30_false_prob: np.ndarray
    rev30_ambiguous_prob: np.ndarray
    rev30_binary_prob: np.ndarray
    rev30_weak_positive_safety_prob: np.ndarray
    trivial_score: np.ndarray


def write_csv(path: Path, rows: list[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def logit_np(prob: np.ndarray) -> np.ndarray:
    clipped = np.clip(prob, 1e-5, 1.0 - 1e-5)
    return np.log(clipped / (1.0 - clipped))


def rev26_score(rows: Rev30Rows) -> np.ndarray:
    return rows.rev26_binary_prob - rows.rev26_ambiguous_prob


def rev30_score(rows: Rev30Rows) -> np.ndarray:
    return rows.rev30_binary_prob - rows.rev30_ambiguous_prob


def weak_positive_mask(rows: Rev30Rows) -> np.ndarray:
    return rows.is_ambiguous_signal & (rows.subtype == "weak_positive_ambiguous")


def feature_matrix(rows: Rev30Rows, include_safety_hint: bool) -> tuple[np.ndarray, list[str]]:
    rev24_margin = rows.rev24_safe_prob - np.maximum(rows.rev24_false_prob, rows.rev24_ambiguous_prob)
    rev26_margin = rev26_score(rows)
    bundle_margin = rows.bundle_positive - rows.bundle_warning
    features = [
        rows.relation,
        rows.support,
        rows.support - rows.relation,
        rows.signed_witness,
        rows.bundle_positive,
        rows.bundle_warning,
        rows.bundle_corroboration,
        rows.bundle_endpoint_compat,
        bundle_margin,
        rows.bundle_positive + rows.bundle_warning,
        np.abs(bundle_margin),
        rows.rev6_score,
        logit_np(rows.rev6_score),
        rows.rev24_safe_prob,
        rows.rev24_false_prob,
        rows.rev24_ambiguous_prob,
        rev24_margin,
        rows.rev26_binary_prob,
        rows.rev26_ambiguous_prob,
        rev26_margin,
    ]
    names = list(BASE_FEATURE_NAMES)
    if include_safety_hint:
        centered = rows.positive_ambiguity_safety_hint - 0.5
        features.extend(
            [
                rows.positive_ambiguity_safety_hint,
                centered,
                centered * bundle_margin,
            ]
        )
        names = list(REV30_FEATURE_NAMES)
    return np.stack(features, axis=1).astype(np.float32), names


def precision_at_budget(score: np.ndarray, labels: np.ndarray, budget: int) -> Dict[str, float | int]:
    order = np.argsort(-score)
    k = min(max(int(budget), 1), len(order))
    selected = np.zeros(len(labels), dtype=bool)
    selected[order[:k]] = True
    tp = float((selected & (labels == 1)).sum())
    fp = float((selected & (labels == 0)).sum())
    fn = float((~selected & (labels == 1)).sum())
    metrics = binary_metrics(tp, fp, fn)
    return {
        "precision_budget": int(k),
        "precision_at_budget": metrics["precision"],
        "budget_recall": metrics["recall"],
        "budget_f1": metrics["f1"],
    }


def scorer_quality_row(
    score: np.ndarray,
    labels: np.ndarray,
    row_name: str,
    budget: int,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "row": row_name,
        "candidate_count": int(len(labels)),
        "positive_count": int((labels == 1).sum()),
        "negative_count": int((labels == 0).sum()),
        "ap": average_precision(score, labels),
        "auroc": auroc(score, labels),
        "score_mean_positive": float(score[labels == 1].mean()) if (labels == 1).any() else None,
        "score_mean_negative": float(score[labels == 0].mean()) if (labels == 0).any() else None,
    }
    row.update(precision_at_budget(score, labels, budget))
    return row


def score_distribution_rows(
    score: np.ndarray,
    labels: np.ndarray,
    row_name: str,
) -> list[Dict[str, Any]]:
    out: list[Dict[str, Any]] = []
    for label, label_name in [(1, "gt_positive"), (0, "gt_negative")]:
        values = score[labels == label]
        row: Dict[str, Any] = {"row": row_name, "class": label_name, "count": int(len(values))}
        if len(values) > 0:
            for q in [0.10, 0.25, 0.50, 0.75, 0.90]:
                row[f"q{int(q * 100):02d}"] = float(np.quantile(values, q))
            row["mean"] = float(values.mean())
        out.append(row)
    return out


def one_feature_rows(test_x: np.ndarray, test_y: np.ndarray, names: list[str]) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for idx, name in enumerate(names):
        score = test_x[:, idx]
        forward_ap = average_precision(score, test_y)
        reverse_ap = average_precision(-score, test_y)
        if reverse_ap is not None and forward_ap is not None and reverse_ap > forward_ap:
            score = -score
            direction = "negative"
            ap = reverse_ap
        else:
            direction = "positive"
            ap = forward_ap
        rows.append(
            {
                "feature": name,
                "best_direction": direction,
                "ap": ap,
                "auroc": auroc(score, test_y),
                "positive_mean": float(test_x[test_y == 1, idx].mean()) if (test_y == 1).any() else None,
                "negative_mean": float(test_x[test_y == 0, idx].mean()) if (test_y == 0).any() else None,
            }
        )
    rows.sort(key=lambda row: -float(row["ap"] or 0.0))
    return rows


def coefficient_rows(fit: ProbeFit, names: list[str], row_name: str) -> list[Dict[str, Any]]:
    weights = fit.model.weight.detach().cpu().numpy().reshape(-1)
    rows = [
        {
            "row": row_name,
            "feature": name,
            "coefficient": float(weights[idx]),
            "abs_coefficient": float(abs(weights[idx])),
        }
        for idx, name in enumerate(names)
    ]
    rows.sort(key=lambda row: -float(row["abs_coefficient"]))
    return rows


def win_rate(positive_scores: np.ndarray, negative_scores: np.ndarray) -> float | None:
    if len(positive_scores) == 0 or len(negative_scores) == 0:
        return None
    return float((positive_scores[:, None] > negative_scores[None, :]).mean())


def base_prediction(rows: Rev30Rows, thresholds: Dict[str, float]) -> np.ndarray:
    return (rows.rev6_score >= threshold_array(rows.variant, thresholds)).astype(np.int64)


def selected_additions(
    rows: Rev30Rows,
    score: np.ndarray,
    thresholds: Dict[str, float],
    budget: int,
) -> np.ndarray:
    base_pred = base_prediction(rows, thresholds)
    candidate_idx = np.flatnonzero(rows.is_candidate & (base_pred == 0))
    chosen = np.zeros_like(rows.target, dtype=bool)
    if len(candidate_idx) == 0 or budget <= 0:
        return chosen
    budget = min(int(budget), len(candidate_idx))
    order = np.argsort(-score[candidate_idx])
    chosen[candidate_idx[order[:budget]]] = True
    return chosen


def prediction_with_additions(
    rows: Rev30Rows,
    additions: np.ndarray,
    thresholds: Dict[str, float],
) -> np.ndarray:
    return np.maximum(base_prediction(rows, thresholds), additions.astype(np.int64))


def top20_budget(rows: Rev30Rows) -> int:
    return max(1, int(round(0.20 * int(rows.is_candidate.sum()))))


def recovery_row(
    rows: Rev30Rows,
    pred: np.ndarray,
    score: np.ndarray,
    row_name: str,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"row": row_name}
    for scope_name, mask in [
        ("overall", np.ones_like(rows.target, dtype=bool)),
        ("clean", rows.variant == "clean"),
        ("noisy", rows.variant == "noisy"),
    ]:
        metrics = edge_metrics_from_arrays(pred[mask], rows.target[mask])
        for key in ["precision", "recall", "f1", "tp", "fp", "fn"]:
            out[f"{scope_name}_{key}"] = metrics[key]

    noisy = rows.variant == "noisy"
    hint_missed_pos = noisy & (rows.target == 1) & (rows.relation < 0.50)
    hint_supported_false = noisy & (rows.target == 0) & (rows.relation >= 0.50)
    out["hint_missed_true_recall"] = float(pred[hint_missed_pos].mean()) if hint_missed_pos.any() else 0.0
    out["hint_missed_avg_score"] = (
        float(score[hint_missed_pos].mean()) if hint_missed_pos.any() else None
    )
    out["hint_supported_fp_error"] = (
        float(pred[hint_supported_false].mean()) if hint_supported_false.any() else 0.0
    )
    out["hm_vs_hard_negative_win_rate"] = win_rate(
        score[hint_missed_pos],
        score[hint_supported_false],
    )
    return out


def rescue_scope_row(
    rows: Rev30Rows,
    additions: np.ndarray,
    row_name: str,
) -> Dict[str, Any]:
    candidate = rows.is_candidate
    selected = additions[candidate].astype(np.int64)
    target = rows.target[candidate]
    metrics = edge_metrics_from_arrays(selected, target)
    weak_positive = weak_positive_mask(rows)
    weak_selected = additions[weak_positive].astype(np.int64)
    weak_target = rows.target[weak_positive]
    weak_metrics = edge_metrics_from_arrays(weak_selected, weak_target)
    low_hint_false = candidate & (rows.candidate_label == 1)
    ambiguous = candidate & (rows.candidate_label == 2)
    safe = candidate & (rows.target == 1)
    return {
        "row": row_name,
        "admitted": int(additions.sum()),
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "rescue_precision": metrics["precision"],
        "rescue_recall": metrics["recall"],
        "rescue_f1": metrics["f1"],
        "safe_admission_rate": float(additions[safe].mean()) if safe.any() else 0.0,
        "low_hint_false_admission_rate": float(additions[low_hint_false].mean())
        if low_hint_false.any()
        else 0.0,
        "ambiguous_admission_rate": float(additions[ambiguous].mean()) if ambiguous.any() else 0.0,
        "weak_positive_selected": int(weak_selected.sum()),
        "weak_positive_precision": weak_metrics["precision"],
        "weak_positive_recall": weak_metrics["recall"],
        "weak_positive_f1": weak_metrics["f1"],
    }


@torch.no_grad()
def collect_rows(
    data_path: str,
    models: Dict[str, Any],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    rescue_relation_max: float,
    rescue_support_min: float,
) -> Rev30Rows:
    dataset = Step30WeakObservationDataset(data_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=step30_weak_observation_collate_fn,
        pin_memory=device.type == "cuda",
    )

    out: Dict[str, list[Any]] = {field: [] for field in Rev30Rows.__dataclass_fields__}
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        variants = [str(v) for v in batch.get("step30_observation_variant", ["unknown"] * len(batch["target_adj"]))]
        target_adj = batch["target_adj"].float()
        node_mask = batch["node_mask"].float()
        relation_hints = batch["weak_relation_hints"].float()
        pair_support_hints = batch["weak_pair_support_hints"].float()
        signed_witness = batch["weak_signed_pair_witness"].float()
        pair_bundle = batch["weak_pair_evidence_bundle"].float()
        safety_hint = batch["weak_positive_ambiguity_safety_hint"].float()
        pair_mask = build_pair_mask(node_mask).bool()

        outputs = {
            name: model(
                weak_slot_features=batch["weak_slot_features"],
                weak_relation_hints=relation_hints,
                weak_pair_support_hints=pair_support_hints,
                weak_signed_pair_witness=signed_witness,
                weak_pair_evidence_bundle=pair_bundle,
                weak_positive_ambiguity_safety_hint=safety_hint,
            )
            for name, model in models.items()
        }
        trivial_outputs = trivial_baseline_outputs(
            weak_slot_features=batch["weak_slot_features"],
            weak_relation_hints=relation_hints,
            num_node_types=models["rev6"].config.num_node_types,
            state_dim=models["rev6"].config.state_dim,
            weak_pair_support_hints=pair_support_hints,
            weak_signed_pair_witness=signed_witness,
            weak_pair_evidence_bundle=pair_bundle,
            weak_positive_ambiguity_safety_hint=safety_hint,
        )

        rev6_prob = torch.sigmoid(outputs["rev6"]["edge_logits"]).detach().cpu().numpy()
        rev17_prob = torch.sigmoid(outputs["rev17"]["edge_logits"]).detach().cpu().numpy()
        trivial_prob = torch.sigmoid(trivial_outputs["edge_logits"]).detach().cpu().numpy()

        rev24_probs = torch.softmax(outputs["rev24"]["rescue_candidate_logits"], dim=-1).detach().cpu().numpy()
        rev26_logits = outputs["rev26"]["rescue_candidate_logits"]
        rev26_probs = torch.softmax(rev26_logits, dim=-1).detach().cpu().numpy()
        rev26_binary = outputs["rev26"].get("rescue_candidate_binary_logits")
        rev26_binary_prob = (
            sigmoid_np(rev26_binary.detach().cpu().numpy())
            if rev26_binary is not None
            else rev26_probs[..., 0]
        )
        rev30_logits = outputs["rev30"]["rescue_candidate_logits"]
        rev30_probs = torch.softmax(rev30_logits, dim=-1).detach().cpu().numpy()
        rev30_binary = outputs["rev30"].get("rescue_candidate_binary_logits")
        rev30_binary_prob = (
            sigmoid_np(rev30_binary.detach().cpu().numpy())
            if rev30_binary is not None
            else rev30_probs[..., 0]
        )
        rev30_weak_positive_tensor = outputs["rev30"].get(
            "weak_positive_ambiguity_safety_logits"
        )
        rev30_weak_positive_prob = (
            sigmoid_np(rev30_weak_positive_tensor.detach().cpu().numpy())
            if rev30_weak_positive_tensor is not None
            else rev30_binary_prob
        )

        for batch_idx in range(int(target_adj.shape[0])):
            n = int(node_mask[batch_idx].sum().item())
            target_np = target_adj[batch_idx, :n, :n].detach().cpu().numpy()
            relation_np = relation_hints[batch_idx, :n, :n].detach().cpu().numpy()
            support_np = pair_support_hints[batch_idx, :n, :n].detach().cpu().numpy()
            witness_np = signed_witness[batch_idx, :n, :n].detach().cpu().numpy()
            bundle_np = pair_bundle[batch_idx, :n, :n, :].detach().cpu().numpy()
            safety_np = safety_hint[batch_idx, :n, :n].detach().cpu().numpy()
            valid_np = pair_mask[batch_idx, :n, :n].detach().cpu().numpy().astype(bool)

            for i in range(n):
                for j in range(i + 1, n):
                    if not bool(valid_np[i, j]):
                        continue
                    relation = float(relation_np[i, j])
                    support = float(support_np[i, j])
                    target = int(target_np[i, j] >= 0.5)
                    bundle = bundle_np[i, j, :]
                    is_candidate = (
                        variants[batch_idx] == "noisy"
                        and relation < rescue_relation_max
                        and support >= rescue_support_min
                    )
                    ambiguous_signal = (
                        is_candidate
                        and (
                            relation >= 0.45
                            or support < 0.65
                            or abs(float(bundle[0]) - float(bundle[1])) < 0.10
                        )
                    )
                    label = -1
                    subtype = "not_ambiguous_signal"
                    if is_candidate:
                        label = rescue_class_label(target, relation, support, bundle)
                    if ambiguous_signal:
                        subtype = subtype_for_candidate(
                            relation=relation,
                            positive=float(bundle[0]),
                            warning=float(bundle[1]),
                            corroboration=float(bundle[2]),
                            endpoint_compat=float(bundle[3]),
                            signed_witness=float(witness_np[i, j]),
                        )

                    out["target"].append(target)
                    out["variant"].append(variants[batch_idx])
                    out["relation"].append(relation)
                    out["support"].append(support)
                    out["signed_witness"].append(float(witness_np[i, j]))
                    out["bundle_positive"].append(float(bundle[0]))
                    out["bundle_warning"].append(float(bundle[1]))
                    out["bundle_corroboration"].append(float(bundle[2]))
                    out["bundle_endpoint_compat"].append(float(bundle[3]))
                    out["positive_ambiguity_safety_hint"].append(float(safety_np[i, j]))
                    out["is_candidate"].append(bool(is_candidate))
                    out["candidate_label"].append(label)
                    out["is_ambiguous_signal"].append(bool(ambiguous_signal))
                    out["subtype"].append(subtype)
                    out["rev6_score"].append(float(rev6_prob[batch_idx, i, j]))
                    out["rev17_score"].append(float(rev17_prob[batch_idx, i, j]))
                    out["rev24_safe_prob"].append(float(rev24_probs[batch_idx, i, j, 0]))
                    out["rev24_false_prob"].append(float(rev24_probs[batch_idx, i, j, 1]))
                    out["rev24_ambiguous_prob"].append(float(rev24_probs[batch_idx, i, j, 2]))
                    out["rev26_binary_prob"].append(float(rev26_binary_prob[batch_idx, i, j]))
                    out["rev26_ambiguous_prob"].append(float(rev26_probs[batch_idx, i, j, 2]))
                    out["rev30_safe_prob"].append(float(rev30_probs[batch_idx, i, j, 0]))
                    out["rev30_false_prob"].append(float(rev30_probs[batch_idx, i, j, 1]))
                    out["rev30_ambiguous_prob"].append(float(rev30_probs[batch_idx, i, j, 2]))
                    out["rev30_binary_prob"].append(float(rev30_binary_prob[batch_idx, i, j]))
                    out["rev30_weak_positive_safety_prob"].append(
                        float(rev30_weak_positive_prob[batch_idx, i, j])
                    )
                    out["trivial_score"].append(float(trivial_prob[batch_idx, i, j]))

    return Rev30Rows(
        target=np.asarray(out["target"], dtype=np.int64),
        variant=np.asarray(out["variant"]),
        relation=np.asarray(out["relation"], dtype=np.float32),
        support=np.asarray(out["support"], dtype=np.float32),
        signed_witness=np.asarray(out["signed_witness"], dtype=np.float32),
        bundle_positive=np.asarray(out["bundle_positive"], dtype=np.float32),
        bundle_warning=np.asarray(out["bundle_warning"], dtype=np.float32),
        bundle_corroboration=np.asarray(out["bundle_corroboration"], dtype=np.float32),
        bundle_endpoint_compat=np.asarray(out["bundle_endpoint_compat"], dtype=np.float32),
        positive_ambiguity_safety_hint=np.asarray(
            out["positive_ambiguity_safety_hint"],
            dtype=np.float32,
        ),
        is_candidate=np.asarray(out["is_candidate"], dtype=bool),
        candidate_label=np.asarray(out["candidate_label"], dtype=np.int64),
        is_ambiguous_signal=np.asarray(out["is_ambiguous_signal"], dtype=bool),
        subtype=np.asarray(out["subtype"]),
        rev6_score=np.asarray(out["rev6_score"], dtype=np.float32),
        rev17_score=np.asarray(out["rev17_score"], dtype=np.float32),
        rev24_safe_prob=np.asarray(out["rev24_safe_prob"], dtype=np.float32),
        rev24_false_prob=np.asarray(out["rev24_false_prob"], dtype=np.float32),
        rev24_ambiguous_prob=np.asarray(out["rev24_ambiguous_prob"], dtype=np.float32),
        rev26_binary_prob=np.asarray(out["rev26_binary_prob"], dtype=np.float32),
        rev26_ambiguous_prob=np.asarray(out["rev26_ambiguous_prob"], dtype=np.float32),
        rev30_safe_prob=np.asarray(out["rev30_safe_prob"], dtype=np.float32),
        rev30_false_prob=np.asarray(out["rev30_false_prob"], dtype=np.float32),
        rev30_ambiguous_prob=np.asarray(out["rev30_ambiguous_prob"], dtype=np.float32),
        rev30_binary_prob=np.asarray(out["rev30_binary_prob"], dtype=np.float32),
        rev30_weak_positive_safety_prob=np.asarray(
            out["rev30_weak_positive_safety_prob"],
            dtype=np.float32,
        ),
        trivial_score=np.asarray(out["trivial_score"], dtype=np.float32),
    )


def train_probe(
    train_rows: Rev30Rows,
    val_rows: Rev30Rows,
    test_rows: Rev30Rows,
    include_safety_hint: bool,
    args: argparse.Namespace,
) -> tuple[ProbeFit, np.ndarray, np.ndarray, list[str]]:
    train_mask = weak_positive_mask(train_rows)
    val_mask = weak_positive_mask(val_rows)
    test_mask = weak_positive_mask(test_rows)
    train_x_all, names = feature_matrix(train_rows, include_safety_hint=include_safety_hint)
    val_x_all, _ = feature_matrix(val_rows, include_safety_hint=include_safety_hint)
    test_x_all, _ = feature_matrix(test_rows, include_safety_hint=include_safety_hint)
    fit = fit_logistic_probe(
        train_x_all[train_mask],
        train_rows.target[train_mask].astype(np.int64),
        val_x_all[val_mask],
        val_rows.target[val_mask].astype(np.int64),
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed + (17 if include_safety_hint else 0),
    )
    test_score = probe_scores(fit, test_x_all[test_mask])
    return fit, test_score, test_x_all[test_mask], names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="data/graph_event_step30_weak_obs_rev30_train.pkl")
    parser.add_argument("--val_path", default="data/graph_event_step30_weak_obs_rev30_val.pkl")
    parser.add_argument("--test_path", default="data/graph_event_step30_weak_obs_rev30_test.pkl")
    parser.add_argument("--rev6_checkpoint", default="checkpoints/step30_encoder_recovery_rev6/best.pt")
    parser.add_argument("--rev17_checkpoint", default="checkpoints/step30_encoder_recovery_rev17/best.pt")
    parser.add_argument("--rev24_checkpoint", default="checkpoints/step30_encoder_recovery_rev24/best.pt")
    parser.add_argument("--rev26_checkpoint", default="checkpoints/step30_encoder_recovery_rev26/best.pt")
    parser.add_argument("--rev30_checkpoint", default="checkpoints/step30_encoder_recovery_rev30/best.pt")
    parser.add_argument("--output_dir", default="artifacts/step30_positive_ambiguity_safety_rev30")
    parser.add_argument("--thresholds", default="clean:0.50,noisy:0.55")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rescue_relation_max", type=float, default=0.50)
    parser.add_argument("--rescue_support_min", type=float, default=0.55)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=130)
    args = parser.parse_args()

    if args.device != "auto":
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    models = {
        "rev6": load_model(args.rev6_checkpoint, device),
        "rev17": load_model(args.rev17_checkpoint, device),
        "rev24": load_model(args.rev24_checkpoint, device),
        "rev26": load_model(args.rev26_checkpoint, device),
        "rev30": load_model(args.rev30_checkpoint, device),
    }
    collect_kwargs = {
        "models": models,
        "device": device,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "rescue_relation_max": args.rescue_relation_max,
        "rescue_support_min": args.rescue_support_min,
    }
    train_rows = collect_rows(args.train_path, **collect_kwargs)
    val_rows = collect_rows(args.val_path, **collect_kwargs)
    test_rows = collect_rows(args.test_path, **collect_kwargs)

    thresholds = parse_thresholds(args.thresholds)
    budget = top20_budget(test_rows)
    weak_mask = weak_positive_mask(test_rows)
    test_y = test_rows.target[weak_mask].astype(np.int64)

    base_fit, base_probe_score, base_test_x, base_names = train_probe(
        train_rows,
        val_rows,
        test_rows,
        include_safety_hint=False,
        args=args,
    )
    safety_fit, safety_probe_score, safety_test_x, safety_names = train_probe(
        train_rows,
        val_rows,
        test_rows,
        include_safety_hint=True,
        args=args,
    )

    rev26_add = selected_additions(test_rows, rev26_score(test_rows), thresholds, budget)
    rev30_add = selected_additions(test_rows, rev30_score(test_rows), thresholds, budget)
    weak_selected_count = int((rev26_add & weak_mask).sum())
    safety_probe_same = rev26_add.copy()
    safety_probe_same[weak_mask] = False
    safety_probe_full_score = rev26_score(test_rows).copy()
    safety_probe_full_score[weak_mask] = safety_probe_score
    weak_candidates = np.flatnonzero(weak_mask & (base_prediction(test_rows, thresholds) == 0))
    if weak_selected_count > 0 and len(weak_candidates) > 0:
        score_full = np.full_like(test_rows.rev6_score, -np.inf)
        score_full[weak_mask] = safety_probe_score
        order = np.argsort(-score_full[weak_candidates])
        chosen = weak_candidates[order[: min(weak_selected_count, len(order))]]
        safety_probe_same[chosen] = True

    scorers = {
        "rev29_current_signal_probe": base_probe_score,
        "rev30_with_safety_probe": safety_probe_score,
        "rev30_model_binary_minus_ambiguous": rev30_score(test_rows)[weak_mask],
        "positive_ambiguity_safety_hint_only": test_rows.positive_ambiguity_safety_hint[weak_mask],
        "rev26_binary_minus_ambiguous": rev26_score(test_rows)[weak_mask],
    }
    probe_quality_rows = [
        scorer_quality_row(score, test_y, name, weak_selected_count)
        for name, score in scorers.items()
    ]
    score_distribution: list[Dict[str, Any]] = []
    for name, score in scorers.items():
        score_distribution.extend(score_distribution_rows(score, test_y, name))

    pred_rev6 = base_prediction(test_rows, thresholds)
    pred_rev17 = (
        test_rows.rev17_score >= threshold_array(test_rows.variant, thresholds)
    ).astype(np.int64)
    pred_rev26 = prediction_with_additions(test_rows, rev26_add, thresholds)
    pred_rev30 = prediction_with_additions(test_rows, rev30_add, thresholds)
    pred_trivial = (
        test_rows.trivial_score >= threshold_array(test_rows.variant, thresholds)
    ).astype(np.int64)
    pred_safety_probe = prediction_with_additions(test_rows, safety_probe_same, thresholds)

    recovery_rows = [
        recovery_row(test_rows, pred_rev6, test_rows.rev6_score, "rev6"),
        recovery_row(test_rows, pred_rev17, test_rows.rev17_score, "rev17_global_bundle"),
        recovery_row(test_rows, pred_rev26, rev26_score(test_rows), "rev26_calibrated"),
        recovery_row(test_rows, pred_rev30, rev30_score(test_rows), "rev30_integrated"),
        recovery_row(test_rows, pred_trivial, test_rows.trivial_score, "trivial_with_rev30_cue"),
        recovery_row(
            test_rows,
            pred_safety_probe,
            safety_probe_full_score,
            "rev30_safety_probe_same_weak_count",
        ),
    ]
    rescue_rows = [
        rescue_scope_row(test_rows, rev26_add, "rev26_calibrated"),
        rescue_scope_row(test_rows, rev30_add, "rev30_integrated"),
        rescue_scope_row(test_rows, safety_probe_same, "rev30_safety_probe_same_weak_count"),
    ]
    feature_rows = one_feature_rows(safety_test_x, test_y, safety_names)
    coefficient_summary = coefficient_rows(base_fit, base_names, "rev29_current_signal_probe")
    coefficient_summary.extend(
        coefficient_rows(safety_fit, safety_names, "rev30_with_safety_probe")
    )

    rev6_noisy_f1 = next(row for row in recovery_rows if row["row"] == "rev6")["noisy_f1"]
    rev30_noisy_f1 = next(row for row in recovery_rows if row["row"] == "rev30_integrated")["noisy_f1"]
    rev29_ap = next(row for row in probe_quality_rows if row["row"] == "rev29_current_signal_probe")["ap"]
    rev30_ap = next(row for row in probe_quality_rows if row["row"] == "rev30_with_safety_probe")["ap"]
    trivial_noisy_f1 = next(row for row in recovery_rows if row["row"] == "trivial_with_rev30_cue")["noisy_f1"]
    gate = {
        "noisy_edge_f1_beats_rev6": bool(rev30_noisy_f1 > rev6_noisy_f1),
        "weak_positive_ap_improves_over_rev29_probe": bool((rev30_ap or 0.0) > (rev29_ap or 0.0)),
        "trivial_clearly_below_encoder": bool(trivial_noisy_f1 + 0.03 < rev30_noisy_f1),
        "backend_rerun": False,
        "backend_rerun_reason": "Step30c was not run; rev30 is recovery/diagnostic-only by task constraint.",
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "probe_quality_summary.csv", probe_quality_rows)
    write_csv(output_dir / "score_distributions.csv", score_distribution)
    write_csv(output_dir / "feature_ranking.csv", feature_rows)
    write_csv(output_dir / "logistic_coefficients.csv", coefficient_summary)
    write_csv(output_dir / "recovery_summary.csv", recovery_rows)
    write_csv(output_dir / "rescue_scope_summary.csv", rescue_rows)
    summary = {
        "step": "step30_rev30_positive_ambiguity_safety",
        "new_cue": "positive_ambiguity_safety_hint",
        "weak_positive_count": int(weak_mask.sum()),
        "weak_positive_positive_rate": float(test_y.mean()) if len(test_y) else None,
        "selected_budget": int(budget),
        "rev26_weak_positive_selected_count": int(weak_selected_count),
        "base_probe": {
            "best_epoch": base_fit.best_epoch,
            "val_ap": base_fit.val_ap,
            "val_auroc": base_fit.val_auroc,
        },
        "safety_probe": {
            "best_epoch": safety_fit.best_epoch,
            "val_ap": safety_fit.val_ap,
            "val_auroc": safety_fit.val_auroc,
        },
        "probe_quality_summary": probe_quality_rows,
        "recovery_summary": recovery_rows,
        "rescue_scope_summary": rescue_rows,
        "gate": gate,
        "args": vars(args),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(clean_json_numbers(summary), f, indent=2)
    print(json.dumps(clean_json_numbers(summary), indent=2))


if __name__ == "__main__":
    main()
