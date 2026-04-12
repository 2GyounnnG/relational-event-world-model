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
    event_families,
    load_model,
    move_batch_to_device,
    threshold_for_variant,
)
from train.eval_step30_rescue_candidate_latent_probe_rev23 import (
    CLASS_NAMES,
    average_precision,
    auroc,
    binary_metrics,
    clean_json_numbers,
    rescue_class_label,
    sigmoid_np,
)


@dataclass
class SplitRows:
    target: np.ndarray
    relation: np.ndarray
    support: np.ndarray
    variant: np.ndarray
    sample_family_mask: Dict[str, np.ndarray]
    is_candidate: np.ndarray
    candidate_labels: np.ndarray
    candidate_target: np.ndarray
    rev6_score: np.ndarray
    rev21_score: np.ndarray
    rev24_safe_score: np.ndarray
    rev24_safe_prob: np.ndarray
    rev24_false_prob: np.ndarray
    rev24_ambiguous_prob: np.ndarray
    rev24_binary_score: np.ndarray
    rev24_binary_prob: np.ndarray
    rev24_ambiguity_score: np.ndarray
    rev24_ambiguity_prob: np.ndarray
    rev24_candidate_class: np.ndarray


def parse_thresholds(value: str) -> Dict[str, float]:
    out: Dict[str, float] = {"default": 0.5}
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        key, raw_val = part.split(":", 1)
        out[key.strip()] = float(raw_val)
    return out


def write_csv(path: Path, rows: list[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def edge_metrics_from_arrays(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    tp = float(((pred == 1) & (target == 1)).sum())
    fp = float(((pred == 1) & (target == 0)).sum())
    fn = float(((pred == 0) & (target == 1)).sum())
    tn = float(((pred == 0) & (target == 0)).sum())
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        **binary_metrics(tp, fp, fn),
    }


def win_rate(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float | None:
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return None
    neg_sorted = np.sort(neg_scores)
    wins = np.searchsorted(neg_sorted, pos_scores, side="left").astype(np.float64)
    ties = (
        np.searchsorted(neg_sorted, pos_scores, side="right")
        - np.searchsorted(neg_sorted, pos_scores, side="left")
    ).astype(np.float64)
    return float((wins + 0.5 * ties).sum() / (len(pos_scores) * len(neg_scores)))


def budget_specs(split: SplitRows) -> Dict[str, int]:
    n = int(split.candidate_target.shape[0])
    rev6_current = int((split.rev6_score[split.is_candidate] >= 0.55).sum())
    rev21_current = int((split.rev21_score[split.is_candidate] >= 0.55).sum())
    return {
        "top_20pct": max(1, int(round(0.20 * n))),
        "rev6_current_admitted": max(1, rev6_current),
        "rev21_current_admitted": max(1, rev21_current),
    }


def select_candidate_additions(
    split: SplitRows,
    base_pred: np.ndarray,
    budget: int,
) -> np.ndarray:
    candidate_indices = np.flatnonzero(split.is_candidate & (base_pred == 0))
    chosen = np.zeros_like(base_pred, dtype=bool)
    if len(candidate_indices) == 0 or budget <= 0:
        return chosen
    budget = min(int(budget), len(candidate_indices))
    order = np.argsort(-split.rev24_safe_score[candidate_indices])
    chosen[candidate_indices[order[:budget]]] = True
    return chosen


def rev24_prediction(split: SplitRows, budget_name: str) -> tuple[np.ndarray, np.ndarray, int]:
    base_pred = (split.rev6_score >= np.where(split.variant == "clean", 0.50, 0.55)).astype(np.int64)
    budgets = budget_specs(split)
    additions = select_candidate_additions(split, base_pred=base_pred, budget=budgets[budget_name])
    pred = np.maximum(base_pred, additions.astype(np.int64))
    return pred, additions, int(budgets[budget_name])


def targeted_metrics(
    split: SplitRows,
    pred: np.ndarray,
    score: np.ndarray,
) -> Dict[str, float | None]:
    noisy = split.variant == "noisy"
    hint_missed_pos = noisy & (split.target == 1) & (split.relation < 0.50)
    hint_supported_false = noisy & (split.target == 0) & (split.relation >= 0.50)
    hm_recall = float(pred[hint_missed_pos].mean()) if hint_missed_pos.any() else 0.0
    hsf_error = float(pred[hint_supported_false].mean()) if hint_supported_false.any() else 0.0
    return {
        "hint_missed_true_recall": hm_recall,
        "hint_missed_avg_score": float(score[hint_missed_pos].mean()) if hint_missed_pos.any() else None,
        "hint_supported_fp_error": hsf_error,
        "hm_vs_hard_negative_win_rate": win_rate(
            score[hint_missed_pos],
            score[hint_supported_false],
        ),
    }


def add_family_rows(
    rows: list[Dict[str, Any]],
    split: SplitRows,
    pred_by_name: Dict[str, np.ndarray],
) -> None:
    for family, mask in split.sample_family_mask.items():
        noisy_mask = mask & (split.variant == "noisy")
        if not noisy_mask.any():
            continue
        for name, pred in pred_by_name.items():
            row = {
                "family": family,
                "row": name,
                "scope": "noisy",
            }
            row.update(edge_metrics_from_arrays(pred[noisy_mask], split.target[noisy_mask]))
            rows.append(row)


@torch.no_grad()
def collect_split(
    data_path: str,
    models: Dict[str, Any],
    device: torch.device,
    thresholds: Dict[str, float],
    batch_size: int,
    num_workers: int,
    rescue_relation_max: float,
    rescue_support_min: float,
) -> SplitRows:
    dataset = Step30WeakObservationDataset(data_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=step30_weak_observation_collate_fn,
        pin_memory=device.type == "cuda",
    )

    target_rows: list[int] = []
    relation_rows: list[float] = []
    support_rows: list[float] = []
    variant_rows: list[str] = []
    candidate_mask_rows: list[bool] = []
    candidate_labels: list[int] = []
    candidate_target: list[int] = []
    rev6_scores: list[float] = []
    rev21_scores: list[float] = []
    rev24_safe_scores: list[float] = []
    rev24_safe_probs: list[float] = []
    rev24_false_probs: list[float] = []
    rev24_ambiguous_probs: list[float] = []
    rev24_binary_scores: list[float] = []
    rev24_binary_probs: list[float] = []
    rev24_ambiguity_scores: list[float] = []
    rev24_ambiguity_probs: list[float] = []
    rev24_candidate_class: list[int] = []
    family_masks: Dict[str, list[bool]] = {}

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        variants = [str(v) for v in batch.get("step30_observation_variant", ["unknown"] * len(batch["target_adj"]))]
        events_list = batch.get("events", [None] * int(batch["target_adj"].shape[0]))
        target_adj = batch["target_adj"].float()
        node_mask = batch["node_mask"].float()
        relation_hints = batch["weak_relation_hints"].float()
        pair_support_hints = batch["weak_pair_support_hints"].float()
        signed_witness = batch["weak_signed_pair_witness"].float()
        pair_bundle = batch["weak_pair_evidence_bundle"].float()
        pair_mask = build_pair_mask(node_mask).bool()

        outputs = {
            name: model(
                weak_slot_features=batch["weak_slot_features"],
                weak_relation_hints=relation_hints,
                weak_pair_support_hints=pair_support_hints,
                weak_signed_pair_witness=signed_witness,
                weak_pair_evidence_bundle=pair_bundle,
            )
            for name, model in models.items()
        }
        rev6_prob = torch.sigmoid(outputs["rev6"]["edge_logits"]).detach().cpu().numpy()
        rev21_prob = torch.sigmoid(outputs["rev21"]["edge_logits"]).detach().cpu().numpy()
        rev24_logits = outputs["rev24"]["rescue_candidate_logits"]
        rev24_safe_logit = outputs["rev24"]["rescue_candidate_safe_logits"].detach().cpu().numpy()
        rev24_probs = torch.softmax(rev24_logits, dim=-1).detach().cpu().numpy()
        rev24_binary_logits_tensor = outputs["rev24"].get("rescue_candidate_binary_logits")
        if rev24_binary_logits_tensor is None:
            rev24_binary_logit = rev24_safe_logit
        else:
            rev24_binary_logit = rev24_binary_logits_tensor.detach().cpu().numpy()
        rev24_binary_prob = sigmoid_np(rev24_binary_logit)
        rev24_ambiguity_logits_tensor = outputs["rev24"].get("rescue_candidate_ambiguity_logits")
        if rev24_ambiguity_logits_tensor is None:
            # Older checkpoints have only the 3-way head; use its ambiguity probability
            # as a backward-compatible risk score.
            rev24_ambiguity_logit = rev24_logits[..., 2].detach().cpu().numpy()
            rev24_ambiguity_prob = rev24_probs[..., 2]
        else:
            rev24_ambiguity_logit = rev24_ambiguity_logits_tensor.detach().cpu().numpy()
            rev24_ambiguity_prob = sigmoid_np(rev24_ambiguity_logit)

        batch_size_i = int(target_adj.shape[0])
        for batch_idx in range(batch_size_i):
            n = int(node_mask[batch_idx].sum().item())
            target_np = target_adj[batch_idx, :n, :n].detach().cpu().numpy()
            relation_np = relation_hints[batch_idx, :n, :n].detach().cpu().numpy()
            support_np = pair_support_hints[batch_idx, :n, :n].detach().cpu().numpy()
            bundle_np = pair_bundle[batch_idx, :n, :n, :].detach().cpu().numpy()
            valid_np = pair_mask[batch_idx, :n, :n].detach().cpu().numpy().astype(bool)
            families = set(event_families(events_list[batch_idx]))
            for i in range(n):
                for j in range(i + 1, n):
                    if not bool(valid_np[i, j]):
                        continue
                    relation = float(relation_np[i, j])
                    support = float(support_np[i, j])
                    target = int(target_np[i, j] >= 0.5)
                    rescue_eligible = (
                        variants[batch_idx] == "noisy"
                        and relation < rescue_relation_max
                        and support >= rescue_support_min
                    )

                    row_idx = len(target_rows)
                    target_rows.append(target)
                    relation_rows.append(relation)
                    support_rows.append(support)
                    variant_rows.append(variants[batch_idx])
                    candidate_mask_rows.append(bool(rescue_eligible))
                    rev6_scores.append(float(rev6_prob[batch_idx, i, j]))
                    rev21_scores.append(float(rev21_prob[batch_idx, i, j]))
                    rev24_safe_scores.append(float(rev24_safe_logit[batch_idx, i, j]))
                    rev24_safe_probs.append(float(rev24_probs[batch_idx, i, j, 0]))
                    rev24_false_probs.append(float(rev24_probs[batch_idx, i, j, 1]))
                    rev24_ambiguous_probs.append(float(rev24_probs[batch_idx, i, j, 2]))
                    rev24_binary_scores.append(float(rev24_binary_logit[batch_idx, i, j]))
                    rev24_binary_probs.append(float(rev24_binary_prob[batch_idx, i, j]))
                    rev24_ambiguity_scores.append(float(rev24_ambiguity_logit[batch_idx, i, j]))
                    rev24_ambiguity_probs.append(float(rev24_ambiguity_prob[batch_idx, i, j]))
                    rev24_candidate_class.append(int(np.argmax(rev24_probs[batch_idx, i, j, :])))

                    for family in families:
                        family_masks.setdefault(family, [False] * row_idx).append(True)
                    for family, values in family_masks.items():
                        if len(values) <= row_idx:
                            values.append(False)

                    if rescue_eligible:
                        label = rescue_class_label(target, relation, support, bundle_np[i, j, :])
                        candidate_labels.append(label)
                        candidate_target.append(int(target == 1))

    family_arrays = {name: np.asarray(values, dtype=bool) for name, values in family_masks.items()}
    return SplitRows(
        target=np.asarray(target_rows, dtype=np.int64),
        relation=np.asarray(relation_rows, dtype=np.float32),
        support=np.asarray(support_rows, dtype=np.float32),
        variant=np.asarray(variant_rows),
        sample_family_mask=family_arrays,
        is_candidate=np.asarray(candidate_mask_rows, dtype=bool),
        candidate_labels=np.asarray(candidate_labels, dtype=np.int64),
        candidate_target=np.asarray(candidate_target, dtype=np.int64),
        rev6_score=np.asarray(rev6_scores, dtype=np.float32),
        rev21_score=np.asarray(rev21_scores, dtype=np.float32),
        rev24_safe_score=np.asarray(rev24_safe_scores, dtype=np.float32),
        rev24_safe_prob=np.asarray(rev24_safe_probs, dtype=np.float32),
        rev24_false_prob=np.asarray(rev24_false_probs, dtype=np.float32),
        rev24_ambiguous_prob=np.asarray(rev24_ambiguous_probs, dtype=np.float32),
        rev24_binary_score=np.asarray(rev24_binary_scores, dtype=np.float32),
        rev24_binary_prob=np.asarray(rev24_binary_probs, dtype=np.float32),
        rev24_ambiguity_score=np.asarray(rev24_ambiguity_scores, dtype=np.float32),
        rev24_ambiguity_prob=np.asarray(rev24_ambiguity_probs, dtype=np.float32),
        rev24_candidate_class=np.asarray(rev24_candidate_class, dtype=np.int64),
    )


def summarize_candidate_classifier(split: SplitRows) -> list[Dict[str, Any]]:
    mask = split.is_candidate
    safe_score = split.rev24_safe_prob[mask]
    binary = split.candidate_target
    rows = [
        {
            "row": "rev24_integrated_safe_prob",
            "ap": average_precision(safe_score, binary),
            "auroc": auroc(safe_score, binary),
            "candidate_count": int(len(binary)),
            "safe_count": int((binary == 1).sum()),
            "unsafe_count": int((binary == 0).sum()),
        }
    ]
    pred_labels = split.rev24_candidate_class[mask]
    for idx, class_name in enumerate(CLASS_NAMES):
        true = split.candidate_labels == idx
        pred = pred_labels == idx
        tp = float((true & pred).sum())
        fp = float((~true & pred).sum())
        fn = float((true & ~pred).sum())
        row = {
            "row": "rev24_integrated_3way",
            "class": class_name,
            "count": int(true.sum()),
            "pred_count": int(pred.sum()),
        }
        row.update(binary_metrics(tp, fp, fn))
        rows.append(row)
    return rows


def summarize_predictions(split: SplitRows, budget_name: str) -> tuple[list[Dict[str, Any]], Dict[str, np.ndarray], np.ndarray]:
    rev6_pred = (split.rev6_score >= np.where(split.variant == "clean", 0.50, 0.55)).astype(np.int64)
    rev21_pred = (split.rev21_score >= np.where(split.variant == "clean", 0.50, 0.55)).astype(np.int64)
    rev24_pred, additions, budget = rev24_prediction(split, budget_name)
    pred_by_name = {
        "rev6": rev6_pred,
        "rev21": rev21_pred,
        "rev24": rev24_pred,
    }
    rows: list[Dict[str, Any]] = []
    for scope_name, scope_mask in {
        "overall": np.ones_like(split.target, dtype=bool),
        "clean": split.variant == "clean",
        "noisy": split.variant == "noisy",
    }.items():
        for name, pred in pred_by_name.items():
            row = {"row": name, "scope": scope_name}
            row.update(edge_metrics_from_arrays(pred[scope_mask], split.target[scope_mask]))
            rows.append(row)

    rows.append(
        {
            "row": "rev24",
            "scope": "decode_selection",
            "budget_name": budget_name,
            "budget": budget,
            "selected_additions": int(additions.sum()),
        }
    )
    add_family_rows(rows, split, pred_by_name)
    return rows, pred_by_name, additions


def summarize_rescue_admission(
    split: SplitRows,
    pred_by_name: Dict[str, np.ndarray],
    rev24_additions: np.ndarray,
) -> list[Dict[str, Any]]:
    rows = []
    candidate = split.is_candidate
    for name, pred in pred_by_name.items():
        pred_scope = pred[candidate].astype(bool)
        row = {
            "row": name,
            "scope": "full_rescue_scope_decode",
            "admitted": int(pred_scope.sum()),
        }
        row.update(edge_metrics_from_arrays(pred_scope.astype(np.int64), split.target[candidate]))
        rows.append(row)

    added = rev24_additions[candidate].astype(bool)
    row = {
        "row": "rev24",
        "scope": "selected_rescue_additions_only",
        "admitted": int(added.sum()),
    }
    row.update(edge_metrics_from_arrays(added.astype(np.int64), split.target[candidate]))
    rows.append(row)

    label_lookup = {
        0: "safe_missed_true_edge",
        1: "low_hint_pair_support_false_admission",
        2: "ambiguous_rescue_candidate",
    }
    for label_idx, label_name in label_lookup.items():
        label_mask = split.candidate_labels == label_idx
        rows.append(
            {
                "row": "rev24",
                "scope": f"selected_additions_{label_name}",
                "candidate_count": int(label_mask.sum()),
                "admitted": int((added & label_mask).sum()),
                "admission_rate": float((added & label_mask).sum() / max(label_mask.sum(), 1)),
            }
        )
    return rows


def summarize_targeted(split: SplitRows, pred_by_name: Dict[str, np.ndarray]) -> list[Dict[str, Any]]:
    score_by_name = {
        "rev6": split.rev6_score,
        "rev21": split.rev21_score,
        # Rank rescue candidates by the learned latent safe score and ordinary
        # pairs by rev6 confidence for a single combined recovery-side score.
        "rev24": np.where(split.is_candidate, split.rev24_safe_prob, split.rev6_score),
    }
    rows = []
    for name, pred in pred_by_name.items():
        row = {"row": name}
        row.update(targeted_metrics(split, pred, score_by_name[name]))
        rows.append(row)
    return rows


def choose_budget(val_split: SplitRows, budget_names: list[str]) -> tuple[str, list[Dict[str, Any]]]:
    rows = []
    best_name = budget_names[0]
    best_score = -float("inf")
    for name in budget_names:
        pred, additions, budget = rev24_prediction(val_split, name)
        noisy = val_split.variant == "noisy"
        metrics = edge_metrics_from_arrays(pred[noisy], val_split.target[noisy])
        candidate = val_split.is_candidate
        add_metrics = edge_metrics_from_arrays(
            additions[candidate].astype(np.int64),
            val_split.target[candidate],
        )
        row = {
            "budget_name": name,
            "budget": budget,
            "selected_additions": int(additions.sum()),
            "noisy_f1": metrics["f1"],
            "noisy_precision": metrics["precision"],
            "noisy_recall": metrics["recall"],
            "addition_precision": add_metrics["precision"],
            "addition_recall": add_metrics["recall"],
        }
        rows.append(row)
        if float(metrics["f1"]) > best_score:
            best_score = float(metrics["f1"])
            best_name = name
    return best_name, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_path", default="data/graph_event_step30_weak_obs_rev17_val.pkl")
    parser.add_argument("--test_path", default="data/graph_event_step30_weak_obs_rev17_test.pkl")
    parser.add_argument("--rev6_checkpoint", default="checkpoints/step30_encoder_recovery_rev6/best.pt")
    parser.add_argument("--rev21_checkpoint", default="checkpoints/step30_encoder_recovery_rev21/best.pt")
    parser.add_argument("--rev24_checkpoint", default="checkpoints/step30_encoder_recovery_rev24/best.pt")
    parser.add_argument("--output_dir", default="artifacts/step30_encoder_recovery_rev24")
    parser.add_argument("--thresholds", default="clean:0.50,noisy:0.55")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rescue_relation_max", type=float, default=0.50)
    parser.add_argument("--rescue_support_min", type=float, default=0.55)
    args = parser.parse_args()

    if args.device != "auto":
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    thresholds = parse_thresholds(args.thresholds)
    models = {
        "rev6": load_model(args.rev6_checkpoint, device),
        "rev21": load_model(args.rev21_checkpoint, device),
        "rev24": load_model(args.rev24_checkpoint, device),
    }

    val_split = collect_split(
        args.val_path,
        models=models,
        device=device,
        thresholds=thresholds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        rescue_relation_max=args.rescue_relation_max,
        rescue_support_min=args.rescue_support_min,
    )
    budget_name, budget_selection_rows = choose_budget(
        val_split,
        ["top_20pct", "rev6_current_admitted", "rev21_current_admitted"],
    )
    test_split = collect_split(
        args.test_path,
        models=models,
        device=device,
        thresholds=thresholds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        rescue_relation_max=args.rescue_relation_max,
        rescue_support_min=args.rescue_support_min,
    )

    recovery_rows, pred_by_name, rev24_additions = summarize_predictions(test_split, budget_name)
    candidate_rows = summarize_candidate_classifier(test_split)
    rescue_rows = summarize_rescue_admission(test_split, pred_by_name, rev24_additions)
    targeted_rows = summarize_targeted(test_split, pred_by_name)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "budget_selection_val.csv", budget_selection_rows)
    write_csv(output_dir / "recovery_summary.csv", recovery_rows)
    write_csv(output_dir / "candidate_classifier_summary.csv", candidate_rows)
    write_csv(output_dir / "rescue_admission_summary.csv", rescue_rows)
    write_csv(output_dir / "targeted_rescue_diagnostics.csv", targeted_rows)

    summary = {
        "step": "step30_rev24_rescue_candidate_latent_integration_probe",
        "backend_rerun": False,
        "backend_rerun_policy": "not run by this recovery-first evaluator",
        "decode_thresholds": thresholds,
        "selected_budget_name": budget_name,
        "budget_selection_val": budget_selection_rows,
        "candidate_classifier_summary": candidate_rows,
        "recovery_summary": recovery_rows,
        "rescue_admission_summary": rescue_rows,
        "targeted_rescue_diagnostics": targeted_rows,
        "rev23_reference": {
            "safe_ap": 0.4293,
            "safe_auroc": 0.7214,
            "rev21_budget_precision": 0.3686,
            "rev21_budget_recall": 0.5837,
            "rev21_budget_f1": 0.4518,
            "recovery_sim_f1_at_rev21_budget": 0.6734,
        },
        "args": vars(args),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(clean_json_numbers(summary), f, indent=2)
    print(json.dumps(clean_json_numbers(summary), indent=2))


if __name__ == "__main__":
    main()
