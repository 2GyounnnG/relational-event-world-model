from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train.eval_step30_encoder_recovery import load_model
from train.eval_step30_rescue_candidate_integration_rev24 import (
    SplitRows,
    add_family_rows,
    budget_specs,
    collect_split,
    edge_metrics_from_arrays,
    targeted_metrics,
    write_csv,
)
from train.eval_step30_rescue_candidate_latent_probe_rev23 import (
    CLASS_NAMES,
    binary_metrics,
    clean_json_numbers,
)


def parse_thresholds(value: str) -> Dict[str, float]:
    out: Dict[str, float] = {"default": 0.5}
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        key, raw_val = part.split(":", 1)
        out[key.strip()] = float(raw_val)
    return out


def base_rev6_prediction(split: SplitRows) -> np.ndarray:
    thresholds = np.where(split.variant == "clean", 0.50, 0.55)
    return (split.rev6_score >= thresholds).astype(np.int64)


def rev21_prediction(split: SplitRows) -> np.ndarray:
    thresholds = np.where(split.variant == "clean", 0.50, 0.55)
    return (split.rev21_score >= thresholds).astype(np.int64)


def class_aware_score(split: SplitRows) -> np.ndarray:
    """Simple rev25 rule: safe probability margin over the strongest reject class."""

    reject_score = np.maximum(split.rev24_false_prob, split.rev24_ambiguous_prob)
    return split.rev24_safe_prob - reject_score


def selected_additions_from_score(
    split: SplitRows,
    score: np.ndarray,
    budget_name: str,
) -> tuple[np.ndarray, int]:
    base_pred = base_rev6_prediction(split)
    budgets = budget_specs(split)
    budget = int(budgets[budget_name])
    candidate_idx = np.flatnonzero(split.is_candidate & (base_pred == 0))
    chosen = np.zeros_like(base_pred, dtype=bool)
    if len(candidate_idx) == 0:
        return chosen, 0
    budget = min(budget, len(candidate_idx))
    order = np.argsort(-score[candidate_idx])
    chosen[candidate_idx[order[:budget]]] = True
    return chosen, budget


def prediction_with_additions(
    split: SplitRows,
    score: np.ndarray,
    budget_name: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    base_pred = base_rev6_prediction(split)
    additions, budget = selected_additions_from_score(split, score=score, budget_name=budget_name)
    return np.maximum(base_pred, additions.astype(np.int64)), additions, budget


def choose_budget(
    val_split: SplitRows,
    budget_names: list[str],
) -> tuple[str, list[Dict[str, Any]]]:
    score = class_aware_score(val_split)
    rows = []
    best_name = budget_names[0]
    best_score = -float("inf")
    for budget_name in budget_names:
        pred, additions, budget = prediction_with_additions(val_split, score=score, budget_name=budget_name)
        noisy = val_split.variant == "noisy"
        noisy_metrics = edge_metrics_from_arrays(pred[noisy], val_split.target[noisy])
        candidate = val_split.is_candidate
        addition_metrics = edge_metrics_from_arrays(
            additions[candidate].astype(np.int64),
            val_split.target[candidate],
        )
        row = {
            "budget_name": budget_name,
            "budget": budget,
            "selected_additions": int(additions.sum()),
            "noisy_precision": noisy_metrics["precision"],
            "noisy_recall": noisy_metrics["recall"],
            "noisy_f1": noisy_metrics["f1"],
            "addition_precision": addition_metrics["precision"],
            "addition_recall": addition_metrics["recall"],
            "addition_f1": addition_metrics["f1"],
        }
        rows.append(row)
        # Keep validation choice recovery-first, with precision as the tiebreaker.
        score_for_selection = float(noisy_metrics["f1"])
        if score_for_selection > best_score:
            best_score = score_for_selection
            best_name = budget_name
    return best_name, rows


def summarize_recovery(
    split: SplitRows,
    budget_name: str,
) -> tuple[list[Dict[str, Any]], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, int]]:
    safe_pred, safe_additions, safe_budget = prediction_with_additions(
        split,
        score=split.rev24_safe_score,
        budget_name=budget_name,
    )
    margin_pred, margin_additions, margin_budget = prediction_with_additions(
        split,
        score=class_aware_score(split),
        budget_name=budget_name,
    )
    pred_by_name = {
        "rev6": base_rev6_prediction(split),
        "rev21": rev21_prediction(split),
        "rev24_safe_only": safe_pred,
        "rev25_class_aware": margin_pred,
    }
    additions_by_name = {
        "rev24_safe_only": safe_additions,
        "rev25_class_aware": margin_additions,
    }
    budgets_by_name = {
        "rev24_safe_only": safe_budget,
        "rev25_class_aware": margin_budget,
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
    add_family_rows(rows, split, pred_by_name)
    return rows, pred_by_name, additions_by_name, budgets_by_name


def class_admission_rates(
    selected: np.ndarray,
    candidate_labels: np.ndarray,
) -> Dict[str, float | int]:
    out: Dict[str, float | int] = {}
    for idx, name in enumerate(CLASS_NAMES):
        mask = candidate_labels == idx
        admitted = int((selected & mask).sum())
        out[f"{name}_count"] = int(mask.sum())
        out[f"{name}_admitted"] = admitted
        out[f"{name}_admission_rate"] = float(admitted / max(int(mask.sum()), 1))
    return out


def summarize_rescue_scope(
    split: SplitRows,
    pred_by_name: Dict[str, np.ndarray],
    additions_by_name: Dict[str, np.ndarray],
    budgets_by_name: Dict[str, int],
) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    candidate = split.is_candidate
    candidate_target = split.target[candidate]
    for name, pred in pred_by_name.items():
        selected = pred[candidate].astype(bool)
        row = {
            "row": name,
            "scope": "full_rescue_scope_decode",
            "budget": budgets_by_name.get(name),
            "admitted": int(selected.sum()),
        }
        row.update(edge_metrics_from_arrays(selected.astype(np.int64), candidate_target))
        row.update(class_admission_rates(selected, split.candidate_labels))
        rows.append(row)
    for name, additions in additions_by_name.items():
        selected = additions[candidate].astype(bool)
        row = {
            "row": name,
            "scope": "selected_rescue_additions_only",
            "budget": budgets_by_name.get(name),
            "admitted": int(selected.sum()),
        }
        row.update(edge_metrics_from_arrays(selected.astype(np.int64), candidate_target))
        row.update(class_admission_rates(selected, split.candidate_labels))
        rows.append(row)
    return rows


def summarize_targeted(
    split: SplitRows,
    pred_by_name: Dict[str, np.ndarray],
) -> list[Dict[str, Any]]:
    score_by_name = {
        "rev6": split.rev6_score,
        "rev21": split.rev21_score,
        "rev24_safe_only": np.where(split.is_candidate, split.rev24_safe_prob, split.rev6_score),
        "rev25_class_aware": np.where(split.is_candidate, class_aware_score(split), split.rev6_score),
    }
    rows = []
    for name, pred in pred_by_name.items():
        row = {"row": name}
        row.update(targeted_metrics(split, pred, score_by_name[name]))
        rows.append(row)
    return rows


def summarize_class_probs(
    split: SplitRows,
    additions_by_name: Dict[str, np.ndarray],
) -> list[Dict[str, Any]]:
    candidate = split.is_candidate
    labels = split.candidate_labels
    probs = {
        "safe_prob": split.rev24_safe_prob[candidate],
        "false_admission_prob": split.rev24_false_prob[candidate],
        "ambiguous_prob": split.rev24_ambiguous_prob[candidate],
        "class_margin": class_aware_score(split)[candidate],
    }
    rows: list[Dict[str, Any]] = []
    for row_name, additions in additions_by_name.items():
        selected = additions[candidate].astype(bool)
        for status, status_mask in {"admitted": selected, "rejected": ~selected}.items():
            for class_idx, class_name in enumerate(CLASS_NAMES):
                mask = status_mask & (labels == class_idx)
                row: Dict[str, Any] = {
                    "row": row_name,
                    "status": status,
                    "true_class": class_name,
                    "count": int(mask.sum()),
                }
                for prob_name, values in probs.items():
                    row[f"mean_{prob_name}"] = float(values[mask].mean()) if mask.any() else None
                rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_path", default="data/graph_event_step30_weak_obs_rev17_val.pkl")
    parser.add_argument("--test_path", default="data/graph_event_step30_weak_obs_rev17_test.pkl")
    parser.add_argument("--rev6_checkpoint", default="checkpoints/step30_encoder_recovery_rev6/best.pt")
    parser.add_argument("--rev21_checkpoint", default="checkpoints/step30_encoder_recovery_rev21/best.pt")
    parser.add_argument("--rev24_checkpoint", default="checkpoints/step30_encoder_recovery_rev24/best.pt")
    parser.add_argument("--output_dir", default="artifacts/step30_encoder_recovery_rev25")
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

    models = {
        "rev6": load_model(args.rev6_checkpoint, device),
        "rev21": load_model(args.rev21_checkpoint, device),
        "rev24": load_model(args.rev24_checkpoint, device),
    }
    thresholds = parse_thresholds(args.thresholds)
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
    budget_name, budget_rows = choose_budget(
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

    recovery_rows, pred_by_name, additions_by_name, budgets_by_name = summarize_recovery(
        test_split,
        budget_name=budget_name,
    )
    rescue_rows = summarize_rescue_scope(
        test_split,
        pred_by_name=pred_by_name,
        additions_by_name=additions_by_name,
        budgets_by_name=budgets_by_name,
    )
    targeted_rows = summarize_targeted(test_split, pred_by_name=pred_by_name)
    class_prob_rows = summarize_class_probs(test_split, additions_by_name=additions_by_name)

    rev6_noisy = next(row for row in recovery_rows if row["row"] == "rev6" and row["scope"] == "noisy")
    rev25_noisy = next(row for row in recovery_rows if row["row"] == "rev25_class_aware" and row["scope"] == "noisy")
    rev24_noisy = next(row for row in recovery_rows if row["row"] == "rev24_safe_only" and row["scope"] == "noisy")
    recovery_gate = {
        "beats_rev6_noisy_f1": bool(rev25_noisy["f1"] > rev6_noisy["f1"]),
        "beats_rev24_noisy_f1": bool(rev25_noisy["f1"] > rev24_noisy["f1"]),
        "rev6_noisy_f1": rev6_noisy["f1"],
        "rev24_noisy_f1": rev24_noisy["f1"],
        "rev25_noisy_f1": rev25_noisy["f1"],
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "budget_selection_val.csv", budget_rows)
    write_csv(output_dir / "recovery_summary.csv", recovery_rows)
    write_csv(output_dir / "rescue_scope_summary.csv", rescue_rows)
    write_csv(output_dir / "targeted_rescue_diagnostics.csv", targeted_rows)
    write_csv(output_dir / "class_probability_admission_summary.csv", class_prob_rows)

    summary = {
        "step": "step30_rev25_class_aware_rescue_admission_probe",
        "backend_rerun": False,
        "backend_rerun_reason": "rev25 is recovery/admission-policy only; Step30c is gated on recovery results.",
        "admission_rule": "class_margin = P(safe_missed_true_edge) - max(P(low_hint_false_admission), P(ambiguous_rescue_candidate))",
        "selected_budget_name": budget_name,
        "budget_selection_val": budget_rows,
        "recovery_gate": recovery_gate,
        "recovery_summary": recovery_rows,
        "rescue_scope_summary": rescue_rows,
        "targeted_rescue_diagnostics": targeted_rows,
        "class_probability_admission_summary": class_prob_rows,
        "args": vars(args),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(clean_json_numbers(summary), f, indent=2)
    print(json.dumps(clean_json_numbers(summary), indent=2))


if __name__ == "__main__":
    main()
