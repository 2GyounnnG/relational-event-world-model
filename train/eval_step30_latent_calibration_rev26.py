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
from train.eval_step30_class_aware_rescue_admission_rev25 import (
    base_rev6_prediction,
    class_aware_score,
    prediction_with_additions,
    rev21_prediction,
)
from train.eval_step30_rescue_candidate_integration_rev24 import (
    SplitRows,
    add_family_rows,
    budget_specs,
    collect_split,
    edge_metrics_from_arrays,
    targeted_metrics,
)
from train.eval_step30_rescue_candidate_latent_probe_rev23 import (
    CLASS_NAMES,
    average_precision,
    auroc,
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


def write_csv(path: Path, rows: list[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def selected_additions_from_score(
    split: SplitRows,
    score: np.ndarray,
    budget_name: str,
) -> tuple[np.ndarray, int]:
    base_pred = base_rev6_prediction(split)
    budget = int(budget_specs(split)[budget_name])
    candidate_idx = np.flatnonzero(split.is_candidate & (base_pred == 0))
    chosen = np.zeros_like(base_pred, dtype=bool)
    if len(candidate_idx) == 0:
        return chosen, 0
    budget = min(budget, len(candidate_idx))
    order = np.argsort(-score[candidate_idx])
    chosen[candidate_idx[order[:budget]]] = True
    return chosen, budget


def pred_from_score(
    split: SplitRows,
    score: np.ndarray,
    budget_name: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    base_pred = base_rev6_prediction(split)
    additions, budget = selected_additions_from_score(split, score=score, budget_name=budget_name)
    return np.maximum(base_pred, additions.astype(np.int64)), additions, budget


def choose_rev26_budget(split: SplitRows, budget_names: list[str]) -> tuple[str, list[Dict[str, Any]]]:
    score = rev26_admission_score(split)
    rows = []
    best_name = budget_names[0]
    best_f1 = -float("inf")
    for budget_name in budget_names:
        pred, additions, budget = pred_from_score(split, score, budget_name)
        noisy = split.variant == "noisy"
        noisy_metrics = edge_metrics_from_arrays(pred[noisy], split.target[noisy])
        candidate = split.is_candidate
        addition_metrics = edge_metrics_from_arrays(
            additions[candidate].astype(np.int64),
            split.target[candidate],
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
        if float(noisy_metrics["f1"]) > best_f1:
            best_f1 = float(noisy_metrics["f1"])
            best_name = budget_name
    return best_name, rows


def rev26_admission_score(split: SplitRows) -> np.ndarray:
    """Use the calibrated safe-vs-false score while still guarding ambiguity."""

    return split.rev24_binary_prob - split.rev24_ambiguous_prob


def safe_vs_false_quality(
    split: SplitRows,
    score: np.ndarray,
    row_name: str,
    precision_budget: int,
) -> Dict[str, Any]:
    mask = split.is_candidate.copy()
    candidate_labels = split.candidate_labels
    full_candidate_idx = np.flatnonzero(mask)
    safe_false_mask_in_candidates = candidate_labels != 2
    full_idx = full_candidate_idx[safe_false_mask_in_candidates]
    labels = (candidate_labels[safe_false_mask_in_candidates] == 0).astype(np.int64)
    scores = score[full_idx]
    order = np.argsort(-scores)
    k = min(max(int(precision_budget), 1), len(order))
    chosen = np.zeros(len(labels), dtype=bool)
    chosen[order[:k]] = True
    tp = float((chosen & (labels == 1)).sum())
    fp = float((chosen & (labels == 0)).sum())
    fn = float((~chosen & (labels == 1)).sum())
    return {
        "row": row_name,
        "scope": "safe_vs_low_hint_false_only",
        "candidate_count": int(len(labels)),
        "safe_count": int((labels == 1).sum()),
        "false_count": int((labels == 0).sum()),
        "ap": average_precision(scores, labels),
        "auroc": auroc(scores, labels),
        "precision_budget": int(k),
        "precision_at_budget": tp / max(float(k), 1.0),
        "budget_recall": tp / max(float((labels == 1).sum()), 1.0),
        "budget_f1": binary_metrics(tp, fp, fn)["f1"],
    }


def class_rows(split: SplitRows, row_name: str) -> list[Dict[str, Any]]:
    pred_labels = split.rev24_candidate_class[split.is_candidate]
    rows = []
    for idx, class_name in enumerate(CLASS_NAMES):
        true = split.candidate_labels == idx
        pred = pred_labels == idx
        tp = float((true & pred).sum())
        fp = float((~true & pred).sum())
        fn = float((true & ~pred).sum())
        row = {
            "row": row_name,
            "class": class_name,
            "count": int(true.sum()),
            "pred_count": int(pred.sum()),
        }
        row.update(binary_metrics(tp, fp, fn))
        rows.append(row)
    return rows


def admission_rates(selected: np.ndarray, labels: np.ndarray) -> Dict[str, float | int]:
    out: Dict[str, float | int] = {}
    for idx, name in enumerate(CLASS_NAMES):
        mask = labels == idx
        admitted = int((selected & mask).sum())
        out[f"{name}_count"] = int(mask.sum())
        out[f"{name}_admitted"] = admitted
        out[f"{name}_admission_rate"] = float(admitted / max(int(mask.sum()), 1))
    return out


def summarize_recovery_and_rescue(
    rev24_split: SplitRows,
    rev26_split: SplitRows,
    budget_name: str,
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]], list[Dict[str, Any]]]:
    rev24_pred, rev24_add, rev24_budget = pred_from_score(
        rev24_split,
        rev24_split.rev24_safe_score,
        budget_name,
    )
    rev25_pred, rev25_add, rev25_budget = pred_from_score(
        rev24_split,
        class_aware_score(rev24_split),
        budget_name,
    )
    rev26_pred, rev26_add, rev26_budget = pred_from_score(
        rev26_split,
        rev26_admission_score(rev26_split),
        budget_name,
    )
    pred_by_name = {
        "rev6": base_rev6_prediction(rev26_split),
        "rev21": rev21_prediction(rev26_split),
        "rev24_safe_only": rev24_pred,
        "rev25_class_aware": rev25_pred,
        "rev26_calibrated": rev26_pred,
    }
    recovery_rows: list[Dict[str, Any]] = []
    for scope_name, scope_mask in {
        "overall": np.ones_like(rev26_split.target, dtype=bool),
        "clean": rev26_split.variant == "clean",
        "noisy": rev26_split.variant == "noisy",
    }.items():
        for name, pred in pred_by_name.items():
            row = {"row": name, "scope": scope_name}
            row.update(edge_metrics_from_arrays(pred[scope_mask], rev26_split.target[scope_mask]))
            recovery_rows.append(row)
    add_family_rows(recovery_rows, rev26_split, pred_by_name)

    additions_by_name = {
        "rev24_safe_only": (rev24_split, rev24_add, rev24_budget),
        "rev25_class_aware": (rev24_split, rev25_add, rev25_budget),
        "rev26_calibrated": (rev26_split, rev26_add, rev26_budget),
    }
    rescue_rows: list[Dict[str, Any]] = []
    candidate = rev26_split.is_candidate
    for name, pred in pred_by_name.items():
        selected = pred[candidate].astype(bool)
        row: Dict[str, Any] = {
            "row": name,
            "scope": "full_rescue_scope_decode",
            "admitted": int(selected.sum()),
        }
        row.update(edge_metrics_from_arrays(selected.astype(np.int64), rev26_split.target[candidate]))
        row.update(admission_rates(selected, rev26_split.candidate_labels))
        rescue_rows.append(row)
    for name, (split, additions, budget) in additions_by_name.items():
        candidate_mask = split.is_candidate
        selected = additions[candidate_mask].astype(bool)
        row = {
            "row": name,
            "scope": "selected_rescue_additions_only",
            "budget": int(budget),
            "admitted": int(selected.sum()),
        }
        row.update(edge_metrics_from_arrays(selected.astype(np.int64), split.target[candidate_mask]))
        row.update(admission_rates(selected, split.candidate_labels))
        rescue_rows.append(row)

    score_by_name = {
        "rev6": rev26_split.rev6_score,
        "rev21": rev26_split.rev21_score,
        "rev24_safe_only": np.where(rev24_split.is_candidate, rev24_split.rev24_safe_prob, rev24_split.rev6_score),
        "rev25_class_aware": np.where(rev24_split.is_candidate, class_aware_score(rev24_split), rev24_split.rev6_score),
        "rev26_calibrated": np.where(rev26_split.is_candidate, rev26_split.rev24_binary_prob, rev26_split.rev6_score),
        "rev26_calibrated_admission": np.where(rev26_split.is_candidate, rev26_admission_score(rev26_split), rev26_split.rev6_score),
    }
    targeted_rows = []
    for name, pred in pred_by_name.items():
        split = rev24_split if name in {"rev24_safe_only", "rev25_class_aware"} else rev26_split
        row = {"row": name}
        score_name = "rev26_calibrated_admission" if name == "rev26_calibrated" else name
        row.update(targeted_metrics(split, pred, score_by_name[score_name]))
        targeted_rows.append(row)
    return recovery_rows, rescue_rows, targeted_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_path", default="data/graph_event_step30_weak_obs_rev17_val.pkl")
    parser.add_argument("--test_path", default="data/graph_event_step30_weak_obs_rev17_test.pkl")
    parser.add_argument("--rev6_checkpoint", default="checkpoints/step30_encoder_recovery_rev6/best.pt")
    parser.add_argument("--rev21_checkpoint", default="checkpoints/step30_encoder_recovery_rev21/best.pt")
    parser.add_argument("--rev24_checkpoint", default="checkpoints/step30_encoder_recovery_rev24/best.pt")
    parser.add_argument("--rev26_checkpoint", default="checkpoints/step30_encoder_recovery_rev26/best.pt")
    parser.add_argument("--output_dir", default="artifacts/step30_encoder_recovery_rev26")
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

    base_models = {
        "rev6": load_model(args.rev6_checkpoint, device),
        "rev21": load_model(args.rev21_checkpoint, device),
    }
    rev24_models = {**base_models, "rev24": load_model(args.rev24_checkpoint, device)}
    rev26_models = {**base_models, "rev24": load_model(args.rev26_checkpoint, device)}
    thresholds = parse_thresholds(args.thresholds)

    rev26_val = collect_split(
        args.val_path,
        models=rev26_models,
        device=device,
        thresholds=thresholds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        rescue_relation_max=args.rescue_relation_max,
        rescue_support_min=args.rescue_support_min,
    )
    budget_name, budget_rows = choose_rev26_budget(
        rev26_val,
        ["top_20pct", "rev6_current_admitted", "rev21_current_admitted"],
    )
    rev24_test = collect_split(
        args.test_path,
        models=rev24_models,
        device=device,
        thresholds=thresholds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        rescue_relation_max=args.rescue_relation_max,
        rescue_support_min=args.rescue_support_min,
    )
    rev26_test = collect_split(
        args.test_path,
        models=rev26_models,
        device=device,
        thresholds=thresholds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        rescue_relation_max=args.rescue_relation_max,
        rescue_support_min=args.rescue_support_min,
    )

    precision_budget = budget_specs(rev26_test)[budget_name]
    candidate_rows = [
        {
            "row": "rev23_probe_reference",
            "scope": "safe_vs_low_hint_false_only",
            "ap": 0.4293,
            "auroc": 0.7214,
            "precision_at_budget": 0.3686,
            "budget_f1": 0.4518,
            "note": "rev23 reported over safe-vs-unsafe rescue candidates; kept as fixed reference",
        },
        safe_vs_false_quality(
            rev24_test,
            rev24_test.rev24_safe_prob,
            "rev24_integrated_safe_prob",
            precision_budget=precision_budget,
        ),
        safe_vs_false_quality(
            rev24_test,
            class_aware_score(rev24_test),
            "rev25_class_margin",
            precision_budget=precision_budget,
        ),
        safe_vs_false_quality(
            rev26_test,
            rev26_test.rev24_binary_prob,
            "rev26_binary_calibrated",
            precision_budget=precision_budget,
        ),
    ]
    candidate_rows.extend(class_rows(rev24_test, "rev24_3way"))
    candidate_rows.extend(class_rows(rev26_test, "rev26_3way"))

    recovery_rows, rescue_rows, targeted_rows = summarize_recovery_and_rescue(
        rev24_test,
        rev26_test,
        budget_name=budget_name,
    )

    rev6_noisy = next(row for row in recovery_rows if row["row"] == "rev6" and row["scope"] == "noisy")
    rev25_noisy = next(row for row in recovery_rows if row["row"] == "rev25_class_aware" and row["scope"] == "noisy")
    rev26_noisy = next(row for row in recovery_rows if row["row"] == "rev26_calibrated" and row["scope"] == "noisy")
    gate = {
        "beats_rev6_noisy_f1": bool(rev26_noisy["f1"] > rev6_noisy["f1"]),
        "beats_rev25_noisy_f1": bool(rev26_noisy["f1"] > rev25_noisy["f1"]),
        "rev6_noisy_f1": rev6_noisy["f1"],
        "rev25_noisy_f1": rev25_noisy["f1"],
        "rev26_noisy_f1": rev26_noisy["f1"],
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "budget_selection_val.csv", budget_rows)
    write_csv(output_dir / "candidate_quality_summary.csv", candidate_rows)
    write_csv(output_dir / "recovery_summary.csv", recovery_rows)
    write_csv(output_dir / "rescue_scope_summary.csv", rescue_rows)
    write_csv(output_dir / "targeted_rescue_diagnostics.csv", targeted_rows)
    summary = {
        "step": "step30_rev26_latent_calibration_objective",
        "backend_rerun": False,
        "backend_rerun_reason": "rev26 is recovery-first; Step30c remains gated on recovery.",
        "objective": "binary safe-vs-low-hint-false calibration head over rescue_candidate_latent",
        "selected_budget_name": budget_name,
        "budget_selection_val": budget_rows,
        "recovery_gate": gate,
        "candidate_quality_summary": candidate_rows,
        "recovery_summary": recovery_rows,
        "rescue_scope_summary": rescue_rows,
        "targeted_rescue_diagnostics": targeted_rows,
        "args": vars(args),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(clean_json_numbers(summary), f, indent=2)
    print(json.dumps(clean_json_numbers(summary), indent=2))


if __name__ == "__main__":
    main()
