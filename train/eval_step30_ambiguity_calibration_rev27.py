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

from train.eval_step30_class_aware_rescue_admission_rev25 import (
    base_rev6_prediction,
    class_aware_score,
    rev21_prediction,
)
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


def rev26_admission_score(split: SplitRows) -> np.ndarray:
    return split.rev24_binary_prob - split.rev24_ambiguous_prob


def rev27_admission_score(split: SplitRows) -> np.ndarray:
    return split.rev24_binary_prob - split.rev24_ambiguity_prob


def choose_rev27_budget(
    split: SplitRows,
    budget_names: list[str],
) -> tuple[str, list[Dict[str, Any]]]:
    score = rev27_admission_score(split)
    rows = []
    best_name = budget_names[0]
    best_score = -float("inf")
    best_precision = -float("inf")
    for budget_name in budget_names:
        pred, additions, budget = pred_from_score(split, score, budget_name)
        noisy = split.variant == "noisy"
        noisy_metrics = edge_metrics_from_arrays(pred[noisy], split.target[noisy])
        candidate = split.is_candidate
        addition_metrics = edge_metrics_from_arrays(
            additions[candidate].astype(np.int64),
            split.target[candidate],
        )
        labels = split.candidate_labels
        selected = additions[candidate].astype(bool)
        ambiguous_mask = labels == 2
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
            "ambiguous_admitted": int((selected & ambiguous_mask).sum()),
            "ambiguous_admission_rate": float(
                (selected & ambiguous_mask).sum() / max(int(ambiguous_mask.sum()), 1)
            ),
        }
        rows.append(row)
        score_for_selection = float(noisy_metrics["f1"])
        precision_for_tiebreak = float(addition_metrics["precision"])
        if (
            score_for_selection > best_score
            or (
                score_for_selection == best_score
                and precision_for_tiebreak > best_precision
            )
        ):
            best_score = score_for_selection
            best_precision = precision_for_tiebreak
            best_name = budget_name
    return best_name, rows


def precision_at_budget(score: np.ndarray, labels: np.ndarray, budget: int) -> Dict[str, float | int]:
    order = np.argsort(-score)
    k = min(max(int(budget), 1), len(order))
    chosen = np.zeros(len(labels), dtype=bool)
    chosen[order[:k]] = True
    tp = float((chosen & (labels == 1)).sum())
    fp = float((chosen & (labels == 0)).sum())
    fn = float((~chosen & (labels == 1)).sum())
    metrics = binary_metrics(tp, fp, fn)
    return {
        "precision_budget": int(k),
        "precision_at_budget": metrics["precision"],
        "budget_recall": metrics["recall"],
        "budget_f1": metrics["f1"],
    }


def safe_vs_false_quality(
    split: SplitRows,
    score: np.ndarray,
    row_name: str,
    precision_budget: int,
) -> Dict[str, Any]:
    full_candidate_idx = np.flatnonzero(split.is_candidate)
    safe_false_mask = split.candidate_labels != 2
    full_idx = full_candidate_idx[safe_false_mask]
    labels = (split.candidate_labels[safe_false_mask] == 0).astype(np.int64)
    scores = score[full_idx]
    row: Dict[str, Any] = {
        "row": row_name,
        "scope": "safe_vs_low_hint_false_only",
        "candidate_count": int(len(labels)),
        "safe_count": int((labels == 1).sum()),
        "false_count": int((labels == 0).sum()),
        "ap": average_precision(scores, labels),
        "auroc": auroc(scores, labels),
    }
    row.update(precision_at_budget(scores, labels, precision_budget))
    return row


def ambiguity_quality(
    split: SplitRows,
    score: np.ndarray,
    row_name: str,
    precision_budget: int,
) -> Dict[str, Any]:
    full_candidate_idx = np.flatnonzero(split.is_candidate)
    labels = (split.candidate_labels == 2).astype(np.int64)
    scores = score[full_candidate_idx]
    row: Dict[str, Any] = {
        "row": row_name,
        "scope": "ambiguity_detection",
        "candidate_count": int(len(labels)),
        "ambiguous_count": int((labels == 1).sum()),
        "non_ambiguous_count": int((labels == 0).sum()),
        "ap": average_precision(scores, labels),
        "auroc": auroc(scores, labels),
    }
    row.update(precision_at_budget(scores, labels, precision_budget))
    return row


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
            "scope": "3way_argmax",
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
    rev27_split: SplitRows,
    budget_name: str,
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]], list[Dict[str, Any]], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
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
    rev27_pred, rev27_add, rev27_budget = pred_from_score(
        rev27_split,
        rev27_admission_score(rev27_split),
        budget_name,
    )
    pred_by_name = {
        "rev6": base_rev6_prediction(rev27_split),
        "rev21": rev21_prediction(rev27_split),
        "rev24_safe_only": rev24_pred,
        "rev25_class_aware": rev25_pred,
        "rev26_calibrated": rev26_pred,
        "rev27_ambiguity_aware": rev27_pred,
    }
    recovery_rows: list[Dict[str, Any]] = []
    for scope_name, scope_mask in {
        "overall": np.ones_like(rev27_split.target, dtype=bool),
        "clean": rev27_split.variant == "clean",
        "noisy": rev27_split.variant == "noisy",
    }.items():
        for name, pred in pred_by_name.items():
            row = {"row": name, "scope": scope_name}
            row.update(edge_metrics_from_arrays(pred[scope_mask], rev27_split.target[scope_mask]))
            recovery_rows.append(row)
    add_family_rows(recovery_rows, rev27_split, pred_by_name)

    additions_by_name = {
        "rev24_safe_only": (rev24_split, rev24_add, rev24_budget),
        "rev25_class_aware": (rev24_split, rev25_add, rev25_budget),
        "rev26_calibrated": (rev26_split, rev26_add, rev26_budget),
        "rev27_ambiguity_aware": (rev27_split, rev27_add, rev27_budget),
    }
    rescue_rows: list[Dict[str, Any]] = []
    for name, pred in pred_by_name.items():
        split = rev27_split
        if name in {"rev24_safe_only", "rev25_class_aware"}:
            split = rev24_split
        elif name == "rev26_calibrated":
            split = rev26_split
        candidate = split.is_candidate
        selected = pred[candidate].astype(bool)
        row: Dict[str, Any] = {
            "row": name,
            "scope": "full_rescue_scope_decode",
            "admitted": int(selected.sum()),
        }
        row.update(edge_metrics_from_arrays(selected.astype(np.int64), split.target[candidate]))
        row.update(admission_rates(selected, split.candidate_labels))
        rescue_rows.append(row)
    for name, (split, additions, budget) in additions_by_name.items():
        candidate = split.is_candidate
        selected = additions[candidate].astype(bool)
        row = {
            "row": name,
            "scope": "selected_rescue_additions_only",
            "budget": int(budget),
            "admitted": int(selected.sum()),
        }
        row.update(edge_metrics_from_arrays(selected.astype(np.int64), split.target[candidate]))
        row.update(admission_rates(selected, split.candidate_labels))
        rescue_rows.append(row)

    score_by_name = {
        "rev6": rev27_split.rev6_score,
        "rev21": rev27_split.rev21_score,
        "rev24_safe_only": np.where(
            rev24_split.is_candidate,
            rev24_split.rev24_safe_prob,
            rev24_split.rev6_score,
        ),
        "rev25_class_aware": np.where(
            rev24_split.is_candidate,
            class_aware_score(rev24_split),
            rev24_split.rev6_score,
        ),
        "rev26_calibrated": np.where(
            rev26_split.is_candidate,
            rev26_admission_score(rev26_split),
            rev26_split.rev6_score,
        ),
        "rev27_ambiguity_aware": np.where(
            rev27_split.is_candidate,
            rev27_admission_score(rev27_split),
            rev27_split.rev6_score,
        ),
    }
    targeted_rows = []
    for name, pred in pred_by_name.items():
        split = rev27_split
        if name in {"rev24_safe_only", "rev25_class_aware"}:
            split = rev24_split
        elif name == "rev26_calibrated":
            split = rev26_split
        row = {"row": name}
        row.update(targeted_metrics(split, pred, score_by_name[name]))
        targeted_rows.append(row)
    addition_arrays = {
        "rev24_safe_only": rev24_add,
        "rev25_class_aware": rev25_add,
        "rev26_calibrated": rev26_add,
        "rev27_ambiguity_aware": rev27_add,
    }
    return recovery_rows, rescue_rows, targeted_rows, pred_by_name, addition_arrays


def class_probability_rows(
    split: SplitRows,
    additions: np.ndarray,
    row_name: str,
) -> list[Dict[str, Any]]:
    candidate = split.is_candidate
    selected = additions[candidate].astype(bool)
    rows: list[Dict[str, Any]] = []
    fields = {
        "safe_prob": split.rev24_safe_prob[candidate],
        "false_prob": split.rev24_false_prob[candidate],
        "softmax_ambiguous_prob": split.rev24_ambiguous_prob[candidate],
        "binary_safe_prob": split.rev24_binary_prob[candidate],
        "ambiguity_risk_prob": split.rev24_ambiguity_prob[candidate],
        "admission_score": rev27_admission_score(split)[candidate],
    }
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_mask = split.candidate_labels == class_idx
        for disposition, mask in {
            "admitted": selected,
            "rejected": ~selected,
        }.items():
            scope = class_mask & mask
            out: Dict[str, Any] = {
                "row": row_name,
                "class": class_name,
                "disposition": disposition,
                "count": int(scope.sum()),
            }
            for field_name, values in fields.items():
                out[f"avg_{field_name}"] = float(values[scope].mean()) if scope.any() else None
            rows.append(out)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_path", default="data/graph_event_step30_weak_obs_rev17_val.pkl")
    parser.add_argument("--test_path", default="data/graph_event_step30_weak_obs_rev17_test.pkl")
    parser.add_argument("--rev6_checkpoint", default="checkpoints/step30_encoder_recovery_rev6/best.pt")
    parser.add_argument("--rev21_checkpoint", default="checkpoints/step30_encoder_recovery_rev21/best.pt")
    parser.add_argument("--rev24_checkpoint", default="checkpoints/step30_encoder_recovery_rev24/best.pt")
    parser.add_argument("--rev26_checkpoint", default="checkpoints/step30_encoder_recovery_rev26/best.pt")
    parser.add_argument("--rev27_checkpoint", default="checkpoints/step30_encoder_recovery_rev27/best.pt")
    parser.add_argument("--output_dir", default="artifacts/step30_encoder_recovery_rev27")
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
    rev27_models = {**base_models, "rev24": load_model(args.rev27_checkpoint, device)}
    thresholds = parse_thresholds(args.thresholds)

    rev27_val = collect_split(
        args.val_path,
        models=rev27_models,
        device=device,
        thresholds=thresholds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        rescue_relation_max=args.rescue_relation_max,
        rescue_support_min=args.rescue_support_min,
    )
    budget_name, budget_rows = choose_rev27_budget(
        rev27_val,
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
    rev27_test = collect_split(
        args.test_path,
        models=rev27_models,
        device=device,
        thresholds=thresholds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        rescue_relation_max=args.rescue_relation_max,
        rescue_support_min=args.rescue_support_min,
    )

    precision_budget = int(budget_specs(rev27_test)[budget_name])
    candidate_rows: list[Dict[str, Any]] = [
        safe_vs_false_quality(
            rev24_test,
            rev24_test.rev24_safe_prob,
            "rev24_integrated_safe_prob",
            precision_budget=precision_budget,
        ),
        safe_vs_false_quality(
            rev26_test,
            rev26_test.rev24_binary_prob,
            "rev26_safe_vs_false_binary",
            precision_budget=precision_budget,
        ),
        safe_vs_false_quality(
            rev27_test,
            rev27_test.rev24_binary_prob,
            "rev27_safe_vs_false_binary",
            precision_budget=precision_budget,
        ),
        ambiguity_quality(
            rev24_test,
            rev24_test.rev24_ambiguous_prob,
            "rev24_softmax_ambiguity_prob",
            precision_budget=precision_budget,
        ),
        ambiguity_quality(
            rev26_test,
            rev26_test.rev24_ambiguous_prob,
            "rev26_softmax_ambiguity_prob",
            precision_budget=precision_budget,
        ),
        ambiguity_quality(
            rev27_test,
            rev27_test.rev24_ambiguity_prob,
            "rev27_ambiguity_risk_head",
            precision_budget=precision_budget,
        ),
    ]
    candidate_rows.extend(class_rows(rev24_test, "rev24_3way"))
    candidate_rows.extend(class_rows(rev26_test, "rev26_3way"))
    candidate_rows.extend(class_rows(rev27_test, "rev27_3way"))

    recovery_rows, rescue_rows, targeted_rows, _pred_by_name, addition_arrays = summarize_recovery_and_rescue(
        rev24_test,
        rev26_test,
        rev27_test,
        budget_name=budget_name,
    )
    class_probability_summary = []
    class_probability_summary.extend(
        class_probability_rows(
            rev26_test,
            addition_arrays["rev26_calibrated"],
            "rev26_calibrated",
        )
    )
    class_probability_summary.extend(
        class_probability_rows(
            rev27_test,
            addition_arrays["rev27_ambiguity_aware"],
            "rev27_ambiguity_aware",
        )
    )

    rev6_noisy = next(row for row in recovery_rows if row["row"] == "rev6" and row["scope"] == "noisy")
    rev26_noisy = next(row for row in recovery_rows if row["row"] == "rev26_calibrated" and row["scope"] == "noisy")
    rev27_noisy = next(row for row in recovery_rows if row["row"] == "rev27_ambiguity_aware" and row["scope"] == "noisy")
    rev26_selected = next(
        row
        for row in rescue_rows
        if row["row"] == "rev26_calibrated" and row["scope"] == "selected_rescue_additions_only"
    )
    rev27_selected = next(
        row
        for row in rescue_rows
        if row["row"] == "rev27_ambiguity_aware" and row["scope"] == "selected_rescue_additions_only"
    )
    gate = {
        "beats_rev6_noisy_f1": bool(rev27_noisy["f1"] > rev6_noisy["f1"]),
        "beats_rev26_noisy_f1": bool(rev27_noisy["f1"] > rev26_noisy["f1"]),
        "reduces_ambiguous_admission_vs_rev26": bool(
            rev27_selected["ambiguous_rescue_candidate_admission_rate"]
            < rev26_selected["ambiguous_rescue_candidate_admission_rate"]
        ),
        "preserves_safe_admission_vs_rev26": bool(
            rev27_selected["safe_missed_true_edge_admission_rate"]
            >= 0.95 * rev26_selected["safe_missed_true_edge_admission_rate"]
        ),
        "rev6_noisy_f1": rev6_noisy["f1"],
        "rev26_noisy_f1": rev26_noisy["f1"],
        "rev27_noisy_f1": rev27_noisy["f1"],
        "rev26_selected_ambiguous_admission_rate": rev26_selected[
            "ambiguous_rescue_candidate_admission_rate"
        ],
        "rev27_selected_ambiguous_admission_rate": rev27_selected[
            "ambiguous_rescue_candidate_admission_rate"
        ],
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "budget_selection_val.csv", budget_rows)
    write_csv(output_dir / "candidate_quality_summary.csv", candidate_rows)
    write_csv(output_dir / "recovery_summary.csv", recovery_rows)
    write_csv(output_dir / "rescue_scope_summary.csv", rescue_rows)
    write_csv(output_dir / "targeted_rescue_diagnostics.csv", targeted_rows)
    write_csv(output_dir / "class_probability_admission_summary.csv", class_probability_summary)
    summary = {
        "step": "step30_rev27_ambiguity_aware_latent_calibration_probe",
        "backend_rerun": False,
        "backend_rerun_reason": "rev27 is recovery-first; Step30c remains gated on recovery.",
        "objective": "ambiguity-risk auxiliary head over integrated rescue_candidate_latent",
        "admission_rule": "rev24/rev26 binary safe score minus rev27 ambiguity-risk probability inside rescue scope",
        "selected_budget_name": budget_name,
        "budget_selection_val": budget_rows,
        "recovery_gate": gate,
        "candidate_quality_summary": candidate_rows,
        "recovery_summary": recovery_rows,
        "rescue_scope_summary": rescue_rows,
        "targeted_rescue_diagnostics": targeted_rows,
        "class_probability_admission_summary": class_probability_summary,
        "args": vars(args),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(clean_json_numbers(summary), f, indent=2)
    print(json.dumps(clean_json_numbers(summary), indent=2))


if __name__ == "__main__":
    main()
