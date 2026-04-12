from __future__ import annotations

import argparse
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
from train.eval_step30_positive_ambiguity_safety_rev30 import (
    Rev30Rows,
    collect_rows,
    recovery_row,
    rescue_scope_row,
    rev26_score,
    rev30_score,
    selected_additions,
    top20_budget,
    weak_positive_mask,
    write_csv,
)
from train.eval_step30_rescue_ambiguity_subtype_probe_rev28 import (
    parse_thresholds,
    threshold_array,
)
from train.eval_step30_rescue_candidate_latent_probe_rev23 import clean_json_numbers


def rev25_score(rows: Rev30Rows) -> np.ndarray:
    return rows.rev24_safe_prob - np.maximum(rows.rev24_false_prob, rows.rev24_ambiguous_prob)


def rev31_score(rows: Rev30Rows) -> np.ndarray:
    score = rev30_score(rows).copy()
    weak_mask = weak_positive_mask(rows)
    score[weak_mask] = rows.rev30_weak_positive_safety_prob[weak_mask]
    return score


def base_prediction(rows: Rev30Rows, thresholds: Dict[str, float]) -> np.ndarray:
    return (rows.rev6_score >= threshold_array(rows.variant, thresholds)).astype(np.int64)


def prediction_with_additions(
    rows: Rev30Rows,
    additions: np.ndarray,
    thresholds: Dict[str, float],
) -> np.ndarray:
    return np.maximum(base_prediction(rows, thresholds), additions.astype(np.int64))


def bucketed_budget_additions(
    rows: Rev30Rows,
    thresholds: Dict[str, float],
    total_budget: int,
    weak_positive_budget: int,
    nonweak_score: np.ndarray,
    weak_positive_score: np.ndarray,
) -> np.ndarray:
    base_pred = base_prediction(rows, thresholds)
    weak_mask = weak_positive_mask(rows)
    eligible = rows.is_candidate & (base_pred == 0)
    weak_candidates = np.flatnonzero(eligible & weak_mask)
    nonweak_candidates = np.flatnonzero(eligible & (~weak_mask))

    weak_budget = min(max(int(weak_positive_budget), 0), len(weak_candidates), int(total_budget))
    nonweak_budget = min(max(int(total_budget) - weak_budget, 0), len(nonweak_candidates))
    chosen = np.zeros_like(rows.target, dtype=bool)
    if weak_budget > 0 and len(weak_candidates) > 0:
        weak_order = np.argsort(-weak_positive_score[weak_candidates])
        chosen[weak_candidates[weak_order[:weak_budget]]] = True
    if nonweak_budget > 0 and len(nonweak_candidates) > 0:
        nonweak_order = np.argsort(-nonweak_score[nonweak_candidates])
        chosen[nonweak_candidates[nonweak_order[:nonweak_budget]]] = True
    return chosen


def calibration_row(
    rows: Rev30Rows,
    additions: np.ndarray,
    thresholds: Dict[str, float],
    row_name: str,
    weak_budget_source: str,
) -> Dict[str, Any]:
    rescue = rescue_scope_row(rows, additions, row_name)
    pred = prediction_with_additions(rows, additions, thresholds)
    noisy = rows.variant == "noisy"
    noisy_metrics = recovery_row(rows, pred, rev31_score(rows), row_name)
    return {
        **rescue,
        "weak_budget_source": weak_budget_source,
        "noisy_precision": noisy_metrics["noisy_precision"],
        "noisy_recall": noisy_metrics["noisy_recall"],
        "noisy_f1": noisy_metrics["noisy_f1"],
        "noisy_tp": noisy_metrics["noisy_tp"],
        "noisy_fp": noisy_metrics["noisy_fp"],
        "noisy_fn": noisy_metrics["noisy_fn"],
        "noisy_candidate_count": int(noisy.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_path", default="data/graph_event_step30_weak_obs_rev30_val.pkl")
    parser.add_argument("--test_path", default="data/graph_event_step30_weak_obs_rev30_test.pkl")
    parser.add_argument("--rev6_checkpoint", default="checkpoints/step30_encoder_recovery_rev6/best.pt")
    parser.add_argument("--rev17_checkpoint", default="checkpoints/step30_encoder_recovery_rev17/best.pt")
    parser.add_argument("--rev24_checkpoint", default="checkpoints/step30_encoder_recovery_rev24/best.pt")
    parser.add_argument("--rev26_checkpoint", default="checkpoints/step30_encoder_recovery_rev26/best.pt")
    parser.add_argument("--rev30_checkpoint", default="checkpoints/step30_encoder_recovery_rev30/best.pt")
    parser.add_argument("--rev31_checkpoint", default="checkpoints/step30_encoder_recovery_rev31/best.pt")
    parser.add_argument("--output_dir", default="artifacts/step30_subtype_head_budget_calibration_rev32")
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

    shared_models = {
        "rev6": load_model(args.rev6_checkpoint, device),
        "rev17": load_model(args.rev17_checkpoint, device),
        "rev24": load_model(args.rev24_checkpoint, device),
        "rev26": load_model(args.rev26_checkpoint, device),
    }
    collect_kwargs = {
        "device": device,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "rescue_relation_max": args.rescue_relation_max,
        "rescue_support_min": args.rescue_support_min,
    }
    val30 = collect_rows(
        args.val_path,
        models={**shared_models, "rev30": load_model(args.rev30_checkpoint, device)},
        **collect_kwargs,
    )
    test30 = collect_rows(
        args.test_path,
        models={**shared_models, "rev30": load_model(args.rev30_checkpoint, device)},
        **collect_kwargs,
    )
    test31 = collect_rows(
        args.test_path,
        models={**shared_models, "rev30": load_model(args.rev31_checkpoint, device)},
        **collect_kwargs,
    )

    thresholds = parse_thresholds(args.thresholds)
    val_budget = top20_budget(val30)
    test_budget = top20_budget(test30)
    val_rev30_add = selected_additions(val30, rev30_score(val30), thresholds, val_budget)
    val_weak_selected = int((val_rev30_add & weak_positive_mask(val30)).sum())
    weak_budget_fraction = val_weak_selected / max(val_budget, 1)
    rev32_weak_budget = int(round(weak_budget_fraction * test_budget))

    rev24_add = selected_additions(test30, test30.rev24_safe_prob, thresholds, test_budget)
    rev25_add = selected_additions(test30, rev25_score(test30), thresholds, test_budget)
    rev26_add = selected_additions(test30, rev26_score(test30), thresholds, test_budget)
    rev30_add = selected_additions(test30, rev30_score(test30), thresholds, test_budget)
    rev31_add = selected_additions(test31, rev31_score(test31), thresholds, test_budget)
    rev31_same_weak_count = bucketed_budget_additions(
        rows=test31,
        thresholds=thresholds,
        total_budget=test_budget,
        weak_positive_budget=int((rev30_add & weak_positive_mask(test30)).sum()),
        nonweak_score=rev30_score(test31),
        weak_positive_score=test31.rev30_weak_positive_safety_prob,
    )
    rev32_add = bucketed_budget_additions(
        rows=test31,
        thresholds=thresholds,
        total_budget=test_budget,
        weak_positive_budget=rev32_weak_budget,
        nonweak_score=rev30_score(test31),
        weak_positive_score=test31.rev30_weak_positive_safety_prob,
    )
    rev32_hybrid_add = bucketed_budget_additions(
        rows=test31,
        thresholds=thresholds,
        total_budget=test_budget,
        weak_positive_budget=rev32_weak_budget,
        nonweak_score=rev30_score(test30),
        weak_positive_score=test31.rev30_weak_positive_safety_prob,
    )

    admission_rows = [
        calibration_row(test30, rev30_add, thresholds, "rev30_integrated", "single_shared_budget"),
        calibration_row(test31, rev31_add, thresholds, "rev31_default", "single_shared_budget"),
        calibration_row(
            test31,
            rev31_same_weak_count,
            thresholds,
            "rev31_same_rev30_test_weak_count",
            "diagnostic_test_count_match",
        ),
        calibration_row(
            test31,
            rev32_add,
            thresholds,
            "rev32_val_matched_weak_budget_rev31_nonweak",
            "val_rev30_weak_fraction",
        ),
        calibration_row(
            test31,
            rev32_hybrid_add,
            thresholds,
            "rev32_val_matched_weak_budget_rev30_nonweak",
            "val_rev30_weak_fraction",
        ),
    ]
    pred_rev6 = base_prediction(test30, thresholds)
    pred_rev26 = prediction_with_additions(test30, rev26_add, thresholds)
    pred_rev30 = prediction_with_additions(test30, rev30_add, thresholds)
    pred_rev31 = prediction_with_additions(test31, rev31_add, thresholds)
    pred_rev32 = prediction_with_additions(test31, rev32_add, thresholds)
    pred_rev32_hybrid = prediction_with_additions(test31, rev32_hybrid_add, thresholds)
    pred_trivial = (
        test30.trivial_score >= threshold_array(test30.variant, thresholds)
    ).astype(np.int64)

    recovery_rows = [
        recovery_row(test30, pred_rev6, test30.rev6_score, "rev6"),
        recovery_row(test30, pred_rev26, rev26_score(test30), "rev26_calibrated"),
        recovery_row(test30, pred_rev30, rev30_score(test30), "rev30_integrated"),
        recovery_row(test31, pred_rev31, rev31_score(test31), "rev31_default"),
        recovery_row(
            test31,
            pred_rev32,
            rev31_score(test31),
            "rev32_val_matched_weak_budget_rev31_nonweak",
        ),
        recovery_row(
            test31,
            pred_rev32_hybrid,
            rev31_score(test31),
            "rev32_val_matched_weak_budget_rev30_nonweak",
        ),
        recovery_row(test30, pred_trivial, test30.trivial_score, "trivial_with_rev30_cue"),
    ]

    rev6_noisy_f1 = next(row for row in recovery_rows if row["row"] == "rev6")["noisy_f1"]
    rev30_noisy_f1 = next(row for row in recovery_rows if row["row"] == "rev30_integrated")["noisy_f1"]
    rev31_noisy_f1 = next(row for row in recovery_rows if row["row"] == "rev31_default")["noisy_f1"]
    rev32_primary_noisy_f1 = next(
        row
        for row in recovery_rows
        if row["row"] == "rev32_val_matched_weak_budget_rev30_nonweak"
    )["noisy_f1"]
    gate = {
        "rev32_primary": "rev32_val_matched_weak_budget_rev30_nonweak",
        "rev32_noisy_edge_f1_beats_rev6": bool(rev32_primary_noisy_f1 > rev6_noisy_f1),
        "rev32_noisy_edge_f1_improves_over_rev30": bool(
            rev32_primary_noisy_f1 > rev30_noisy_f1
        ),
        "rev32_noisy_edge_f1_improves_over_rev31": bool(
            rev32_primary_noisy_f1 > rev31_noisy_f1
        ),
        "backend_rerun": False,
        "backend_rerun_reason": "Step30c was not run; rev32 did not clear the recovery gate.",
    }
    calibration = {
        "val_budget": int(val_budget),
        "val_rev30_weak_positive_selected": int(val_weak_selected),
        "val_rev30_weak_positive_budget_fraction": float(weak_budget_fraction),
        "test_budget": int(test_budget),
        "rev32_test_weak_positive_budget": int(rev32_weak_budget),
        "rev30_test_weak_positive_selected": int((rev30_add & weak_positive_mask(test30)).sum()),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "admission_calibration_summary.csv", admission_rows)
    write_csv(output_dir / "recovery_summary.csv", recovery_rows)
    summary = {
        "step": "step30_rev32_subtype_head_budget_calibration",
        "calibration_rule": "partition rescue budget into weak-positive and non-weak-positive buckets; weak-positive budget fraction selected on val from rev30 retained admissions",
        "calibration": calibration,
        "admission_calibration_summary": admission_rows,
        "recovery_summary": recovery_rows,
        "gate": gate,
        "args": vars(args),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(clean_json_numbers(summary), f, indent=2)
    print(json.dumps(clean_json_numbers(summary), indent=2))


if __name__ == "__main__":
    main()
