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
    scorer_quality_row,
    selected_additions,
    top20_budget,
    weak_positive_mask,
    write_csv,
)
from train.eval_step30_rescue_ambiguity_subtype_probe_rev28 import (
    parse_thresholds,
    prediction_with_additions,
    threshold_array,
)
from train.eval_step30_rescue_candidate_latent_probe_rev23 import clean_json_numbers


def rev25_score(rows: Rev30Rows) -> np.ndarray:
    return rows.rev24_safe_prob - np.maximum(rows.rev24_false_prob, rows.rev24_ambiguous_prob)


def rev31_score(rows: Rev30Rows) -> np.ndarray:
    score = rev30_score(rows).copy()
    score[weak_positive_mask(rows)] = rows.rev30_weak_positive_safety_prob[weak_positive_mask(rows)]
    return score


def base_prediction(rows: Rev30Rows, thresholds: Dict[str, float]) -> np.ndarray:
    return (rows.rev6_score >= threshold_array(rows.variant, thresholds)).astype(np.int64)


def load_rev30_offline_probe_row(path: str) -> Dict[str, Any]:
    summary_path = Path(path)
    if summary_path.is_dir():
        summary_path = summary_path / "summary.json"
    with open(summary_path) as f:
        summary = json.load(f)
    for row in summary["probe_quality_summary"]:
        if row["row"] == "rev30_with_safety_probe":
            out = dict(row)
            out["row"] = "rev30_offline_probe"
            return out
    raise ValueError(f"Could not find rev30_with_safety_probe in {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", default="data/graph_event_step30_weak_obs_rev30_test.pkl")
    parser.add_argument("--rev6_checkpoint", default="checkpoints/step30_encoder_recovery_rev6/best.pt")
    parser.add_argument("--rev17_checkpoint", default="checkpoints/step30_encoder_recovery_rev17/best.pt")
    parser.add_argument("--rev24_checkpoint", default="checkpoints/step30_encoder_recovery_rev24/best.pt")
    parser.add_argument("--rev26_checkpoint", default="checkpoints/step30_encoder_recovery_rev26/best.pt")
    parser.add_argument("--rev30_checkpoint", default="checkpoints/step30_encoder_recovery_rev30/best.pt")
    parser.add_argument("--rev31_checkpoint", default="checkpoints/step30_encoder_recovery_rev31/best.pt")
    parser.add_argument("--rev30_summary", default="artifacts/step30_positive_ambiguity_safety_rev30")
    parser.add_argument("--output_dir", default="artifacts/step30_subtype_scoped_safety_rev31")
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
    rows30 = collect_rows(
        args.test_path,
        models={**shared_models, "rev30": load_model(args.rev30_checkpoint, device)},
        **collect_kwargs,
    )
    rows31 = collect_rows(
        args.test_path,
        models={**shared_models, "rev30": load_model(args.rev31_checkpoint, device)},
        **collect_kwargs,
    )

    thresholds = parse_thresholds(args.thresholds)
    budget = top20_budget(rows30)
    rev24_add = selected_additions(rows30, rows30.rev24_safe_prob, thresholds, budget)
    rev25_add = selected_additions(rows30, rev25_score(rows30), thresholds, budget)
    rev26_add = selected_additions(rows30, rev26_score(rows30), thresholds, budget)
    rev30_add = selected_additions(rows30, rev30_score(rows30), thresholds, budget)
    rev31_add = selected_additions(rows31, rev31_score(rows31), thresholds, budget)
    rev31_same_weak_count_add = rev30_add.copy()
    weak_mask31 = weak_positive_mask(rows31)
    weak_count_rev30 = int((rev30_add & weak_positive_mask(rows30)).sum())
    rev31_same_weak_count_add[weak_positive_mask(rows30)] = False
    weak_candidates = np.flatnonzero(
        weak_mask31 & (base_prediction(rows31, thresholds) == 0)
    )
    if weak_count_rev30 > 0 and len(weak_candidates) > 0:
        order = np.argsort(-rows31.rev30_weak_positive_safety_prob[weak_candidates])
        chosen = weak_candidates[order[: min(weak_count_rev30, len(order))]]
        rev31_same_weak_count_add[chosen] = True

    weak_mask30 = weak_positive_mask(rows30)
    weak_budget = int((rev26_add & weak_mask30).sum())
    weak_labels = rows30.target[weak_mask30].astype(np.int64)

    probe_rows = [
        load_rev30_offline_probe_row(args.rev30_summary),
        scorer_quality_row(
            rev30_score(rows30)[weak_mask30],
            weak_labels,
            "rev30_integrated",
            weak_budget,
        ),
        scorer_quality_row(
            rows31.rev30_weak_positive_safety_prob[weak_mask31],
            rows31.target[weak_mask31].astype(np.int64),
            "rev31_integrated_subtype_head",
            weak_budget,
        ),
        scorer_quality_row(
            rev31_score(rows31)[weak_mask31],
            rows31.target[weak_mask31].astype(np.int64),
            "rev31_integrated_admission_score",
            weak_budget,
        ),
    ]

    rescue_rows = [
        rescue_scope_row(rows30, rev24_add, "rev24_safe_only"),
        rescue_scope_row(rows30, rev25_add, "rev25_class_aware"),
        rescue_scope_row(rows30, rev26_add, "rev26_calibrated"),
        rescue_scope_row(rows30, rev30_add, "rev30_integrated"),
        rescue_scope_row(rows31, rev31_add, "rev31_subtype_scoped"),
        rescue_scope_row(
            rows31,
            rev31_same_weak_count_add,
            "rev31_head_same_rev30_weak_count",
        ),
    ]

    pred_rev6 = base_prediction(rows30, thresholds)
    pred_rev26 = prediction_with_additions(rows30, rev26_add, thresholds)
    pred_rev30 = prediction_with_additions(rows30, rev30_add, thresholds)
    pred_rev31 = prediction_with_additions(rows31, rev31_add, thresholds)
    pred_rev31_same_weak_count = prediction_with_additions(
        rows31,
        rev31_same_weak_count_add,
        thresholds,
    )
    pred_trivial = (
        rows30.trivial_score >= threshold_array(rows30.variant, thresholds)
    ).astype(np.int64)
    recovery_rows = [
        recovery_row(rows30, pred_rev6, rows30.rev6_score, "rev6"),
        recovery_row(rows30, pred_rev26, rev26_score(rows30), "rev26_calibrated"),
        recovery_row(rows30, pred_rev30, rev30_score(rows30), "rev30_integrated"),
        recovery_row(rows31, pred_rev31, rev31_score(rows31), "rev31_subtype_scoped"),
        recovery_row(
            rows31,
            pred_rev31_same_weak_count,
            rev31_score(rows31),
            "rev31_head_same_rev30_weak_count",
        ),
        recovery_row(rows30, pred_trivial, rows30.trivial_score, "trivial_with_rev30_cue"),
    ]

    rev6_noisy_f1 = next(row for row in recovery_rows if row["row"] == "rev6")["noisy_f1"]
    rev30_noisy_f1 = next(row for row in recovery_rows if row["row"] == "rev30_integrated")["noisy_f1"]
    rev31_noisy_f1 = next(row for row in recovery_rows if row["row"] == "rev31_subtype_scoped")["noisy_f1"]
    rev30_probe_ap = next(row for row in probe_rows if row["row"] == "rev30_offline_probe")["ap"]
    rev30_integrated_ap = next(row for row in probe_rows if row["row"] == "rev30_integrated")["ap"]
    rev31_integrated_ap = next(
        row for row in probe_rows if row["row"] == "rev31_integrated_subtype_head"
    )["ap"]
    gate = {
        "rev31_noisy_edge_f1_beats_rev6": bool(rev31_noisy_f1 > rev6_noisy_f1),
        "rev31_noisy_edge_f1_improves_over_rev30": bool(rev31_noisy_f1 > rev30_noisy_f1),
        "rev31_weak_positive_ap_improves_over_rev30_integrated": bool(
            (rev31_integrated_ap or 0.0) > (rev30_integrated_ap or 0.0)
        ),
        "rev31_closes_probe_gap": bool(
            abs(float(rev30_probe_ap or 0.0) - float(rev31_integrated_ap or 0.0))
            < abs(float(rev30_probe_ap or 0.0) - float(rev30_integrated_ap or 0.0))
        ),
        "backend_rerun": False,
        "backend_rerun_reason": "Step30c was not run; rev31 is recovery-side only.",
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "probe_quality_summary.csv", probe_rows)
    write_csv(output_dir / "rescue_scope_summary.csv", rescue_rows)
    write_csv(output_dir / "recovery_summary.csv", recovery_rows)
    summary = {
        "step": "step30_rev31_subtype_scoped_safety_objective",
        "objective": "binary safety head/loss only inside weak_positive_ambiguous",
        "selected_budget": int(budget),
        "rev26_weak_positive_selected_count": int(weak_budget),
        "probe_quality_summary": probe_rows,
        "rescue_scope_summary": rescue_rows,
        "recovery_summary": recovery_rows,
        "gate": gate,
        "args": vars(args),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(clean_json_numbers(summary), f, indent=2)
    print(json.dumps(clean_json_numbers(summary), indent=2))


if __name__ == "__main__":
    main()
