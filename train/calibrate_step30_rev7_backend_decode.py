from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train.eval_step30_frozen_backend_integration import evaluate


def parse_threshold_grid(grid_arg: str) -> list[float]:
    values: list[float] = []
    for part in grid_arg.split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    if not values:
        raise ValueError("--threshold_grid must contain at least one value")
    return values


def metric(row: Dict[str, Any], key: str) -> float:
    value = row.get(key)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0.0
    return float(value)


def backend_alignment_score(row: Dict[str, Any]) -> float:
    # Decode calibration is selected for backend utility, not edge-F1 alone.
    # The weights intentionally prioritize proposal coverage/OOS while keeping
    # a small amount of pressure on downstream edit and context behavior.
    return (
        0.40 * metric(row, "proposal_edge_recall")
        + 0.30 * (1.0 - metric(row, "out_of_scope_miss"))
        + 0.10 * metric(row, "changed_edge")
        + 0.10 * metric(row, "add")
        + 0.10 * metric(row, "context_edge")
    )


def candidate_row(payload: Dict[str, Any], backend: str, variant: str) -> Dict[str, Any]:
    for row in payload["summary_rows"]:
        if (
            row["backend"] == backend
            and row["input_mode"] == "encoder_recovered"
            and row["observation_variant"] == variant
            and row["group_type"] == "overall"
            and row["group_name"] == "all"
        ):
            return row
    raise ValueError(f"Missing candidate row for backend={backend!r}, variant={variant!r}")


def build_eval_args(args: argparse.Namespace, threshold: float) -> Namespace:
    return Namespace(
        data_path=args.calibration_data_path,
        encoder_checkpoint_path=args.encoder_checkpoint_path,
        backends=args.backend,
        clean_proposal_checkpoint_path=args.clean_proposal_checkpoint_path,
        w012_checkpoint_path=args.w012_checkpoint_path,
        noisy_proposal_checkpoint_path=args.noisy_proposal_checkpoint_path,
        rft1_checkpoint_path=args.rft1_checkpoint_path,
        clean_node_threshold=args.clean_node_threshold,
        clean_edge_threshold=args.clean_edge_threshold,
        noisy_node_threshold=args.noisy_node_threshold,
        noisy_edge_threshold=args.noisy_edge_threshold,
        recovered_edge_threshold=args.default_threshold,
        recovered_edge_thresholds_by_variant=f"default:{args.default_threshold},clean:{args.clean_threshold},noisy:{threshold}",
        decode_mode="threshold",
        rescue_variants="noisy",
        rescue_relation_max=0.5,
        rescue_support_min=0.55,
        rescue_budget_fraction=0.0,
        rescue_score_mode="raw",
        rescue_support_weight=0.5,
        rescue_relation_weight=0.25,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        limit_batches=args.limit_batches,
    )


def write_csv(path: Path, rows: list[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration_data_path", type=str, required=True)
    parser.add_argument("--encoder_checkpoint_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="artifacts/step30_frozen_backend_integration_rev7/calibration")
    parser.add_argument("--backend", type=str, default="rft1_p2")
    parser.add_argument("--calibration_backend_name", type=str, default="rft1_calibrated_p2")
    parser.add_argument("--observation_variant", type=str, default="noisy")
    parser.add_argument("--threshold_grid", type=str, default="0.35,0.40,0.45,0.50,0.55,0.60")
    parser.add_argument("--default_threshold", type=float, default=0.5)
    parser.add_argument("--clean_threshold", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--limit_batches", type=int, default=None)
    parser.add_argument("--clean_proposal_checkpoint_path", type=str, default="checkpoints/scope_proposal_node_edge_flipw2/best.pt")
    parser.add_argument("--w012_checkpoint_path", type=str, default="checkpoints/fp_keep_w012/best.pt")
    parser.add_argument("--noisy_proposal_checkpoint_path", type=str, default="checkpoints/proposal_noisy_obs_p2/best.pt")
    parser.add_argument("--rft1_checkpoint_path", type=str, default="checkpoints/step6_noisy_rewrite_rft1/best.pt")
    parser.add_argument("--clean_node_threshold", type=float, default=0.20)
    parser.add_argument("--clean_edge_threshold", type=float, default=0.15)
    parser.add_argument("--noisy_node_threshold", type=float, default=0.15)
    parser.add_argument("--noisy_edge_threshold", type=float, default=0.10)
    args = parser.parse_args()

    rows: list[Dict[str, Any]] = []
    for threshold in parse_threshold_grid(args.threshold_grid):
        payload = evaluate(build_eval_args(args, threshold))
        row = candidate_row(payload, backend=args.calibration_backend_name, variant=args.observation_variant)
        rows.append(
            {
                "threshold": threshold,
                "backend_alignment_score": backend_alignment_score(row),
                "full_edge": row.get("full_edge"),
                "context_edge": row.get("context_edge"),
                "changed_edge": row.get("changed_edge"),
                "add": row.get("add"),
                "delete": row.get("delete"),
                "proposal_edge_recall": row.get("proposal_edge_recall"),
                "out_of_scope_miss": row.get("out_of_scope_miss"),
                "recovery_edge_precision": row.get("recovery_edge_precision"),
                "recovery_edge_recall": row.get("recovery_edge_recall"),
                "recovery_edge_f1": row.get("recovery_edge_f1"),
            }
        )

    best = max(
        rows,
        key=lambda item: (
            float(item["backend_alignment_score"]),
            float(item["proposal_edge_recall"] or 0.0),
            -abs(float(item["threshold"]) - args.default_threshold),
        ),
    )
    payload = {
        "metadata": {
            "calibration_data_path": args.calibration_data_path,
            "encoder_checkpoint_path": args.encoder_checkpoint_path,
            "backend": args.backend,
            "calibration_backend_name": args.calibration_backend_name,
            "observation_variant": args.observation_variant,
            "threshold_grid": parse_threshold_grid(args.threshold_grid),
            "score_definition": (
                "0.40*proposal_edge_recall + 0.30*(1-out_of_scope_miss) + "
                "0.10*changed_edge + 0.10*add + 0.10*context_edge"
            ),
            "selected_thresholds_by_variant": {
                "default": args.default_threshold,
                "clean": args.clean_threshold,
                "noisy": best["threshold"],
            },
        },
        "selected": best,
        "candidates": rows,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "backend_decode_calibration.json", "w") as f:
        json.dump(payload, f, indent=2)
    write_csv(output_dir / "backend_decode_calibration.csv", rows)
    print(json.dumps(payload, indent=2))
    print(f"wrote JSON: {output_dir / 'backend_decode_calibration.json'}")
    print(f"wrote CSV: {output_dir / 'backend_decode_calibration.csv'}")


if __name__ == "__main__":
    main()
