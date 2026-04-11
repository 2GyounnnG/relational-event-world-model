from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train.eval_noisy_structured_observation import build_loader, get_device, resolve_path, save_json
from train.eval_scope_edit_localization_gap import slugify
from train.eval_step18_fallback_gate_frontier import (
    BASE_MODE,
    KEEP_FRACTIONS,
    ORACLE_CHOOSE_MODE,
    ORACLE_FALSE_SCOPE_MODE,
    STEP9C_MODE,
    THRESHOLD_GATE_MODE,
    compute_global_keep_score_thresholds,
    evaluate_step18,
    format_keep_fraction,
)
from train.eval_step9_gated_edge_completion import load_completion_model, load_proposal_model, load_rewrite_model
from train.train_step17_rescue_fallback_gate import load_fallback_gate_model


def vanilla_mode(frac: float) -> str:
    return f"step18_vanilla_topk_keep_{format_keep_fraction(frac)}"


def ranking_mode(frac: float) -> str:
    return f"step19_ranking_topk_keep_{format_keep_fraction(frac)}"


def source_topk_mode(frac: float) -> str:
    return f"step18_topk_keep_{format_keep_fraction(frac)}"


def merge_frontiers(vanilla: Dict[str, Any], ranking: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for section in ["proposal_side", "downstream", "decomposition", "chooser_behavior"]:
        merged[section] = {}
        for mode in [BASE_MODE, STEP9C_MODE, THRESHOLD_GATE_MODE, ORACLE_FALSE_SCOPE_MODE, ORACLE_CHOOSE_MODE]:
            if mode in vanilla[section]:
                merged[section][mode] = vanilla[section][mode]
        for frac in KEEP_FRACTIONS:
            src = source_topk_mode(frac)
            if src in vanilla[section]:
                merged[section][vanilla_mode(frac)] = vanilla[section][src]
            if src in ranking[section]:
                merged[section][ranking_mode(frac)] = ranking[section][src]

    merged["ranking_diagnostics"] = {
        "vanilla_bce": vanilla["ranking_diagnostics"],
        "pairwise_keep_ranking": ranking["ranking_diagnostics"],
    }
    merged["deltas_vs_step9c"] = compute_deltas_vs_step9c(merged["downstream"])
    return merged


def compute_deltas_vs_step9c(downstream_results: Dict[str, Any]) -> Dict[str, Any]:
    step9c = downstream_results[STEP9C_MODE]["overall"]
    return {
        mode: {
            key: None if metrics["overall"].get(key) is None else metrics["overall"][key] - step9c[key]
            for key in ["full_edge", "context_edge", "changed_edge", "add", "delete"]
        }
        for mode, metrics in downstream_results.items()
    }


def write_summary_csv(path: Path, payload: Dict[str, Any]) -> None:
    rows = []
    results = payload["results"]
    for mode, metrics_by_group in results["downstream"].items():
        down = metrics_by_group["overall"]
        decomp = results.get("decomposition", {}).get(mode, {}).get("overall", {})
        chooser = results.get("chooser_behavior", {}).get(mode, {}).get("overall", {})
        delta = results.get("deltas_vs_step9c", {}).get(mode, {})
        rows.append(
            {
                "mode": mode,
                "full_edge": down.get("full_edge"),
                "context_edge": down.get("context_edge"),
                "changed_edge": down.get("changed_edge"),
                "add": down.get("add"),
                "delete": down.get("delete"),
                "delta_full_edge_vs_step9c": delta.get("full_edge"),
                "delta_context_edge_vs_step9c": delta.get("context_edge"),
                "delta_changed_edge_vs_step9c": delta.get("changed_edge"),
                "delta_add_vs_step9c": delta.get("add"),
                "delta_delete_vs_step9c": delta.get("delete"),
                "actual_keep_fraction": chooser.get("actual_keep_fraction"),
                "chooser_target_precision_among_kept": chooser.get("chooser_target_precision_among_kept"),
                "chooser_target_recall": chooser.get("chooser_target_recall"),
                "gt_changed_precision_among_kept": chooser.get("gt_changed_precision_among_kept"),
                "gt_false_scope_fraction_among_kept": chooser.get("gt_false_scope_fraction_among_kept"),
                "false_scope_preserve_rate": decomp.get("preserve_rate_rescued_false_scope_budget"),
                "true_changed_correct_rate": decomp.get("correct_edit_rate_rescued_true_changed_budget"),
            }
        )
    for label, rank in results.get("ranking_diagnostics", {}).items():
        rows.append(
            {
                "mode": f"ranking_diagnostics::{label}",
                "full_edge": None,
                "context_edge": None,
                "changed_edge": None,
                "add": None,
                "delete": None,
                "delta_full_edge_vs_step9c": None,
                "delta_context_edge_vs_step9c": None,
                "delta_changed_edge_vs_step9c": None,
                "delta_add_vs_step9c": None,
                "delta_delete_vs_step9c": None,
                "actual_keep_fraction": None,
                "chooser_target_precision_among_kept": rank["overall"].get("chooser_target_ap"),
                "chooser_target_recall": rank["overall"].get("chooser_target_auroc"),
                "gt_changed_precision_among_kept": None,
                "gt_false_scope_fraction_among_kept": None,
                "false_scope_preserve_rate": None,
                "true_changed_correct_rate": None,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal_checkpoint_path", type=str, required=True)
    parser.add_argument("--completion_checkpoint_path", type=str, required=True)
    parser.add_argument("--rewrite_checkpoint_path", type=str, required=True)
    parser.add_argument("--vanilla_gate_checkpoint_path", type=str, required=True)
    parser.add_argument("--ranking_gate_checkpoint_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--split_name", type=str, default="test")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--artifact_dir", type=str, default="artifacts/step19_ranking_chooser")
    parser.add_argument("--node_threshold", type=float, default=0.20)
    parser.add_argument("--edge_threshold", type=float, default=0.15)
    parser.add_argument("--completion_threshold", type=float, default=0.50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = get_device(args.device)
    proposal_checkpoint_path = resolve_path(args.proposal_checkpoint_path)
    completion_checkpoint_path = resolve_path(args.completion_checkpoint_path)
    rewrite_checkpoint_path = resolve_path(args.rewrite_checkpoint_path)
    vanilla_gate_checkpoint_path = resolve_path(args.vanilla_gate_checkpoint_path)
    ranking_gate_checkpoint_path = resolve_path(args.ranking_gate_checkpoint_path)
    data_path = resolve_path(args.data_path)
    artifact_dir = resolve_path(args.artifact_dir)

    _, loader = build_loader(str(data_path), args.batch_size, args.num_workers, pin_memory=(device.type == "cuda"))
    proposal_model = load_proposal_model(proposal_checkpoint_path, device)
    completion_model = load_completion_model(completion_checkpoint_path, device)
    rewrite_model, use_proposal_conditioning = load_rewrite_model(rewrite_checkpoint_path, device)
    vanilla_gate_model = load_fallback_gate_model(vanilla_gate_checkpoint_path, device)
    ranking_gate_model = load_fallback_gate_model(ranking_gate_checkpoint_path, device)

    vanilla_thresholds = compute_global_keep_score_thresholds(
        proposal_model=proposal_model,
        completion_model=completion_model,
        rewrite_model=rewrite_model,
        gate_model=vanilla_gate_model,
        loader=loader,
        device=device,
        node_threshold=args.node_threshold,
        edge_threshold=args.edge_threshold,
        use_proposal_conditioning=use_proposal_conditioning,
        keep_fractions=KEEP_FRACTIONS,
    )
    vanilla_results = evaluate_step18(
        proposal_model=proposal_model,
        completion_model=completion_model,
        rewrite_model=rewrite_model,
        gate_model=vanilla_gate_model,
        loader=loader,
        device=device,
        node_threshold=args.node_threshold,
        edge_threshold=args.edge_threshold,
        completion_threshold=args.completion_threshold,
        use_proposal_conditioning=use_proposal_conditioning,
        keep_fractions=KEEP_FRACTIONS,
        keep_score_thresholds=vanilla_thresholds,
    )
    ranking_thresholds = compute_global_keep_score_thresholds(
        proposal_model=proposal_model,
        completion_model=completion_model,
        rewrite_model=rewrite_model,
        gate_model=ranking_gate_model,
        loader=loader,
        device=device,
        node_threshold=args.node_threshold,
        edge_threshold=args.edge_threshold,
        use_proposal_conditioning=use_proposal_conditioning,
        keep_fractions=KEEP_FRACTIONS,
    )
    ranking_results = evaluate_step18(
        proposal_model=proposal_model,
        completion_model=completion_model,
        rewrite_model=rewrite_model,
        gate_model=ranking_gate_model,
        loader=loader,
        device=device,
        node_threshold=args.node_threshold,
        edge_threshold=args.edge_threshold,
        completion_threshold=args.completion_threshold,
        use_proposal_conditioning=use_proposal_conditioning,
        keep_fractions=KEEP_FRACTIONS,
        keep_score_thresholds=ranking_thresholds,
    )
    results = merge_frontiers(vanilla_results, ranking_results)

    run_name = args.run_name or slugify(f"{args.split_name}_{rewrite_checkpoint_path.parent.name}")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifact_dir / f"{run_name}.json"
    csv_path = artifact_dir / f"{run_name}.csv"
    payload = {
        "metadata": {
            "proposal_checkpoint_path": str(proposal_checkpoint_path),
            "completion_checkpoint_path": str(completion_checkpoint_path),
            "rewrite_checkpoint_path": str(rewrite_checkpoint_path),
            "vanilla_gate_checkpoint_path": str(vanilla_gate_checkpoint_path),
            "ranking_gate_checkpoint_path": str(ranking_gate_checkpoint_path),
            "data_path": str(data_path),
            "split_name": args.split_name,
            "node_threshold": args.node_threshold,
            "edge_threshold": args.edge_threshold,
            "completion_threshold": args.completion_threshold,
            "rescue_budget_fraction": 0.10,
            "keep_fraction_grid": KEEP_FRACTIONS,
            "vanilla_keep_score_thresholds": {str(k): v for k, v in vanilla_thresholds.items()},
            "ranking_keep_score_thresholds": {str(k): v for k, v in ranking_thresholds.items()},
            "chooser_objectives_compared": ["vanilla_bce", "pairwise_keep_ranking"],
            "rewrite_uses_proposal_conditioning": use_proposal_conditioning,
        },
        "results": results,
    }
    save_json(out_path, payload)
    write_summary_csv(csv_path, payload)
    compact = {
        "downstream_overall": {mode: metrics["overall"] for mode, metrics in results["downstream"].items()},
        "chooser_behavior_overall": {mode: metrics["overall"] for mode, metrics in results["chooser_behavior"].items()},
        "ranking_diagnostics": {key: value["overall"] for key, value in results["ranking_diagnostics"].items()},
        "deltas_vs_step9c": results["deltas_vs_step9c"],
    }
    print(f"device: {device}")
    print(f"saved json: {out_path}")
    print(f"saved csv: {csv_path}")
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
