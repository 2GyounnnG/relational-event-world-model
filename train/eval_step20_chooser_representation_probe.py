from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train.eval_edge_scope_miss_anatomy import make_group_names
from train.eval_noisy_structured_observation import build_loader, get_device, move_batch_to_device, require_keys, resolve_path, save_json
from train.eval_scope_edit_localization_gap import slugify
from train.eval_step10_rescued_scope_rewrite_decomp import bool_pred_adj, finalize_decomp_bucket, init_decomp_bucket, update_decomp_sample
from train.eval_step12_guard_ranking_interaction import FIXED_RESCUE_BUDGET_FRACTION, build_ranked_outputs
from train.eval_step16_rescue_aware_rewrite_probe import clone_rewrite_with_edge_logits
from train.eval_step17_rescue_fallback_gate import apply_choose_step_gate
from train.eval_step18_fallback_gate_frontier import (
    KEEP_FRACTIONS,
    STEP9C_MODE,
    compute_deltas_vs_step9c,
    finalize_chooser_bucket,
    finalize_rank_bucket,
    format_grouped_results,
    format_keep_fraction,
    init_chooser_bucket,
    init_rank_bucket,
    update_chooser_bucket,
    update_rank_bucket,
)
from train.eval_step9_gated_edge_completion import (
    EDGE_COMPLETION_OFF,
    apply_edge_completion_mode,
    build_downstream_sample_stats,
    finalize_downstream_bucket,
    get_base_proposal_outputs,
    init_downstream_bucket,
    load_completion_model,
    load_proposal_model,
    load_rewrite_model,
    update_bucket,
)
from train.train_step17_rescue_fallback_gate import build_gate_targets
from train.train_step20_chooser_representation_probe import (
    COMPACT_INTERFACE_SCORES,
    ENRICHED_RESCUE_LOCAL_CONTEXT,
    build_step20_feature_tensor,
    load_step20_probe_model,
)


COMPACT_PREFIX = "step20_compact_probe"
ENRICHED_PREFIX = "step20_enriched_probe"
FEATURE_LABELS = {
    COMPACT_PREFIX: COMPACT_INTERFACE_SCORES,
    ENRICHED_PREFIX: ENRICHED_RESCUE_LOCAL_CONTEXT,
}


def mode_name(prefix: str, frac: float) -> str:
    return f"{prefix}_topk_keep_{format_keep_fraction(frac)}"


def compute_deltas(downstream_results: Dict[str, Any]) -> Dict[str, Any]:
    step9c = downstream_results[STEP9C_MODE]["overall"]
    return {
        mode: {
            key: None if metrics["overall"].get(key) is None else metrics["overall"][key] - step9c[key]
            for key in ["full_edge", "context_edge", "changed_edge", "add", "delete"]
        }
        for mode, metrics in downstream_results.items()
    }


@torch.no_grad()
def compute_global_thresholds(
    proposal_model: torch.nn.Module,
    completion_model: torch.nn.Module,
    rewrite_model: torch.nn.Module,
    compact_probe: torch.nn.Module,
    enriched_probe: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    node_threshold: float,
    edge_threshold: float,
    use_proposal_conditioning: bool,
    keep_fractions: Iterable[float],
) -> Dict[str, Dict[float, float]]:
    score_chunks = {COMPACT_PREFIX: [], ENRICHED_PREFIX: []}
    for batch in loader:
        require_keys(batch, ["node_feats", "adj", "next_adj", "node_mask", "changed_edges", "event_scope_union_edges"])
        batch = move_batch_to_device(batch, device)
        base_outputs = get_base_proposal_outputs(proposal_model, batch, node_threshold, edge_threshold)
        valid_edge_mask = base_outputs["valid_edge_mask"].bool()
        completion_logits = completion_model(base_outputs["node_latents"], base_outputs["edge_scope_logits"])
        completion_probs = torch.sigmoid(completion_logits) * valid_edge_mask.float()
        mode_base = apply_edge_completion_mode(base_outputs, EDGE_COMPLETION_OFF, None, 0.5)
        mode_step9c = build_ranked_outputs(base_outputs, STEP9C_MODE, completion_probs, completion_probs)
        rewrite_base = rewrite_model(
            node_feats=mode_base["input_node_feats"],
            adj=mode_base["input_adj"],
            scope_node_mask=mode_base["pred_scope_nodes"].float(),
            scope_edge_mask=mode_base["final_pred_scope_edges"].float(),
            proposal_node_probs=mode_base["proposal_node_probs"] if use_proposal_conditioning else None,
            proposal_edge_probs=mode_base["final_proposal_edge_probs"] if use_proposal_conditioning else None,
        )
        rewrite_step9c = rewrite_model(
            node_feats=mode_step9c["input_node_feats"],
            adj=mode_step9c["input_adj"],
            scope_node_mask=mode_step9c["pred_scope_nodes"].float(),
            scope_edge_mask=mode_step9c["final_pred_scope_edges"].float(),
            proposal_node_probs=mode_step9c["proposal_node_probs"] if use_proposal_conditioning else None,
            proposal_edge_probs=mode_step9c["final_proposal_edge_probs"] if use_proposal_conditioning else None,
        )
        rescued = mode_step9c["rescued_edges"].bool() & valid_edge_mask
        if rescued.float().sum().item() <= 0:
            continue
        compact_features = build_step20_feature_tensor(batch, base_outputs, completion_logits, rewrite_base, rewrite_step9c, COMPACT_INTERFACE_SCORES)
        enriched_features = build_step20_feature_tensor(batch, base_outputs, completion_logits, rewrite_base, rewrite_step9c, ENRICHED_RESCUE_LOCAL_CONTEXT)
        compact_scores = torch.sigmoid(compact_probe(compact_features)) * valid_edge_mask.float()
        enriched_scores = torch.sigmoid(enriched_probe(enriched_features)) * valid_edge_mask.float()
        score_chunks[COMPACT_PREFIX].append(compact_scores[rescued].detach().cpu())
        score_chunks[ENRICHED_PREFIX].append(enriched_scores[rescued].detach().cpu())

    thresholds: Dict[str, Dict[float, float]] = {}
    for prefix, chunks in score_chunks.items():
        if not chunks:
            thresholds[prefix] = {frac: float("inf") for frac in keep_fractions}
            continue
        scores = torch.cat(chunks).float()
        sorted_scores = torch.sort(scores, descending=True).values
        thresholds[prefix] = {}
        for frac in keep_fractions:
            keep_count = int(scores.numel() * frac)
            thresholds[prefix][frac] = float("inf") if keep_count <= 0 else sorted_scores[min(keep_count, scores.numel()) - 1].item()
    return thresholds


@torch.no_grad()
def evaluate_step20(
    proposal_model: torch.nn.Module,
    completion_model: torch.nn.Module,
    rewrite_model: torch.nn.Module,
    compact_probe: torch.nn.Module,
    enriched_probe: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    node_threshold: float,
    edge_threshold: float,
    use_proposal_conditioning: bool,
    thresholds: Dict[str, Dict[float, float]],
) -> Dict[str, Any]:
    mode_prefixes = [COMPACT_PREFIX, ENRICHED_PREFIX]
    all_modes = [mode_name(prefix, frac) for prefix in mode_prefixes for frac in KEEP_FRACTIONS]
    downstream_buckets = {mode: defaultdict(init_downstream_bucket) for mode in all_modes}
    decomp_buckets = {mode: defaultdict(init_decomp_bucket) for mode in all_modes}
    chooser_buckets = {mode: defaultdict(init_chooser_bucket) for mode in all_modes}
    rank_buckets = {COMPACT_PREFIX: defaultdict(init_rank_bucket), ENRICHED_PREFIX: defaultdict(init_rank_bucket)}

    for batch in loader:
        require_keys(batch, ["node_feats", "adj", "next_adj", "node_mask", "changed_edges", "event_scope_union_edges", "events"])
        batch = move_batch_to_device(batch, device)
        base_outputs = get_base_proposal_outputs(proposal_model, batch, node_threshold, edge_threshold)
        valid_edge_mask = base_outputs["valid_edge_mask"].bool()
        completion_logits = completion_model(base_outputs["node_latents"], base_outputs["edge_scope_logits"])
        completion_probs = torch.sigmoid(completion_logits) * valid_edge_mask.float()
        mode_base = apply_edge_completion_mode(base_outputs, EDGE_COMPLETION_OFF, None, 0.5)
        mode_step9c = build_ranked_outputs(base_outputs, STEP9C_MODE, completion_probs, completion_probs)
        rewrite_base = rewrite_model(
            node_feats=mode_base["input_node_feats"],
            adj=mode_base["input_adj"],
            scope_node_mask=mode_base["pred_scope_nodes"].float(),
            scope_edge_mask=mode_base["final_pred_scope_edges"].float(),
            proposal_node_probs=mode_base["proposal_node_probs"] if use_proposal_conditioning else None,
            proposal_edge_probs=mode_base["final_proposal_edge_probs"] if use_proposal_conditioning else None,
        )
        rewrite_step9c = rewrite_model(
            node_feats=mode_step9c["input_node_feats"],
            adj=mode_step9c["input_adj"],
            scope_node_mask=mode_step9c["pred_scope_nodes"].float(),
            scope_edge_mask=mode_step9c["final_pred_scope_edges"].float(),
            proposal_node_probs=mode_step9c["proposal_node_probs"] if use_proposal_conditioning else None,
            proposal_edge_probs=mode_step9c["final_proposal_edge_probs"] if use_proposal_conditioning else None,
        )
        target_choose_step, rescued, base_correct, step_correct = build_gate_targets(
            batch=batch,
            valid_edge_mask=valid_edge_mask,
            rescued_edges=mode_step9c["rescued_edges"],
            rewrite_base=rewrite_base,
            rewrite_step9c=rewrite_step9c,
        )
        compact_features = build_step20_feature_tensor(batch, base_outputs, completion_logits, rewrite_base, rewrite_step9c, COMPACT_INTERFACE_SCORES)
        enriched_features = build_step20_feature_tensor(batch, base_outputs, completion_logits, rewrite_base, rewrite_step9c, ENRICHED_RESCUE_LOCAL_CONTEXT)
        scores_by_prefix = {
            COMPACT_PREFIX: torch.sigmoid(compact_probe(compact_features)) * valid_edge_mask.float(),
            ENRICHED_PREFIX: torch.sigmoid(enriched_probe(enriched_features)) * valid_edge_mask.float(),
        }
        rewrite_by_mode: Dict[str, Dict[str, torch.Tensor]] = {}
        choose_by_mode: Dict[str, torch.Tensor] = {}
        for prefix, scores in scores_by_prefix.items():
            for frac in KEEP_FRACTIONS:
                mode = mode_name(prefix, frac)
                choose_mask = (scores >= thresholds[prefix][frac]) & rescued & valid_edge_mask
                choose_by_mode[mode] = choose_mask
                logits = apply_choose_step_gate(
                    rescued_edges=rescued,
                    choose_step_mask=choose_mask,
                    base_edge_logits_full=rewrite_base["edge_logits_full"],
                    step_edge_logits_full=rewrite_step9c["edge_logits_full"],
                )
                rewrite_by_mode[mode] = clone_rewrite_with_edge_logits(rewrite_step9c, logits)

        pred_adj_base = bool_pred_adj(rewrite_base["edge_logits_full"], valid_edge_mask)
        pred_adj_by_mode = {
            mode: bool_pred_adj(outputs["edge_logits_full"], valid_edge_mask)
            for mode, outputs in rewrite_by_mode.items()
        }

        for sample_idx in range(batch["node_feats"].shape[0]):
            group_names = make_group_names(batch, sample_idx)
            groups = ["overall"] + [name for name in group_names if name.startswith("event_type::")]
            if "step6a_corruption_setting" in batch:
                groups.append(f"corruption::{batch['step6a_corruption_setting'][sample_idx]}")

            for prefix, scores in scores_by_prefix.items():
                for group in groups:
                    update_rank_bucket(rank_buckets[prefix][group], scores, target_choose_step, rescued, valid_edge_mask, sample_idx)

            for mode, outputs in rewrite_by_mode.items():
                down_stats = build_downstream_sample_stats(batch, outputs, valid_edge_mask, sample_idx)
                for group in groups:
                    update_bucket(downstream_buckets[mode][group], down_stats)
                    update_decomp_sample(
                        bucket=decomp_buckets[mode][group],
                        batch=batch,
                        base_pred_edges=base_outputs["pred_scope_edges"],
                        rescued_edges=mode_step9c["rescued_edges"],
                        pred_adj_base=pred_adj_base,
                        pred_adj_budget=pred_adj_by_mode[mode],
                        valid_edge_mask=valid_edge_mask,
                        sample_idx=sample_idx,
                    )
                    update_chooser_bucket(
                        bucket=chooser_buckets[mode][group],
                        batch=batch,
                        valid_edge_mask=valid_edge_mask,
                        rescued_edges=rescued,
                        choose_step_mask=choose_by_mode[mode],
                        target_choose_step=target_choose_step,
                        base_correct=base_correct,
                        step_correct=step_correct,
                        sample_idx=sample_idx,
                    )

    downstream = {
        mode: format_grouped_results({k: finalize_downstream_bucket(v) for k, v in buckets.items()}, finalize_downstream_bucket(init_downstream_bucket()))
        for mode, buckets in downstream_buckets.items()
    }
    decomp = {
        mode: format_grouped_results({k: finalize_decomp_bucket(v) for k, v in buckets.items()}, finalize_decomp_bucket(init_decomp_bucket()))
        for mode, buckets in decomp_buckets.items()
    }
    chooser = {
        mode: format_grouped_results({k: finalize_chooser_bucket(v) for k, v in buckets.items()}, finalize_chooser_bucket(init_chooser_bucket()))
        for mode, buckets in chooser_buckets.items()
    }
    ranking = {
        prefix: format_grouped_results({k: finalize_rank_bucket(v) for k, v in buckets.items()}, finalize_rank_bucket(init_rank_bucket()))
        for prefix, buckets in rank_buckets.items()
    }
    return {"downstream": downstream, "decomposition": decomp, "chooser_behavior": chooser, "ranking_diagnostics": ranking}


def merge_with_reference(reference: Dict[str, Any], step20: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(reference["results"])
    for section in ["downstream", "decomposition", "chooser_behavior"]:
        merged.setdefault(section, {}).update(step20.get(section, {}))
    merged.setdefault("ranking_diagnostics", {}).update(step20["ranking_diagnostics"])
    merged["deltas_vs_step9c"] = compute_deltas(merged["downstream"])
    return merged


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
    parser.add_argument("--compact_probe_checkpoint_path", type=str, required=True)
    parser.add_argument("--enriched_probe_checkpoint_path", type=str, required=True)
    parser.add_argument("--reference_step19_json_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--split_name", type=str, default="test")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--artifact_dir", type=str, default="artifacts/step20_chooser_representation_probe")
    parser.add_argument("--node_threshold", type=float, default=0.20)
    parser.add_argument("--edge_threshold", type=float, default=0.15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = get_device(args.device)
    proposal_checkpoint_path = resolve_path(args.proposal_checkpoint_path)
    completion_checkpoint_path = resolve_path(args.completion_checkpoint_path)
    rewrite_checkpoint_path = resolve_path(args.rewrite_checkpoint_path)
    compact_probe_checkpoint_path = resolve_path(args.compact_probe_checkpoint_path)
    enriched_probe_checkpoint_path = resolve_path(args.enriched_probe_checkpoint_path)
    reference_step19_json_path = resolve_path(args.reference_step19_json_path)
    data_path = resolve_path(args.data_path)
    artifact_dir = resolve_path(args.artifact_dir)

    reference_payload = json.load(open(reference_step19_json_path, encoding="utf-8"))
    _, loader = build_loader(str(data_path), args.batch_size, args.num_workers, pin_memory=(device.type == "cuda"))
    proposal_model = load_proposal_model(proposal_checkpoint_path, device)
    completion_model = load_completion_model(completion_checkpoint_path, device)
    rewrite_model, use_proposal_conditioning = load_rewrite_model(rewrite_checkpoint_path, device)
    compact_probe = load_step20_probe_model(compact_probe_checkpoint_path, device)
    enriched_probe = load_step20_probe_model(enriched_probe_checkpoint_path, device)

    thresholds = compute_global_thresholds(
        proposal_model=proposal_model,
        completion_model=completion_model,
        rewrite_model=rewrite_model,
        compact_probe=compact_probe,
        enriched_probe=enriched_probe,
        loader=loader,
        device=device,
        node_threshold=args.node_threshold,
        edge_threshold=args.edge_threshold,
        use_proposal_conditioning=use_proposal_conditioning,
        keep_fractions=KEEP_FRACTIONS,
    )
    step20_results = evaluate_step20(
        proposal_model=proposal_model,
        completion_model=completion_model,
        rewrite_model=rewrite_model,
        compact_probe=compact_probe,
        enriched_probe=enriched_probe,
        loader=loader,
        device=device,
        node_threshold=args.node_threshold,
        edge_threshold=args.edge_threshold,
        use_proposal_conditioning=use_proposal_conditioning,
        thresholds=thresholds,
    )
    results = merge_with_reference(reference_payload, step20_results)

    run_name = args.run_name or slugify(f"{args.split_name}_{rewrite_checkpoint_path.parent.name}")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifact_dir / f"{run_name}.json"
    csv_path = artifact_dir / f"{run_name}.csv"
    payload = {
        "metadata": {
            "proposal_checkpoint_path": str(proposal_checkpoint_path),
            "completion_checkpoint_path": str(completion_checkpoint_path),
            "rewrite_checkpoint_path": str(rewrite_checkpoint_path),
            "compact_probe_checkpoint_path": str(compact_probe_checkpoint_path),
            "enriched_probe_checkpoint_path": str(enriched_probe_checkpoint_path),
            "reference_step19_json_path": str(reference_step19_json_path),
            "data_path": str(data_path),
            "split_name": args.split_name,
            "node_threshold": args.node_threshold,
            "edge_threshold": args.edge_threshold,
            "rescue_budget_fraction": FIXED_RESCUE_BUDGET_FRACTION,
            "keep_fraction_grid": KEEP_FRACTIONS,
            "feature_bundles": [COMPACT_INTERFACE_SCORES, ENRICHED_RESCUE_LOCAL_CONTEXT],
            "objective": "plain_bce",
            "rewrite_uses_proposal_conditioning": use_proposal_conditioning,
            "step20_keep_score_thresholds": {
                prefix: {str(frac): value for frac, value in frac_map.items()}
                for prefix, frac_map in thresholds.items()
            },
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
