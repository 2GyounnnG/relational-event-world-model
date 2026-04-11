from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train.eval_edge_scope_miss_anatomy import make_group_names
from train.eval_noisy_structured_observation import (
    build_loader,
    get_device,
    move_batch_to_device,
    require_keys,
    resolve_path,
    save_json,
)
from train.eval_scope_edit_localization_gap import safe_div, slugify
from train.eval_step10_rescued_scope_rewrite_decomp import (
    bool_pred_adj,
    finalize_decomp_bucket,
    init_decomp_bucket,
    update_decomp_sample,
)
from train.eval_step12_guard_ranking_interaction import FIXED_RESCUE_BUDGET_FRACTION, build_ranked_outputs
from train.eval_step16_rescue_aware_rewrite_probe import (
    apply_oracle_false_scope_fallback,
    clone_rewrite_with_edge_logits,
)
from train.eval_step17_rescue_fallback_gate import apply_choose_step_gate
from train.eval_step9_gated_edge_completion import (
    EDGE_COMPLETION_OFF,
    apply_edge_completion_mode,
    build_downstream_sample_stats,
    build_proposal_sample_stats,
    finalize_downstream_bucket,
    finalize_proposal_bucket,
    get_base_proposal_outputs,
    init_downstream_bucket,
    init_proposal_bucket,
    load_completion_model,
    load_proposal_model,
    load_rewrite_model,
    update_bucket,
)
from train.eval_step9_rescue_frontier import average_precision, auroc
from train.train_step17_rescue_fallback_gate import (
    build_gate_feature_tensor,
    build_gate_targets,
    load_fallback_gate_model,
)


BASE_MODE = "base"
STEP9C_MODE = "step9c_completion_only"
THRESHOLD_GATE_MODE = "step17_thresholded_gate"
ORACLE_FALSE_SCOPE_MODE = "step9c_oracle_false_scope_fallback"
ORACLE_CHOOSE_MODE = "oracle_choose_better_of_two_paths"
KEEP_FRACTIONS = [0.02, 0.05, 0.10, 0.20]
TOPK_MODES = [f"step18_topk_keep_{str(frac).replace('.', 'p')}" for frac in KEEP_FRACTIONS]
MODE_ORDER = [
    BASE_MODE,
    STEP9C_MODE,
    THRESHOLD_GATE_MODE,
    *TOPK_MODES,
    ORACLE_FALSE_SCOPE_MODE,
    ORACLE_CHOOSE_MODE,
]
DECOMP_MODES = [STEP9C_MODE, THRESHOLD_GATE_MODE, *TOPK_MODES, ORACLE_FALSE_SCOPE_MODE, ORACLE_CHOOSE_MODE]
CHOOSER_MODES = [THRESHOLD_GATE_MODE, *TOPK_MODES, ORACLE_CHOOSE_MODE]


def format_keep_fraction(frac: float) -> str:
    return str(frac).replace(".", "p")


def mode_for_keep_fraction(frac: float) -> str:
    return f"step18_topk_keep_{format_keep_fraction(frac)}"


def build_topk_keep_mask(
    gate_scores: torch.Tensor,
    rescued_edges: torch.Tensor,
    valid_edge_mask: torch.Tensor,
    keep_fraction: float,
) -> torch.Tensor:
    rescued = rescued_edges.bool() & valid_edge_mask.bool()
    selected = torch.zeros_like(rescued)
    for batch_idx in range(rescued.shape[0]):
        candidate_indices = rescued[batch_idx].nonzero(as_tuple=False)
        num_candidates = candidate_indices.shape[0]
        if num_candidates <= 0:
            continue
        keep_budget = int(num_candidates * keep_fraction)
        if keep_budget <= 0:
            continue
        scores = gate_scores[batch_idx][rescued[batch_idx]]
        topk = torch.topk(scores, k=min(keep_budget, num_candidates), largest=True).indices
        chosen = candidate_indices[topk]
        selected[batch_idx, chosen[:, 0], chosen[:, 1]] = True
    return selected & valid_edge_mask.bool()


@torch.no_grad()
def compute_global_keep_score_thresholds(
    proposal_model: torch.nn.Module,
    completion_model: torch.nn.Module,
    rewrite_model: torch.nn.Module,
    gate_model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    node_threshold: float,
    edge_threshold: float,
    use_proposal_conditioning: bool,
    keep_fractions: Iterable[float],
) -> Dict[float, float]:
    """Find dataset-level top-k score cutoffs over the fixed rescued-edge set.

    中文说明：这里的 keep_fraction 是全测试集 rescued edges 的比例，而不是
    每个 sample 内部取整；否则 2%/5% 在小局部图上会退化成 0 条边。
    """
    proposal_model.eval()
    completion_model.eval()
    rewrite_model.eval()
    gate_model.eval()
    score_chunks: list[torch.Tensor] = []
    for batch in loader:
        require_keys(batch, ["node_feats", "adj", "next_adj", "node_mask", "changed_edges", "event_scope_union_edges"])
        batch = move_batch_to_device(batch, device)
        base_outputs = get_base_proposal_outputs(
            proposal_model=proposal_model,
            batch=batch,
            node_threshold=node_threshold,
            edge_threshold=edge_threshold,
        )
        valid_edge_mask = base_outputs["valid_edge_mask"].bool()
        completion_logits = completion_model(
            node_latents=base_outputs["node_latents"],
            base_edge_logits=base_outputs["edge_scope_logits"],
        )
        completion_probs = torch.sigmoid(completion_logits) * valid_edge_mask.float()
        base_mode_outputs = apply_edge_completion_mode(
            base_outputs=base_outputs,
            edge_completion_mode=EDGE_COMPLETION_OFF,
            completion_model=None,
            completion_threshold=0.50,
        )
        step9c_outputs = build_ranked_outputs(
            base_outputs=base_outputs,
            mode_name=STEP9C_MODE,
            ranking_scores=completion_probs,
            completion_probs=completion_probs,
        )
        rewrite_base = rewrite_model(
            node_feats=base_mode_outputs["input_node_feats"],
            adj=base_mode_outputs["input_adj"],
            scope_node_mask=base_mode_outputs["pred_scope_nodes"].float(),
            scope_edge_mask=base_mode_outputs["final_pred_scope_edges"].float(),
            proposal_node_probs=base_mode_outputs["proposal_node_probs"] if use_proposal_conditioning else None,
            proposal_edge_probs=base_mode_outputs["final_proposal_edge_probs"] if use_proposal_conditioning else None,
        )
        rewrite_step9c = rewrite_model(
            node_feats=step9c_outputs["input_node_feats"],
            adj=step9c_outputs["input_adj"],
            scope_node_mask=step9c_outputs["pred_scope_nodes"].float(),
            scope_edge_mask=step9c_outputs["final_pred_scope_edges"].float(),
            proposal_node_probs=step9c_outputs["proposal_node_probs"] if use_proposal_conditioning else None,
            proposal_edge_probs=step9c_outputs["final_proposal_edge_probs"] if use_proposal_conditioning else None,
        )
        features = build_gate_feature_tensor(
            base_outputs=base_outputs,
            completion_logits=completion_logits,
            rewrite_base=rewrite_base,
            rewrite_step9c=rewrite_step9c,
        )
        gate_scores = torch.sigmoid(gate_model(features)) * valid_edge_mask.float()
        rescued = step9c_outputs["rescued_edges"].bool() & valid_edge_mask
        if rescued.float().sum().item() > 0:
            score_chunks.append(gate_scores[rescued].detach().cpu())
    if not score_chunks:
        return {frac: float("inf") for frac in keep_fractions}
    scores = torch.cat(score_chunks).float()
    thresholds: Dict[float, float] = {}
    num_scores = scores.numel()
    for frac in keep_fractions:
        keep_count = int(num_scores * frac)
        if keep_count <= 0:
            thresholds[frac] = float("inf")
            continue
        sorted_scores = torch.sort(scores, descending=True).values
        thresholds[frac] = sorted_scores[min(keep_count, num_scores) - 1].item()
    return thresholds


def init_chooser_bucket() -> Dict[str, float]:
    return {
        "rescued_total": 0.0,
        "kept_total": 0.0,
        "fallback_total": 0.0,
        "target_positive_total": 0.0,
        "kept_target_positive_total": 0.0,
        "kept_changed_total": 0.0,
        "kept_false_scope_total": 0.0,
        "kept_event_scope_total": 0.0,
        "chosen_correct_total": 0.0,
    }


def update_chooser_bucket(
    bucket: Dict[str, float],
    batch: Dict[str, Any],
    valid_edge_mask: torch.Tensor,
    rescued_edges: torch.Tensor,
    choose_step_mask: torch.Tensor,
    target_choose_step: torch.Tensor,
    base_correct: torch.Tensor,
    step_correct: torch.Tensor,
    sample_idx: int,
) -> None:
    valid = valid_edge_mask[sample_idx].bool()
    rescued = rescued_edges[sample_idx].bool() & valid
    kept = choose_step_mask[sample_idx].bool() & rescued
    fallback = rescued & (~kept)
    target_positive = (target_choose_step[sample_idx] > 0.5) & rescued
    changed = (batch["changed_edges"][sample_idx] > 0.5) & valid
    event_scope = (batch["event_scope_union_edges"][sample_idx] > 0.5) & valid
    false_scope = (~event_scope) & valid
    chosen_correct = torch.where(kept, step_correct[sample_idx], base_correct[sample_idx])

    bucket["rescued_total"] += rescued.float().sum().item()
    bucket["kept_total"] += kept.float().sum().item()
    bucket["fallback_total"] += fallback.float().sum().item()
    bucket["target_positive_total"] += target_positive.float().sum().item()
    bucket["kept_target_positive_total"] += (kept & target_positive).float().sum().item()
    bucket["kept_changed_total"] += (kept & changed).float().sum().item()
    bucket["kept_false_scope_total"] += (kept & false_scope).float().sum().item()
    bucket["kept_event_scope_total"] += (kept & event_scope).float().sum().item()
    bucket["chosen_correct_total"] += (chosen_correct & rescued).float().sum().item()


def finalize_chooser_bucket(bucket: Dict[str, float]) -> Dict[str, Any]:
    rescued = bucket["rescued_total"]
    kept = bucket["kept_total"]
    target_pos = bucket["target_positive_total"]
    return {
        "rescued_total": int(rescued),
        "kept_total": int(kept),
        "fallback_total": int(bucket["fallback_total"]),
        "actual_keep_fraction": safe_div(kept, rescued),
        "fallback_fraction": safe_div(bucket["fallback_total"], rescued),
        "target_positive_fraction": safe_div(target_pos, rescued),
        "chooser_target_precision_among_kept": safe_div(bucket["kept_target_positive_total"], kept),
        "chooser_target_recall": safe_div(bucket["kept_target_positive_total"], target_pos),
        "gt_changed_precision_among_kept": safe_div(bucket["kept_changed_total"], kept),
        "gt_false_scope_fraction_among_kept": safe_div(bucket["kept_false_scope_total"], kept),
        "gt_event_scope_precision_among_kept": safe_div(bucket["kept_event_scope_total"], kept),
        "chosen_correct_rate_on_rescued": safe_div(bucket["chosen_correct_total"], rescued),
    }


def init_rank_bucket() -> Dict[str, list[torch.Tensor]]:
    return {
        "scores": [],
        "target_labels": [],
    }


def update_rank_bucket(
    bucket: Dict[str, list[torch.Tensor]],
    gate_scores: torch.Tensor,
    target_choose_step: torch.Tensor,
    rescued_edges: torch.Tensor,
    valid_edge_mask: torch.Tensor,
    sample_idx: int,
) -> None:
    mask = rescued_edges[sample_idx].bool() & valid_edge_mask[sample_idx].bool()
    if mask.float().sum().item() <= 0:
        return
    bucket["scores"].append(gate_scores[sample_idx][mask].detach().cpu())
    bucket["target_labels"].append((target_choose_step[sample_idx][mask] > 0.5).detach().cpu())


def finalize_rank_bucket(bucket: Dict[str, list[torch.Tensor]]) -> Dict[str, Any]:
    if not bucket["scores"]:
        return {
            "rescued_candidate_count": 0,
            "chooser_target_positive_count": 0,
            "chooser_target_ap": None,
            "chooser_target_auroc": None,
        }
    scores = torch.cat(bucket["scores"]).float()
    labels = torch.cat(bucket["target_labels"]).bool()
    return {
        "rescued_candidate_count": int(scores.numel()),
        "chooser_target_positive_count": int(labels.float().sum().item()),
        "chooser_target_positive_fraction": safe_div(labels.float().sum().item(), scores.numel()),
        "chooser_target_ap": average_precision(scores, labels),
        "chooser_target_auroc": auroc(scores, labels),
    }


def format_grouped_results(finalized: Dict[str, Any], empty_overall: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "overall": finalized.get("overall", empty_overall),
        "by_event_type": {
            name.split("event_type::", 1)[1]: value
            for name, value in finalized.items()
            if name.startswith("event_type::")
        },
        "by_corruption_setting": {
            name.split("corruption::", 1)[1]: value
            for name, value in finalized.items()
            if name.startswith("corruption::")
        },
    }


def compute_deltas_vs_step9c(downstream_results: Dict[str, Any]) -> Dict[str, Any]:
    step9c = downstream_results[STEP9C_MODE]["overall"]
    return {
        mode: {
            key: None if metrics["overall"].get(key) is None else metrics["overall"][key] - step9c[key]
            for key in ["full_edge", "context_edge", "changed_edge", "add", "delete"]
        }
        for mode, metrics in downstream_results.items()
    }


@torch.no_grad()
def evaluate_step18(
    proposal_model: torch.nn.Module,
    completion_model: torch.nn.Module,
    rewrite_model: torch.nn.Module,
    gate_model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    node_threshold: float,
    edge_threshold: float,
    completion_threshold: float,
    use_proposal_conditioning: bool,
    keep_fractions: Iterable[float],
    keep_score_thresholds: Dict[float, float],
) -> Dict[str, Any]:
    proposal_model.eval()
    completion_model.eval()
    rewrite_model.eval()
    gate_model.eval()

    proposal_buckets = {mode: defaultdict(init_proposal_bucket) for mode in MODE_ORDER}
    downstream_buckets = {mode: defaultdict(init_downstream_bucket) for mode in MODE_ORDER}
    decomp_buckets = {mode: defaultdict(init_decomp_bucket) for mode in DECOMP_MODES}
    chooser_buckets = {mode: defaultdict(init_chooser_bucket) for mode in CHOOSER_MODES}
    rank_buckets = defaultdict(init_rank_bucket)

    for batch in loader:
        require_keys(
            batch,
            [
                "node_feats",
                "adj",
                "next_adj",
                "node_mask",
                "changed_edges",
                "event_scope_union_edges",
                "events",
            ],
        )
        batch = move_batch_to_device(batch, device)
        base_outputs = get_base_proposal_outputs(
            proposal_model=proposal_model,
            batch=batch,
            node_threshold=node_threshold,
            edge_threshold=edge_threshold,
        )
        valid_edge_mask = base_outputs["valid_edge_mask"].bool()
        completion_logits = completion_model(
            node_latents=base_outputs["node_latents"],
            base_edge_logits=base_outputs["edge_scope_logits"],
        )
        completion_probs = torch.sigmoid(completion_logits) * valid_edge_mask.float()
        mode_outputs: Dict[str, Dict[str, torch.Tensor]] = {
            BASE_MODE: apply_edge_completion_mode(
                base_outputs=base_outputs,
                edge_completion_mode=EDGE_COMPLETION_OFF,
                completion_model=None,
                completion_threshold=completion_threshold,
            ),
            STEP9C_MODE: build_ranked_outputs(
                base_outputs=base_outputs,
                mode_name=STEP9C_MODE,
                ranking_scores=completion_probs,
                completion_probs=completion_probs,
            ),
        }
        for mode in [THRESHOLD_GATE_MODE, *TOPK_MODES, ORACLE_FALSE_SCOPE_MODE, ORACLE_CHOOSE_MODE]:
            mode_outputs[mode] = {**mode_outputs[STEP9C_MODE], "edge_completion_mode": mode}

        rewrite_outputs_by_mode: Dict[str, Dict[str, torch.Tensor]] = {}
        for mode in [BASE_MODE, STEP9C_MODE]:
            outputs = mode_outputs[mode]
            rewrite_outputs_by_mode[mode] = rewrite_model(
                node_feats=outputs["input_node_feats"],
                adj=outputs["input_adj"],
                scope_node_mask=outputs["pred_scope_nodes"].float(),
                scope_edge_mask=outputs["final_pred_scope_edges"].float(),
                proposal_node_probs=outputs["proposal_node_probs"] if use_proposal_conditioning else None,
                proposal_edge_probs=outputs["final_proposal_edge_probs"] if use_proposal_conditioning else None,
            )

        features = build_gate_feature_tensor(
            base_outputs=base_outputs,
            completion_logits=completion_logits,
            rewrite_base=rewrite_outputs_by_mode[BASE_MODE],
            rewrite_step9c=rewrite_outputs_by_mode[STEP9C_MODE],
        )
        gate_scores = torch.sigmoid(gate_model(features)) * valid_edge_mask.float()
        rescued_edges = mode_outputs[STEP9C_MODE]["rescued_edges"].bool()
        threshold_choose = (gate_scores >= 0.5) & rescued_edges
        target_choose_step, rescued, base_correct, step_correct = build_gate_targets(
            batch=batch,
            valid_edge_mask=valid_edge_mask,
            rescued_edges=rescued_edges,
            rewrite_base=rewrite_outputs_by_mode[BASE_MODE],
            rewrite_step9c=rewrite_outputs_by_mode[STEP9C_MODE],
        )
        oracle_choose = (target_choose_step > 0.5) & rescued

        choose_masks: Dict[str, torch.Tensor] = {
            THRESHOLD_GATE_MODE: threshold_choose,
            ORACLE_CHOOSE_MODE: oracle_choose,
        }
        for frac in keep_fractions:
            threshold = keep_score_thresholds[frac]
            choose_masks[mode_for_keep_fraction(frac)] = (gate_scores >= threshold) & rescued_edges & valid_edge_mask

        fallback_logits = apply_oracle_false_scope_fallback(
            batch=batch,
            valid_edge_mask=valid_edge_mask,
            rescued_edges=rescued_edges,
            base_edge_logits_full=rewrite_outputs_by_mode[BASE_MODE]["edge_logits_full"],
            budget_edge_logits_full=rewrite_outputs_by_mode[STEP9C_MODE]["edge_logits_full"],
        )
        rewrite_outputs_by_mode[ORACLE_FALSE_SCOPE_MODE] = clone_rewrite_with_edge_logits(
            rewrite_outputs_by_mode[STEP9C_MODE],
            fallback_logits,
        )
        for mode, choose_mask in choose_masks.items():
            logits = apply_choose_step_gate(
                rescued_edges=rescued_edges,
                choose_step_mask=choose_mask,
                base_edge_logits_full=rewrite_outputs_by_mode[BASE_MODE]["edge_logits_full"],
                step_edge_logits_full=rewrite_outputs_by_mode[STEP9C_MODE]["edge_logits_full"],
            )
            rewrite_outputs_by_mode[mode] = clone_rewrite_with_edge_logits(
                rewrite_outputs_by_mode[STEP9C_MODE],
                logits,
            )

        pred_adj_base = bool_pred_adj(rewrite_outputs_by_mode[BASE_MODE]["edge_logits_full"], valid_edge_mask)
        pred_adj_by_mode = {
            mode: bool_pred_adj(rewrite_outputs_by_mode[mode]["edge_logits_full"], valid_edge_mask)
            for mode in DECOMP_MODES
        }

        for sample_idx in range(batch["node_feats"].shape[0]):
            group_names = make_group_names(batch, sample_idx)
            groups = ["overall"] + [name for name in group_names if name.startswith("event_type::")]
            if "step6a_corruption_setting" in batch:
                groups.append(f"corruption::{batch['step6a_corruption_setting'][sample_idx]}")

            for group in groups:
                update_rank_bucket(
                    bucket=rank_buckets[group],
                    gate_scores=gate_scores,
                    target_choose_step=target_choose_step,
                    rescued_edges=rescued_edges,
                    valid_edge_mask=valid_edge_mask,
                    sample_idx=sample_idx,
                )

            for mode in MODE_ORDER:
                outputs = mode_outputs[mode]
                prop_stats = build_proposal_sample_stats(
                    batch=batch,
                    final_pred_edges=outputs["final_pred_scope_edges"],
                    rescued_edges=outputs["rescued_edges"],
                    valid_edge_mask=valid_edge_mask,
                    sample_idx=sample_idx,
                )
                down_stats = build_downstream_sample_stats(
                    batch=batch,
                    rewrite_outputs=rewrite_outputs_by_mode[mode],
                    valid_edge_mask=valid_edge_mask,
                    sample_idx=sample_idx,
                )
                for group in groups:
                    update_bucket(proposal_buckets[mode][group], prop_stats)
                    update_bucket(downstream_buckets[mode][group], down_stats)

            for mode in DECOMP_MODES:
                for group in groups:
                    update_decomp_sample(
                        bucket=decomp_buckets[mode][group],
                        batch=batch,
                        base_pred_edges=base_outputs["pred_scope_edges"],
                        rescued_edges=mode_outputs[mode]["rescued_edges"],
                        pred_adj_base=pred_adj_base,
                        pred_adj_budget=pred_adj_by_mode[mode],
                        valid_edge_mask=valid_edge_mask,
                        sample_idx=sample_idx,
                    )

            for mode, choose_mask in choose_masks.items():
                for group in groups:
                    update_chooser_bucket(
                        bucket=chooser_buckets[mode][group],
                        batch=batch,
                        valid_edge_mask=valid_edge_mask,
                        rescued_edges=rescued_edges,
                        choose_step_mask=choose_mask,
                        target_choose_step=target_choose_step,
                        base_correct=base_correct,
                        step_correct=step_correct,
                        sample_idx=sample_idx,
                    )

    proposal_results: Dict[str, Any] = {}
    downstream_results: Dict[str, Any] = {}
    decomp_results: Dict[str, Any] = {}
    chooser_results: Dict[str, Any] = {}
    for mode in MODE_ORDER:
        finalized_prop = {name: finalize_proposal_bucket(bucket) for name, bucket in proposal_buckets[mode].items()}
        finalized_down = {name: finalize_downstream_bucket(bucket) for name, bucket in downstream_buckets[mode].items()}
        proposal_results[mode] = format_grouped_results(finalized_prop, finalize_proposal_bucket(init_proposal_bucket()))
        downstream_results[mode] = format_grouped_results(finalized_down, finalize_downstream_bucket(init_downstream_bucket()))
    for mode in DECOMP_MODES:
        finalized_decomp = {name: finalize_decomp_bucket(bucket) for name, bucket in decomp_buckets[mode].items()}
        decomp_results[mode] = format_grouped_results(finalized_decomp, finalize_decomp_bucket(init_decomp_bucket()))
    for mode in CHOOSER_MODES:
        finalized_chooser = {name: finalize_chooser_bucket(bucket) for name, bucket in chooser_buckets[mode].items()}
        chooser_results[mode] = format_grouped_results(finalized_chooser, finalize_chooser_bucket(init_chooser_bucket()))
    finalized_rank = {name: finalize_rank_bucket(bucket) for name, bucket in rank_buckets.items()}
    rank_results = format_grouped_results(
        finalized_rank,
        finalize_rank_bucket(init_rank_bucket()),
    )

    return {
        "proposal_side": proposal_results,
        "downstream": downstream_results,
        "decomposition": decomp_results,
        "chooser_behavior": chooser_results,
        "ranking_diagnostics": rank_results,
        "deltas_vs_step9c": compute_deltas_vs_step9c(downstream_results),
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
                "group": "overall",
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
                "false_scope_preserve_rate": decomp.get("preserve_rate_rescued_false_scope_budget"),
                "true_changed_correct_rate": decomp.get("correct_edit_rate_rescued_true_changed_budget"),
                "actual_keep_fraction": chooser.get("actual_keep_fraction"),
                "chooser_target_precision_among_kept": chooser.get("chooser_target_precision_among_kept"),
                "chooser_target_recall": chooser.get("chooser_target_recall"),
                "gt_changed_precision_among_kept": chooser.get("gt_changed_precision_among_kept"),
                "gt_false_scope_fraction_among_kept": chooser.get("gt_false_scope_fraction_among_kept"),
                "chosen_correct_rate_on_rescued": chooser.get("chosen_correct_rate_on_rescued"),
            }
        )
    rank = results.get("ranking_diagnostics", {}).get("overall", {})
    rows.append(
        {
            "mode": "ranking_diagnostics",
            "group": "overall",
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
            "false_scope_preserve_rate": None,
            "true_changed_correct_rate": None,
            "actual_keep_fraction": None,
            "chooser_target_precision_among_kept": rank.get("chooser_target_ap"),
            "chooser_target_recall": rank.get("chooser_target_auroc"),
            "gt_changed_precision_among_kept": None,
            "gt_false_scope_fraction_among_kept": None,
            "chosen_correct_rate_on_rescued": None,
        }
    )
    fieldnames = list(rows[0].keys()) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal_checkpoint_path", type=str, required=True)
    parser.add_argument("--completion_checkpoint_path", type=str, required=True)
    parser.add_argument("--rewrite_checkpoint_path", type=str, required=True)
    parser.add_argument("--fallback_gate_checkpoint_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--split_name", type=str, default="test")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--artifact_dir", type=str, default="artifacts/step18_fallback_gate_frontier")
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
    fallback_gate_checkpoint_path = resolve_path(args.fallback_gate_checkpoint_path)
    data_path = resolve_path(args.data_path)
    artifact_dir = resolve_path(args.artifact_dir)

    _, loader = build_loader(str(data_path), args.batch_size, args.num_workers, pin_memory=(device.type == "cuda"))
    proposal_model = load_proposal_model(proposal_checkpoint_path, device)
    completion_model = load_completion_model(completion_checkpoint_path, device)
    rewrite_model, use_proposal_conditioning = load_rewrite_model(rewrite_checkpoint_path, device)
    gate_model = load_fallback_gate_model(fallback_gate_checkpoint_path, device)
    keep_score_thresholds = compute_global_keep_score_thresholds(
        proposal_model=proposal_model,
        completion_model=completion_model,
        rewrite_model=rewrite_model,
        gate_model=gate_model,
        loader=loader,
        device=device,
        node_threshold=args.node_threshold,
        edge_threshold=args.edge_threshold,
        use_proposal_conditioning=use_proposal_conditioning,
        keep_fractions=KEEP_FRACTIONS,
    )

    results = evaluate_step18(
        proposal_model=proposal_model,
        completion_model=completion_model,
        rewrite_model=rewrite_model,
        gate_model=gate_model,
        loader=loader,
        device=device,
        node_threshold=args.node_threshold,
        edge_threshold=args.edge_threshold,
        completion_threshold=args.completion_threshold,
        use_proposal_conditioning=use_proposal_conditioning,
        keep_fractions=KEEP_FRACTIONS,
        keep_score_thresholds=keep_score_thresholds,
    )

    run_name = args.run_name or slugify(f"{args.split_name}_{rewrite_checkpoint_path.parent.name}")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifact_dir / f"{run_name}.json"
    csv_path = artifact_dir / f"{run_name}.csv"
    payload = {
        "metadata": {
            "proposal_checkpoint_path": str(proposal_checkpoint_path),
            "completion_checkpoint_path": str(completion_checkpoint_path),
            "rewrite_checkpoint_path": str(rewrite_checkpoint_path),
            "fallback_gate_checkpoint_path": str(fallback_gate_checkpoint_path),
            "data_path": str(data_path),
            "split_name": args.split_name,
            "node_threshold": args.node_threshold,
            "edge_threshold": args.edge_threshold,
            "completion_threshold": args.completion_threshold,
            "rescue_budget_fraction": FIXED_RESCUE_BUDGET_FRACTION,
            "keep_fraction_grid": KEEP_FRACTIONS,
            "keep_score_thresholds": {str(key): value for key, value in keep_score_thresholds.items()},
            "chooser_operating_point": "rank_rescued_edges_by_step17_keep_score_then_keep_top_fraction",
            "tie_rule_in_training": "fallback_to_base_when_base_and_step9c_have_equal_correctness",
            "rewrite_uses_proposal_conditioning": use_proposal_conditioning,
        },
        "results": results,
    }
    save_json(out_path, payload)
    write_summary_csv(csv_path, payload)
    compact = {
        "downstream_overall": {mode: metrics["overall"] for mode, metrics in results["downstream"].items()},
        "chooser_behavior_overall": {mode: metrics["overall"] for mode, metrics in results["chooser_behavior"].items()},
        "decomposition_overall": {mode: metrics["overall"] for mode, metrics in results["decomposition"].items()},
        "ranking_diagnostics_overall": results["ranking_diagnostics"]["overall"],
        "deltas_vs_step9c": results["deltas_vs_step9c"],
    }
    print(f"device: {device}")
    print(f"saved json: {out_path}")
    print(f"saved csv: {csv_path}")
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
