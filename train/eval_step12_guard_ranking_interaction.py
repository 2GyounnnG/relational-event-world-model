from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

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
    finalize_decomp_bucket,
    init_decomp_bucket,
    update_decomp_sample,
)
from train.eval_step11_guarded_internal_completion import (
    EventScopeGuardHead,
    load_guard_model,
)
from train.eval_step9_gated_edge_completion import (
    EDGE_COMPLETION_NAIVE,
    EDGE_COMPLETION_OFF,
    GatedInternalEdgeCompletionHead,
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


FIXED_RESCUE_BUDGET_FRACTION = 0.10

BASE_MODE = "base"
COMPLETION_ONLY_MODE = "completion_only"
GUARD_ONLY_MODE = "guard_only"
PRODUCT_MODE = "product"
GUARD_THEN_COMPLETION_MODE = "guard_then_completion"
ORACLE_GUARD_THEN_COMPLETION_MODE = "oracle_guard_then_completion"
NAIVE_MODE = "naive_induced_closure_reference"

RANKING_MODES = [
    COMPLETION_ONLY_MODE,
    GUARD_ONLY_MODE,
    PRODUCT_MODE,
    GUARD_THEN_COMPLETION_MODE,
    ORACLE_GUARD_THEN_COMPLETION_MODE,
]
MODE_ORDER = [BASE_MODE] + RANKING_MODES + [NAIVE_MODE]


def init_top_tail_bucket() -> Dict[str, Any]:
    return {
        "candidate_total": 0.0,
        "budget_total": 0.0,
        "selected_total": 0.0,
        "oracle_positive_candidate_total": 0.0,
        "changed_positive_candidate_total": 0.0,
        "selected_oracle_positive_total": 0.0,
        "selected_changed_positive_total": 0.0,
        "scores": [],
        "oracle_labels": [],
        "changed_labels": [],
    }


def update_top_tail_bucket(
    bucket: Dict[str, Any],
    candidates_2d: torch.Tensor,
    selected_2d: torch.Tensor,
    ranking_scores_2d: torch.Tensor,
    oracle_scope_2d: torch.Tensor,
    changed_edges_2d: torch.Tensor,
) -> None:
    candidates = candidates_2d.bool()
    selected = selected_2d.bool() & candidates
    oracle = oracle_scope_2d.bool() & candidates
    changed = changed_edges_2d.bool() & candidates
    candidate_count = int(candidates.float().sum().item())
    budget = int(candidate_count * FIXED_RESCUE_BUDGET_FRACTION)
    if candidate_count <= 0:
        return
    bucket["candidate_total"] += candidate_count
    bucket["budget_total"] += budget
    bucket["selected_total"] += selected.float().sum().item()
    bucket["oracle_positive_candidate_total"] += oracle.float().sum().item()
    bucket["changed_positive_candidate_total"] += changed.float().sum().item()
    bucket["selected_oracle_positive_total"] += (selected & oracle).float().sum().item()
    bucket["selected_changed_positive_total"] += (selected & changed).float().sum().item()
    bucket["scores"].append(ranking_scores_2d[candidates].detach().float().cpu())
    bucket["oracle_labels"].append(oracle[candidates].detach().bool().cpu())
    bucket["changed_labels"].append(changed[candidates].detach().bool().cpu())


def finalize_top_tail_bucket(bucket: Dict[str, Any]) -> Dict[str, Any]:
    result = {
        "candidate_total": int(bucket["candidate_total"]),
        "budget_total": int(bucket["budget_total"]),
        "selected_total": int(bucket["selected_total"]),
        "event_scope_positive_candidate_total": int(bucket["oracle_positive_candidate_total"]),
        "changed_positive_candidate_total": int(bucket["changed_positive_candidate_total"]),
        "event_scope_precision_at_budget": safe_div(
            bucket["selected_oracle_positive_total"], bucket["selected_total"]
        ),
        "changed_edge_precision_at_budget": safe_div(
            bucket["selected_changed_positive_total"], bucket["selected_total"]
        ),
        "event_scope_recall_at_budget": safe_div(
            bucket["selected_oracle_positive_total"], bucket["oracle_positive_candidate_total"]
        ),
        "changed_edge_recall_at_budget": safe_div(
            bucket["selected_changed_positive_total"], bucket["changed_positive_candidate_total"]
        ),
    }
    if not bucket["scores"]:
        result.update(
            {
                "event_scope_ap": None,
                "event_scope_auroc": None,
                "changed_edge_ap": None,
                "changed_edge_auroc": None,
            }
        )
        return result
    scores = torch.cat(bucket["scores"]).float()
    oracle_labels = torch.cat(bucket["oracle_labels"]).bool()
    changed_labels = torch.cat(bucket["changed_labels"]).bool()
    result.update(
        {
            "event_scope_ap": average_precision(scores, oracle_labels),
            "event_scope_auroc": auroc(scores, oracle_labels),
            "changed_edge_ap": average_precision(scores, changed_labels),
            "changed_edge_auroc": auroc(scores, changed_labels),
        }
    )
    return result


def lexicographic_topk_indices(
    primary_scores: torch.Tensor,
    secondary_scores: torch.Tensor,
    k: int,
) -> torch.Tensor:
    if k <= 0 or primary_scores.numel() <= 0:
        return torch.empty(0, device=primary_scores.device, dtype=torch.long)
    secondary_order = torch.argsort(secondary_scores, descending=True, stable=True)
    primary_order = torch.argsort(primary_scores[secondary_order], descending=True, stable=True)
    return secondary_order[primary_order[: min(k, primary_scores.numel())]]


def build_ranked_outputs(
    base_outputs: Dict[str, torch.Tensor],
    mode_name: str,
    ranking_scores: torch.Tensor,
    completion_probs: torch.Tensor,
    primary_scores: Optional[torch.Tensor] = None,
    secondary_scores: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    candidates = base_outputs["rescue_candidates"].bool()
    selected = torch.zeros_like(candidates)
    batch_size = candidates.shape[0]
    for batch_idx in range(batch_size):
        candidate_indices = candidates[batch_idx].nonzero(as_tuple=False)
        num_candidates = candidate_indices.shape[0]
        if num_candidates <= 0:
            continue
        budget = int(num_candidates * FIXED_RESCUE_BUDGET_FRACTION)
        if budget <= 0:
            continue
        if primary_scores is not None and secondary_scores is not None:
            primary = primary_scores[batch_idx][candidates[batch_idx]]
            secondary = secondary_scores[batch_idx][candidates[batch_idx]]
            topk = lexicographic_topk_indices(primary, secondary, budget)
        else:
            scores = ranking_scores[batch_idx][candidates[batch_idx]]
            topk = torch.topk(scores, k=min(budget, num_candidates), largest=True).indices
        chosen = candidate_indices[topk]
        selected[batch_idx, chosen[:, 0], chosen[:, 1]] = True

    valid_edge_mask = base_outputs["valid_edge_mask"].bool()
    final_pred_edges = (base_outputs["pred_scope_edges"] | selected) & valid_edge_mask
    # The ranking mode controls admission only. Once admitted, the unchanged
    # Step 9 completion confidence is passed to rewrite for a clean comparison.
    final_edge_probs = torch.where(
        selected,
        torch.maximum(base_outputs["proposal_edge_probs"], completion_probs),
        base_outputs["proposal_edge_probs"],
    )
    return {
        **base_outputs,
        "edge_completion_mode": mode_name,
        "rescued_edges": selected & valid_edge_mask,
        "completion_probs": completion_probs * valid_edge_mask.float(),
        "ranking_scores": ranking_scores * valid_edge_mask.float(),
        "final_pred_scope_edges": final_pred_edges,
        "final_proposal_edge_probs": final_edge_probs * valid_edge_mask.float(),
    }


def mode_ranking_score(
    mode: str,
    completion_probs: torch.Tensor,
    guard_probs: torch.Tensor,
    oracle_scope_mask: torch.Tensor,
) -> torch.Tensor:
    if mode == COMPLETION_ONLY_MODE:
        return completion_probs
    if mode == GUARD_ONLY_MODE:
        return guard_probs
    if mode == PRODUCT_MODE:
        return completion_probs * guard_probs
    if mode == GUARD_THEN_COMPLETION_MODE:
        return guard_probs + (completion_probs * 1.0e-6)
    if mode == ORACLE_GUARD_THEN_COMPLETION_MODE:
        return oracle_scope_mask.float() + (completion_probs * 1.0e-6)
    raise ValueError(f"Unknown ranking mode: {mode}")


def add_efficiency_metrics(proposal_results: Dict[str, Any]) -> None:
    base = proposal_results[BASE_MODE]["overall"]
    naive = proposal_results[NAIVE_MODE]["overall"]
    for mode, metrics_by_group in proposal_results.items():
        if mode in {BASE_MODE, NAIVE_MODE}:
            continue
        metrics = metrics_by_group["overall"]
        metrics["recall_recovery_fraction_vs_naive"] = safe_div(
            metrics["proposal_changed_region_recall_edge"] - base["proposal_changed_region_recall_edge"],
            naive["proposal_changed_region_recall_edge"] - base["proposal_changed_region_recall_edge"],
        )
        metrics["cost_fraction_vs_naive"] = safe_div(
            metrics["edge_pred_total"] - base["edge_pred_total"],
            naive["edge_pred_total"] - base["edge_pred_total"],
        )


def add_event_scope_precision(decomp_results: Dict[str, Any]) -> None:
    for mode_results in decomp_results.values():
        for section in ["overall", "by_event_type", "by_corruption_setting"]:
            if section == "overall":
                buckets = [mode_results.get("overall", {})]
            else:
                buckets = list(mode_results.get(section, {}).values())
            for bucket in buckets:
                if not bucket:
                    continue
                false_fraction = bucket.get("rescued_false_scope_fraction")
                if false_fraction is not None:
                    bucket["rescued_event_scope_precision"] = 1.0 - false_fraction
                bucket["rescued_changed_edge_precision"] = bucket.get("rescued_true_changed_fraction")


@torch.no_grad()
def evaluate_ranking_interaction(
    proposal_model: torch.nn.Module,
    completion_model: GatedInternalEdgeCompletionHead,
    guard_model: EventScopeGuardHead,
    rewrite_model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    node_threshold: float,
    edge_threshold: float,
    completion_threshold: float,
    use_proposal_conditioning: bool,
) -> Dict[str, Any]:
    proposal_model.eval()
    completion_model.eval()
    guard_model.eval()
    rewrite_model.eval()

    proposal_buckets = {mode: defaultdict(init_proposal_bucket) for mode in MODE_ORDER}
    downstream_buckets = {mode: defaultdict(init_downstream_bucket) for mode in MODE_ORDER}
    decomp_buckets = {
        mode: defaultdict(init_decomp_bucket)
        for mode in RANKING_MODES
    }
    top_tail_buckets = {
        mode: defaultdict(init_top_tail_bucket)
        for mode in RANKING_MODES
    }

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
        completion_logits = completion_model(
            node_latents=base_outputs["node_latents"],
            base_edge_logits=base_outputs["edge_scope_logits"],
        )
        completion_probs = torch.sigmoid(completion_logits) * base_outputs["valid_edge_mask"].float()
        guard_logits = guard_model(
            node_latents=base_outputs["node_latents"],
            base_edge_logits=base_outputs["edge_scope_logits"],
            completion_logits=completion_logits,
        )
        guard_probs = torch.sigmoid(guard_logits) * base_outputs["valid_edge_mask"].float()
        valid_edge_mask = base_outputs["valid_edge_mask"].bool()
        oracle_scope_mask = (batch["event_scope_union_edges"] > 0.5) & valid_edge_mask

        mode_outputs: Dict[str, Dict[str, torch.Tensor]] = {
            BASE_MODE: apply_edge_completion_mode(
                base_outputs=base_outputs,
                edge_completion_mode=EDGE_COMPLETION_OFF,
                completion_model=None,
                completion_threshold=completion_threshold,
            ),
            NAIVE_MODE: apply_edge_completion_mode(
                base_outputs=base_outputs,
                edge_completion_mode=EDGE_COMPLETION_NAIVE,
                completion_model=None,
                completion_threshold=completion_threshold,
            ),
        }
        ranking_scores_by_mode: Dict[str, torch.Tensor] = {}
        for mode in RANKING_MODES:
            ranking_scores = mode_ranking_score(
                mode=mode,
                completion_probs=completion_probs,
                guard_probs=guard_probs,
                oracle_scope_mask=oracle_scope_mask,
            )
            ranking_scores_by_mode[mode] = ranking_scores
            primary = None
            secondary = None
            if mode == GUARD_THEN_COMPLETION_MODE:
                primary = guard_probs
                secondary = completion_probs
            elif mode == ORACLE_GUARD_THEN_COMPLETION_MODE:
                primary = oracle_scope_mask.float()
                secondary = completion_probs
            mode_outputs[mode] = build_ranked_outputs(
                base_outputs=base_outputs,
                mode_name=mode,
                ranking_scores=ranking_scores,
                completion_probs=completion_probs,
                primary_scores=primary,
                secondary_scores=secondary,
            )

        rewrite_outputs_by_mode = {}
        for mode, outputs in mode_outputs.items():
            rewrite_outputs_by_mode[mode] = rewrite_model(
                node_feats=outputs["input_node_feats"],
                adj=outputs["input_adj"],
                scope_node_mask=outputs["pred_scope_nodes"].float(),
                scope_edge_mask=outputs["final_pred_scope_edges"].float(),
                proposal_node_probs=outputs["proposal_node_probs"] if use_proposal_conditioning else None,
                proposal_edge_probs=outputs["final_proposal_edge_probs"] if use_proposal_conditioning else None,
            )

        from train.eval_step10_rescued_scope_rewrite_decomp import bool_pred_adj

        pred_adj_base = bool_pred_adj(rewrite_outputs_by_mode[BASE_MODE]["edge_logits_full"], valid_edge_mask)
        pred_adj_by_mode = {
            mode: bool_pred_adj(outputs["edge_logits_full"], valid_edge_mask)
            for mode, outputs in rewrite_outputs_by_mode.items()
            if mode in RANKING_MODES
        }

        batch_size = batch["node_feats"].shape[0]
        for sample_idx in range(batch_size):
            group_names = make_group_names(batch, sample_idx)
            groups = ["overall"] + [name for name in group_names if name.startswith("event_type::")]
            if "step6a_corruption_setting" in batch:
                groups.append(f"corruption::{batch['step6a_corruption_setting'][sample_idx]}")

            candidates_2d = base_outputs["rescue_candidates"][sample_idx]
            oracle_scope_2d = batch["event_scope_union_edges"][sample_idx]
            changed_edges_2d = batch["changed_edges"][sample_idx]

            for mode, outputs in mode_outputs.items():
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

            for mode in RANKING_MODES:
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
                    update_top_tail_bucket(
                        bucket=top_tail_buckets[mode][group],
                        candidates_2d=candidates_2d,
                        selected_2d=mode_outputs[mode]["rescued_edges"][sample_idx],
                        ranking_scores_2d=ranking_scores_by_mode[mode][sample_idx],
                        oracle_scope_2d=oracle_scope_2d,
                        changed_edges_2d=changed_edges_2d,
                    )

    proposal_results: dict[str, Any] = {}
    downstream_results: dict[str, Any] = {}
    decomp_results: dict[str, Any] = {}
    top_tail_results: dict[str, Any] = {}
    for mode in MODE_ORDER:
        finalized_prop = {
            name: finalize_proposal_bucket(bucket) for name, bucket in proposal_buckets[mode].items()
        }
        finalized_down = {
            name: finalize_downstream_bucket(bucket) for name, bucket in downstream_buckets[mode].items()
        }
        proposal_results[mode] = {
            "overall": finalized_prop.get("overall", finalize_proposal_bucket(init_proposal_bucket())),
            "by_event_type": {
                name.split("event_type::", 1)[1]: value
                for name, value in finalized_prop.items()
                if name.startswith("event_type::")
            },
            "by_corruption_setting": {
                name.split("corruption::", 1)[1]: value
                for name, value in finalized_prop.items()
                if name.startswith("corruption::")
            },
        }
        downstream_results[mode] = {
            "overall": finalized_down.get("overall", finalize_downstream_bucket(init_downstream_bucket())),
            "by_event_type": {
                name.split("event_type::", 1)[1]: value
                for name, value in finalized_down.items()
                if name.startswith("event_type::")
            },
            "by_corruption_setting": {
                name.split("corruption::", 1)[1]: value
                for name, value in finalized_down.items()
                if name.startswith("corruption::")
            },
        }

    for mode in RANKING_MODES:
        finalized_decomp = {
            name: finalize_decomp_bucket(bucket) for name, bucket in decomp_buckets[mode].items()
        }
        finalized_top_tail = {
            name: finalize_top_tail_bucket(bucket) for name, bucket in top_tail_buckets[mode].items()
        }
        decomp_results[mode] = {
            "overall": finalized_decomp.get("overall", finalize_decomp_bucket(init_decomp_bucket())),
            "by_event_type": {
                name.split("event_type::", 1)[1]: value
                for name, value in finalized_decomp.items()
                if name.startswith("event_type::")
            },
            "by_corruption_setting": {
                name.split("corruption::", 1)[1]: value
                for name, value in finalized_decomp.items()
                if name.startswith("corruption::")
            },
        }
        top_tail_results[mode] = {
            "overall": finalized_top_tail.get("overall", finalize_top_tail_bucket(init_top_tail_bucket())),
            "by_event_type": {
                name.split("event_type::", 1)[1]: value
                for name, value in finalized_top_tail.items()
                if name.startswith("event_type::")
            },
            "by_corruption_setting": {
                name.split("corruption::", 1)[1]: value
                for name, value in finalized_top_tail.items()
                if name.startswith("corruption::")
            },
        }

    add_efficiency_metrics(proposal_results)
    add_event_scope_precision(decomp_results)
    return {
        "proposal_side": proposal_results,
        "downstream": downstream_results,
        "decomposition": decomp_results,
        "top_tail_diagnostics": top_tail_results,
    }


def write_summary_csv(path: Path, payload: Dict[str, Any]) -> None:
    rows = []
    results = payload["results"]
    for mode, prop in results["proposal_side"].items():
        overall = prop["overall"]
        down = results["downstream"][mode]["overall"]
        decomp = results.get("decomposition", {}).get(mode, {}).get("overall", {})
        top_tail = results.get("top_tail_diagnostics", {}).get(mode, {}).get("overall", {})
        rows.append(
            {
                "mode": mode,
                "group": "overall",
                "edge_recall": overall.get("proposal_changed_region_recall_edge"),
                "out_of_scope_miss_edge": overall.get("out_of_scope_miss_edge"),
                "edge_pred_total": overall.get("edge_pred_total"),
                "recall_recovery_fraction_vs_naive": overall.get("recall_recovery_fraction_vs_naive"),
                "cost_fraction_vs_naive": overall.get("cost_fraction_vs_naive"),
                "rescued_event_scope_precision": decomp.get("rescued_event_scope_precision"),
                "rescued_changed_edge_precision": decomp.get("rescued_changed_edge_precision"),
                "rescued_true_changed_fraction": decomp.get("rescued_true_changed_fraction"),
                "rescued_true_scope_context_fraction": decomp.get("rescued_true_scope_context_fraction"),
                "rescued_false_scope_fraction": decomp.get("rescued_false_scope_fraction"),
                "event_scope_precision_at_budget": top_tail.get("event_scope_precision_at_budget"),
                "changed_edge_precision_at_budget": top_tail.get("changed_edge_precision_at_budget"),
                "event_scope_recall_at_budget": top_tail.get("event_scope_recall_at_budget"),
                "changed_edge_recall_at_budget": top_tail.get("changed_edge_recall_at_budget"),
                "event_scope_ap": top_tail.get("event_scope_ap"),
                "event_scope_auroc": top_tail.get("event_scope_auroc"),
                "changed_edge_ap": top_tail.get("changed_edge_ap"),
                "changed_edge_auroc": top_tail.get("changed_edge_auroc"),
                "full_edge": down.get("full_edge"),
                "changed_edge": down.get("changed_edge"),
                "context_edge": down.get("context_edge"),
                "add": down.get("add"),
                "delete": down.get("delete"),
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
    parser.add_argument("--guard_checkpoint_path", type=str, required=True)
    parser.add_argument("--rewrite_checkpoint_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--split_name", type=str, default="test")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--artifact_dir", type=str, default="artifacts/step12_guard_ranking_interaction")
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
    guard_checkpoint_path = resolve_path(args.guard_checkpoint_path)
    rewrite_checkpoint_path = resolve_path(args.rewrite_checkpoint_path)
    data_path = resolve_path(args.data_path)
    artifact_dir = resolve_path(args.artifact_dir)

    _, loader = build_loader(
        data_path=str(data_path),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    proposal_model = load_proposal_model(proposal_checkpoint_path, device)
    completion_model = load_completion_model(completion_checkpoint_path, device)
    guard_model = load_guard_model(guard_checkpoint_path, device)
    rewrite_model, use_proposal_conditioning = load_rewrite_model(rewrite_checkpoint_path, device)

    results = evaluate_ranking_interaction(
        proposal_model=proposal_model,
        completion_model=completion_model,
        guard_model=guard_model,
        rewrite_model=rewrite_model,
        loader=loader,
        device=device,
        node_threshold=args.node_threshold,
        edge_threshold=args.edge_threshold,
        completion_threshold=args.completion_threshold,
        use_proposal_conditioning=use_proposal_conditioning,
    )

    run_name = args.run_name or slugify(f"{args.split_name}_{rewrite_checkpoint_path.parent.name}")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifact_dir / f"{run_name}.json"
    csv_path = artifact_dir / f"{run_name}.csv"
    payload = {
        "metadata": {
            "proposal_checkpoint_path": str(proposal_checkpoint_path),
            "completion_checkpoint_path": str(completion_checkpoint_path),
            "guard_checkpoint_path": str(guard_checkpoint_path),
            "rewrite_checkpoint_path": str(rewrite_checkpoint_path),
            "data_path": str(data_path),
            "split_name": args.split_name,
            "node_threshold": args.node_threshold,
            "edge_threshold": args.edge_threshold,
            "completion_threshold": args.completion_threshold,
            "rescue_budget_fraction": FIXED_RESCUE_BUDGET_FRACTION,
            "ranking_modes": RANKING_MODES,
            "rewrite_uses_proposal_conditioning": use_proposal_conditioning,
        },
        "results": results,
    }
    save_json(out_path, payload)
    write_summary_csv(csv_path, payload)

    compact = {
        "proposal_overall": {
            mode: metrics["overall"] for mode, metrics in results["proposal_side"].items()
        },
        "downstream_overall": {
            mode: metrics["overall"] for mode, metrics in results["downstream"].items()
        },
        "selection_overall": {
            mode: metrics["overall"] for mode, metrics in results["top_tail_diagnostics"].items()
        },
        "rescue_mix_overall": {
            mode: metrics["overall"] for mode, metrics in results["decomposition"].items()
        },
    }
    print(f"device: {device}")
    print(f"saved json: {out_path}")
    print(f"saved csv: {csv_path}")
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

