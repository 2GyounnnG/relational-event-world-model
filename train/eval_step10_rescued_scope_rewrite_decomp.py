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
from train.eval_step9_gated_edge_completion import (
    EDGE_COMPLETION_OFF,
    apply_edge_completion_mode,
    build_downstream_sample_stats,
    get_base_proposal_outputs,
    init_downstream_bucket,
    load_completion_model,
    load_proposal_model,
    load_rewrite_model,
    update_bucket,
)
from train.eval_step9_rescue_frontier import build_budgeted_outputs, finalize_downstream_bucket


FIXED_RESCUE_BUDGET_FRACTION = 0.10


def init_decomp_bucket() -> Dict[str, float]:
    return {
        "num_samples": 0.0,
        "rescued_total": 0.0,
        "rescued_true_changed_total": 0.0,
        "rescued_true_scope_context_total": 0.0,
        "rescued_false_scope_total": 0.0,
        "rescued_unclassified_total": 0.0,
        "rescued_true_changed_correct_budget": 0.0,
        "rescued_true_changed_correct_base": 0.0,
        "rescued_true_scope_context_correct_budget": 0.0,
        "rescued_true_scope_context_correct_base": 0.0,
        "rescued_false_scope_correct_budget": 0.0,
        "rescued_false_scope_correct_base": 0.0,
        "base_scope_context_total": 0.0,
        "base_scope_context_correct_base": 0.0,
        "base_scope_context_correct_budget": 0.0,
        "all_context_total": 0.0,
        "all_context_correct_base": 0.0,
        "all_context_correct_budget": 0.0,
        "rescued_context_total": 0.0,
        "rescued_context_correct_base": 0.0,
        "rescued_context_correct_budget": 0.0,
    }


def bool_pred_adj(edge_logits: torch.Tensor, valid_edge_mask: torch.Tensor) -> torch.Tensor:
    pred_adj = torch.sigmoid(edge_logits) >= 0.5
    pred_adj = (pred_adj | pred_adj.transpose(1, 2)) & valid_edge_mask
    return pred_adj.bool()


def correct_count(pred_adj: torch.Tensor, target_adj: torch.Tensor, mask: torch.Tensor) -> float:
    return ((pred_adj == target_adj) & mask).float().sum().item()


def update_decomp_sample(
    bucket: Dict[str, float],
    batch: Dict[str, Any],
    base_pred_edges: torch.Tensor,
    rescued_edges: torch.Tensor,
    pred_adj_base: torch.Tensor,
    pred_adj_budget: torch.Tensor,
    valid_edge_mask: torch.Tensor,
    sample_idx: int,
) -> None:
    valid = valid_edge_mask[sample_idx].bool()
    changed = (batch["changed_edges"][sample_idx] > 0.5) & valid
    event_scope = (batch["event_scope_union_edges"][sample_idx] > 0.5) & valid
    target_adj = (batch["next_adj"][sample_idx] > 0.5) & valid
    base_scope = base_pred_edges[sample_idx].bool() & valid
    rescued = rescued_edges[sample_idx].bool() & valid
    context = (~changed) & valid

    # Exclusive rescued-edge classes. Changed edges get priority even if a data bug
    # ever left them outside event_scope_union_edges.
    rescued_true_changed = rescued & changed
    rescued_true_scope_context = rescued & (~changed) & event_scope
    rescued_false_scope = rescued & (~changed) & (~event_scope)
    classified = rescued_true_changed | rescued_true_scope_context | rescued_false_scope
    rescued_unclassified = rescued & (~classified)

    base_scope_context = base_scope & context
    rescued_context = rescued & context

    pred_base = pred_adj_base[sample_idx] & valid
    pred_budget = pred_adj_budget[sample_idx] & valid

    bucket["num_samples"] += 1.0
    bucket["rescued_total"] += rescued.float().sum().item()
    bucket["rescued_true_changed_total"] += rescued_true_changed.float().sum().item()
    bucket["rescued_true_scope_context_total"] += rescued_true_scope_context.float().sum().item()
    bucket["rescued_false_scope_total"] += rescued_false_scope.float().sum().item()
    bucket["rescued_unclassified_total"] += rescued_unclassified.float().sum().item()

    bucket["rescued_true_changed_correct_budget"] += correct_count(
        pred_budget, target_adj, rescued_true_changed
    )
    bucket["rescued_true_changed_correct_base"] += correct_count(
        pred_base, target_adj, rescued_true_changed
    )
    bucket["rescued_true_scope_context_correct_budget"] += correct_count(
        pred_budget, target_adj, rescued_true_scope_context
    )
    bucket["rescued_true_scope_context_correct_base"] += correct_count(
        pred_base, target_adj, rescued_true_scope_context
    )
    bucket["rescued_false_scope_correct_budget"] += correct_count(
        pred_budget, target_adj, rescued_false_scope
    )
    bucket["rescued_false_scope_correct_base"] += correct_count(
        pred_base, target_adj, rescued_false_scope
    )

    bucket["base_scope_context_total"] += base_scope_context.float().sum().item()
    bucket["base_scope_context_correct_base"] += correct_count(
        pred_base, target_adj, base_scope_context
    )
    bucket["base_scope_context_correct_budget"] += correct_count(
        pred_budget, target_adj, base_scope_context
    )

    bucket["all_context_total"] += context.float().sum().item()
    bucket["all_context_correct_base"] += correct_count(pred_base, target_adj, context)
    bucket["all_context_correct_budget"] += correct_count(pred_budget, target_adj, context)

    bucket["rescued_context_total"] += rescued_context.float().sum().item()
    bucket["rescued_context_correct_base"] += correct_count(pred_base, target_adj, rescued_context)
    bucket["rescued_context_correct_budget"] += correct_count(
        pred_budget, target_adj, rescued_context
    )


def finalize_decomp_bucket(bucket: Dict[str, float]) -> Dict[str, Any]:
    rescued_total = bucket["rescued_total"]
    true_changed = bucket["rescued_true_changed_total"]
    true_scope_context = bucket["rescued_true_scope_context_total"]
    false_scope = bucket["rescued_false_scope_total"]

    def preserve_rate(correct_key: str, total: float) -> Optional[float]:
        return safe_div(bucket[correct_key], total)

    true_changed_budget = preserve_rate("rescued_true_changed_correct_budget", true_changed)
    true_changed_base = preserve_rate("rescued_true_changed_correct_base", true_changed)
    true_scope_context_budget = preserve_rate(
        "rescued_true_scope_context_correct_budget", true_scope_context
    )
    true_scope_context_base = preserve_rate(
        "rescued_true_scope_context_correct_base", true_scope_context
    )
    false_scope_budget = preserve_rate("rescued_false_scope_correct_budget", false_scope)
    false_scope_base = preserve_rate("rescued_false_scope_correct_base", false_scope)

    base_scope_context_base = preserve_rate(
        "base_scope_context_correct_base", bucket["base_scope_context_total"]
    )
    base_scope_context_budget = preserve_rate(
        "base_scope_context_correct_budget", bucket["base_scope_context_total"]
    )
    all_context_base = preserve_rate("all_context_correct_base", bucket["all_context_total"])
    all_context_budget = preserve_rate("all_context_correct_budget", bucket["all_context_total"])
    rescued_context_base = preserve_rate(
        "rescued_context_correct_base", bucket["rescued_context_total"]
    )
    rescued_context_budget = preserve_rate(
        "rescued_context_correct_budget", bucket["rescued_context_total"]
    )

    total_context_base_errors = bucket["all_context_total"] - bucket["all_context_correct_base"]
    total_context_budget_errors = bucket["all_context_total"] - bucket["all_context_correct_budget"]
    rescued_true_scope_base_errors = (
        true_scope_context - bucket["rescued_true_scope_context_correct_base"]
    )
    rescued_true_scope_budget_errors = (
        true_scope_context - bucket["rescued_true_scope_context_correct_budget"]
    )
    rescued_false_scope_base_errors = false_scope - bucket["rescued_false_scope_correct_base"]
    rescued_false_scope_budget_errors = false_scope - bucket["rescued_false_scope_correct_budget"]
    base_scope_context_base_errors = (
        bucket["base_scope_context_total"] - bucket["base_scope_context_correct_base"]
    )
    base_scope_context_budget_errors = (
        bucket["base_scope_context_total"] - bucket["base_scope_context_correct_budget"]
    )

    extra_context_errors = total_context_budget_errors - total_context_base_errors
    true_scope_extra = rescued_true_scope_budget_errors - rescued_true_scope_base_errors
    false_scope_extra = rescued_false_scope_budget_errors - rescued_false_scope_base_errors
    spillover_extra = base_scope_context_budget_errors - base_scope_context_base_errors

    return {
        "num_samples": int(bucket["num_samples"]),
        "rescued_total": int(rescued_total),
        "rescued_true_changed_total": int(true_changed),
        "rescued_true_scope_context_total": int(true_scope_context),
        "rescued_false_scope_total": int(false_scope),
        "rescued_unclassified_total": int(bucket["rescued_unclassified_total"]),
        "rescued_true_changed_fraction": safe_div(true_changed, rescued_total),
        "rescued_true_scope_context_fraction": safe_div(true_scope_context, rescued_total),
        "rescued_false_scope_fraction": safe_div(false_scope, rescued_total),
        "correct_edit_rate_rescued_true_changed_budget": true_changed_budget,
        "correct_edit_rate_rescued_true_changed_base": true_changed_base,
        "correct_edit_delta_rescued_true_changed": (
            None
            if true_changed_budget is None or true_changed_base is None
            else true_changed_budget - true_changed_base
        ),
        "preserve_rate_rescued_true_scope_context_budget": true_scope_context_budget,
        "preserve_rate_rescued_true_scope_context_base": true_scope_context_base,
        "preserve_drop_rescued_true_scope_context": (
            None
            if true_scope_context_budget is None or true_scope_context_base is None
            else true_scope_context_base - true_scope_context_budget
        ),
        "over_edit_rate_rescued_true_scope_context_budget": (
            None if true_scope_context_budget is None else 1.0 - true_scope_context_budget
        ),
        "preserve_rate_rescued_false_scope_budget": false_scope_budget,
        "preserve_rate_rescued_false_scope_base": false_scope_base,
        "preserve_drop_rescued_false_scope": (
            None
            if false_scope_budget is None or false_scope_base is None
            else false_scope_base - false_scope_budget
        ),
        "over_edit_rate_rescued_false_scope_budget": (
            None if false_scope_budget is None else 1.0 - false_scope_budget
        ),
        "base_scope_context_total": int(bucket["base_scope_context_total"]),
        "base_scope_context_preserve_rate_base": base_scope_context_base,
        "base_scope_context_preserve_rate_budget": base_scope_context_budget,
        "spillover_context_drop": (
            None
            if base_scope_context_base is None or base_scope_context_budget is None
            else base_scope_context_base - base_scope_context_budget
        ),
        "all_context_preserve_rate_base": all_context_base,
        "all_context_preserve_rate_budget": all_context_budget,
        "all_context_drop": (
            None if all_context_base is None or all_context_budget is None else all_context_base - all_context_budget
        ),
        "rescued_context_preserve_rate_base": rescued_context_base,
        "rescued_context_preserve_rate_budget": rescued_context_budget,
        "rescued_context_drop": (
            None
            if rescued_context_base is None or rescued_context_budget is None
            else rescued_context_base - rescued_context_budget
        ),
        "extra_context_errors_total": extra_context_errors,
        "extra_context_errors_rescued_true_scope_context": true_scope_extra,
        "extra_context_errors_rescued_false_scope": false_scope_extra,
        "extra_context_errors_base_scope_spillover": spillover_extra,
        "extra_context_error_share_rescued_true_scope_context": safe_div(
            true_scope_extra, extra_context_errors
        ),
        "extra_context_error_share_rescued_false_scope": safe_div(
            false_scope_extra, extra_context_errors
        ),
        "extra_context_error_share_base_scope_spillover": safe_div(
            spillover_extra, extra_context_errors
        ),
    }


@torch.no_grad()
def evaluate_decomp(
    proposal_model: torch.nn.Module,
    rewrite_model: torch.nn.Module,
    completion_model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    node_threshold: float,
    edge_threshold: float,
    completion_threshold: float,
    use_proposal_conditioning: bool,
) -> Dict[str, Any]:
    proposal_model.eval()
    rewrite_model.eval()
    completion_model.eval()

    decomp_buckets: dict[str, Dict[str, float]] = defaultdict(init_decomp_bucket)
    downstream_buckets: dict[str, dict[str, Dict[str, float]]] = {
        "base": defaultdict(init_downstream_bucket),
        "budget_0.10": defaultdict(init_downstream_bucket),
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

        base_mode_outputs = apply_edge_completion_mode(
            base_outputs=base_outputs,
            edge_completion_mode=EDGE_COMPLETION_OFF,
            completion_model=None,
            completion_threshold=completion_threshold,
        )
        budget_outputs = build_budgeted_outputs(
            base_outputs=base_outputs,
            completion_probs=completion_probs,
            rescue_budget_fraction=FIXED_RESCUE_BUDGET_FRACTION,
        )

        rewrite_base = rewrite_model(
            node_feats=base_mode_outputs["input_node_feats"],
            adj=base_mode_outputs["input_adj"],
            scope_node_mask=base_mode_outputs["pred_scope_nodes"].float(),
            scope_edge_mask=base_mode_outputs["final_pred_scope_edges"].float(),
            proposal_node_probs=base_mode_outputs["proposal_node_probs"] if use_proposal_conditioning else None,
            proposal_edge_probs=base_mode_outputs["final_proposal_edge_probs"] if use_proposal_conditioning else None,
        )
        rewrite_budget = rewrite_model(
            node_feats=budget_outputs["input_node_feats"],
            adj=budget_outputs["input_adj"],
            scope_node_mask=budget_outputs["pred_scope_nodes"].float(),
            scope_edge_mask=budget_outputs["final_pred_scope_edges"].float(),
            proposal_node_probs=budget_outputs["proposal_node_probs"] if use_proposal_conditioning else None,
            proposal_edge_probs=budget_outputs["final_proposal_edge_probs"] if use_proposal_conditioning else None,
        )

        valid_edge_mask = base_outputs["valid_edge_mask"].bool()
        pred_adj_base = bool_pred_adj(rewrite_base["edge_logits_full"], valid_edge_mask)
        pred_adj_budget = bool_pred_adj(rewrite_budget["edge_logits_full"], valid_edge_mask)
        batch_size = batch["node_feats"].shape[0]

        for sample_idx in range(batch_size):
            group_names = make_group_names(batch, sample_idx)
            groups = ["overall"] + [name for name in group_names if name.startswith("event_type::")]
            if "step6a_corruption_setting" in batch:
                groups.append(f"corruption::{batch['step6a_corruption_setting'][sample_idx]}")

            for group in groups:
                update_decomp_sample(
                    bucket=decomp_buckets[group],
                    batch=batch,
                    base_pred_edges=base_outputs["pred_scope_edges"],
                    rescued_edges=budget_outputs["rescued_edges"],
                    pred_adj_base=pred_adj_base,
                    pred_adj_budget=pred_adj_budget,
                    valid_edge_mask=valid_edge_mask,
                    sample_idx=sample_idx,
                )
                update_bucket(
                    downstream_buckets["base"][group],
                    build_downstream_sample_stats(
                        batch=batch,
                        rewrite_outputs=rewrite_base,
                        valid_edge_mask=valid_edge_mask,
                        sample_idx=sample_idx,
                    ),
                )
                update_bucket(
                    downstream_buckets["budget_0.10"][group],
                    build_downstream_sample_stats(
                        batch=batch,
                        rewrite_outputs=rewrite_budget,
                        valid_edge_mask=valid_edge_mask,
                        sample_idx=sample_idx,
                    ),
                )

    finalized_decomp = {
        name: finalize_decomp_bucket(bucket) for name, bucket in decomp_buckets.items()
    }
    finalized_downstream = {
        mode: {name: finalize_downstream_bucket(bucket) for name, bucket in buckets.items()}
        for mode, buckets in downstream_buckets.items()
    }
    return {
        "decomposition": {
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
        },
        "downstream": {
            mode: {
                "overall": values.get("overall", finalize_downstream_bucket(init_downstream_bucket())),
                "by_event_type": {
                    name.split("event_type::", 1)[1]: value
                    for name, value in values.items()
                    if name.startswith("event_type::")
                },
                "by_corruption_setting": {
                    name.split("corruption::", 1)[1]: value
                    for name, value in values.items()
                    if name.startswith("corruption::")
                },
            }
            for mode, values in finalized_downstream.items()
        },
    }


def write_csv(path: Path, payload: Dict[str, Any]) -> None:
    rows = []
    decomp = payload["results"]["decomposition"]
    groups = {"overall": decomp["overall"]}
    groups.update({f"event_type::{k}": v for k, v in decomp.get("by_event_type", {}).items()})
    for group_name, metrics in groups.items():
        rows.append(
            {
                "group": group_name,
                "rescued_total": metrics.get("rescued_total"),
                "rescued_true_changed_fraction": metrics.get("rescued_true_changed_fraction"),
                "rescued_true_scope_context_fraction": metrics.get("rescued_true_scope_context_fraction"),
                "rescued_false_scope_fraction": metrics.get("rescued_false_scope_fraction"),
                "correct_edit_rate_rescued_true_changed_budget": metrics.get(
                    "correct_edit_rate_rescued_true_changed_budget"
                ),
                "preserve_rate_rescued_true_scope_context_budget": metrics.get(
                    "preserve_rate_rescued_true_scope_context_budget"
                ),
                "preserve_rate_rescued_false_scope_budget": metrics.get(
                    "preserve_rate_rescued_false_scope_budget"
                ),
                "base_scope_context_preserve_rate_base": metrics.get(
                    "base_scope_context_preserve_rate_base"
                ),
                "base_scope_context_preserve_rate_budget": metrics.get(
                    "base_scope_context_preserve_rate_budget"
                ),
                "spillover_context_drop": metrics.get("spillover_context_drop"),
                "all_context_drop": metrics.get("all_context_drop"),
                "extra_context_errors_rescued_true_scope_context": metrics.get(
                    "extra_context_errors_rescued_true_scope_context"
                ),
                "extra_context_errors_rescued_false_scope": metrics.get(
                    "extra_context_errors_rescued_false_scope"
                ),
                "extra_context_errors_base_scope_spillover": metrics.get(
                    "extra_context_errors_base_scope_spillover"
                ),
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
    parser.add_argument("--rewrite_checkpoint_path", type=str, required=True)
    parser.add_argument("--completion_checkpoint_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--split_name", type=str, default="test")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--artifact_dir", type=str, default="artifacts/step10_rescued_scope_rewrite_decomp")
    parser.add_argument("--node_threshold", type=float, default=0.20)
    parser.add_argument("--edge_threshold", type=float, default=0.15)
    parser.add_argument("--completion_threshold", type=float, default=0.50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    proposal_checkpoint_path = resolve_path(args.proposal_checkpoint_path)
    rewrite_checkpoint_path = resolve_path(args.rewrite_checkpoint_path)
    completion_checkpoint_path = resolve_path(args.completion_checkpoint_path)
    data_path = resolve_path(args.data_path)
    artifact_dir = resolve_path(args.artifact_dir)
    device = get_device(args.device)

    _, loader = build_loader(
        data_path=str(data_path),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    proposal_model = load_proposal_model(proposal_checkpoint_path, device)
    rewrite_model, use_proposal_conditioning = load_rewrite_model(rewrite_checkpoint_path, device)
    completion_model = load_completion_model(completion_checkpoint_path, device)

    results = evaluate_decomp(
        proposal_model=proposal_model,
        rewrite_model=rewrite_model,
        completion_model=completion_model,
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
            "rewrite_checkpoint_path": str(rewrite_checkpoint_path),
            "completion_checkpoint_path": str(completion_checkpoint_path),
            "data_path": str(data_path),
            "split_name": args.split_name,
            "node_threshold": args.node_threshold,
            "edge_threshold": args.edge_threshold,
            "completion_threshold": args.completion_threshold,
            "rescue_budget_fraction": FIXED_RESCUE_BUDGET_FRACTION,
            "comparison": "base proposal vs fixed-budget 10% internal completion",
            "rewrite_uses_proposal_conditioning": use_proposal_conditioning,
        },
        "results": results,
    }
    save_json(out_path, payload)
    write_csv(csv_path, payload)

    compact = {
        "decomposition_overall": results["decomposition"]["overall"],
        "downstream_overall": {
            mode: values["overall"] for mode, values in results["downstream"].items()
        },
    }
    print(f"device: {device}")
    print(f"saved json: {out_path}")
    print(f"saved csv: {csv_path}")
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
