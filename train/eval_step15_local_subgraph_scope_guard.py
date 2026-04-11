from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

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
from train.eval_scope_edit_localization_gap import slugify
from train.eval_step10_rescued_scope_rewrite_decomp import (
    bool_pred_adj,
    finalize_decomp_bucket,
    init_decomp_bucket,
    update_decomp_sample,
)
from train.eval_step11_guarded_internal_completion import load_guard_model
from train.eval_step12_guard_ranking_interaction import (
    FIXED_RESCUE_BUDGET_FRACTION,
    add_efficiency_metrics,
    add_event_scope_precision,
    build_ranked_outputs,
    finalize_top_tail_bucket,
    init_top_tail_bucket,
    update_top_tail_bucket,
)
from train.eval_step14_guard_representation_probe import score_candidates_with_probe
from train.eval_step9_gated_edge_completion import (
    EDGE_COMPLETION_NAIVE,
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
from train.train_step14_guard_representation_probe import (
    ENRICHED_LOCAL_CONTEXT,
    load_probe_model,
)
from train.train_step15_local_subgraph_scope_guard import load_local_guard_model


BASE_MODE = "base"
STEP9C_MODE = "step9c_completion_only"
STEP14_ENRICHED_MODE = "step14_enriched_local_context_probe"
STEP15_MODE = "step15_learned_local_subgraph_guard"
ORACLE_MODE = "oracle_event_scope_guard_budget_0.10"
NAIVE_MODE = "naive_induced_closure_reference"
MODE_ORDER = [
    BASE_MODE,
    STEP9C_MODE,
    STEP14_ENRICHED_MODE,
    STEP15_MODE,
    ORACLE_MODE,
    NAIVE_MODE,
]
RESCUE_MODES = [STEP9C_MODE, STEP14_ENRICHED_MODE, STEP15_MODE, ORACLE_MODE]


@torch.no_grad()
def score_candidates_with_local_guard(
    local_guard_model: torch.nn.Module,
    base_outputs: Dict[str, torch.Tensor],
    completion_logits: torch.Tensor,
) -> torch.Tensor:
    logits = local_guard_model(
        node_latents=base_outputs["node_latents"],
        input_adj=base_outputs["input_adj"],
        pred_scope_nodes=base_outputs["pred_scope_nodes"],
        valid_edge_mask=base_outputs["valid_edge_mask"],
        node_scope_logits=base_outputs["node_scope_logits"],
        proposal_node_probs=base_outputs["proposal_node_probs"],
        base_edge_logits=base_outputs["edge_scope_logits"],
        proposal_edge_probs=base_outputs["proposal_edge_probs"],
        completion_logits=completion_logits,
    )
    return torch.sigmoid(logits) * base_outputs["valid_edge_mask"].float()


@torch.no_grad()
def evaluate_step15(
    proposal_model: torch.nn.Module,
    completion_model: torch.nn.Module,
    reference_guard_model: torch.nn.Module,
    enriched_probe_model: torch.nn.Module,
    local_guard_model: torch.nn.Module,
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
    reference_guard_model.eval()
    enriched_probe_model.eval()
    local_guard_model.eval()
    rewrite_model.eval()

    proposal_buckets = {mode: defaultdict(init_proposal_bucket) for mode in MODE_ORDER}
    downstream_buckets = {mode: defaultdict(init_downstream_bucket) for mode in MODE_ORDER}
    decomp_buckets = {mode: defaultdict(init_decomp_bucket) for mode in RESCUE_MODES}
    top_tail_buckets = {mode: defaultdict(init_top_tail_bucket) for mode in RESCUE_MODES}

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
        valid_edge_mask = base_outputs["valid_edge_mask"].bool()
        completion_probs = torch.sigmoid(completion_logits) * valid_edge_mask.float()
        reference_guard_logits = reference_guard_model(
            node_latents=base_outputs["node_latents"],
            base_edge_logits=base_outputs["edge_scope_logits"],
            completion_logits=completion_logits,
        )
        enriched_probe_probs = score_candidates_with_probe(
            probe_model=enriched_probe_model,
            feature_bundle=ENRICHED_LOCAL_CONTEXT,
            batch=batch,
            base_outputs=base_outputs,
            completion_logits=completion_logits,
            reference_guard_logits=reference_guard_logits,
        )
        local_guard_probs = score_candidates_with_local_guard(
            local_guard_model=local_guard_model,
            base_outputs=base_outputs,
            completion_logits=completion_logits,
        )
        oracle_scope_mask = (batch["event_scope_union_edges"] > 0.5) & valid_edge_mask
        oracle_ranking = oracle_scope_mask.float() + (completion_probs * 1.0e-6)

        ranking_scores_by_mode = {
            STEP9C_MODE: completion_probs,
            STEP14_ENRICHED_MODE: enriched_probe_probs,
            STEP15_MODE: local_guard_probs,
            ORACLE_MODE: oracle_ranking,
        }
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
                ranking_scores=ranking_scores_by_mode[STEP9C_MODE],
                completion_probs=completion_probs,
            ),
            STEP14_ENRICHED_MODE: build_ranked_outputs(
                base_outputs=base_outputs,
                mode_name=STEP14_ENRICHED_MODE,
                ranking_scores=ranking_scores_by_mode[STEP14_ENRICHED_MODE],
                completion_probs=completion_probs,
            ),
            STEP15_MODE: build_ranked_outputs(
                base_outputs=base_outputs,
                mode_name=STEP15_MODE,
                ranking_scores=ranking_scores_by_mode[STEP15_MODE],
                completion_probs=completion_probs,
            ),
            ORACLE_MODE: build_ranked_outputs(
                base_outputs=base_outputs,
                mode_name=ORACLE_MODE,
                ranking_scores=ranking_scores_by_mode[ORACLE_MODE],
                completion_probs=completion_probs,
                primary_scores=oracle_scope_mask.float(),
                secondary_scores=completion_probs,
            ),
            NAIVE_MODE: apply_edge_completion_mode(
                base_outputs=base_outputs,
                edge_completion_mode=EDGE_COMPLETION_NAIVE,
                completion_model=None,
                completion_threshold=completion_threshold,
            ),
        }

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

        pred_adj_base = bool_pred_adj(rewrite_outputs_by_mode[BASE_MODE]["edge_logits_full"], valid_edge_mask)
        pred_adj_by_mode = {
            mode: bool_pred_adj(rewrite_outputs_by_mode[mode]["edge_logits_full"], valid_edge_mask)
            for mode in RESCUE_MODES
        }

        for sample_idx in range(batch["node_feats"].shape[0]):
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

            for mode in RESCUE_MODES:
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

    for mode in RESCUE_MODES:
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
                "false_scope_extra_context_errors": decomp.get("extra_context_errors_rescued_false_scope"),
                "true_scope_extra_context_errors": decomp.get("extra_context_errors_rescued_true_scope_context"),
                "spillover_context_drop": decomp.get("spillover_context_drop"),
                "event_scope_precision_at_budget": top_tail.get("event_scope_precision_at_budget"),
                "event_scope_recall_at_budget": top_tail.get("event_scope_recall_at_budget"),
                "changed_edge_precision_at_budget": top_tail.get("changed_edge_precision_at_budget"),
                "changed_edge_recall_at_budget": top_tail.get("changed_edge_recall_at_budget"),
                "event_scope_ap": top_tail.get("event_scope_ap"),
                "event_scope_auroc": top_tail.get("event_scope_auroc"),
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
    parser.add_argument("--reference_guard_checkpoint_path", type=str, required=True)
    parser.add_argument("--enriched_probe_checkpoint_path", type=str, required=True)
    parser.add_argument("--local_guard_checkpoint_path", type=str, required=True)
    parser.add_argument("--rewrite_checkpoint_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--split_name", type=str, default="test")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--artifact_dir", type=str, default="artifacts/step15_local_subgraph_scope_guard")
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
    reference_guard_checkpoint_path = resolve_path(args.reference_guard_checkpoint_path)
    enriched_probe_checkpoint_path = resolve_path(args.enriched_probe_checkpoint_path)
    local_guard_checkpoint_path = resolve_path(args.local_guard_checkpoint_path)
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
    reference_guard_model = load_guard_model(reference_guard_checkpoint_path, device)
    enriched_probe_model = load_probe_model(enriched_probe_checkpoint_path, device)
    local_guard_model = load_local_guard_model(local_guard_checkpoint_path, device)
    rewrite_model, use_proposal_conditioning = load_rewrite_model(rewrite_checkpoint_path, device)

    results = evaluate_step15(
        proposal_model=proposal_model,
        completion_model=completion_model,
        reference_guard_model=reference_guard_model,
        enriched_probe_model=enriched_probe_model,
        local_guard_model=local_guard_model,
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
            "reference_guard_checkpoint_path": str(reference_guard_checkpoint_path),
            "enriched_probe_checkpoint_path": str(enriched_probe_checkpoint_path),
            "local_guard_checkpoint_path": str(local_guard_checkpoint_path),
            "rewrite_checkpoint_path": str(rewrite_checkpoint_path),
            "data_path": str(data_path),
            "split_name": args.split_name,
            "node_threshold": args.node_threshold,
            "edge_threshold": args.edge_threshold,
            "completion_threshold": args.completion_threshold,
            "rescue_budget_fraction": FIXED_RESCUE_BUDGET_FRACTION,
            "guard_representation": "learned_local_subgraph_guard",
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
        "decomposition_overall": {
            mode: metrics["overall"] for mode, metrics in results["decomposition"].items()
        },
        "top_tail_overall": {
            mode: metrics["overall"] for mode, metrics in results["top_tail_diagnostics"].items()
        },
    }
    print(f"device: {device}")
    print(f"saved json: {out_path}")
    print(f"saved csv: {csv_path}")
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

