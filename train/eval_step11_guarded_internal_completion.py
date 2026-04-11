from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.baselines import build_mlp
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
from train.eval_step9_rescue_frontier import build_budgeted_outputs


FIXED_RESCUE_BUDGET_FRACTION = 0.10
BASE_MODE = "base"
STEP9C_MODE = "budget_completion_0.10"
GUARDED_MODE = "guarded_budget_0.10"
ORACLE_GUARD_MODE = "oracle_event_scope_guard_budget_0.10"
NAIVE_MODE = "naive_induced_closure_reference"


@dataclass
class EventScopeGuardConfig:
    proposal_hidden_dim: int
    hidden_dim: int = 128
    head_layers: int = 2
    dropout: float = 0.0
    include_base_edge_features: bool = True
    include_completion_features: bool = True


class EventScopeGuardHead(nn.Module):
    """
    Small residual guard for internal completion candidates.

    It predicts event-scope membership for candidate edge slots. It does not
    alter node proposal, base edge proposal, completion scoring, or rewrite.
    中文说明：guard 只学习“这个内部候选边是不是 event scope”，不是 changed-edge-only head。
    """

    def __init__(self, config: EventScopeGuardConfig):
        super().__init__()
        self.config = config
        extra_dim = 0
        if config.include_base_edge_features:
            extra_dim += 2
        if config.include_completion_features:
            extra_dim += 2
        in_dim = config.proposal_hidden_dim * 4 + extra_dim
        self.head = build_mlp(
            in_dim=in_dim,
            hidden_dim=config.hidden_dim,
            out_dim=1,
            num_layers=config.head_layers,
            dropout=config.dropout,
        )

    def forward(
        self,
        node_latents: torch.Tensor,
        base_edge_logits: torch.Tensor,
        completion_logits: torch.Tensor,
    ) -> torch.Tensor:
        bsz, num_nodes, hidden_dim = node_latents.shape
        h_i = node_latents.unsqueeze(2).expand(bsz, num_nodes, num_nodes, hidden_dim)
        h_j = node_latents.unsqueeze(1).expand(bsz, num_nodes, num_nodes, hidden_dim)
        features = [
            h_i,
            h_j,
            torch.abs(h_i - h_j),
            h_i * h_j,
        ]
        if self.config.include_base_edge_features:
            base_logits = base_edge_logits.clamp(min=-20.0, max=20.0)
            features.extend([base_logits.unsqueeze(-1), torch.sigmoid(base_logits).unsqueeze(-1)])
        if self.config.include_completion_features:
            comp_logits = completion_logits.clamp(min=-20.0, max=20.0)
            features.extend([comp_logits.unsqueeze(-1), torch.sigmoid(comp_logits).unsqueeze(-1)])

        pair_feat = torch.cat(features, dim=-1)
        logits = self.head(pair_feat).squeeze(-1)
        logits = 0.5 * (logits + logits.transpose(1, 2))
        diag_mask = torch.eye(num_nodes, device=node_latents.device, dtype=torch.bool).unsqueeze(0)
        return logits.masked_fill(diag_mask, -1e9)


def save_guard_checkpoint(
    path: Path,
    model: EventScopeGuardHead,
    args: argparse.Namespace,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "model_config": asdict(model.config),
        "model_state_dict": model.state_dict(),
        "args": vars(args),
    }
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_guard_model(checkpoint_path: Path, device: torch.device) -> EventScopeGuardHead:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = EventScopeGuardHead(EventScopeGuardConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def build_ranked_budget_outputs(
    base_outputs: Dict[str, torch.Tensor],
    ranking_scores: torch.Tensor,
    mode_name: str,
    eligible_mask: Optional[torch.Tensor] = None,
    selected_edge_probs: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    all_candidates = base_outputs["rescue_candidates"].bool()
    candidates = all_candidates if eligible_mask is None else (all_candidates & eligible_mask.bool())
    selected = torch.zeros_like(all_candidates)
    batch_size = all_candidates.shape[0]
    for batch_idx in range(batch_size):
        candidate_indices = candidates[batch_idx].nonzero(as_tuple=False)
        if candidate_indices.shape[0] <= 0:
            continue
        total_internal_candidates = int(all_candidates[batch_idx].float().sum().item())
        budget = int(total_internal_candidates * FIXED_RESCUE_BUDGET_FRACTION)
        if budget <= 0:
            continue
        scores = ranking_scores[batch_idx][candidates[batch_idx]]
        topk = torch.topk(scores, k=min(budget, candidate_indices.shape[0]), largest=True).indices
        chosen = candidate_indices[topk]
        selected[batch_idx, chosen[:, 0], chosen[:, 1]] = True

    valid_edge_mask = base_outputs["valid_edge_mask"].bool()
    final_pred_edges = (base_outputs["pred_scope_edges"] | selected) & valid_edge_mask
    rescue_probs = ranking_scores if selected_edge_probs is None else selected_edge_probs
    final_edge_probs = torch.where(
        selected,
        torch.maximum(base_outputs["proposal_edge_probs"], rescue_probs),
        base_outputs["proposal_edge_probs"],
    )
    return {
        **base_outputs,
        "edge_completion_mode": mode_name,
        "rescued_edges": selected & valid_edge_mask,
        "completion_probs": ranking_scores * valid_edge_mask.float(),
        "final_pred_scope_edges": final_pred_edges,
        "final_proposal_edge_probs": final_edge_probs * valid_edge_mask.float(),
    }


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


@torch.no_grad()
def evaluate_guarded_completion(
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

    mode_order = [BASE_MODE, STEP9C_MODE, GUARDED_MODE, ORACLE_GUARD_MODE, NAIVE_MODE]
    proposal_buckets = {mode: defaultdict(init_proposal_bucket) for mode in mode_order}
    downstream_buckets = {mode: defaultdict(init_downstream_bucket) for mode in mode_order}
    decomp_buckets = {
        mode: defaultdict(init_decomp_bucket)
        for mode in [STEP9C_MODE, GUARDED_MODE, ORACLE_GUARD_MODE]
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
        combined_scores = completion_probs * guard_probs
        oracle_scope_mask = (batch["event_scope_union_edges"] > 0.5) & base_outputs["valid_edge_mask"].bool()

        mode_outputs = {
            BASE_MODE: apply_edge_completion_mode(
                base_outputs=base_outputs,
                edge_completion_mode=EDGE_COMPLETION_OFF,
                completion_model=None,
                completion_threshold=completion_threshold,
            ),
            STEP9C_MODE: build_budgeted_outputs(
                base_outputs=base_outputs,
                completion_probs=completion_probs,
                rescue_budget_fraction=FIXED_RESCUE_BUDGET_FRACTION,
            ),
            GUARDED_MODE: build_ranked_budget_outputs(
                base_outputs=base_outputs,
                ranking_scores=combined_scores,
                mode_name=GUARDED_MODE,
                selected_edge_probs=completion_probs,
            ),
            ORACLE_GUARD_MODE: build_ranked_budget_outputs(
                base_outputs=base_outputs,
                ranking_scores=completion_probs,
                mode_name=ORACLE_GUARD_MODE,
                eligible_mask=oracle_scope_mask,
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

        valid_edge_mask = base_outputs["valid_edge_mask"].bool()
        # Imported lazily to avoid a circular conceptual dependency in the top-level imports.
        from train.eval_step10_rescued_scope_rewrite_decomp import bool_pred_adj

        pred_adj_base = bool_pred_adj(rewrite_outputs_by_mode[BASE_MODE]["edge_logits_full"], valid_edge_mask)
        pred_adj_by_mode = {
            mode: bool_pred_adj(outputs["edge_logits_full"], valid_edge_mask)
            for mode, outputs in rewrite_outputs_by_mode.items()
            if mode != BASE_MODE
        }

        batch_size = batch["node_feats"].shape[0]
        for sample_idx in range(batch_size):
            group_names = make_group_names(batch, sample_idx)
            groups = ["overall"] + [name for name in group_names if name.startswith("event_type::")]
            if "step6a_corruption_setting" in batch:
                groups.append(f"corruption::{batch['step6a_corruption_setting'][sample_idx]}")

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

            for mode in [STEP9C_MODE, GUARDED_MODE, ORACLE_GUARD_MODE]:
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

    proposal_results: dict[str, Any] = {}
    downstream_results: dict[str, Any] = {}
    decomp_results: dict[str, Any] = {}
    for mode in mode_order:
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

    for mode in [STEP9C_MODE, GUARDED_MODE, ORACLE_GUARD_MODE]:
        finalized_decomp = {
            name: finalize_decomp_bucket(bucket) for name, bucket in decomp_buckets[mode].items()
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

    add_efficiency_metrics(proposal_results)
    return {
        "proposal_side": proposal_results,
        "downstream": downstream_results,
        "decomposition": decomp_results,
    }


def write_summary_csv(path: Path, payload: Dict[str, Any]) -> None:
    rows = []
    for mode, prop in payload["results"]["proposal_side"].items():
        overall = prop["overall"]
        down = payload["results"]["downstream"][mode]["overall"]
        decomp = payload["results"].get("decomposition", {}).get(mode, {}).get("overall", {})
        rows.append(
            {
                "mode": mode,
                "group": "overall",
                "edge_recall": overall.get("proposal_changed_region_recall_edge"),
                "out_of_scope_miss_edge": overall.get("out_of_scope_miss_edge"),
                "edge_pred_total": overall.get("edge_pred_total"),
                "recall_recovery_fraction_vs_naive": overall.get("recall_recovery_fraction_vs_naive"),
                "cost_fraction_vs_naive": overall.get("cost_fraction_vs_naive"),
                "rescued_true_changed_fraction": decomp.get("rescued_true_changed_fraction"),
                "rescued_true_scope_context_fraction": decomp.get("rescued_true_scope_context_fraction"),
                "rescued_false_scope_fraction": decomp.get("rescued_false_scope_fraction"),
                "false_scope_extra_context_errors": decomp.get("extra_context_errors_rescued_false_scope"),
                "true_scope_extra_context_errors": decomp.get("extra_context_errors_rescued_true_scope_context"),
                "spillover_context_drop": decomp.get("spillover_context_drop"),
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
    parser.add_argument("--artifact_dir", type=str, default="artifacts/step11_guarded_internal_completion")
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

    results = evaluate_guarded_completion(
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
            "guard_mode": "learned_event_scope_guard",
            "combined_score": "completion_score * guard_score",
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
    }
    print(f"device: {device}")
    print(f"saved json: {out_path}")
    print(f"saved csv: {csv_path}")
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
