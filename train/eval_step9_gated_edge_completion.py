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
from models.oracle_local_delta import (
    EDGE_DELTA_ADD,
    EDGE_DELTA_DELETE,
    EDGE_DELTA_KEEP,
    OracleLocalDeltaRewriteConfig,
    OracleLocalDeltaRewriteModel,
    build_edge_delta_targets,
    build_valid_edge_mask,
)
from models.proposal import ScopeProposalConfig, ScopeProposalModel
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


EDGE_COMPLETION_OFF = "off"
EDGE_COMPLETION_NAIVE = "naive_induced_closure"
EDGE_COMPLETION_LEARNED = "learned_gated_internal_completion"
EDGE_COMPLETION_MODES = (
    EDGE_COMPLETION_OFF,
    EDGE_COMPLETION_NAIVE,
    EDGE_COMPLETION_LEARNED,
)


@dataclass
class EdgeCompletionConfig:
    proposal_hidden_dim: int
    hidden_dim: int = 128
    head_layers: int = 2
    dropout: float = 0.0
    include_base_edge_features: bool = True


class GatedInternalEdgeCompletionHead(nn.Module):
    """
    Minimal residual edge-completion head.

    The head does not replace the proposal edge logits. It only scores candidate
    edge slots whose endpoints are already inside predicted node scope; the
    evaluator unions accepted rescues with the base proposal edge scope.
    中文说明：这个 head 只补“节点 scope 内部漏掉的边”，不扩节点、不改 rewrite。
    """

    def __init__(self, config: EdgeCompletionConfig):
        super().__init__()
        self.config = config
        edge_extra_dim = 2 if config.include_base_edge_features else 0
        in_dim = config.proposal_hidden_dim * 4 + edge_extra_dim
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
            clamped_logits = base_edge_logits.clamp(min=-20.0, max=20.0)
            features.extend(
                [
                    clamped_logits.unsqueeze(-1),
                    torch.sigmoid(clamped_logits).unsqueeze(-1),
                ]
            )
        pair_feat = torch.cat(features, dim=-1)
        logits = self.head(pair_feat).squeeze(-1)
        logits = 0.5 * (logits + logits.transpose(1, 2))
        diag_mask = torch.eye(num_nodes, device=node_latents.device, dtype=torch.bool).unsqueeze(0)
        return logits.masked_fill(diag_mask, -1e9)


def load_proposal_model(checkpoint_path: Path, device: torch.device) -> ScopeProposalModel:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = ScopeProposalModel(ScopeProposalConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def load_rewrite_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[OracleLocalDeltaRewriteModel, bool]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = OracleLocalDeltaRewriteModel(
        OracleLocalDeltaRewriteConfig(**checkpoint["model_config"])
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    use_proposal_conditioning = bool(
        checkpoint.get("model_config", {}).get("use_proposal_conditioning", False)
    )
    return model, use_proposal_conditioning


def save_completion_checkpoint(
    path: Path,
    model: GatedInternalEdgeCompletionHead,
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


def load_completion_model(
    checkpoint_path: Path,
    device: torch.device,
) -> GatedInternalEdgeCompletionHead:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = GatedInternalEdgeCompletionHead(
        EdgeCompletionConfig(**checkpoint["model_config"])
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def get_base_proposal_outputs(
    proposal_model: ScopeProposalModel,
    batch: Dict[str, Any],
    node_threshold: float,
    edge_threshold: float,
) -> Dict[str, torch.Tensor]:
    input_node_feats = batch.get("obs_node_feats", batch["node_feats"])
    input_adj = batch.get("obs_adj", batch["adj"])
    node_mask = batch["node_mask"].bool()
    valid_edge_mask = build_valid_edge_mask(batch["node_mask"]).bool()

    proposal_outputs = proposal_model(node_feats=input_node_feats, adj=input_adj)
    node_scope_logits = proposal_outputs.get("node_scope_logits", proposal_outputs["scope_logits"])
    proposal_node_probs = torch.sigmoid(node_scope_logits) * batch["node_mask"]
    pred_scope_nodes = (proposal_node_probs >= node_threshold) & node_mask

    if "edge_scope_logits" in proposal_outputs:
        edge_scope_logits = proposal_outputs["edge_scope_logits"]
        proposal_edge_probs = torch.sigmoid(edge_scope_logits) * valid_edge_mask.float()
        pred_scope_edges = (proposal_edge_probs >= edge_threshold) & valid_edge_mask
    else:
        proposal_edge_probs = (
            proposal_node_probs.unsqueeze(2) * proposal_node_probs.unsqueeze(1) * valid_edge_mask.float()
        )
        pred_scope_edges = (
            pred_scope_nodes.unsqueeze(2) & pred_scope_nodes.unsqueeze(1) & valid_edge_mask
        )
        edge_scope_logits = torch.logit(proposal_edge_probs.clamp(min=1e-6, max=1.0 - 1e-6))
        edge_scope_logits = edge_scope_logits.masked_fill(~valid_edge_mask, -1e9)

    node_induced_edges = pred_scope_nodes.unsqueeze(2) & pred_scope_nodes.unsqueeze(1) & valid_edge_mask
    rescue_candidates = node_induced_edges & (~pred_scope_edges) & valid_edge_mask

    return {
        "input_node_feats": input_node_feats,
        "input_adj": input_adj,
        "node_latents": proposal_outputs["node_latents"],
        "node_scope_logits": node_scope_logits,
        "edge_scope_logits": edge_scope_logits,
        "proposal_node_probs": proposal_node_probs,
        "proposal_edge_probs": proposal_edge_probs,
        "pred_scope_nodes": pred_scope_nodes,
        "pred_scope_edges": pred_scope_edges,
        "valid_edge_mask": valid_edge_mask,
        "node_induced_edges": node_induced_edges,
        "rescue_candidates": rescue_candidates,
    }


def apply_edge_completion_mode(
    base_outputs: Dict[str, torch.Tensor],
    edge_completion_mode: str,
    completion_model: Optional[GatedInternalEdgeCompletionHead],
    completion_threshold: float,
) -> Dict[str, torch.Tensor]:
    base_pred_edges = base_outputs["pred_scope_edges"]
    base_edge_probs = base_outputs["proposal_edge_probs"]
    valid_edge_mask = base_outputs["valid_edge_mask"]
    node_induced_edges = base_outputs["node_induced_edges"]
    rescue_candidates = base_outputs["rescue_candidates"]

    if edge_completion_mode == EDGE_COMPLETION_OFF:
        final_edge_probs = base_edge_probs
        final_pred_edges = base_pred_edges
        completion_probs = torch.zeros_like(base_edge_probs)
        rescued_edges = torch.zeros_like(base_pred_edges)
    elif edge_completion_mode == EDGE_COMPLETION_NAIVE:
        final_pred_edges = base_pred_edges | node_induced_edges
        final_edge_probs = torch.maximum(base_edge_probs, node_induced_edges.float())
        completion_probs = node_induced_edges.float()
        rescued_edges = node_induced_edges & (~base_pred_edges) & valid_edge_mask
    elif edge_completion_mode == EDGE_COMPLETION_LEARNED:
        if completion_model is None:
            raise ValueError("learned_gated_internal_completion requires --completion_checkpoint_path")
        completion_logits = completion_model(
            node_latents=base_outputs["node_latents"],
            base_edge_logits=base_outputs["edge_scope_logits"],
        )
        completion_probs = torch.sigmoid(completion_logits) * valid_edge_mask.float()
        rescued_edges = (completion_probs >= completion_threshold) & rescue_candidates
        final_pred_edges = base_pred_edges | rescued_edges
        final_edge_probs = torch.where(
            rescued_edges,
            torch.maximum(base_edge_probs, completion_probs),
            base_edge_probs,
        )
    else:
        raise ValueError(f"Unsupported edge_completion_mode: {edge_completion_mode}")

    return {
        **base_outputs,
        "edge_completion_mode": edge_completion_mode,
        "completion_probs": completion_probs,
        "rescued_edges": rescued_edges,
        "final_pred_scope_edges": final_pred_edges & valid_edge_mask,
        "final_proposal_edge_probs": final_edge_probs * valid_edge_mask.float(),
    }


def init_proposal_bucket() -> Dict[str, float]:
    return {
        "num_samples": 0.0,
        "edge_changed_total": 0.0,
        "edge_changed_covered": 0.0,
        "edge_pred_total": 0.0,
        "edge_pred_context_total": 0.0,
        "edge_out_scope_changed_total": 0.0,
        "rescued_edge_total": 0.0,
        "rescued_changed_edge_total": 0.0,
    }


def init_downstream_bucket() -> Dict[str, float]:
    return {
        "num_samples": 0.0,
        "full_edge_correct": 0.0,
        "full_edge_total": 0.0,
        "changed_edge_correct": 0.0,
        "changed_edge_total": 0.0,
        "context_edge_correct": 0.0,
        "context_edge_total": 0.0,
        "delta_correct": 0.0,
        "delta_total": 0.0,
        "keep_correct": 0.0,
        "keep_total": 0.0,
        "add_correct": 0.0,
        "add_total": 0.0,
        "delete_correct": 0.0,
        "delete_total": 0.0,
    }


def update_bucket(bucket: Dict[str, float], stats: Dict[str, float]) -> None:
    for key, value in stats.items():
        bucket[key] += value


def finalize_proposal_bucket(bucket: Dict[str, float]) -> Dict[str, Any]:
    return {
        "num_samples": int(bucket["num_samples"]),
        "proposal_changed_region_recall_edge": safe_div(
            bucket["edge_changed_covered"], bucket["edge_changed_total"]
        ),
        "out_of_scope_miss_edge": safe_div(
            bucket["edge_out_scope_changed_total"], bucket["edge_changed_total"]
        ),
        "proposal_scope_excess_ratio_edge": safe_div(
            bucket["edge_pred_context_total"], bucket["edge_pred_total"]
        ),
        "edge_changed_total": int(bucket["edge_changed_total"]),
        "edge_pred_total": int(bucket["edge_pred_total"]),
        "edge_out_scope_changed_total": int(bucket["edge_out_scope_changed_total"]),
        "rescued_edge_total": int(bucket["rescued_edge_total"]),
        "rescued_changed_edge_total": int(bucket["rescued_changed_edge_total"]),
    }


def finalize_downstream_bucket(bucket: Dict[str, float]) -> Dict[str, Any]:
    return {
        "num_samples": int(bucket["num_samples"]),
        "full_edge": safe_div(bucket["full_edge_correct"], bucket["full_edge_total"]),
        "changed_edge": safe_div(bucket["changed_edge_correct"], bucket["changed_edge_total"]),
        "context_edge": safe_div(bucket["context_edge_correct"], bucket["context_edge_total"]),
        "delta_all": safe_div(bucket["delta_correct"], bucket["delta_total"]),
        "keep": safe_div(bucket["keep_correct"], bucket["keep_total"]),
        "add": safe_div(bucket["add_correct"], bucket["add_total"]),
        "delete": safe_div(bucket["delete_correct"], bucket["delete_total"]),
        "edge_changed_total": int(bucket["changed_edge_total"]),
        "edge_context_total": int(bucket["context_edge_total"]),
    }


def add_efficiency_metrics(results: Dict[str, Any]) -> None:
    for section in ["overall", "by_event_type"]:
        base_section = results[EDGE_COMPLETION_OFF].get(section, {})
        naive_section = results[EDGE_COMPLETION_NAIVE].get(section, {})
        learned_section = results[EDGE_COMPLETION_LEARNED].get(section, {})
        if section == "overall":
            group_names = ["overall"]
            base_lookup = {"overall": base_section}
            naive_lookup = {"overall": naive_section}
            learned_lookup = {"overall": learned_section}
        else:
            group_names = sorted(set(base_section) | set(naive_section) | set(learned_section))
            base_lookup = base_section
            naive_lookup = naive_section
            learned_lookup = learned_section

        for group_name in group_names:
            base = base_lookup.get(group_name, {})
            naive = naive_lookup.get(group_name, {})
            learned = learned_lookup.get(group_name, {})
            base_recall = base.get("proposal_changed_region_recall_edge")
            naive_recall = naive.get("proposal_changed_region_recall_edge")
            learned_recall = learned.get("proposal_changed_region_recall_edge")
            base_size = base.get("edge_pred_total")
            naive_size = naive.get("edge_pred_total")
            learned_size = learned.get("edge_pred_total")
            recall_recovery = None
            cost_fraction = None
            if None not in (base_recall, naive_recall, learned_recall):
                recall_recovery = safe_div(learned_recall - base_recall, naive_recall - base_recall)
            if None not in (base_size, naive_size, learned_size):
                cost_fraction = safe_div(learned_size - base_size, naive_size - base_size)
            learned["recall_recovery_fraction_vs_naive"] = recall_recovery
            learned["cost_fraction_vs_naive"] = cost_fraction


def build_proposal_sample_stats(
    batch: Dict[str, Any],
    final_pred_edges: torch.Tensor,
    rescued_edges: torch.Tensor,
    valid_edge_mask: torch.Tensor,
    sample_idx: int,
) -> Dict[str, float]:
    changed_edges = (batch["changed_edges"][sample_idx] > 0.5) & valid_edge_mask[sample_idx]
    context_edges = (~changed_edges) & valid_edge_mask[sample_idx]
    pred_edges = final_pred_edges[sample_idx] & valid_edge_mask[sample_idx]
    rescued = rescued_edges[sample_idx] & valid_edge_mask[sample_idx]
    return {
        "num_samples": 1.0,
        "edge_changed_total": changed_edges.float().sum().item(),
        "edge_changed_covered": (changed_edges & pred_edges).float().sum().item(),
        "edge_pred_total": pred_edges.float().sum().item(),
        "edge_pred_context_total": (context_edges & pred_edges).float().sum().item(),
        "edge_out_scope_changed_total": (changed_edges & (~pred_edges)).float().sum().item(),
        "rescued_edge_total": rescued.float().sum().item(),
        "rescued_changed_edge_total": (rescued & changed_edges).float().sum().item(),
    }


def build_downstream_sample_stats(
    batch: Dict[str, Any],
    rewrite_outputs: Dict[str, torch.Tensor],
    valid_edge_mask: torch.Tensor,
    sample_idx: int,
) -> Dict[str, float]:
    pred_adj = (torch.sigmoid(rewrite_outputs["edge_logits_full"][sample_idx]) >= 0.5)
    pred_adj = ((pred_adj | pred_adj.transpose(0, 1)) & valid_edge_mask[sample_idx]).bool()
    target_adj = (batch["next_adj"][sample_idx] > 0.5) & valid_edge_mask[sample_idx]
    current_adj = batch["adj"][sample_idx : sample_idx + 1]
    pred_delta = build_edge_delta_targets(current_adj, pred_adj.float().unsqueeze(0))[0]
    target_delta = build_edge_delta_targets(current_adj, batch["next_adj"][sample_idx : sample_idx + 1])[0]
    changed_edges = (batch["changed_edges"][sample_idx] > 0.5) & valid_edge_mask[sample_idx]
    context_edges = (~changed_edges) & valid_edge_mask[sample_idx]
    correct_edges = pred_adj == target_adj
    correct_delta = pred_delta == target_delta
    add_mask = valid_edge_mask[sample_idx] & (target_delta == EDGE_DELTA_ADD)
    delete_mask = valid_edge_mask[sample_idx] & (target_delta == EDGE_DELTA_DELETE)
    keep_mask = valid_edge_mask[sample_idx] & (target_delta == EDGE_DELTA_KEEP)

    return {
        "num_samples": 1.0,
        "full_edge_correct": (correct_edges & valid_edge_mask[sample_idx]).float().sum().item(),
        "full_edge_total": valid_edge_mask[sample_idx].float().sum().item(),
        "changed_edge_correct": (correct_edges & changed_edges).float().sum().item(),
        "changed_edge_total": changed_edges.float().sum().item(),
        "context_edge_correct": (correct_edges & context_edges).float().sum().item(),
        "context_edge_total": context_edges.float().sum().item(),
        "delta_correct": (correct_delta & valid_edge_mask[sample_idx]).float().sum().item(),
        "delta_total": valid_edge_mask[sample_idx].float().sum().item(),
        "keep_correct": (correct_delta & keep_mask).float().sum().item(),
        "keep_total": keep_mask.float().sum().item(),
        "add_correct": (correct_delta & add_mask).float().sum().item(),
        "add_total": add_mask.float().sum().item(),
        "delete_correct": (correct_delta & delete_mask).float().sum().item(),
        "delete_total": delete_mask.float().sum().item(),
    }


@torch.no_grad()
def evaluate_step9(
    proposal_model: ScopeProposalModel,
    rewrite_model: Optional[OracleLocalDeltaRewriteModel],
    completion_model: Optional[GatedInternalEdgeCompletionHead],
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    node_threshold: float,
    edge_threshold: float,
    completion_threshold: float,
    use_proposal_conditioning: bool = True,
) -> Dict[str, Any]:
    proposal_model.eval()
    if rewrite_model is not None:
        rewrite_model.eval()
    if completion_model is not None:
        completion_model.eval()

    proposal_buckets: dict[str, dict[str, Dict[str, float]]] = {
        mode: defaultdict(init_proposal_bucket) for mode in EDGE_COMPLETION_MODES
    }
    downstream_buckets: dict[str, dict[str, Dict[str, float]]] = {
        mode: defaultdict(init_downstream_bucket) for mode in EDGE_COMPLETION_MODES
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
        valid_edge_mask = base_outputs["valid_edge_mask"]
        batch_size = batch["node_feats"].shape[0]

        mode_outputs: dict[str, Dict[str, torch.Tensor]] = {}
        for mode in EDGE_COMPLETION_MODES:
            if mode == EDGE_COMPLETION_LEARNED and completion_model is None:
                continue
            mode_outputs[mode] = apply_edge_completion_mode(
                base_outputs=base_outputs,
                edge_completion_mode=mode,
                completion_model=completion_model,
                completion_threshold=completion_threshold,
            )

        rewrite_outputs_by_mode: dict[str, Dict[str, torch.Tensor]] = {}
        if rewrite_model is not None:
            for mode, outputs in mode_outputs.items():
                rewrite_outputs_by_mode[mode] = rewrite_model(
                    node_feats=outputs["input_node_feats"],
                    adj=outputs["input_adj"],
                    scope_node_mask=outputs["pred_scope_nodes"].float(),
                    scope_edge_mask=outputs["final_pred_scope_edges"].float(),
                    proposal_node_probs=outputs["proposal_node_probs"] if use_proposal_conditioning else None,
                    proposal_edge_probs=outputs["final_proposal_edge_probs"] if use_proposal_conditioning else None,
                )

        for sample_idx in range(batch_size):
            group_names = make_group_names(batch, sample_idx)
            event_type_groups = [name for name in group_names if name.startswith("event_type::")]
            summary_groups = ["overall"] + event_type_groups
            if "step6a_corruption_setting" in batch:
                setting = str(batch["step6a_corruption_setting"][sample_idx])
                summary_groups.append(f"corruption::{setting}")
                summary_groups.extend(
                    name for name in group_names if name.startswith(f"corruption::{setting}::event_type::")
                )

            for mode, outputs in mode_outputs.items():
                proposal_stats = build_proposal_sample_stats(
                    batch=batch,
                    final_pred_edges=outputs["final_pred_scope_edges"],
                    rescued_edges=outputs["rescued_edges"],
                    valid_edge_mask=valid_edge_mask,
                    sample_idx=sample_idx,
                )
                for group_name in summary_groups:
                    update_bucket(proposal_buckets[mode][group_name], proposal_stats)

                if rewrite_model is not None:
                    downstream_stats = build_downstream_sample_stats(
                        batch=batch,
                        rewrite_outputs=rewrite_outputs_by_mode[mode],
                        valid_edge_mask=valid_edge_mask,
                        sample_idx=sample_idx,
                    )
                    for group_name in summary_groups:
                        update_bucket(downstream_buckets[mode][group_name], downstream_stats)

    proposal_results: dict[str, Dict[str, Any]] = {}
    downstream_results: dict[str, Dict[str, Any]] = {}
    for mode in EDGE_COMPLETION_MODES:
        if mode == EDGE_COMPLETION_LEARNED and completion_model is None:
            continue
        finalized = {name: finalize_proposal_bucket(bucket) for name, bucket in proposal_buckets[mode].items()}
        proposal_results[mode] = {
            "overall": finalized.get("overall", finalize_proposal_bucket(init_proposal_bucket())),
            "by_event_type": {
                name.split("event_type::", 1)[1]: value
                for name, value in finalized.items()
                if name.startswith("event_type::")
            },
            "by_corruption_setting": {
                name.split("corruption::", 1)[1]: value
                for name, value in finalized.items()
                if name.startswith("corruption::") and "::event_type::" not in name
            },
            "by_corruption_event_type": {
                name: value
                for name, value in finalized.items()
                if name.startswith("corruption::") and "::event_type::" in name
            },
        }

        if rewrite_model is not None:
            finalized_downstream = {
                name: finalize_downstream_bucket(bucket)
                for name, bucket in downstream_buckets[mode].items()
            }
            downstream_results[mode] = {
                "overall": finalized_downstream.get(
                    "overall", finalize_downstream_bucket(init_downstream_bucket())
                ),
                "by_event_type": {
                    name.split("event_type::", 1)[1]: value
                    for name, value in finalized_downstream.items()
                    if name.startswith("event_type::")
                },
                "by_corruption_setting": {
                    name.split("corruption::", 1)[1]: value
                    for name, value in finalized_downstream.items()
                    if name.startswith("corruption::") and "::event_type::" not in name
                },
                "by_corruption_event_type": {
                    name: value
                    for name, value in finalized_downstream.items()
                    if name.startswith("corruption::") and "::event_type::" in name
                },
            }

    if completion_model is not None:
        add_efficiency_metrics(proposal_results)

    return {
        "proposal_side": proposal_results,
        "downstream": downstream_results,
    }


def write_summary_csv(path: Path, payload: Dict[str, Any]) -> None:
    rows = []
    for mode, metrics in payload["results"]["proposal_side"].items():
        overall = metrics["overall"]
        rows.append(
            {
                "section": "proposal_overall",
                "mode": mode,
                "group": "overall",
                "edge_recall": overall.get("proposal_changed_region_recall_edge"),
                "out_of_scope_miss_edge": overall.get("out_of_scope_miss_edge"),
                "scope_excess_ratio_edge": overall.get("proposal_scope_excess_ratio_edge"),
                "edge_pred_total": overall.get("edge_pred_total"),
                "recall_recovery_fraction_vs_naive": overall.get("recall_recovery_fraction_vs_naive"),
                "cost_fraction_vs_naive": overall.get("cost_fraction_vs_naive"),
                "full_edge": None,
                "changed_edge": None,
                "context_edge": None,
                "add": None,
                "delete": None,
            }
        )
        for event_type, event_metrics in metrics.get("by_event_type", {}).items():
            if event_type not in {"edge_add", "edge_delete"}:
                continue
            rows.append(
                {
                    "section": "proposal_event_type",
                    "mode": mode,
                    "group": event_type,
                    "edge_recall": event_metrics.get("proposal_changed_region_recall_edge"),
                    "out_of_scope_miss_edge": event_metrics.get("out_of_scope_miss_edge"),
                    "scope_excess_ratio_edge": event_metrics.get("proposal_scope_excess_ratio_edge"),
                    "edge_pred_total": event_metrics.get("edge_pred_total"),
                    "recall_recovery_fraction_vs_naive": event_metrics.get(
                        "recall_recovery_fraction_vs_naive"
                    ),
                    "cost_fraction_vs_naive": event_metrics.get("cost_fraction_vs_naive"),
                    "full_edge": None,
                    "changed_edge": None,
                    "context_edge": None,
                    "add": None,
                    "delete": None,
                }
            )

    for mode, metrics in payload["results"].get("downstream", {}).items():
        overall = metrics["overall"]
        rows.append(
            {
                "section": "downstream_overall",
                "mode": mode,
                "group": "overall",
                "edge_recall": None,
                "out_of_scope_miss_edge": None,
                "scope_excess_ratio_edge": None,
                "edge_pred_total": None,
                "recall_recovery_fraction_vs_naive": None,
                "cost_fraction_vs_naive": None,
                "full_edge": overall.get("full_edge"),
                "changed_edge": overall.get("changed_edge"),
                "context_edge": overall.get("context_edge"),
                "add": overall.get("add"),
                "delete": overall.get("delete"),
            }
        )

    fieldnames = [
        "section",
        "mode",
        "group",
        "edge_recall",
        "out_of_scope_miss_edge",
        "scope_excess_ratio_edge",
        "edge_pred_total",
        "recall_recovery_fraction_vs_naive",
        "cost_fraction_vs_naive",
        "full_edge",
        "changed_edge",
        "context_edge",
        "add",
        "delete",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal_checkpoint_path", type=str, required=True)
    parser.add_argument("--rewrite_checkpoint_path", type=str, required=True)
    parser.add_argument("--completion_checkpoint_path", type=str, default="")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--split_name", type=str, default="test")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--artifact_dir", type=str, default="artifacts/step9_gated_edge_completion")
    parser.add_argument("--node_threshold", type=float, default=0.20)
    parser.add_argument("--edge_threshold", type=float, default=0.15)
    parser.add_argument("--completion_threshold", type=float, default=0.50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    proposal_checkpoint_path = resolve_path(args.proposal_checkpoint_path)
    rewrite_checkpoint_path = resolve_path(args.rewrite_checkpoint_path)
    completion_checkpoint_path = resolve_path(args.completion_checkpoint_path) if args.completion_checkpoint_path else None
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
    completion_model = (
        load_completion_model(completion_checkpoint_path, device)
        if completion_checkpoint_path is not None
        else None
    )

    results = evaluate_step9(
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
    out_path = artifact_dir / f"{run_name}.json"
    csv_path = artifact_dir / f"{run_name}.csv"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "proposal_checkpoint_path": str(proposal_checkpoint_path),
            "rewrite_checkpoint_path": str(rewrite_checkpoint_path),
            "completion_checkpoint_path": str(completion_checkpoint_path) if completion_checkpoint_path else None,
            "data_path": str(data_path),
            "split_name": args.split_name,
            "node_threshold": args.node_threshold,
            "edge_threshold": args.edge_threshold,
            "completion_threshold": args.completion_threshold,
            "edge_completion_modes": list(EDGE_COMPLETION_MODES),
            "rewrite_uses_proposal_conditioning": use_proposal_conditioning,
            "notes": [
                "node proposal is unchanged",
                "learned completion only rescues edge slots whose endpoints are already inside predicted node scope",
                "rewrite checkpoint is unchanged; proposal edge probabilities are updated only for completed edges",
            ],
        },
        "results": results,
    }
    save_json(out_path, payload)
    write_summary_csv(csv_path, payload)

    print(f"device: {device}")
    print(f"saved json: {out_path}")
    print(f"saved csv: {csv_path}")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
