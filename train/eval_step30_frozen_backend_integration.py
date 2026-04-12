from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.step30_dataset import Step30WeakObservationDataset, step30_weak_observation_collate_fn
from models.encoder_step30 import Step30EncoderConfig, Step30WeakObservationEncoder, build_pair_mask
from models.oracle_local_delta import (
    EDGE_DELTA_ADD,
    EDGE_DELTA_DELETE,
    EDGE_DELTA_KEEP,
    OracleLocalDeltaRewriteConfig,
    OracleLocalDeltaRewriteModel,
    build_edge_delta_targets,
)
from models.proposal import ScopeProposalConfig, ScopeProposalModel
from train.utils_step30_decode import (
    hard_adj_selective_rescue,
    load_rich_rescue_scorer,
    load_rescue_safety_scorer,
)


SYSTEM_METRICS = [
    "full_edge",
    "context_edge",
    "changed_edge",
    "add",
    "delete",
    "full_type",
    "full_state_mae",
]
RECOVERY_METRICS = [
    "recovery_node_type_accuracy",
    "recovery_node_state_mae",
    "recovery_node_state_mse",
    "recovery_edge_accuracy",
    "recovery_edge_precision",
    "recovery_edge_recall",
    "recovery_edge_f1",
    "decoded_edge_density",
    "gt_edge_density",
    "edge_fp_count",
    "edge_fn_count",
]
DELTA_METRICS = [
    "full_edge",
    "context_edge",
    "changed_edge",
    "add",
    "delete",
    "proposal_edge_recall",
    "out_of_scope_miss",
]


@dataclass
class BackendSpec:
    name: str
    proposal_checkpoint_path: Path
    rewrite_checkpoint_path: Path
    node_threshold: float
    edge_threshold: float
    proposal_model: ScopeProposalModel
    rewrite_model: OracleLocalDeltaRewriteModel
    use_proposal_conditioning: bool


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def get_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if has_mps:
        return torch.device("mps")
    return torch.device("cpu")


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def safe_div(num: float, den: float) -> Optional[float]:
    if den <= 0:
        return None
    return num / den


def accuracy_from_mask(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> Optional[float]:
    mask_bool = mask.bool()
    total = mask_bool.float().sum().item()
    if total <= 0:
        return None
    correct = ((pred == target) & mask_bool).float().sum().item()
    return correct / total


def mae_from_mask(pred: torch.Tensor, target: torch.Tensor, node_mask: torch.Tensor) -> Optional[float]:
    mask = node_mask.unsqueeze(-1).float()
    total = mask.sum().item() * pred.shape[-1]
    if total <= 0:
        return None
    return ((pred - target).abs() * mask).sum().item() / total


def binary_scope_counts(pred_mask: torch.Tensor, true_mask: torch.Tensor, valid_mask: torch.Tensor) -> Dict[str, float]:
    pred = pred_mask.bool() & valid_mask.bool()
    true = true_mask.bool() & valid_mask.bool()
    tp = (pred & true).float().sum().item()
    pred_pos = pred.float().sum().item()
    true_pos = true.float().sum().item()
    excess = (pred & (~true)).float().sum().item()
    return {"tp": tp, "pred_pos": pred_pos, "true_pos": true_pos, "excess": excess}


def hard_adj_from_logits(edge_logits: torch.Tensor, threshold: float | torch.Tensor = 0.5) -> torch.Tensor:
    pred_adj = (torch.sigmoid(edge_logits) >= threshold).float()
    pred_adj = ((pred_adj + pred_adj.transpose(1, 2)) > 0.5).float()
    diag_mask = torch.eye(pred_adj.shape[1], device=pred_adj.device, dtype=torch.bool).unsqueeze(0)
    return pred_adj.masked_fill(diag_mask, 0.0)


def hard_adj_from_scores(edge_scores: torch.Tensor, threshold: float | torch.Tensor = 0.5) -> torch.Tensor:
    pred_adj = (edge_scores >= threshold).float()
    pred_adj = ((pred_adj + pred_adj.transpose(1, 2)) > 0.5).float()
    diag_mask = torch.eye(pred_adj.shape[1], device=pred_adj.device, dtype=torch.bool).unsqueeze(0)
    return pred_adj.masked_fill(diag_mask, 0.0)


def logit_from_scores(edge_scores: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    edge_scores = edge_scores.clamp(eps, 1.0 - eps)
    return torch.log(edge_scores / (1.0 - edge_scores))


def parse_variant_thresholds(spec: Optional[str]) -> Optional[Dict[str, float]]:
    if not spec:
        return None
    thresholds: Dict[str, float] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(
                "--recovered_edge_thresholds_by_variant entries must look like clean:0.5,noisy:0.55"
            )
        key, value = part.split(":", 1)
        thresholds[key.strip()] = float(value)
    return thresholds


def threshold_for_batch_variants(
    variants: Iterable[Any],
    default_threshold: float,
    thresholds_by_variant: Optional[Dict[str, float]],
    device: torch.device,
) -> float | torch.Tensor:
    if not thresholds_by_variant:
        return default_threshold
    values = [
        float(thresholds_by_variant.get(str(variant), thresholds_by_variant.get("default", default_threshold)))
        for variant in variants
    ]
    return torch.tensor(values, dtype=torch.float32, device=device).view(-1, 1, 1)


def load_encoder(checkpoint_path: Path, device: torch.device) -> Step30WeakObservationEncoder:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config_dict = dict(checkpoint["config"])
    config_dict.setdefault("use_relation_hint_in_edge_head", False)
    config_dict.setdefault("use_relation_logit_residual", False)
    config_dict.setdefault("relation_logit_residual_scale", 1.0)
    config_dict.setdefault("use_trust_denoising_edge_decoder", False)
    config_dict.setdefault("use_pair_support_hints", False)
    config_dict.setdefault("use_pair_evidence_bundle", False)
    config_dict.setdefault("pair_evidence_bundle_dim", 0)
    config_dict.setdefault("use_rescue_scoped_pair_evidence_bundle", False)
    config_dict.setdefault("rescue_scoped_bundle_relation_max", 0.5)
    config_dict.setdefault("rescue_scoped_bundle_residual_scale", 0.5)
    config_dict.setdefault("use_rescue_safety_aux_head", False)
    config_dict.setdefault("use_signed_pair_witness", False)
    config_dict.setdefault("use_signed_pair_witness_in_edge_head", True)
    config_dict.setdefault("use_signed_pair_witness_correction", False)
    config_dict.setdefault("signed_pair_witness_correction_scale", 0.5)
    model = Step30WeakObservationEncoder(Step30EncoderConfig(**config_dict)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def load_proposal(checkpoint_path: Path, device: torch.device) -> ScopeProposalModel:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = ScopeProposalModel(ScopeProposalConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def load_rewrite(checkpoint_path: Path, device: torch.device) -> tuple[OracleLocalDeltaRewriteModel, bool]:
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


def build_backend_specs(args: argparse.Namespace, device: torch.device) -> list[BackendSpec]:
    specs = []
    requested = {name.strip() for name in args.backends.split(",") if name.strip()}
    if "w012" in requested:
        proposal_path = resolve_path(args.clean_proposal_checkpoint_path)
        rewrite_path = resolve_path(args.w012_checkpoint_path)
        rewrite_model, use_conditioning = load_rewrite(rewrite_path, device)
        specs.append(
            BackendSpec(
                name="w012",
                proposal_checkpoint_path=proposal_path,
                rewrite_checkpoint_path=rewrite_path,
                node_threshold=args.clean_node_threshold,
                edge_threshold=args.clean_edge_threshold,
                proposal_model=load_proposal(proposal_path, device),
                rewrite_model=rewrite_model,
                use_proposal_conditioning=use_conditioning,
            )
        )
    if "rft1_p2" in requested:
        proposal_path = resolve_path(args.noisy_proposal_checkpoint_path)
        rewrite_path = resolve_path(args.rft1_checkpoint_path)
        rewrite_model, use_conditioning = load_rewrite(rewrite_path, device)
        specs.append(
            BackendSpec(
                name="rft1_calibrated_p2",
                proposal_checkpoint_path=proposal_path,
                rewrite_checkpoint_path=rewrite_path,
                node_threshold=args.noisy_node_threshold,
                edge_threshold=args.noisy_edge_threshold,
                proposal_model=load_proposal(proposal_path, device),
                rewrite_model=rewrite_model,
                use_proposal_conditioning=use_conditioning,
            )
        )
    if not specs:
        raise ValueError(f"No known backends requested from --backends={args.backends!r}")
    return specs


@torch.no_grad()
def decode_inputs(
    batch: Dict[str, Any],
    encoder_model: Step30WeakObservationEncoder,
    edge_threshold: float,
    edge_thresholds_by_variant: Optional[Dict[str, float]] = None,
    decode_mode: str = "threshold",
    rescue_variants: Optional[set[str]] = None,
    rescue_relation_max: float = 0.5,
    rescue_support_min: float = 0.55,
    rescue_budget_fraction: float = 0.0,
    rescue_score_mode: str = "raw",
    rescue_support_weight: float = 0.5,
    rescue_relation_weight: float = 0.25,
    rescue_safety_scorer: Optional[Dict[str, Any]] = None,
    rich_rescue_scorer: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, torch.Tensor]]:
    target_node_feats = batch["target_node_feats"]
    weak_slot_features = batch["weak_slot_features"]
    weak_relation_hints = batch["weak_relation_hints"]
    weak_pair_support_hints = batch.get("weak_pair_support_hints")
    weak_signed_pair_witness = batch.get("weak_signed_pair_witness")
    weak_pair_evidence_bundle = batch.get("weak_pair_evidence_bundle")
    variants = batch.get("step30_observation_variant", ["unknown"] * target_node_feats.shape[0])
    edge_threshold_value = threshold_for_batch_variants(
        variants=variants,
        default_threshold=edge_threshold,
        thresholds_by_variant=edge_thresholds_by_variant,
        device=target_node_feats.device,
    )
    num_node_types = int(encoder_model.config.num_node_types)
    state_dim = int(encoder_model.config.state_dim)

    encoder_outputs = encoder_model(
        weak_slot_features,
        weak_relation_hints,
        weak_pair_support_hints=weak_pair_support_hints,
        weak_signed_pair_witness=weak_signed_pair_witness,
        weak_pair_evidence_bundle=weak_pair_evidence_bundle,
    )
    encoder_type = encoder_outputs["type_logits"].argmax(dim=-1).float().unsqueeze(-1)
    encoder_state = encoder_outputs["state_pred"]
    if decode_mode == "selective_rescue":
        encoder_adj = hard_adj_selective_rescue(
            edge_logits=encoder_outputs["edge_logits"],
            relation_hints=weak_relation_hints,
            pair_support_hints=weak_pair_support_hints,
            node_mask=batch["node_mask"],
            variants=variants,
            base_threshold=edge_threshold_value,
            rescue_variants=rescue_variants or {"noisy"},
            rescue_relation_max=rescue_relation_max,
            rescue_support_min=rescue_support_min,
            rescue_budget_fraction=rescue_budget_fraction,
            rescue_score_mode=rescue_score_mode,
            rescue_support_weight=rescue_support_weight,
            rescue_relation_weight=rescue_relation_weight,
            rescue_safety_scorer=rescue_safety_scorer,
            node_latents=encoder_outputs.get("node_latents"),
            rescue_aux_logits=encoder_outputs.get("rescue_safety_logits"),
            rich_rescue_scorer=rich_rescue_scorer,
        )
    else:
        encoder_adj = hard_adj_from_logits(encoder_outputs["edge_logits"], threshold=edge_threshold_value)

    type_hint = weak_slot_features[:, :, :num_node_types]
    trivial_type = type_hint.argmax(dim=-1).float().unsqueeze(-1)
    state_start = num_node_types + 1
    trivial_state = weak_slot_features[:, :, state_start : state_start + state_dim]
    trivial_edge_scores = weak_relation_hints
    if weak_pair_support_hints is not None:
        trivial_edge_scores = 0.5 * (weak_relation_hints + weak_pair_support_hints)
    if weak_signed_pair_witness is not None:
        witness_scores = (0.5 + 0.5 * weak_signed_pair_witness).clamp(0.0, 1.0)
        trivial_edge_scores = 0.5 * (trivial_edge_scores + witness_scores)
    if weak_pair_evidence_bundle is not None:
        bundle_scores = (
            0.35 * weak_pair_evidence_bundle[..., 0]
            + 0.30 * (1.0 - weak_pair_evidence_bundle[..., 1])
            + 0.20 * weak_pair_evidence_bundle[..., 2]
            + 0.15 * weak_pair_evidence_bundle[..., 3]
        ).clamp(0.0, 1.0)
        trivial_edge_scores = 0.5 * trivial_edge_scores + 0.5 * bundle_scores
    if decode_mode == "selective_rescue":
        trivial_adj = hard_adj_selective_rescue(
            edge_logits=logit_from_scores(trivial_edge_scores),
            relation_hints=weak_relation_hints,
            pair_support_hints=weak_pair_support_hints,
            node_mask=batch["node_mask"],
            variants=variants,
            base_threshold=edge_threshold_value,
            rescue_variants=rescue_variants or {"noisy"},
            rescue_relation_max=rescue_relation_max,
            rescue_support_min=rescue_support_min,
            rescue_budget_fraction=rescue_budget_fraction,
            rescue_score_mode="raw" if rescue_score_mode in {"rich_learned", "aux"} else rescue_score_mode,
            rescue_support_weight=rescue_support_weight,
            rescue_relation_weight=rescue_relation_weight,
            rescue_safety_scorer=rescue_safety_scorer,
        )
    else:
        trivial_adj = hard_adj_from_scores(trivial_edge_scores, threshold=edge_threshold_value)

    return {
        "gt_structured": {
            "node_feats": target_node_feats,
            "adj": batch["target_adj"],
        },
        "encoder_recovered": {
            "node_feats": torch.cat([encoder_type, encoder_state], dim=-1),
            "adj": encoder_adj,
        },
        "trivial_recovered": {
            "node_feats": torch.cat([trivial_type, trivial_state], dim=-1),
            "adj": trivial_adj,
        },
    }


@torch.no_grad()
def recovery_metrics_for_decoded(
    decoded_node_feats: torch.Tensor,
    decoded_adj: torch.Tensor,
    target_node_feats: torch.Tensor,
    target_adj: torch.Tensor,
    node_mask: torch.Tensor,
) -> tuple[list[Dict[str, Any]], Counter[str]]:
    pair_mask = build_pair_mask(node_mask).bool()
    decoded_type = decoded_node_feats[:, :, 0].long()
    target_type = target_node_feats[:, :, 0].long()
    decoded_state = decoded_node_feats[:, :, 1:]
    target_state = target_node_feats[:, :, 1:]
    decoded_adj_bool = decoded_adj > 0.5
    target_adj_bool = target_adj > 0.5
    records: list[Dict[str, Any]] = []
    type_confusion: Counter[str] = Counter()

    for idx in range(decoded_node_feats.shape[0]):
        node_mask_i = node_mask[idx].bool()
        pair_mask_i = pair_mask[idx]
        edge_fp = (decoded_adj_bool[idx] & (~target_adj_bool[idx]) & pair_mask_i).float().sum().item()
        edge_fn = ((~decoded_adj_bool[idx]) & target_adj_bool[idx] & pair_mask_i).float().sum().item()
        edge_tp = (decoded_adj_bool[idx] & target_adj_bool[idx] & pair_mask_i).float().sum().item()
        edge_pred_pos = (decoded_adj_bool[idx] & pair_mask_i).float().sum().item()
        edge_true_pos = (target_adj_bool[idx] & pair_mask_i).float().sum().item()
        edge_precision = safe_div(edge_tp, edge_pred_pos)
        edge_recall = safe_div(edge_tp, edge_true_pos)
        if edge_precision is None or edge_recall is None or edge_precision + edge_recall <= 0:
            edge_f1 = None
        else:
            edge_f1 = 2.0 * edge_precision * edge_recall / (edge_precision + edge_recall)
        state_abs = (decoded_state[idx] - target_state[idx]).abs() * node_mask_i.unsqueeze(-1).float()
        state_sq = ((decoded_state[idx] - target_state[idx]) ** 2) * node_mask_i.unsqueeze(-1).float()
        state_total = node_mask_i.float().sum().item() * decoded_state.shape[-1]
        type_acc = accuracy_from_mask(decoded_type[idx], target_type[idx], node_mask_i)
        for true_t, pred_t in zip(target_type[idx][node_mask_i].tolist(), decoded_type[idx][node_mask_i].tolist()):
            type_confusion[f"{int(true_t)}->{int(pred_t)}"] += 1
        records.append(
            {
                "recovery_node_type_accuracy": type_acc,
                "recovery_node_state_mae": state_abs.sum().item() / state_total if state_total > 0 else None,
                "recovery_node_state_mse": state_sq.sum().item() / state_total if state_total > 0 else None,
                "recovery_edge_accuracy": accuracy_from_mask(decoded_adj_bool[idx], target_adj_bool[idx], pair_mask_i),
                "recovery_edge_precision": edge_precision,
                "recovery_edge_recall": edge_recall,
                "recovery_edge_f1": edge_f1,
                "decoded_edge_density": safe_div(edge_pred_pos, pair_mask_i.float().sum().item()),
                "gt_edge_density": safe_div(edge_true_pos, pair_mask_i.float().sum().item()),
                "edge_fp_count": edge_fp,
                "edge_fn_count": edge_fn,
            }
        )
    return records, type_confusion


def quality_label(value: Optional[float], high: float, mid: float, invert: bool = False) -> str:
    if value is None:
        return "na"
    score = -value if invert else value
    high_cmp = -high if invert else high
    mid_cmp = -mid if invert else mid
    if score >= high_cmp:
        return "high"
    if score >= mid_cmp:
        return "mid"
    return "low"


def recovery_quality_bucket(record: Dict[str, Any]) -> str:
    node_bucket = quality_label(record.get("recovery_node_type_accuracy"), high=0.90, mid=0.70)
    edge_bucket = quality_label(record.get("recovery_edge_f1"), high=0.80, mid=0.50)
    return f"node_{node_bucket}__edge_{edge_bucket}"


def event_families(events_item: Any) -> list[str]:
    if not isinstance(events_item, list):
        return ["unknown"]
    families = sorted({str(event.get("event_type", "unknown")) for event in events_item if isinstance(event, dict)})
    return families or ["unknown"]


@torch.no_grad()
def backend_forward_records(
    backend: BackendSpec,
    input_node_feats: torch.Tensor,
    input_adj: torch.Tensor,
    batch: Dict[str, Any],
    recovery_records: list[Dict[str, Any]],
) -> list[Dict[str, Any]]:
    node_mask = batch["node_mask"].bool()
    valid_edge_mask = build_pair_mask(batch["node_mask"]).bool()
    target_next_node_feats = batch["next_node_feats"]
    target_next_adj = batch["next_adj"]
    clean_current_node_feats = batch["target_node_feats"]
    clean_current_adj = batch["target_adj"]

    proposal_outputs = backend.proposal_model(input_node_feats, input_adj)
    node_scope_logits = proposal_outputs.get("node_scope_logits", proposal_outputs["scope_logits"])
    proposal_node_probs = torch.sigmoid(node_scope_logits) * batch["node_mask"]
    pred_scope_nodes = (proposal_node_probs >= backend.node_threshold) & node_mask
    if "edge_scope_logits" in proposal_outputs:
        proposal_edge_probs = torch.sigmoid(proposal_outputs["edge_scope_logits"]) * valid_edge_mask.float()
        pred_scope_edges = (proposal_edge_probs >= backend.edge_threshold) & valid_edge_mask
    else:
        proposal_edge_probs = proposal_node_probs.unsqueeze(2) * proposal_node_probs.unsqueeze(1) * valid_edge_mask.float()
        pred_scope_edges = pred_scope_nodes.unsqueeze(2) & pred_scope_nodes.unsqueeze(1) & valid_edge_mask

    rewrite_outputs = backend.rewrite_model(
        node_feats=input_node_feats,
        adj=input_adj,
        scope_node_mask=pred_scope_nodes.float(),
        scope_edge_mask=pred_scope_edges.float(),
        proposal_node_probs=proposal_node_probs if backend.use_proposal_conditioning else None,
        proposal_edge_probs=proposal_edge_probs if backend.use_proposal_conditioning else None,
    )

    pred_type = rewrite_outputs["type_logits_full"].argmax(dim=-1)
    pred_state = rewrite_outputs["state_pred_full"]
    pred_adj = hard_adj_from_logits(rewrite_outputs["edge_logits_full"], threshold=0.5)
    pred_adj_bool = pred_adj > 0.5
    target_adj_bool = target_next_adj > 0.5
    target_type = target_next_node_feats[:, :, 0].long()
    target_state = target_next_node_feats[:, :, 1:]
    changed_nodes = batch.get("changed_nodes", torch.any((clean_current_node_feats - target_next_node_feats).abs() > 1e-6, dim=-1)).bool()
    changed_edges = batch.get("changed_edges", (clean_current_adj != target_next_adj).float()).bool() & valid_edge_mask
    context_edges = (~changed_edges) & valid_edge_mask
    target_delta = build_edge_delta_targets(clean_current_adj, target_next_adj)
    pred_delta = build_edge_delta_targets(clean_current_adj, pred_adj)
    oracle_node_scope = batch.get("event_scope_union_nodes", torch.zeros_like(batch["node_mask"])).bool() & node_mask
    oracle_edge_scope = batch.get("event_scope_union_edges", torch.zeros_like(valid_edge_mask.float())).bool() & valid_edge_mask

    records = []
    for idx in range(input_node_feats.shape[0]):
        node_counts = binary_scope_counts(pred_scope_nodes[idx], oracle_node_scope[idx], node_mask[idx])
        edge_counts = binary_scope_counts(pred_scope_edges[idx], oracle_edge_scope[idx], valid_edge_mask[idx])
        changed_counts = binary_scope_counts(pred_scope_edges[idx], changed_edges[idx], valid_edge_mask[idx])
        record = {
            "observation_variant": str(batch.get("step30_observation_variant", ["unknown"] * input_node_feats.shape[0])[idx]),
            "events": batch.get("events", [None] * input_node_feats.shape[0])[idx],
            "full_edge": accuracy_from_mask(pred_adj_bool[idx], target_adj_bool[idx], valid_edge_mask[idx]),
            "context_edge": accuracy_from_mask(pred_adj_bool[idx], target_adj_bool[idx], context_edges[idx]),
            "changed_edge": accuracy_from_mask(pred_adj_bool[idx], target_adj_bool[idx], changed_edges[idx]),
            "add": accuracy_from_mask(
                pred_delta[idx],
                target_delta[idx],
                valid_edge_mask[idx] & (target_delta[idx] == EDGE_DELTA_ADD),
            ),
            "delete": accuracy_from_mask(
                pred_delta[idx],
                target_delta[idx],
                valid_edge_mask[idx] & (target_delta[idx] == EDGE_DELTA_DELETE),
            ),
            "keep": accuracy_from_mask(
                pred_delta[idx],
                target_delta[idx],
                valid_edge_mask[idx] & (target_delta[idx] == EDGE_DELTA_KEEP),
            ),
            "full_type": accuracy_from_mask(pred_type[idx], target_type[idx], node_mask[idx]),
            "full_state_mae": mae_from_mask(pred_state[idx], target_state[idx], node_mask[idx]),
            "proposal_node_scope_tp": node_counts["tp"],
            "proposal_node_scope_pred_pos": node_counts["pred_pos"],
            "proposal_node_scope_true_pos": node_counts["true_pos"],
            "proposal_node_scope_excess": node_counts["excess"],
            "proposal_edge_scope_tp": edge_counts["tp"],
            "proposal_edge_scope_pred_pos": edge_counts["pred_pos"],
            "proposal_edge_scope_true_pos": edge_counts["true_pos"],
            "proposal_edge_scope_excess": edge_counts["excess"],
            "proposal_changed_edge_tp": changed_counts["tp"],
            "proposal_changed_edge_true_pos": changed_counts["true_pos"],
            **recovery_records[idx],
        }
        record["recovery_quality_bucket"] = recovery_quality_bucket(record)
        records.append(record)
    return records


def summarize_records(records: list[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"count": len(records)}
    for metric in SYSTEM_METRICS + RECOVERY_METRICS:
        values = [float(record[metric]) for record in records if record.get(metric) is not None]
        summary[metric] = sum(values) / len(values) if values else None

    node_tp = sum(float(record.get("proposal_node_scope_tp", 0.0) or 0.0) for record in records)
    node_pred = sum(float(record.get("proposal_node_scope_pred_pos", 0.0) or 0.0) for record in records)
    node_true = sum(float(record.get("proposal_node_scope_true_pos", 0.0) or 0.0) for record in records)
    node_excess = sum(float(record.get("proposal_node_scope_excess", 0.0) or 0.0) for record in records)
    edge_tp = sum(float(record.get("proposal_edge_scope_tp", 0.0) or 0.0) for record in records)
    edge_pred = sum(float(record.get("proposal_edge_scope_pred_pos", 0.0) or 0.0) for record in records)
    edge_true = sum(float(record.get("proposal_edge_scope_true_pos", 0.0) or 0.0) for record in records)
    edge_excess = sum(float(record.get("proposal_edge_scope_excess", 0.0) or 0.0) for record in records)
    changed_tp = sum(float(record.get("proposal_changed_edge_tp", 0.0) or 0.0) for record in records)
    changed_true = sum(float(record.get("proposal_changed_edge_true_pos", 0.0) or 0.0) for record in records)
    summary["proposal_node_recall"] = safe_div(node_tp, node_true)
    summary["proposal_node_precision"] = safe_div(node_tp, node_pred)
    summary["proposal_node_scope_excess_ratio"] = safe_div(node_excess, node_pred)
    summary["proposal_edge_recall"] = safe_div(edge_tp, edge_true)
    summary["proposal_edge_precision"] = safe_div(edge_tp, edge_pred)
    summary["proposal_edge_scope_excess_ratio"] = safe_div(edge_excess, edge_pred)
    summary["proposal_changed_edge_recall"] = safe_div(changed_tp, changed_true)
    changed_recall = summary["proposal_changed_edge_recall"]
    summary["out_of_scope_miss"] = None if changed_recall is None else 1.0 - changed_recall
    return summary


def grouped_summaries(records: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    groups: Dict[tuple[str, str, str, str, str], list[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        base = (
            str(record["backend"]),
            str(record["input_mode"]),
            str(record["observation_variant"]),
        )
        groups[(*base, "overall", "all")].append(record)
        for event_family in event_families(record.get("events")):
            groups[(*base, "event_family", event_family)].append(record)
        groups[(*base, "recovery_quality_bucket", str(record["recovery_quality_bucket"]))].append(record)

    rows = []
    for (backend, input_mode, obs_variant, group_type, group_name), group_records in sorted(groups.items()):
        rows.append(
            {
                "backend": backend,
                "input_mode": input_mode,
                "observation_variant": obs_variant,
                "group_type": group_type,
                "group_name": group_name,
                **summarize_records(group_records),
            }
        )
    return rows


def add_deltas(rows: list[Dict[str, Any]]) -> None:
    gt_lookup = {
        (row["backend"], row["observation_variant"], row["group_type"], row["group_name"]): row
        for row in rows
        if row["input_mode"] == "gt_structured"
    }
    for row in rows:
        gt_row = gt_lookup.get(
            (row["backend"], row["observation_variant"], row["group_type"], row["group_name"])
        )
        for metric in DELTA_METRICS:
            key = f"delta_{metric}_vs_gt"
            if row["input_mode"] == "gt_structured" or gt_row is None:
                row[key] = 0.0 if row["input_mode"] == "gt_structured" else None
                continue
            current_value = row.get(metric)
            gt_value = gt_row.get(metric)
            row[key] = None if current_value is None or gt_value is None else current_value - gt_value


def clean_json_numbers(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: clean_json_numbers(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [clean_json_numbers(value) for value in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def write_csv(path: Path, rows: list[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def update_confusion(
    target: Dict[str, Counter[str]],
    key: str,
    confusion: Counter[str],
) -> None:
    target.setdefault(key, Counter())
    target[key].update(confusion)


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    device = get_device(args.device)
    data_path = resolve_path(args.data_path)
    encoder_path = resolve_path(args.encoder_checkpoint_path)
    recovered_thresholds_by_variant = parse_variant_thresholds(args.recovered_edge_thresholds_by_variant)
    rescue_variants = {part.strip() for part in args.rescue_variants.split(",") if part.strip()}
    dataset = Step30WeakObservationDataset(data_path)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=step30_weak_observation_collate_fn,
        pin_memory=device.type == "cuda",
    )
    encoder_model = load_encoder(encoder_path, device)
    rescue_safety_scorer = load_rescue_safety_scorer(args.rescue_safety_scorer_path)
    rich_rescue_scorer = load_rich_rescue_scorer(args.rich_rescue_scorer_path)
    backends = build_backend_specs(args, device)

    all_records: list[Dict[str, Any]] = []
    type_confusion_by_mode_variant: Dict[str, Counter[str]] = {}

    for batch_idx, batch in enumerate(loader):
        if args.limit_batches is not None and batch_idx >= args.limit_batches:
            break
        batch = move_batch_to_device(batch, device)
        decoded_inputs = decode_inputs(
            batch=batch,
            encoder_model=encoder_model,
            edge_threshold=args.recovered_edge_threshold,
            edge_thresholds_by_variant=recovered_thresholds_by_variant,
            decode_mode=args.decode_mode,
            rescue_variants=rescue_variants,
            rescue_relation_max=args.rescue_relation_max,
            rescue_support_min=args.rescue_support_min,
            rescue_budget_fraction=args.rescue_budget_fraction,
            rescue_score_mode=args.rescue_score_mode,
            rescue_support_weight=args.rescue_support_weight,
            rescue_relation_weight=args.rescue_relation_weight,
            rescue_safety_scorer=rescue_safety_scorer,
            rich_rescue_scorer=rich_rescue_scorer,
        )
        recovery_by_mode: Dict[str, list[Dict[str, Any]]] = {}
        for input_mode, decoded in decoded_inputs.items():
            recovery_records, confusion = recovery_metrics_for_decoded(
                decoded_node_feats=decoded["node_feats"],
                decoded_adj=decoded["adj"],
                target_node_feats=batch["target_node_feats"],
                target_adj=batch["target_adj"],
                node_mask=batch["node_mask"],
            )
            recovery_by_mode[input_mode] = recovery_records
            variants = batch.get("step30_observation_variant", ["unknown"] * len(recovery_records))
            for variant in sorted(set(str(v) for v in variants)):
                idxs = [i for i, v in enumerate(variants) if str(v) == variant]
                variant_confusion = Counter()
                decoded_type = decoded["node_feats"][:, :, 0].long()
                target_type = batch["target_node_feats"][:, :, 0].long()
                for idx in idxs:
                    mask = batch["node_mask"][idx].bool()
                    for true_t, pred_t in zip(target_type[idx][mask].tolist(), decoded_type[idx][mask].tolist()):
                        variant_confusion[f"{int(true_t)}->{int(pred_t)}"] += 1
                update_confusion(type_confusion_by_mode_variant, f"{input_mode}|{variant}", variant_confusion)

        for backend in backends:
            for input_mode, decoded in decoded_inputs.items():
                records = backend_forward_records(
                    backend=backend,
                    input_node_feats=decoded["node_feats"],
                    input_adj=decoded["adj"],
                    batch=batch,
                    recovery_records=recovery_by_mode[input_mode],
                )
                for record in records:
                    record["backend"] = backend.name
                    record["input_mode"] = input_mode
                all_records.extend(records)

    rows = grouped_summaries(all_records)
    add_deltas(rows)
    payload = {
        "metadata": {
            "data_path": str(data_path),
            "encoder_checkpoint_path": str(encoder_path),
            "batch_size": args.batch_size,
            "limit_batches": args.limit_batches,
            "recovered_edge_threshold": args.recovered_edge_threshold,
            "recovered_edge_thresholds_by_variant": recovered_thresholds_by_variant,
            "decode_mode": args.decode_mode,
            "rescue_variants": sorted(rescue_variants),
            "rescue_relation_max": args.rescue_relation_max,
            "rescue_support_min": args.rescue_support_min,
            "rescue_budget_fraction": args.rescue_budget_fraction,
            "rescue_score_mode": args.rescue_score_mode,
            "rescue_support_weight": args.rescue_support_weight,
            "rescue_relation_weight": args.rescue_relation_weight,
            "rescue_safety_scorer_path": args.rescue_safety_scorer_path,
            "rich_rescue_scorer_path": args.rich_rescue_scorer_path,
            "input_modes": ["gt_structured", "encoder_recovered", "trivial_recovered"],
            "backend_specs": [
                {
                    "name": backend.name,
                    "proposal_checkpoint_path": str(backend.proposal_checkpoint_path),
                    "rewrite_checkpoint_path": str(backend.rewrite_checkpoint_path),
                    "node_threshold": backend.node_threshold,
                    "edge_threshold": backend.edge_threshold,
                    "rewrite_uses_proposal_conditioning": backend.use_proposal_conditioning,
                }
                for backend in backends
            ],
            "notes": (
                "Step30c uses hard decoded graph_t_hat only. No soft adjacency, confidence-aware "
                "backend change, or end-to-end training is used."
            ),
        },
        "summary_rows": rows,
        "type_confusion_by_input_mode_variant": {
            key: dict(sorted(counter.items())) for key, counter in sorted(type_confusion_by_mode_variant.items())
        },
    }
    return clean_json_numbers(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--encoder_checkpoint_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="artifacts/step30_frozen_backend_integration")
    parser.add_argument("--backends", type=str, default="w012,rft1_p2")
    parser.add_argument("--clean_proposal_checkpoint_path", type=str, default="checkpoints/scope_proposal_node_edge_flipw2/best.pt")
    parser.add_argument("--w012_checkpoint_path", type=str, default="checkpoints/fp_keep_w012/best.pt")
    parser.add_argument("--noisy_proposal_checkpoint_path", type=str, default="checkpoints/proposal_noisy_obs_p2/best.pt")
    parser.add_argument("--rft1_checkpoint_path", type=str, default="checkpoints/step6_noisy_rewrite_rft1/best.pt")
    parser.add_argument("--clean_node_threshold", type=float, default=0.20)
    parser.add_argument("--clean_edge_threshold", type=float, default=0.15)
    parser.add_argument("--noisy_node_threshold", type=float, default=0.15)
    parser.add_argument("--noisy_edge_threshold", type=float, default=0.10)
    parser.add_argument("--recovered_edge_threshold", type=float, default=0.5)
    parser.add_argument(
        "--recovered_edge_thresholds_by_variant",
        type=str,
        default=None,
        help="Optional comma-separated thresholds such as default:0.5,clean:0.5,noisy:0.55.",
    )
    parser.add_argument("--decode_mode", choices=["threshold", "selective_rescue"], default="threshold")
    parser.add_argument("--rescue_variants", type=str, default="noisy")
    parser.add_argument("--rescue_relation_max", type=float, default=0.5)
    parser.add_argument("--rescue_support_min", type=float, default=0.55)
    parser.add_argument("--rescue_budget_fraction", type=float, default=0.0)
    parser.add_argument("--rescue_score_mode", choices=["raw", "guarded", "learned", "rich_learned", "aux"], default="raw")
    parser.add_argument("--rescue_support_weight", type=float, default=0.5)
    parser.add_argument("--rescue_relation_weight", type=float, default=0.25)
    parser.add_argument("--rescue_safety_scorer_path", type=str, default=None)
    parser.add_argument("--rich_rescue_scorer_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--limit_batches", type=int, default=None)
    args = parser.parse_args()

    payload = evaluate(args)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "summary.json"
    csv_path = output_dir / "summary.csv"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    write_csv(csv_path, payload["summary_rows"])
    print(f"wrote JSON: {json_path}")
    print(f"wrote CSV: {csv_path}")
    overall_rows = [
        row for row in payload["summary_rows"]
        if row["group_type"] == "overall" and row["group_name"] == "all"
    ]
    print(json.dumps(overall_rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
