from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch

RESCUE_SAFETY_FEATURE_NAMES = [
    "edge_score",
    "edge_logit",
    "relation_hint",
    "pair_support_hint",
    "edge_margin",
    "support_minus_relation",
    "support_margin",
    "relation_gap",
    "edge_score_times_support",
    "edge_score_times_relation_gap",
]


def load_rescue_safety_scorer(path: str | Path | None) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        scorer = json.load(f)
    if list(scorer.get("feature_names", [])) != RESCUE_SAFETY_FEATURE_NAMES:
        raise ValueError(f"Unexpected rescue safety scorer feature_names in {path}")
    return scorer


def load_rich_rescue_scorer(path: str | Path | None) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    scorer = torch.load(path, map_location="cpu")
    if scorer.get("model_type") != "tiny_mlp":
        raise ValueError(f"Unexpected rich rescue scorer model_type in {path}")
    return scorer


def parse_variant_thresholds(spec: Optional[str]) -> Optional[Dict[str, float]]:
    if not spec:
        return None
    thresholds: Dict[str, float] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError("variant thresholds must look like default:0.5,clean:0.5,noisy:0.55")
        key, value = part.split(":", 1)
        thresholds[key.strip()] = float(value)
    return thresholds


def threshold_for_variant(edge_threshold: Any, variant: str) -> float:
    if isinstance(edge_threshold, dict):
        return float(edge_threshold.get(variant, edge_threshold.get("default", 0.5)))
    return float(edge_threshold)


def threshold_tensor_for_variants(
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


def hard_adj_from_scores(edge_scores: torch.Tensor, threshold: float | torch.Tensor = 0.5) -> torch.Tensor:
    pred_adj = (edge_scores >= threshold).float()
    pred_adj = ((pred_adj + pred_adj.transpose(1, 2)) > 0.5).float()
    diag_mask = torch.eye(pred_adj.shape[1], device=pred_adj.device, dtype=torch.bool).unsqueeze(0)
    return pred_adj.masked_fill(diag_mask, 0.0)


def hard_adj_from_logits(edge_logits: torch.Tensor, threshold: float | torch.Tensor = 0.5) -> torch.Tensor:
    return hard_adj_from_scores(torch.sigmoid(edge_logits), threshold=threshold)


def upper_pair_mask(node_mask: torch.Tensor) -> torch.Tensor:
    valid = node_mask.bool()
    pair_mask = valid.unsqueeze(2) & valid.unsqueeze(1)
    num_nodes = node_mask.shape[1]
    upper = torch.triu(
        torch.ones(num_nodes, num_nodes, device=node_mask.device, dtype=torch.bool),
        diagonal=1,
    )
    return pair_mask & upper.unsqueeze(0)


def rescue_safety_feature_tensor(
    edge_logits: torch.Tensor,
    relation_hints: torch.Tensor,
    pair_support_hints: torch.Tensor,
    base_threshold: float | torch.Tensor,
    rescue_relation_max: float,
    rescue_support_min: float,
) -> torch.Tensor:
    edge_scores = torch.sigmoid(edge_logits)
    threshold = base_threshold
    if not isinstance(threshold, torch.Tensor):
        threshold = torch.tensor(float(threshold), device=edge_scores.device, dtype=edge_scores.dtype)
    threshold = threshold.to(device=edge_scores.device, dtype=edge_scores.dtype)
    threshold = torch.zeros_like(edge_scores) + threshold
    edge_logit = edge_logits.clamp(-10.0, 10.0)
    relation_gap = float(rescue_relation_max) - relation_hints
    support_margin = pair_support_hints - float(rescue_support_min)
    return torch.stack(
        [
            edge_scores,
            edge_logit,
            relation_hints,
            pair_support_hints,
            edge_scores - threshold,
            pair_support_hints - relation_hints,
            support_margin,
            relation_gap,
            edge_scores * pair_support_hints,
            edge_scores * relation_gap,
        ],
        dim=-1,
    )


def rescue_safety_score_matrix(
    edge_logits: torch.Tensor,
    relation_hints: torch.Tensor,
    pair_support_hints: torch.Tensor,
    base_threshold: float | torch.Tensor,
    rescue_relation_max: float,
    rescue_support_min: float,
    rescue_safety_scorer: Dict[str, Any],
) -> torch.Tensor:
    feature_names = list(rescue_safety_scorer.get("feature_names", []))
    if feature_names != RESCUE_SAFETY_FEATURE_NAMES:
        raise ValueError(
            "rescue safety scorer feature_names do not match current Step30 rev10 features"
        )
    features = rescue_safety_feature_tensor(
        edge_logits=edge_logits,
        relation_hints=relation_hints,
        pair_support_hints=pair_support_hints,
        base_threshold=base_threshold,
        rescue_relation_max=rescue_relation_max,
        rescue_support_min=rescue_support_min,
    )
    dtype = features.dtype
    device = features.device
    mean = torch.tensor(rescue_safety_scorer["feature_mean"], dtype=dtype, device=device)
    std = torch.tensor(rescue_safety_scorer["feature_std"], dtype=dtype, device=device).clamp_min(1e-6)
    weights = torch.tensor(rescue_safety_scorer["weights"], dtype=dtype, device=device)
    bias = torch.tensor(float(rescue_safety_scorer["bias"]), dtype=dtype, device=device)
    return ((features - mean) / std * weights).sum(dim=-1) + bias


def rich_rescue_feature_tensor(
    node_latents: torch.Tensor,
    edge_logits: torch.Tensor,
    relation_hints: torch.Tensor,
    pair_support_hints: torch.Tensor,
    node_mask: torch.Tensor,
    base_threshold: float | torch.Tensor,
    rescue_relation_max: float,
    rescue_support_min: float,
) -> torch.Tensor:
    batch_size, num_nodes, hidden_dim = node_latents.shape
    h_i = node_latents.unsqueeze(2).expand(batch_size, num_nodes, num_nodes, hidden_dim)
    h_j = node_latents.unsqueeze(1).expand(batch_size, num_nodes, num_nodes, hidden_dim)
    scalar_features = rescue_safety_feature_tensor(
        edge_logits=edge_logits,
        relation_hints=relation_hints,
        pair_support_hints=pair_support_hints,
        base_threshold=base_threshold,
        rescue_relation_max=rescue_relation_max,
        rescue_support_min=rescue_support_min,
    )
    edge_scores = torch.sigmoid(edge_logits)
    pair_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    diag_mask = torch.eye(num_nodes, device=node_latents.device, dtype=edge_scores.dtype).unsqueeze(0)
    pair_mask = pair_mask * (1.0 - diag_mask)
    relation_degree = (relation_hints * pair_mask).sum(dim=-1)
    support_degree = (pair_support_hints * pair_mask).sum(dim=-1)
    score_degree = (edge_scores * pair_mask).sum(dim=-1)
    relation_common = torch.matmul(relation_hints * pair_mask, relation_hints * pair_mask)
    support_common = torch.matmul(pair_support_hints * pair_mask, pair_support_hints * pair_mask)
    local_features = torch.stack(
        [
            relation_degree.unsqueeze(2).expand(batch_size, num_nodes, num_nodes),
            relation_degree.unsqueeze(1).expand(batch_size, num_nodes, num_nodes),
            support_degree.unsqueeze(2).expand(batch_size, num_nodes, num_nodes),
            support_degree.unsqueeze(1).expand(batch_size, num_nodes, num_nodes),
            score_degree.unsqueeze(2).expand(batch_size, num_nodes, num_nodes),
            score_degree.unsqueeze(1).expand(batch_size, num_nodes, num_nodes),
            relation_common,
            support_common,
        ],
        dim=-1,
    )
    return torch.cat(
        [
            scalar_features,
            h_i,
            h_j,
            torch.abs(h_i - h_j),
            h_i * h_j,
            local_features,
        ],
        dim=-1,
    )


def rich_rescue_score_matrix(
    node_latents: torch.Tensor,
    edge_logits: torch.Tensor,
    relation_hints: torch.Tensor,
    pair_support_hints: torch.Tensor,
    node_mask: torch.Tensor,
    base_threshold: float | torch.Tensor,
    rescue_relation_max: float,
    rescue_support_min: float,
    rich_rescue_scorer: Dict[str, Any],
) -> torch.Tensor:
    features = rich_rescue_feature_tensor(
        node_latents=node_latents,
        edge_logits=edge_logits,
        relation_hints=relation_hints,
        pair_support_hints=pair_support_hints,
        node_mask=node_mask,
        base_threshold=base_threshold,
        rescue_relation_max=rescue_relation_max,
        rescue_support_min=rescue_support_min,
    )
    dtype = features.dtype
    device = features.device
    mean = rich_rescue_scorer["feature_mean"].to(device=device, dtype=dtype)
    std = rich_rescue_scorer["feature_std"].to(device=device, dtype=dtype).clamp_min(1e-6)
    x = (features - mean) / std
    layers = rich_rescue_scorer["layers"]
    w1 = layers["linear1.weight"].to(device=device, dtype=dtype)
    b1 = layers["linear1.bias"].to(device=device, dtype=dtype)
    w2 = layers["linear2.weight"].to(device=device, dtype=dtype)
    b2 = layers["linear2.bias"].to(device=device, dtype=dtype)
    hidden = torch.relu(torch.matmul(x, w1.t()) + b1)
    return (torch.matmul(hidden, w2.t()) + b2).squeeze(-1)


def hard_adj_selective_rescue(
    edge_logits: torch.Tensor,
    relation_hints: torch.Tensor,
    pair_support_hints: Optional[torch.Tensor],
    node_mask: torch.Tensor,
    variants: Iterable[Any],
    base_threshold: float | torch.Tensor,
    rescue_variants: set[str],
    rescue_relation_max: float,
    rescue_support_min: float,
    rescue_budget_fraction: float,
    rescue_score_mode: str = "raw",
    rescue_support_weight: float = 0.5,
    rescue_relation_weight: float = 0.25,
    rescue_safety_scorer: Optional[Dict[str, Any]] = None,
    node_latents: Optional[torch.Tensor] = None,
    rescue_aux_logits: Optional[torch.Tensor] = None,
    rich_rescue_scorer: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    edge_scores = torch.sigmoid(edge_logits)
    base_adj = hard_adj_from_scores(edge_scores, threshold=base_threshold).bool()
    if pair_support_hints is None or rescue_budget_fraction <= 0.0:
        return base_adj.float()

    upper_mask = upper_pair_mask(node_mask)
    rescue_adj = torch.zeros_like(base_adj)
    variant_list = [str(variant) for variant in variants]
    for batch_idx, variant in enumerate(variant_list):
        if variant not in rescue_variants:
            continue
        valid_upper = upper_mask[batch_idx]
        valid_pair_count = int(valid_upper.sum().item())
        if valid_pair_count <= 0:
            continue
        budget = int(math.ceil(valid_pair_count * float(rescue_budget_fraction)))
        if budget <= 0:
            continue
        candidate_mask = (
            valid_upper
            & (~base_adj[batch_idx])
            & (relation_hints[batch_idx] < float(rescue_relation_max))
            & (pair_support_hints[batch_idx] >= float(rescue_support_min))
        )
        candidate_idx = candidate_mask.nonzero(as_tuple=False)
        if candidate_idx.numel() == 0:
            continue
        if rescue_score_mode == "raw":
            score_matrix = edge_scores[batch_idx]
        elif rescue_score_mode == "guarded":
            support_bonus = pair_support_hints[batch_idx] - float(rescue_support_min)
            relation_bonus = float(rescue_relation_max) - relation_hints[batch_idx]
            score_matrix = (
                edge_scores[batch_idx]
                + float(rescue_support_weight) * support_bonus
                + float(rescue_relation_weight) * relation_bonus
            )
        elif rescue_score_mode == "learned":
            if rescue_safety_scorer is None:
                raise ValueError("rescue_score_mode='learned' requires rescue_safety_scorer")
            score_matrix = rescue_safety_score_matrix(
                edge_logits=edge_logits[batch_idx : batch_idx + 1],
                relation_hints=relation_hints[batch_idx : batch_idx + 1],
                pair_support_hints=pair_support_hints[batch_idx : batch_idx + 1],
                base_threshold=(
                    base_threshold[batch_idx : batch_idx + 1]
                    if isinstance(base_threshold, torch.Tensor) and base_threshold.ndim > 0
                    else base_threshold
                ),
                rescue_relation_max=rescue_relation_max,
                rescue_support_min=rescue_support_min,
                rescue_safety_scorer=rescue_safety_scorer,
            )[0]
        elif rescue_score_mode == "aux":
            if rescue_aux_logits is None:
                raise ValueError("rescue_score_mode='aux' requires rescue_aux_logits")
            score_matrix = rescue_aux_logits[batch_idx]
        elif rescue_score_mode == "rich_learned":
            if rich_rescue_scorer is None or node_latents is None:
                raise ValueError("rescue_score_mode='rich_learned' requires rich_rescue_scorer and node_latents")
            score_matrix = rich_rescue_score_matrix(
                node_latents=node_latents[batch_idx : batch_idx + 1],
                edge_logits=edge_logits[batch_idx : batch_idx + 1],
                relation_hints=relation_hints[batch_idx : batch_idx + 1],
                pair_support_hints=pair_support_hints[batch_idx : batch_idx + 1],
                node_mask=node_mask[batch_idx : batch_idx + 1],
                base_threshold=(
                    base_threshold[batch_idx : batch_idx + 1]
                    if isinstance(base_threshold, torch.Tensor) and base_threshold.ndim > 0
                    else base_threshold
                ),
                rescue_relation_max=rescue_relation_max,
                rescue_support_min=rescue_support_min,
                rich_rescue_scorer=rich_rescue_scorer,
            )[0]
        else:
            raise ValueError(f"Unknown rescue_score_mode: {rescue_score_mode}")
        candidate_scores = score_matrix[candidate_mask]
        top_k = min(budget, int(candidate_scores.numel()))
        selected = torch.topk(candidate_scores, k=top_k).indices
        selected_idx = candidate_idx[selected]
        rescue_adj[batch_idx, selected_idx[:, 0], selected_idx[:, 1]] = True
        rescue_adj[batch_idx, selected_idx[:, 1], selected_idx[:, 0]] = True

    pred_adj = base_adj | rescue_adj
    diag_mask = torch.eye(pred_adj.shape[1], device=pred_adj.device, dtype=torch.bool).unsqueeze(0)
    return pred_adj.masked_fill(diag_mask, False).float()
