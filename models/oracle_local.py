from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.baselines import SimpleGraphEncoder, build_mlp


@dataclass
class OracleLocalRewriteConfig:
    node_feat_dim: int
    num_node_types: int = 3
    type_dim: int = 1
    state_dim: int = 4
    hidden_dim: int = 128
    msg_pass_layers: int = 3
    node_mlp_layers: int = 2
    edge_mlp_layers: int = 2
    dropout: float = 0.0
    copy_logit_value: float = 10.0


def build_valid_edge_mask(node_mask: torch.Tensor) -> torch.Tensor:
    """
    node_mask: [B, N]
    return:    [B, N, N] with diagonal removed
    """
    bsz, num_nodes = node_mask.shape
    pair_mask = node_mask.unsqueeze(2) * node_mask.unsqueeze(1)
    diag = torch.eye(num_nodes, device=node_mask.device, dtype=pair_mask.dtype).unsqueeze(0)
    return pair_mask * (1.0 - diag)


def masked_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    mask = mask.unsqueeze(-1)
    sq_err = (pred - target) ** 2
    sq_err = sq_err * mask
    return sq_err.sum() / (mask.sum() * pred.shape[-1] + eps)


def masked_type_ce_loss(
    type_logits: torch.Tensor,
    target_type: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    type_logits: [B, N, C]
    target_type: [B, N] integer labels
    mask:        [B, N]
    """
    bsz, num_nodes, num_classes = type_logits.shape
    ce = F.cross_entropy(
        type_logits.reshape(bsz * num_nodes, num_classes),
        target_type.reshape(bsz * num_nodes),
        reduction="none",
    ).reshape(bsz, num_nodes)
    if weights is not None:
        ce = ce * weights
    ce = ce * mask
    if weights is not None:
        denom = (weights * mask).sum()
    else:
        denom = mask.sum()
    return ce.sum() / (denom + eps)


def masked_edge_bce_loss_from_pair_mask(
    edge_logits: torch.Tensor,
    target_adj: torch.Tensor,
    pair_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    edge_logits: [B, N, N]
    target_adj:  [B, N, N]
    pair_mask:   [B, N, N]
    """
    bce = F.binary_cross_entropy_with_logits(
        edge_logits,
        target_adj.float(),
        reduction="none",
    )
    bce = bce * pair_mask
    return bce.sum() / (pair_mask.sum() + eps)


class OracleLocalRewriteModel(nn.Module):
    """
    Stage 1 oracle local rewrite baseline.

    Design:
    - encode the full current graph
    - predict typed local patch candidates for all nodes/edges
    - train only on oracle local scope
    - merge local predictions back into a full-graph prediction by
      copying current-state values outside scope
    """

    def __init__(self, config: OracleLocalRewriteConfig):
        super().__init__()
        self.config = config

        self.encoder = SimpleGraphEncoder(
            node_feat_dim=config.node_feat_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.msg_pass_layers,
            dropout=config.dropout,
        )

        self.type_head = build_mlp(
            in_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            out_dim=config.num_node_types,
            num_layers=config.node_mlp_layers,
            dropout=config.dropout,
        )
        self.state_head = build_mlp(
            in_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            out_dim=config.state_dim,
            num_layers=config.node_mlp_layers,
            dropout=config.dropout,
        )

        edge_in_dim = config.hidden_dim * 4
        self.edge_head = build_mlp(
            in_dim=edge_in_dim,
            hidden_dim=config.hidden_dim,
            out_dim=1,
            num_layers=config.edge_mlp_layers,
            dropout=config.dropout,
        )

    def forward(
        self,
        node_feats: torch.Tensor,
        adj: torch.Tensor,
        scope_node_mask: torch.Tensor,
        scope_edge_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            node_feats:       [B, N, F]
            adj:              [B, N, N]
            scope_node_mask:  [B, N]
            scope_edge_mask:  [B, N, N]

        Returns:
            local predictions + merge-back full predictions
        """
        h = self.encoder(node_feats, adj)

        type_logits_local = self.type_head(h)          # [B, N, C]
        state_pred_local = self.state_head(h)          # [B, N, state_dim]
        edge_logits_local = self.predict_edges_from_nodes(h)  # [B, N, N]

        type_logits_full = self.merge_type_logits(
            node_feats=node_feats,
            type_logits_local=type_logits_local,
            scope_node_mask=scope_node_mask,
        )
        state_pred_full = self.merge_state_pred(
            node_feats=node_feats,
            state_pred_local=state_pred_local,
            scope_node_mask=scope_node_mask,
        )
        edge_logits_full = self.merge_edge_logits(
            adj=adj,
            edge_logits_local=edge_logits_local,
            scope_edge_mask=scope_edge_mask,
        )

        return {
            "node_latents": h,
            "type_logits_local": type_logits_local,
            "state_pred_local": state_pred_local,
            "edge_logits_local": edge_logits_local,
            "type_logits_full": type_logits_full,
            "state_pred_full": state_pred_full,
            "edge_logits_full": edge_logits_full,
        }

    def predict_edges_from_nodes(self, h: torch.Tensor) -> torch.Tensor:
        """
        Build pairwise edge logits from node latents.

        Args:
            h: [B, N, H]

        Returns:
            edge_logits: [B, N, N]
        """
        bsz, num_nodes, hidden_dim = h.shape

        h_i = h.unsqueeze(2).expand(bsz, num_nodes, num_nodes, hidden_dim)
        h_j = h.unsqueeze(1).expand(bsz, num_nodes, num_nodes, hidden_dim)

        pair_feat = torch.cat(
            [
                h_i,
                h_j,
                torch.abs(h_i - h_j),
                h_i * h_j,
            ],
            dim=-1,
        )

        edge_logits = self.edge_head(pair_feat).squeeze(-1)
        edge_logits = 0.5 * (edge_logits + edge_logits.transpose(1, 2))

        diag_mask = torch.eye(num_nodes, device=h.device, dtype=torch.bool).unsqueeze(0)
        edge_logits = edge_logits.masked_fill(diag_mask, -1e9)
        return edge_logits

    def make_copy_type_logits(self, node_feats: torch.Tensor) -> torch.Tensor:
        """
        Convert current node type ids into deterministic logits so that
        full-graph type evaluation can reuse the same metric functions.
        """
        current_type = node_feats[:, :, 0].long().clamp(min=0, max=self.config.num_node_types - 1)
        bsz, num_nodes = current_type.shape

        logits = torch.full(
            (bsz, num_nodes, self.config.num_node_types),
            fill_value=-self.config.copy_logit_value,
            device=node_feats.device,
            dtype=torch.float32,
        )
        logits.scatter_(2, current_type.unsqueeze(-1), self.config.copy_logit_value)
        return logits

    def make_copy_state_pred(self, node_feats: torch.Tensor) -> torch.Tensor:
        return node_feats[:, :, 1:]

    def make_copy_edge_logits(self, adj: torch.Tensor) -> torch.Tensor:
        logits = torch.where(
            adj > 0.5,
            torch.full_like(adj, self.config.copy_logit_value),
            torch.full_like(adj, -self.config.copy_logit_value),
        )

        num_nodes = adj.shape[1]
        diag_mask = torch.eye(num_nodes, device=adj.device, dtype=torch.bool).unsqueeze(0)
        logits = logits.masked_fill(diag_mask, -1e9)
        return logits

    def merge_type_logits(
        self,
        node_feats: torch.Tensor,
        type_logits_local: torch.Tensor,
        scope_node_mask: torch.Tensor,
    ) -> torch.Tensor:
        copy_type_logits = self.make_copy_type_logits(node_feats)
        scope_bool = scope_node_mask.unsqueeze(-1).bool()
        return torch.where(scope_bool, type_logits_local, copy_type_logits)

    def merge_state_pred(
        self,
        node_feats: torch.Tensor,
        state_pred_local: torch.Tensor,
        scope_node_mask: torch.Tensor,
    ) -> torch.Tensor:
        copy_state_pred = self.make_copy_state_pred(node_feats)
        scope = scope_node_mask.unsqueeze(-1)
        return scope * state_pred_local + (1.0 - scope) * copy_state_pred

    def merge_edge_logits(
        self,
        adj: torch.Tensor,
        edge_logits_local: torch.Tensor,
        scope_edge_mask: torch.Tensor,
    ) -> torch.Tensor:
        copy_edge_logits = self.make_copy_edge_logits(adj)
        edge_logits_full = torch.where(scope_edge_mask.bool(), edge_logits_local, copy_edge_logits)

        num_nodes = adj.shape[1]
        diag_mask = torch.eye(num_nodes, device=adj.device, dtype=torch.bool).unsqueeze(0)
        edge_logits_full = edge_logits_full.masked_fill(diag_mask, -1e9)
        return edge_logits_full


def oracle_local_rewrite_loss(
    outputs: Dict[str, torch.Tensor],
    current_node_feats: torch.Tensor,
    target_node_feats: torch.Tensor,
    target_adj: torch.Tensor,
    node_mask: torch.Tensor,
    scope_node_mask: torch.Tensor,
    scope_edge_mask: torch.Tensor,
    edge_loss_weight: float = 1.0,
    type_loss_weight: float = 1.0,
    state_loss_weight: float = 1.0,
    type_flip_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    Scope-only training loss.
    """
    current_type = current_node_feats[:, :, 0].long()
    target_type = target_node_feats[:, :, 0].long()
    target_state = target_node_feats[:, :, 1:]

    scoped_node_mask = node_mask * scope_node_mask
    scoped_edge_mask = build_valid_edge_mask(node_mask) * scope_edge_mask
    flip_target_mask = (current_type != target_type).to(outputs["type_logits_local"].dtype)
    type_weights = 1.0 + (type_flip_weight - 1.0) * flip_target_mask

    type_loss = masked_type_ce_loss(
        outputs["type_logits_local"],
        target_type,
        scoped_node_mask,
        weights=type_weights,
    )
    state_loss = masked_mse_loss(
        outputs["state_pred_local"],
        target_state,
        scoped_node_mask,
    )
    edge_loss = masked_edge_bce_loss_from_pair_mask(
        outputs["edge_logits_local"],
        target_adj,
        scoped_edge_mask,
    )

    total_loss = (
        type_loss_weight * type_loss
        + state_loss_weight * state_loss
        + edge_loss_weight * edge_loss
    )

    return {
        "total_loss": total_loss,
        "type_loss": type_loss,
        "state_loss": state_loss,
        "edge_loss": edge_loss,
    }


def oracle_full_prediction_loss(
    outputs: Dict[str, torch.Tensor],
    target_node_feats: torch.Tensor,
    target_adj: torch.Tensor,
    node_mask: torch.Tensor,
    edge_loss_weight: float = 1.0,
    type_loss_weight: float = 1.0,
    state_loss_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    Full-graph loss after merge-back.
    This is not the training objective, but it gives a directly comparable
    validation statistic against the typed global baseline.
    """
    target_type = target_node_feats[:, :, 0].long()
    target_state = target_node_feats[:, :, 1:]
    valid_edge_mask = build_valid_edge_mask(node_mask)

    type_loss = masked_type_ce_loss(
        outputs["type_logits_full"],
        target_type,
        node_mask,
    )
    state_loss = masked_mse_loss(
        outputs["state_pred_full"],
        target_state,
        node_mask,
    )
    edge_loss = masked_edge_bce_loss_from_pair_mask(
        outputs["edge_logits_full"],
        target_adj,
        valid_edge_mask,
    )

    total_loss = (
        type_loss_weight * type_loss
        + state_loss_weight * state_loss
        + edge_loss_weight * edge_loss
    )

    return {
        "total_loss": total_loss,
        "type_loss": type_loss,
        "state_loss": state_loss,
        "edge_loss": edge_loss,
    }
