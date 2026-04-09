# models/baselines.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------
# Utility
# ------------------------------------------------------------

def build_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    num_layers: int = 2,
    dropout: float = 0.0,
) -> nn.Sequential:
    """
    Simple MLP builder.

    Args:
        in_dim: input feature dimension
        hidden_dim: hidden layer dimension
        out_dim: output feature dimension
        num_layers: total number of linear layers
        dropout: dropout probability between hidden layers
    """
    assert num_layers >= 1

    layers = []
    if num_layers == 1:
        layers.append(nn.Linear(in_dim, out_dim))
        return nn.Sequential(*layers)

    layers.append(nn.Linear(in_dim, hidden_dim))
    layers.append(nn.ReLU())

    if dropout > 0:
        layers.append(nn.Dropout(dropout))

    for _ in range(num_layers - 2):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

@dataclass
class GlobalBaselineConfig:
    node_feat_dim: int
    num_node_types: int = 3
    type_dim: int = 1
    state_dim: int = 4
    hidden_dim: int = 128
    msg_pass_layers: int = 3
    node_mlp_layers: int = 2
    edge_mlp_layers: int = 2
    dropout: float = 0.0


# ------------------------------------------------------------
# Graph Encoder
# ------------------------------------------------------------

class GraphMessagePassingLayer(nn.Module):
    """
    A very lightweight message passing layer.

    Input:
        x   : [B, N, D]
        adj : [B, N, N]   (0/1 or weighted adjacency)

    Update rule:
        aggregated = adj @ x
        new_x = MLP([x, aggregated])

    This is intentionally simple for Stage 1.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.update = build_mlp(
            in_dim=hidden_dim * 2,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            num_layers=2,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x:   [B, N, D]
        # adj: [B, N, N]
        aggregated = torch.matmul(adj, x)  # [B, N, D]
        h = torch.cat([x, aggregated], dim=-1)
        h = self.update(h)
        h = self.norm(h + x)  # residual
        return h


class SimpleGraphEncoder(nn.Module):
    """
    Encode node features using repeated message passing.

    Output:
        node_latents: [B, N, H]
    """

    def __init__(
        self,
        node_feat_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_proj = nn.Linear(node_feat_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [
                GraphMessagePassingLayer(hidden_dim, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, node_feats: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_feats: [B, N, F]
            adj:        [B, N, N]

        Returns:
            node_latents: [B, N, H]
        """
        x = self.input_proj(node_feats)
        for layer in self.layers:
            x = layer(x, adj)
        return x


# ------------------------------------------------------------
# Global Transition Baseline
# ------------------------------------------------------------

class GlobalTransitionBaseline(nn.Module):
    """
    Whole-graph transition baseline.

    Given current graph state (node features + adjacency),
    predict:
        1) next node features
        2) next adjacency logits

    Stage 1 design choice:
    - Keep this baseline simple and strong enough to test learnability.
    - No explicit event proposal.
    - No explicit locality bottleneck.
    """

    def __init__(self, config: GlobalBaselineConfig):
        super().__init__()
        self.config = config

        self.encoder = SimpleGraphEncoder(
            node_feat_dim=config.node_feat_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.msg_pass_layers,
            dropout=config.dropout,
        )

        # Predict next node features from encoded node states

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
        # Predict next edge existence from pairwise node representations
        # pair feature = [h_i, h_j, |h_i-h_j|, h_i*h_j]
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
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            node_feats: [B, N, F]
            adj:        [B, N, N]

        Returns:
            dict with:
                node_pred       : [B, N, F]
                edge_logits     : [B, N, N]
                node_latents    : [B, N, H]
        """
        h = self.encoder(node_feats, adj)             # [B, N, H]
        type_logits = self.type_head(h)                 # [B, N, C]
        state_pred = self.state_head(h)                 # [B, N, state_dim]
        edge_logits = self.predict_edges_from_nodes(h)  # [B, N, N]

        return {
            "type_logits": type_logits,
            "state_pred": state_pred,
            "edge_logits": edge_logits,
            "node_latents": h,
        }
    
    def predict_edges_from_nodes(self, h: torch.Tensor) -> torch.Tensor:
        """
        Build pairwise edge logits from node latents.

        Args:
            h: [B, N, H]

        Returns:
            edge_logits: [B, N, N]
        """
        B, N, H = h.shape

        h_i = h.unsqueeze(2).expand(B, N, N, H)
        h_j = h.unsqueeze(1).expand(B, N, N, H)

        pair_feat = torch.cat(
            [
                h_i,
                h_j,
                torch.abs(h_i - h_j),
                h_i * h_j,
            ],
            dim=-1,
        )  # [B, N, N, 4H]

        edge_logits = self.edge_head(pair_feat).squeeze(-1)  # [B, N, N]

        # Optional: enforce symmetric edge logits for undirected graphs
        edge_logits = 0.5 * (edge_logits + edge_logits.transpose(1, 2))

        # Usually we don't want self-loops unless dataset includes them
        diag_mask = torch.eye(N, device=h.device, dtype=torch.bool).unsqueeze(0)
        edge_logits = edge_logits.masked_fill(diag_mask, -1e9)

        return edge_logits


# ------------------------------------------------------------
# Loss
# ------------------------------------------------------------

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


def masked_edge_bce_loss(
    edge_logits: torch.Tensor,
    target_adj: torch.Tensor,
    node_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    B, N, _ = edge_logits.shape

    pair_mask = node_mask.unsqueeze(2) * node_mask.unsqueeze(1)

    diag = torch.eye(N, device=edge_logits.device).unsqueeze(0)
    pair_mask = pair_mask * (1.0 - diag)

    bce = F.binary_cross_entropy_with_logits(
        edge_logits,
        target_adj.float(),
        reduction="none",
    )

    bce = bce * pair_mask
    return bce.sum() / (pair_mask.sum() + eps)


def masked_type_ce_loss(
    type_logits: torch.Tensor,
    target_type: torch.Tensor,
    node_mask: torch.Tensor,
    weights: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    type_logits: [B, N, C]
    target_type: [B, N]   integer labels
    node_mask:   [B, N]
    """
    B, N, C = type_logits.shape

    ce = F.cross_entropy(
        type_logits.reshape(B * N, C),
        target_type.reshape(B * N),
        reduction="none",
    ).reshape(B, N)

    if weights is not None:
        ce = ce * weights
    ce = ce * node_mask
    if weights is not None:
        denom = (weights * node_mask).sum()
    else:
        denom = node_mask.sum()
    return ce.sum() / (denom + eps)


def global_baseline_loss(
    outputs: Dict[str, torch.Tensor],
    current_node_feats: torch.Tensor,
    target_node_feats: torch.Tensor,
    target_adj: torch.Tensor,
    node_mask: torch.Tensor,
    edge_loss_weight: float = 1.0,
    type_loss_weight: float = 1.0,
    state_loss_weight: float = 1.0,
    type_flip_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    target_node_feats layout:
        [:, :, 0]   -> discrete type id
        [:, :, 1:]  -> continuous state
    """
    type_logits = outputs["type_logits"]   # [B, N, C]
    state_pred = outputs["state_pred"]     # [B, N, 4]
    edge_logits = outputs["edge_logits"]   # [B, N, N]

    current_type = current_node_feats[:, :, 0].long()
    target_type = target_node_feats[:, :, 0].long()
    target_state = target_node_feats[:, :, 1:]
    flip_target_mask = (current_type != target_type).to(type_logits.dtype)
    type_weights = 1.0 + (type_flip_weight - 1.0) * flip_target_mask

    type_loss = masked_type_ce_loss(type_logits, target_type, node_mask, weights=type_weights)
    state_loss = masked_mse_loss(state_pred, target_state, node_mask)
    edge_loss = masked_edge_bce_loss(edge_logits, target_adj, node_mask)

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
