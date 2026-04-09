from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.baselines import SimpleGraphEncoder, build_mlp


@dataclass
class ScopeProposalConfig:
    node_feat_dim: int
    hidden_dim: int = 128
    msg_pass_layers: int = 3
    head_layers: int = 2
    dropout: float = 0.0


def masked_scope_bce_loss(
    scope_logits: torch.Tensor,
    target_scope: torch.Tensor,
    node_mask: torch.Tensor,
    weights: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(
        scope_logits,
        target_scope.float(),
        reduction="none",
    )
    if weights is not None:
        bce = bce * weights
    bce = bce * node_mask
    if weights is not None:
        denom = (weights * node_mask).sum()
    else:
        denom = node_mask.sum()
    return bce.sum() / (denom + eps)


def masked_edge_scope_bce_loss(
    edge_scope_logits: torch.Tensor,
    target_edge_scope: torch.Tensor,
    pair_mask: torch.Tensor,
    pos_weight: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    pos_weight_tensor = torch.tensor(pos_weight, device=edge_scope_logits.device, dtype=edge_scope_logits.dtype)
    bce = F.binary_cross_entropy_with_logits(
        edge_scope_logits,
        target_edge_scope.float(),
        pos_weight=pos_weight_tensor,
        reduction="none",
    )
    bce = bce * pair_mask
    return bce.sum() / (pair_mask.sum() + eps)


class ScopeProposalModel(nn.Module):
    """
    Minimal node + edge scope proposal model for Stage 1.
    """

    def __init__(self, config: ScopeProposalConfig):
        super().__init__()
        self.config = config

        self.encoder = SimpleGraphEncoder(
            node_feat_dim=config.node_feat_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.msg_pass_layers,
            dropout=config.dropout,
        )
        self.scope_head = build_mlp(
            in_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            out_dim=1,
            num_layers=config.head_layers,
            dropout=config.dropout,
        )
        edge_in_dim = config.hidden_dim * 4
        self.edge_scope_head = build_mlp(
            in_dim=edge_in_dim,
            hidden_dim=config.hidden_dim,
            out_dim=1,
            num_layers=config.head_layers,
            dropout=config.dropout,
        )

    def forward(
        self,
        node_feats: torch.Tensor,
        adj: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        node_latents = self.encoder(node_feats, adj)
        node_scope_logits = self.scope_head(node_latents).squeeze(-1)

        bsz, num_nodes, hidden_dim = node_latents.shape
        h_i = node_latents.unsqueeze(2).expand(bsz, num_nodes, num_nodes, hidden_dim)
        h_j = node_latents.unsqueeze(1).expand(bsz, num_nodes, num_nodes, hidden_dim)
        edge_pair_feat = torch.cat(
            [
                h_i,
                h_j,
                torch.abs(h_i - h_j),
                h_i * h_j,
            ],
            dim=-1,
        )
        edge_scope_logits = self.edge_scope_head(edge_pair_feat).squeeze(-1)
        edge_scope_logits = 0.5 * (edge_scope_logits + edge_scope_logits.transpose(1, 2))

        diag_mask = torch.eye(num_nodes, device=node_feats.device, dtype=torch.bool).unsqueeze(0)
        edge_scope_logits = edge_scope_logits.masked_fill(diag_mask, -1e9)
        return {
            "node_latents": node_latents,
            "node_scope_logits": node_scope_logits,
            "scope_logits": node_scope_logits,
            "edge_scope_logits": edge_scope_logits,
        }


def scope_proposal_loss(
    outputs: Dict[str, torch.Tensor],
    target_node_scope: torch.Tensor,
    target_edge_scope: torch.Tensor,
    node_mask: torch.Tensor,
    pair_mask: torch.Tensor,
    node_scope_loss_weight: float = 1.0,
    edge_scope_loss_weight: float = 1.0,
    edge_scope_pos_weight: float = 1.0,
    node_scope_weights: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    node_scope_loss = masked_scope_bce_loss(
        outputs["node_scope_logits"],
        target_node_scope,
        node_mask,
        weights=node_scope_weights,
    )
    edge_scope_loss = masked_edge_scope_bce_loss(
        outputs["edge_scope_logits"],
        target_edge_scope,
        pair_mask,
        pos_weight=edge_scope_pos_weight,
    )
    total_loss = node_scope_loss_weight * node_scope_loss + edge_scope_loss_weight * edge_scope_loss
    return {
        "total_loss": total_loss,
        "node_scope_loss": node_scope_loss,
        "edge_scope_loss": edge_scope_loss,
    }
