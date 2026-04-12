from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

import torch
import torch.nn as nn

from models.baselines import SimpleGraphEncoder, build_mlp
from models.encoder_step30 import logit_from_hint


@dataclass
class Step31MultiViewEncoderConfig:
    obs_slot_dim: int
    num_views: int = 3
    pair_evidence_bundle_dim: int = 4
    num_node_types: int = 3
    state_dim: int = 4
    hidden_dim: int = 128
    msg_pass_layers: int = 3
    node_head_layers: int = 2
    edge_head_layers: int = 2
    dropout: float = 0.0
    use_relation_logit_residual: bool = True
    relation_logit_residual_scale: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Step31MultiViewObservationEncoder(nn.Module):
    """Small shared-view encoder with explicit cross-view fusion.

    Each view is encoded by the same graph encoder. The fused representation is
    intentionally simple: mean and standard deviation over view latents, plus
    mean/std evidence features in the edge head. This keeps Step31 about the
    multi-view evidence substrate, not a large new architecture family.
    """

    def __init__(self, config: Step31MultiViewEncoderConfig):
        super().__init__()
        self.config = config
        self.view_encoder = SimpleGraphEncoder(
            node_feat_dim=config.obs_slot_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.msg_pass_layers,
            dropout=config.dropout,
        )
        self.node_fusion = build_mlp(
            in_dim=config.hidden_dim * 2,
            hidden_dim=config.hidden_dim,
            out_dim=config.hidden_dim,
            num_layers=2,
            dropout=config.dropout,
        )
        self.type_head = build_mlp(
            in_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            out_dim=config.num_node_types,
            num_layers=config.node_head_layers,
            dropout=config.dropout,
        )
        self.state_head = build_mlp(
            in_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            out_dim=config.state_dim,
            num_layers=config.node_head_layers,
            dropout=config.dropout,
        )

        evidence_dim = 8 + 2 + int(config.pair_evidence_bundle_dim) * 2
        edge_in_dim = config.hidden_dim * 4 + evidence_dim
        self.edge_head = build_mlp(
            in_dim=edge_in_dim,
            hidden_dim=config.hidden_dim,
            out_dim=1,
            num_layers=config.edge_head_layers,
            dropout=config.dropout,
        )

    def forward(
        self,
        multi_view_slot_features: torch.Tensor,
        multi_view_relation_hints: torch.Tensor,
        multi_view_pair_support_hints: torch.Tensor,
        multi_view_signed_pair_witness: torch.Tensor,
        multi_view_pair_evidence_bundle: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size, num_views, num_nodes, obs_dim = multi_view_slot_features.shape
        flat_slots = multi_view_slot_features.reshape(batch_size * num_views, num_nodes, obs_dim)
        flat_relation = multi_view_relation_hints.reshape(
            batch_size * num_views,
            num_nodes,
            num_nodes,
        )
        view_latents = self.view_encoder(flat_slots, flat_relation)
        view_latents = view_latents.reshape(
            batch_size,
            num_views,
            num_nodes,
            int(self.config.hidden_dim),
        )
        latent_mean = view_latents.mean(dim=1)
        latent_std = view_latents.std(dim=1, unbiased=False)
        node_latents = self.node_fusion(torch.cat([latent_mean, latent_std], dim=-1))

        type_logits = self.type_head(node_latents)
        state_pred = self.state_head(node_latents)
        edge_logits, edge_aux = self.predict_edges_from_nodes(
            node_latents=node_latents,
            multi_view_relation_hints=multi_view_relation_hints,
            multi_view_pair_support_hints=multi_view_pair_support_hints,
            multi_view_signed_pair_witness=multi_view_signed_pair_witness,
            multi_view_pair_evidence_bundle=multi_view_pair_evidence_bundle,
        )
        return {
            "node_latents": node_latents,
            "view_node_latents": view_latents,
            "type_logits": type_logits,
            "state_pred": state_pred,
            "edge_logits": edge_logits,
            **edge_aux,
        }

    def predict_edges_from_nodes(
        self,
        node_latents: torch.Tensor,
        multi_view_relation_hints: torch.Tensor,
        multi_view_pair_support_hints: torch.Tensor,
        multi_view_signed_pair_witness: torch.Tensor,
        multi_view_pair_evidence_bundle: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size, num_nodes, hidden_dim = node_latents.shape
        h_i = node_latents.unsqueeze(2).expand(batch_size, num_nodes, num_nodes, hidden_dim)
        h_j = node_latents.unsqueeze(1).expand(batch_size, num_nodes, num_nodes, hidden_dim)

        relation_mean = multi_view_relation_hints.mean(dim=1)
        relation_std = multi_view_relation_hints.std(dim=1, unbiased=False)
        relation_min = multi_view_relation_hints.min(dim=1).values
        relation_max = multi_view_relation_hints.max(dim=1).values
        support_mean = multi_view_pair_support_hints.mean(dim=1)
        support_std = multi_view_pair_support_hints.std(dim=1, unbiased=False)
        support_min = multi_view_pair_support_hints.min(dim=1).values
        support_max = multi_view_pair_support_hints.max(dim=1).values
        witness_mean = multi_view_signed_pair_witness.mean(dim=1)
        witness_std = multi_view_signed_pair_witness.std(dim=1, unbiased=False)
        bundle_mean = multi_view_pair_evidence_bundle.mean(dim=1)
        bundle_std = multi_view_pair_evidence_bundle.std(dim=1, unbiased=False)

        evidence_features = torch.cat(
            [
                relation_mean.unsqueeze(-1),
                relation_std.unsqueeze(-1),
                relation_min.unsqueeze(-1),
                relation_max.unsqueeze(-1),
                support_mean.unsqueeze(-1),
                support_std.unsqueeze(-1),
                support_min.unsqueeze(-1),
                support_max.unsqueeze(-1),
                witness_mean.unsqueeze(-1),
                witness_std.unsqueeze(-1),
                bundle_mean,
                bundle_std,
            ],
            dim=-1,
        )
        pair_features = torch.cat(
            [
                h_i,
                h_j,
                torch.abs(h_i - h_j),
                h_i * h_j,
                evidence_features,
            ],
            dim=-1,
        )
        edge_logits = self.edge_head(pair_features).squeeze(-1)
        if self.config.use_relation_logit_residual:
            residual_hint = 0.5 * (relation_mean + support_mean)
            edge_logits = edge_logits + float(self.config.relation_logit_residual_scale) * logit_from_hint(
                residual_hint
            )

        edge_logits = 0.5 * (edge_logits + edge_logits.transpose(1, 2))
        diag_mask = torch.eye(num_nodes, device=edge_logits.device, dtype=torch.bool).unsqueeze(0)
        edge_logits = edge_logits.masked_fill(diag_mask, -1e9)

        edge_aux = {
            "multi_view_relation_mean": relation_mean.masked_fill(diag_mask, 0.0),
            "multi_view_relation_std": relation_std.masked_fill(diag_mask, 0.0),
            "multi_view_pair_support_mean": support_mean.masked_fill(diag_mask, 0.0),
            "multi_view_pair_support_std": support_std.masked_fill(diag_mask, 0.0),
            "multi_view_signed_witness_mean": witness_mean.masked_fill(diag_mask, 0.0),
            "multi_view_signed_witness_std": witness_std.masked_fill(diag_mask, 0.0),
        }
        return edge_logits, edge_aux
