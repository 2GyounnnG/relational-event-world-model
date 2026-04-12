from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.baselines import SimpleGraphEncoder, build_mlp


@dataclass
class Step30EncoderConfig:
    obs_slot_dim: int
    num_node_types: int = 3
    state_dim: int = 4
    hidden_dim: int = 128
    msg_pass_layers: int = 3
    node_head_layers: int = 2
    edge_head_layers: int = 2
    dropout: float = 0.0
    use_relation_hint_in_edge_head: bool = True
    use_relation_logit_residual: bool = False
    relation_logit_residual_scale: float = 1.0
    use_trust_denoising_edge_decoder: bool = False
    use_pair_support_hints: bool = False
    use_pair_evidence_bundle: bool = False
    pair_evidence_bundle_dim: int = 0
    use_rescue_scoped_pair_evidence_bundle: bool = False
    rescue_scoped_bundle_relation_max: float = 0.5
    rescue_scoped_bundle_residual_scale: float = 0.5
    use_rescue_safety_aux_head: bool = False
    use_rescue_candidate_latent_head: bool = False
    rescue_candidate_latent_dim: int = 32
    rescue_candidate_relation_max: float = 0.5
    rescue_candidate_support_min: float = 0.55
    use_rescue_candidate_binary_calibration_head: bool = False
    use_rescue_candidate_ambiguity_head: bool = False
    use_positive_ambiguity_safety_hint: bool = False
    positive_ambiguity_safety_projection_scale: float = 1.0
    use_weak_positive_ambiguity_safety_head: bool = False
    use_signed_pair_witness: bool = False
    use_signed_pair_witness_in_edge_head: bool = True
    use_signed_pair_witness_correction: bool = False
    signed_pair_witness_correction_scale: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Step30WeakObservationEncoder(nn.Module):
    """
    Minimal Step30 encoder.

    Input is slot-aligned weak observation:
    - weak_slot_features: [B, N, F_obs]
    - weak_relation_hints: [B, N, N]

    Output is a graph-like belief over the current structured graph.
    """

    def __init__(self, config: Step30EncoderConfig):
        super().__init__()
        self.config = config
        self.encoder = SimpleGraphEncoder(
            node_feat_dim=config.obs_slot_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.msg_pass_layers,
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
        edge_in_dim = config.hidden_dim * 4
        if config.use_relation_hint_in_edge_head:
            edge_in_dim += 1
        if config.use_pair_support_hints:
            edge_in_dim += 1
        if config.use_pair_evidence_bundle:
            edge_in_dim += int(config.pair_evidence_bundle_dim)
        if config.use_signed_pair_witness and config.use_signed_pair_witness_in_edge_head:
            edge_in_dim += 1
        self.edge_head = build_mlp(
            in_dim=edge_in_dim,
            hidden_dim=config.hidden_dim,
            out_dim=1,
            num_layers=config.edge_head_layers,
            dropout=config.dropout,
        )
        if config.use_trust_denoising_edge_decoder:
            self.edge_trust_head = build_mlp(
                in_dim=edge_in_dim,
                hidden_dim=config.hidden_dim,
                out_dim=1,
                num_layers=config.edge_head_layers,
                dropout=config.dropout,
            )
        if config.use_rescue_safety_aux_head:
            self.rescue_safety_head = build_mlp(
                in_dim=edge_in_dim,
                hidden_dim=config.hidden_dim,
                out_dim=1,
                num_layers=config.edge_head_layers,
                dropout=config.dropout,
            )
        if config.use_rescue_candidate_latent_head:
            rescue_candidate_in_dim = config.hidden_dim * 4 + 6 + int(config.pair_evidence_bundle_dim)
            self.rescue_candidate_latent = build_mlp(
                in_dim=rescue_candidate_in_dim,
                hidden_dim=max(32, config.hidden_dim // 2),
                out_dim=int(config.rescue_candidate_latent_dim),
                num_layers=2,
                dropout=config.dropout,
            )
            self.rescue_candidate_classifier = nn.Linear(
                int(config.rescue_candidate_latent_dim),
                3,
            )
            if config.use_rescue_candidate_binary_calibration_head:
                self.rescue_candidate_binary_calibration_head = nn.Linear(
                    int(config.rescue_candidate_latent_dim),
                    1,
                )
            if config.use_rescue_candidate_ambiguity_head:
                self.rescue_candidate_ambiguity_head = nn.Linear(
                    int(config.rescue_candidate_latent_dim),
                    1,
                )
            if config.use_positive_ambiguity_safety_hint:
                self.positive_ambiguity_safety_projection = nn.Linear(
                    1,
                    int(config.rescue_candidate_latent_dim),
                )
                nn.init.zeros_(self.positive_ambiguity_safety_projection.weight)
                if self.positive_ambiguity_safety_projection.bias is not None:
                    nn.init.zeros_(self.positive_ambiguity_safety_projection.bias)
            if config.use_weak_positive_ambiguity_safety_head:
                self.weak_positive_ambiguity_safety_head = nn.Linear(
                    int(config.rescue_candidate_latent_dim),
                    1,
                )
        if config.use_signed_pair_witness_correction:
            self.signed_pair_witness_correction_head = build_mlp(
                in_dim=7,
                hidden_dim=max(16, config.hidden_dim // 4),
                out_dim=1,
                num_layers=2,
                dropout=config.dropout,
            )
            self._zero_init_last_linear(self.signed_pair_witness_correction_head)
        if config.use_rescue_scoped_pair_evidence_bundle:
            self.pair_evidence_rescue_head = build_mlp(
                in_dim=int(config.pair_evidence_bundle_dim) + 3,
                hidden_dim=max(16, config.hidden_dim // 4),
                out_dim=1,
                num_layers=2,
                dropout=config.dropout,
            )
            self._zero_init_last_linear(self.pair_evidence_rescue_head)
        if config.use_relation_logit_residual:
            self._zero_init_edge_residual_head()

    @staticmethod
    def _zero_init_last_linear(module: nn.Module) -> None:
        for submodule in reversed(list(module.modules())):
            if isinstance(submodule, nn.Linear):
                nn.init.zeros_(submodule.weight)
                if submodule.bias is not None:
                    nn.init.zeros_(submodule.bias)
                return

    def _zero_init_edge_residual_head(self) -> None:
        """Start rev2 edge prediction at the relation-hint baseline, then learn a residual."""
        self._zero_init_last_linear(self.edge_head)

    def forward(
        self,
        weak_slot_features: torch.Tensor,
        weak_relation_hints: torch.Tensor,
        weak_pair_support_hints: torch.Tensor | None = None,
        weak_signed_pair_witness: torch.Tensor | None = None,
        weak_pair_evidence_bundle: torch.Tensor | None = None,
        weak_positive_ambiguity_safety_hint: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        node_latents = self.encoder(weak_slot_features, weak_relation_hints)
        type_logits = self.type_head(node_latents)
        state_pred = self.state_head(node_latents)
        edge_result = self.predict_edges_from_nodes(
            node_latents,
            weak_relation_hints,
            pair_support_hints=weak_pair_support_hints,
            signed_pair_witness=weak_signed_pair_witness,
            pair_evidence_bundle=weak_pair_evidence_bundle,
            positive_ambiguity_safety_hint=weak_positive_ambiguity_safety_hint,
        )
        if isinstance(edge_result, tuple):
            edge_logits, edge_aux = edge_result
        else:
            edge_logits = edge_result
            edge_aux = {}
        return {
            "node_latents": node_latents,
            "type_logits": type_logits,
            "state_pred": state_pred,
            "edge_logits": edge_logits,
            **edge_aux,
        }

    def predict_edges_from_nodes(
        self,
        node_latents: torch.Tensor,
        relation_hints: torch.Tensor | None = None,
        pair_support_hints: torch.Tensor | None = None,
        signed_pair_witness: torch.Tensor | None = None,
        pair_evidence_bundle: torch.Tensor | None = None,
        positive_ambiguity_safety_hint: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size, num_nodes, hidden_dim = node_latents.shape
        h_i = node_latents.unsqueeze(2).expand(batch_size, num_nodes, num_nodes, hidden_dim)
        h_j = node_latents.unsqueeze(1).expand(batch_size, num_nodes, num_nodes, hidden_dim)
        pair_parts = [h_i, h_j, torch.abs(h_i - h_j), h_i * h_j]
        if self.config.use_relation_hint_in_edge_head:
            if relation_hints is None:
                raise ValueError("relation_hints are required when use_relation_hint_in_edge_head=True")
            pair_parts.append(relation_hints.unsqueeze(-1))
        if self.config.use_pair_support_hints:
            if pair_support_hints is None:
                raise ValueError("pair_support_hints are required when use_pair_support_hints=True")
            pair_parts.append(pair_support_hints.unsqueeze(-1))
        if self.config.use_pair_evidence_bundle:
            if pair_evidence_bundle is None:
                raise ValueError("pair_evidence_bundle is required when use_pair_evidence_bundle=True")
            pair_parts.append(pair_evidence_bundle)
        if self.config.use_signed_pair_witness and self.config.use_signed_pair_witness_in_edge_head:
            if signed_pair_witness is None:
                raise ValueError("signed_pair_witness is required when use_signed_pair_witness=True")
            pair_parts.append(signed_pair_witness.unsqueeze(-1))
        pair_features = torch.cat(pair_parts, dim=-1)
        edge_logits = self.edge_head(pair_features).squeeze(-1)
        edge_aux: Dict[str, torch.Tensor] = {}
        if self.config.use_rescue_safety_aux_head:
            edge_aux["rescue_safety_logits"] = self.rescue_safety_head(pair_features).squeeze(-1)
        if self.config.use_rescue_candidate_latent_head:
            if relation_hints is None:
                raise ValueError(
                    "relation_hints are required when use_rescue_candidate_latent_head=True"
                )
            if pair_support_hints is None:
                support_part = torch.zeros_like(relation_hints) + 0.5
            else:
                support_part = pair_support_hints
            if signed_pair_witness is None:
                signed_part = torch.zeros_like(relation_hints)
            else:
                signed_part = signed_pair_witness
            if int(self.config.pair_evidence_bundle_dim) > 0:
                if pair_evidence_bundle is None:
                    raise ValueError(
                        "pair_evidence_bundle is required when use_rescue_candidate_latent_head=True "
                        "and pair_evidence_bundle_dim > 0"
                    )
                bundle_part = pair_evidence_bundle
            else:
                bundle_part = relation_hints.new_zeros(
                    batch_size,
                    num_nodes,
                    num_nodes,
                    0,
                )
            rescue_candidate_features = torch.cat(
                [
                    h_i,
                    h_j,
                    torch.abs(h_i - h_j),
                    h_i * h_j,
                    relation_hints.unsqueeze(-1),
                    support_part.unsqueeze(-1),
                    (support_part - relation_hints).unsqueeze(-1),
                    signed_part.unsqueeze(-1),
                    edge_logits.unsqueeze(-1),
                    torch.sigmoid(edge_logits).unsqueeze(-1),
                    bundle_part,
                ],
                dim=-1,
            )
            rescue_candidate_latent = self.rescue_candidate_latent(rescue_candidate_features)
            if self.config.use_positive_ambiguity_safety_hint:
                if positive_ambiguity_safety_hint is None:
                    raise ValueError(
                        "positive_ambiguity_safety_hint is required when "
                        "use_positive_ambiguity_safety_hint=True"
                    )
                centered_safety = (positive_ambiguity_safety_hint - 0.5).unsqueeze(-1)
                safety_delta = self.positive_ambiguity_safety_projection(centered_safety)
                rescue_candidate_latent = (
                    rescue_candidate_latent
                    + float(self.config.positive_ambiguity_safety_projection_scale) * safety_delta
                )
                edge_aux["positive_ambiguity_safety_hint"] = positive_ambiguity_safety_hint
            rescue_candidate_logits = self.rescue_candidate_classifier(rescue_candidate_latent)
            edge_aux["rescue_candidate_logits"] = rescue_candidate_logits
            edge_aux["rescue_candidate_safe_logits"] = rescue_candidate_logits[..., 0]
            # Compatibility with existing selective-rescue decode helpers: aux mode
            # can rank rescue candidates by this learned safe-candidate logit.
            edge_aux["rescue_safety_logits"] = rescue_candidate_logits[..., 0]
            if self.config.use_rescue_candidate_binary_calibration_head:
                binary_logits = self.rescue_candidate_binary_calibration_head(
                    rescue_candidate_latent
                ).squeeze(-1)
                edge_aux["rescue_candidate_binary_logits"] = binary_logits
                # Rev26 uses this calibrated safe-vs-false score for aux decode.
                edge_aux["rescue_safety_logits"] = binary_logits
            if self.config.use_rescue_candidate_ambiguity_head:
                ambiguity_logits = self.rescue_candidate_ambiguity_head(
                    rescue_candidate_latent
                ).squeeze(-1)
                edge_aux["rescue_candidate_ambiguity_logits"] = ambiguity_logits
            if self.config.use_weak_positive_ambiguity_safety_head:
                weak_positive_logits = self.weak_positive_ambiguity_safety_head(
                    rescue_candidate_latent
                ).squeeze(-1)
                edge_aux["weak_positive_ambiguity_safety_logits"] = weak_positive_logits
        if self.config.use_relation_logit_residual:
            if relation_hints is None:
                raise ValueError("relation_hints are required when use_relation_logit_residual=True")
            residual_hint = relation_hints
            if self.config.use_pair_support_hints and pair_support_hints is not None:
                residual_hint = 0.5 * (relation_hints + pair_support_hints)
            relation_logits = logit_from_hint(residual_hint)
            if self.config.use_trust_denoising_edge_decoder:
                override_logits = edge_logits
                trust_logits = self.edge_trust_head(pair_features).squeeze(-1)
                trust = torch.sigmoid(trust_logits)
                # Trust-aware denoising: follow the relation hint where reliable, and
                # route unreliable pairs to a learned adjacency override path. This lets
                # the model fix both hint-supported false positives and hint-missed edges.
                edge_logits = (
                    trust * float(self.config.relation_logit_residual_scale) * relation_logits
                    + (1.0 - trust) * override_logits
                )
                edge_aux["edge_trust_logits"] = trust_logits
                edge_aux["edge_override_logits"] = override_logits
            else:
                edge_logits = edge_logits + float(self.config.relation_logit_residual_scale) * relation_logits
        if self.config.use_signed_pair_witness_correction:
            if signed_pair_witness is None:
                raise ValueError(
                    "signed_pair_witness is required when use_signed_pair_witness_correction=True"
                )
            if relation_hints is None:
                relation_part = torch.zeros_like(signed_pair_witness)
            else:
                relation_part = relation_hints
            if pair_support_hints is None:
                support_part = torch.zeros_like(signed_pair_witness) + 0.5
            else:
                support_part = pair_support_hints
            positive_witness = signed_pair_witness.clamp_min(0.0)
            negative_witness = (-signed_pair_witness).clamp_min(0.0)
            correction_features = torch.stack(
                [
                    signed_pair_witness,
                    positive_witness,
                    negative_witness,
                    signed_pair_witness.abs(),
                    relation_part,
                    support_part,
                    support_part - relation_part,
                ],
                dim=-1,
            )
            correction_logits = self.signed_pair_witness_correction_head(correction_features).squeeze(-1)
            correction_logits = torch.tanh(correction_logits)
            edge_logits = edge_logits + float(self.config.signed_pair_witness_correction_scale) * correction_logits
            edge_aux["signed_pair_witness_correction_logits"] = correction_logits
            edge_aux["signed_pair_witness"] = signed_pair_witness
        if self.config.use_rescue_scoped_pair_evidence_bundle:
            if pair_evidence_bundle is None:
                raise ValueError(
                    "pair_evidence_bundle is required when use_rescue_scoped_pair_evidence_bundle=True"
                )
            if relation_hints is None:
                raise ValueError(
                    "relation_hints are required when use_rescue_scoped_pair_evidence_bundle=True"
                )
            if pair_support_hints is None:
                support_part = torch.zeros_like(relation_hints) + 0.5
            else:
                support_part = pair_support_hints
            rescue_features = torch.cat(
                [
                    pair_evidence_bundle,
                    relation_hints.unsqueeze(-1),
                    support_part.unsqueeze(-1),
                    (support_part - relation_hints).unsqueeze(-1),
                ],
                dim=-1,
            )
            rescue_residual = self.pair_evidence_rescue_head(rescue_features).squeeze(-1)
            rescue_residual = torch.tanh(rescue_residual)
            rescue_scope = (relation_hints < float(self.config.rescue_scoped_bundle_relation_max)).float()
            edge_logits = (
                edge_logits
                + float(self.config.rescue_scoped_bundle_residual_scale)
                * rescue_scope
                * rescue_residual
            )
            edge_aux["pair_evidence_rescue_residual"] = rescue_scope * rescue_residual
        edge_logits = 0.5 * (edge_logits + edge_logits.transpose(1, 2))
        for key, value in list(edge_aux.items()):
            edge_aux[key] = 0.5 * (value + value.transpose(1, 2))
        diag_mask = torch.eye(num_nodes, device=node_latents.device, dtype=torch.bool).unsqueeze(0)
        edge_logits = edge_logits.masked_fill(diag_mask, -1e9)
        for key, value in list(edge_aux.items()):
            value_diag_mask = diag_mask
            while value_diag_mask.dim() < value.dim():
                value_diag_mask = value_diag_mask.unsqueeze(-1)
            edge_aux[key] = value.masked_fill(value_diag_mask, 0.0)
        if edge_aux:
            return edge_logits, edge_aux
        return edge_logits


def logit_from_hint(hint: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    hint = hint.clamp(eps, 1.0 - eps)
    return torch.log(hint / (1.0 - hint))


def build_pair_mask(node_mask: torch.Tensor) -> torch.Tensor:
    batch_size, num_nodes = node_mask.shape
    pair_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    diag = torch.eye(num_nodes, device=node_mask.device, dtype=node_mask.dtype).unsqueeze(0)
    return pair_mask * (1.0 - diag)


def masked_type_loss(
    type_logits: torch.Tensor,
    target_type: torch.Tensor,
    node_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    batch_size, num_nodes, num_classes = type_logits.shape
    ce = F.cross_entropy(
        type_logits.reshape(batch_size * num_nodes, num_classes),
        target_type.reshape(batch_size * num_nodes),
        reduction="none",
    ).reshape(batch_size, num_nodes)
    ce = ce * node_mask
    return ce.sum() / (node_mask.sum() + eps)


def masked_state_mse_loss(
    state_pred: torch.Tensor,
    target_state: torch.Tensor,
    node_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    mask = node_mask.unsqueeze(-1)
    sq_err = (state_pred - target_state) ** 2
    sq_err = sq_err * mask
    return sq_err.sum() / (mask.sum() * state_pred.shape[-1] + eps)


def masked_edge_bce_loss(
    edge_logits: torch.Tensor,
    target_adj: torch.Tensor,
    node_mask: torch.Tensor,
    edge_pos_weight: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    pair_mask = build_pair_mask(node_mask)
    pos_weight = torch.tensor(edge_pos_weight, device=edge_logits.device, dtype=edge_logits.dtype)
    bce = F.binary_cross_entropy_with_logits(
        edge_logits,
        target_adj.float(),
        pos_weight=pos_weight,
        reduction="none",
    )
    bce = bce * pair_mask
    return bce.sum() / (pair_mask.sum() + eps)


def masked_hint_hard_negative_bce_loss(
    edge_logits: torch.Tensor,
    target_adj: torch.Tensor,
    node_mask: torch.Tensor,
    relation_hints: torch.Tensor,
    hard_negative_hint_threshold: float = 0.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    pair_mask = build_pair_mask(node_mask)
    target = target_adj.float()
    hard_negative_mask = (
        pair_mask
        * (target < 0.5).float()
        * (relation_hints >= hard_negative_hint_threshold).float()
    )
    bce = F.binary_cross_entropy_with_logits(
        edge_logits,
        torch.zeros_like(edge_logits),
        reduction="none",
    )
    return (bce * hard_negative_mask).sum() / (hard_negative_mask.sum() + eps)


def masked_edge_ranking_loss(
    edge_logits: torch.Tensor,
    target_adj: torch.Tensor,
    node_mask: torch.Tensor,
    relation_hints: torch.Tensor,
    hard_negative_hint_threshold: float = 0.5,
    margin: float = 0.5,
) -> torch.Tensor:
    pair_mask = build_pair_mask(node_mask)
    target = target_adj.float()
    positive_mask = (pair_mask * (target > 0.5).float()).bool()
    hard_negative_mask = (
        pair_mask
        * (target < 0.5).float()
        * (relation_hints >= hard_negative_hint_threshold).float()
    ).bool()

    losses = []
    for batch_idx in range(edge_logits.shape[0]):
        pos_scores = edge_logits[batch_idx][positive_mask[batch_idx]]
        neg_scores = edge_logits[batch_idx][hard_negative_mask[batch_idx]]
        if pos_scores.numel() == 0 or neg_scores.numel() == 0:
            continue
        pair_losses = F.softplus(float(margin) - pos_scores[:, None] + neg_scores[None, :])
        losses.append(pair_losses.mean())

    if not losses:
        return edge_logits.new_zeros(())
    return torch.stack(losses).mean()


def masked_hint_trust_loss(
    trust_logits: torch.Tensor,
    target_adj: torch.Tensor,
    node_mask: torch.Tensor,
    relation_hints: torch.Tensor,
    hint_threshold: float = 0.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    pair_mask = build_pair_mask(node_mask)
    target = target_adj.float()
    hint_binary = (relation_hints >= hint_threshold).float()
    hint_correct = (hint_binary == target).float()
    bce = F.binary_cross_entropy_with_logits(
        trust_logits,
        hint_correct,
        reduction="none",
    )
    return (bce * pair_mask).sum() / (pair_mask.sum() + eps)


def masked_missed_edge_recovery_loss(
    edge_logits: torch.Tensor,
    target_adj: torch.Tensor,
    node_mask: torch.Tensor,
    relation_hints: torch.Tensor,
    missed_hint_threshold: float = 0.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    pair_mask = build_pair_mask(node_mask)
    missed_positive_mask = (
        pair_mask
        * (target_adj.float() > 0.5).float()
        * (relation_hints < missed_hint_threshold).float()
    )
    bce = F.binary_cross_entropy_with_logits(
        edge_logits,
        torch.ones_like(edge_logits),
        reduction="none",
    )
    return (bce * missed_positive_mask).sum() / (missed_positive_mask.sum() + eps)


def masked_rescue_safety_aux_loss(
    rescue_safety_logits: torch.Tensor,
    target_adj: torch.Tensor,
    node_mask: torch.Tensor,
    relation_hints: torch.Tensor,
    pair_support_hints: torch.Tensor,
    rescue_relation_max: float = 0.5,
    rescue_support_min: float = 0.55,
    rescue_pos_weight: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    pair_mask = build_pair_mask(node_mask)
    candidate_mask = (
        pair_mask
        * (relation_hints < float(rescue_relation_max)).float()
        * (pair_support_hints >= float(rescue_support_min)).float()
    )
    pos_weight = torch.tensor(rescue_pos_weight, device=rescue_safety_logits.device, dtype=rescue_safety_logits.dtype)
    bce = F.binary_cross_entropy_with_logits(
        rescue_safety_logits,
        target_adj.float(),
        pos_weight=pos_weight,
        reduction="none",
    )
    return (bce * candidate_mask).sum() / (candidate_mask.sum() + eps)


def masked_signed_witness_correction_loss(
    correction_logits: torch.Tensor,
    target_adj: torch.Tensor,
    node_mask: torch.Tensor,
    relation_hints: torch.Tensor,
    pair_support_hints: torch.Tensor,
    signed_pair_witness: torch.Tensor,
    rescue_relation_max: float = 0.5,
    rescue_support_min: float = 0.55,
    witness_active_min: float = 0.25,
    ambiguous_weight: float = 0.25,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Shape the witness correction head without making it dominate edge BCE.

    Low-hint true edges get positive correction; low-hint, support-backed false
    admissions get negative correction; ambiguous weak-witness cases are softly
    regularized toward zero.
    """

    pair_mask = build_pair_mask(node_mask)
    target = target_adj.float()
    low_relation = (relation_hints < float(rescue_relation_max)).float()
    support_candidate = (pair_support_hints >= float(rescue_support_min)).float()
    active_witness = (signed_pair_witness.abs() >= float(witness_active_min)).float()

    safe_rescue = pair_mask * low_relation * (target > 0.5).float() * active_witness
    unsafe_false = (
        pair_mask
        * low_relation
        * support_candidate
        * (target < 0.5).float()
        * active_witness
    )
    ambiguous = (
        pair_mask
        * low_relation
        * (1.0 - active_witness)
        * (
            (target > 0.5).float()
            + support_candidate * (target < 0.5).float()
        ).clamp(max=1.0)
    )

    supervised_mask = safe_rescue + unsafe_false + float(ambiguous_weight) * ambiguous
    target_correction = safe_rescue - unsafe_false
    # Ambiguous rows keep target 0, with reduced weight through supervised_mask.
    sq_err = (correction_logits - target_correction) ** 2
    return (sq_err * supervised_mask).sum() / (supervised_mask.sum() + eps)


def masked_false_admission_correction_penalty(
    correction_logits: torch.Tensor,
    target_adj: torch.Tensor,
    node_mask: torch.Tensor,
    relation_hints: torch.Tensor,
    pair_support_hints: torch.Tensor,
    rescue_relation_max: float = 0.5,
    rescue_support_min: float = 0.55,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Directly discourage positive correction on unsafe rescue-like negatives."""

    pair_mask = build_pair_mask(node_mask)
    unsafe_false_mask = (
        pair_mask
        * (relation_hints < float(rescue_relation_max)).float()
        * (pair_support_hints >= float(rescue_support_min)).float()
        * (target_adj.float() < 0.5).float()
    )
    positive_correction = F.relu(correction_logits)
    penalty = positive_correction ** 2
    return (penalty * unsafe_false_mask).sum() / (unsafe_false_mask.sum() + eps)


def masked_rescue_residual_contrast_suppression_loss(
    rescue_residual: torch.Tensor,
    target_adj: torch.Tensor,
    node_mask: torch.Tensor,
    relation_hints: torch.Tensor,
    pair_support_hints: torch.Tensor,
    rescue_relation_max: float = 0.5,
    rescue_support_min: float = 0.55,
    margin: float = 0.25,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Shape only the rescue residual inside the low-relation rescue scope.

    The objective is intentionally small and direct:
    - unsafe rescue negatives should not receive positive residual boosts;
    - safe missed positives should keep at least a modest positive residual;
    - batch means should maintain a small safe-vs-unsafe residual gap.
    """

    pair_mask = build_pair_mask(node_mask)
    rescue_mask = (
        pair_mask
        * (relation_hints < float(rescue_relation_max)).float()
        * (pair_support_hints >= float(rescue_support_min)).float()
    )
    safe_mask = rescue_mask * (target_adj.float() > 0.5).float()
    unsafe_mask = rescue_mask * (target_adj.float() < 0.5).float()

    unsafe_positive_penalty = (F.relu(rescue_residual) ** 2 * unsafe_mask).sum() / (
        unsafe_mask.sum() + eps
    )
    safe_floor_penalty = (F.relu(float(margin) - rescue_residual) ** 2 * safe_mask).sum() / (
        safe_mask.sum() + eps
    )

    safe_count = safe_mask.sum()
    unsafe_count = unsafe_mask.sum()
    if safe_count.item() > 0 and unsafe_count.item() > 0:
        safe_mean = (rescue_residual * safe_mask).sum() / (safe_count + eps)
        unsafe_mean = (rescue_residual * unsafe_mask).sum() / (unsafe_count + eps)
        contrast_penalty = F.relu(float(margin) - (safe_mean - unsafe_mean)) ** 2
    else:
        contrast_penalty = rescue_residual.new_zeros(())

    return unsafe_positive_penalty + 0.5 * safe_floor_penalty + 0.25 * contrast_penalty


def masked_safe_rescue_residual_preservation_loss(
    rescue_residual: torch.Tensor,
    target_adj: torch.Tensor,
    node_mask: torch.Tensor,
    relation_hints: torch.Tensor,
    pair_support_hints: torch.Tensor,
    rescue_relation_max: float = 0.5,
    rescue_support_min: float = 0.55,
    safe_floor: float = 0.45,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Preserve positive residual on safe low-hint rescue positives only."""

    pair_mask = build_pair_mask(node_mask)
    safe_mask = (
        pair_mask
        * (relation_hints < float(rescue_relation_max)).float()
        * (pair_support_hints >= float(rescue_support_min)).float()
        * (target_adj.float() > 0.5).float()
    )
    floor_penalty = F.relu(float(safe_floor) - rescue_residual) ** 2
    return (floor_penalty * safe_mask).sum() / (safe_mask.sum() + eps)


def masked_rescue_candidate_latent_loss(
    rescue_candidate_logits: torch.Tensor,
    target_adj: torch.Tensor,
    node_mask: torch.Tensor,
    relation_hints: torch.Tensor,
    pair_support_hints: torch.Tensor,
    pair_evidence_bundle: torch.Tensor | None = None,
    rescue_relation_max: float = 0.5,
    rescue_support_min: float = 0.55,
    ambiguous_weight: float = 0.35,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Train a first-class rescue-candidate latent inside the rescue scope.

    Classes are intentionally small and diagnostic:
    0: safe missed true edge,
    1: low-hint pair-support false admission,
    2: ambiguous rescue candidate.
    """

    pair_mask = build_pair_mask(node_mask)
    target = target_adj.float()
    candidate_mask = (
        pair_mask
        * (relation_hints < float(rescue_relation_max)).float()
        * (pair_support_hints >= float(rescue_support_min)).float()
    ).bool()
    if not candidate_mask.any():
        return rescue_candidate_logits.new_zeros(())

    # Keep a middle class for low-confidence rescue negatives so the probe does
    # not learn a brittle binary boundary over noisy weak evidence.
    ambiguous_negative = (
        (target < 0.5)
        & (
            (relation_hints >= 0.40)
            | (pair_support_hints < 0.65)
        )
    )
    if pair_evidence_bundle is not None and pair_evidence_bundle.shape[-1] >= 2:
        positive_support = pair_evidence_bundle[..., 0]
        warning_support = pair_evidence_bundle[..., 1]
        ambiguous_negative = ambiguous_negative | (
            (target < 0.5) & ((positive_support - warning_support).abs() < 0.12)
        )

    class_target = torch.full_like(target_adj.long(), 2)
    class_target[target > 0.5] = 0
    class_target[(target < 0.5) & (~ambiguous_negative)] = 1

    flat_logits = rescue_candidate_logits[candidate_mask]
    flat_target = class_target[candidate_mask]
    class_weights = rescue_candidate_logits.new_tensor([1.5, 1.0, float(ambiguous_weight)])
    return F.cross_entropy(flat_logits, flat_target, weight=class_weights, reduction="mean")


def masked_rescue_candidate_binary_calibration_loss(
    binary_logits: torch.Tensor,
    target_adj: torch.Tensor,
    node_mask: torch.Tensor,
    relation_hints: torch.Tensor,
    pair_support_hints: torch.Tensor,
    pair_evidence_bundle: torch.Tensor | None = None,
    rescue_relation_max: float = 0.5,
    rescue_support_min: float = 0.55,
    pos_weight: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Calibrate safe-vs-low-hint-false separation only.

    Ambiguous rescue negatives are deliberately ignored here: rev25 showed that
    decode-side class margins can reduce ambiguity, but the remaining failure is
    the safe-vs-low-hint-false boundary.
    """

    pair_mask = build_pair_mask(node_mask)
    target = target_adj.float()
    candidate_mask = (
        pair_mask
        * (relation_hints < float(rescue_relation_max)).float()
        * (pair_support_hints >= float(rescue_support_min)).float()
    ).bool()
    ambiguous_negative = (
        (target < 0.5)
        & (
            (relation_hints >= 0.40)
            | (pair_support_hints < 0.65)
        )
    )
    if pair_evidence_bundle is not None and pair_evidence_bundle.shape[-1] >= 2:
        positive_support = pair_evidence_bundle[..., 0]
        warning_support = pair_evidence_bundle[..., 1]
        ambiguous_negative = ambiguous_negative | (
            (target < 0.5) & ((positive_support - warning_support).abs() < 0.12)
        )

    safe_mask = candidate_mask & (target > 0.5)
    false_mask = candidate_mask & (target < 0.5) & (~ambiguous_negative)
    supervised_mask = safe_mask | false_mask
    if not supervised_mask.any():
        return binary_logits.new_zeros(())

    binary_target = safe_mask.float()
    weight = torch.ones_like(binary_logits)
    weight = torch.where(safe_mask, weight * float(pos_weight), weight)
    bce = F.binary_cross_entropy_with_logits(
        binary_logits,
        binary_target,
        reduction="none",
    )
    return (bce * weight * supervised_mask.float()).sum() / (
        (weight * supervised_mask.float()).sum() + eps
    )


def masked_rescue_candidate_ambiguity_loss(
    ambiguity_logits: torch.Tensor,
    target_adj: torch.Tensor,
    node_mask: torch.Tensor,
    relation_hints: torch.Tensor,
    pair_support_hints: torch.Tensor,
    pair_evidence_bundle: torch.Tensor | None = None,
    rescue_relation_max: float = 0.5,
    rescue_support_min: float = 0.55,
    pos_weight: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Predict ambiguous rescue negatives as a first-class admission risk."""

    pair_mask = build_pair_mask(node_mask)
    target = target_adj.float()
    candidate_mask = (
        pair_mask
        * (relation_hints < float(rescue_relation_max)).float()
        * (pair_support_hints >= float(rescue_support_min)).float()
    ).bool()
    if not candidate_mask.any():
        return ambiguity_logits.new_zeros(())

    ambiguous_target = (
        (target < 0.5)
        & (
            (relation_hints >= 0.40)
            | (pair_support_hints < 0.65)
        )
    )
    if pair_evidence_bundle is not None and pair_evidence_bundle.shape[-1] >= 2:
        positive_support = pair_evidence_bundle[..., 0]
        warning_support = pair_evidence_bundle[..., 1]
        ambiguous_target = ambiguous_target | (
            (target < 0.5) & ((positive_support - warning_support).abs() < 0.12)
        )

    pos_weight_tensor = ambiguity_logits.new_tensor(float(pos_weight))
    bce = F.binary_cross_entropy_with_logits(
        ambiguity_logits,
        ambiguous_target.float(),
        pos_weight=pos_weight_tensor,
        reduction="none",
    )
    return (bce * candidate_mask.float()).sum() / (candidate_mask.float().sum() + eps)


def weak_positive_ambiguity_mask(
    target_adj: torch.Tensor,
    node_mask: torch.Tensor,
    relation_hints: torch.Tensor,
    pair_support_hints: torch.Tensor,
    pair_evidence_bundle: torch.Tensor,
    signed_pair_witness: torch.Tensor | None = None,
    rescue_relation_max: float = 0.5,
    rescue_support_min: float = 0.55,
) -> torch.Tensor:
    pair_mask = build_pair_mask(node_mask).bool()
    positive_support = pair_evidence_bundle[..., 0]
    warning_support = pair_evidence_bundle[..., 1]
    corroboration = pair_evidence_bundle[..., 2]
    endpoint_compat = pair_evidence_bundle[..., 3]
    signed_support = (
        signed_pair_witness >= 0.05
        if signed_pair_witness is not None
        else torch.zeros_like(relation_hints, dtype=torch.bool)
    )
    positive_looking = (
        (positive_support - warning_support >= 0.12)
        & (
            (corroboration >= 0.45)
            | (endpoint_compat >= 0.50)
            | signed_support
        )
    )
    ambiguous_signal = (
        (relation_hints >= 0.45)
        | (pair_support_hints < 0.65)
        | ((positive_support - warning_support).abs() < 0.10)
    )
    return (
        pair_mask
        & (relation_hints < float(rescue_relation_max))
        & (pair_support_hints >= float(rescue_support_min))
        & positive_looking
        & ambiguous_signal
    )


def masked_weak_positive_ambiguity_safety_loss(
    weak_positive_logits: torch.Tensor,
    target_adj: torch.Tensor,
    node_mask: torch.Tensor,
    relation_hints: torch.Tensor,
    pair_support_hints: torch.Tensor,
    pair_evidence_bundle: torch.Tensor,
    signed_pair_witness: torch.Tensor | None = None,
    rescue_relation_max: float = 0.5,
    rescue_support_min: float = 0.55,
    pos_weight: float = 2.4,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Binary safe-vs-false loss only inside weak-positive ambiguity."""

    mask = weak_positive_ambiguity_mask(
        target_adj=target_adj,
        node_mask=node_mask,
        relation_hints=relation_hints,
        pair_support_hints=pair_support_hints,
        pair_evidence_bundle=pair_evidence_bundle,
        signed_pair_witness=signed_pair_witness,
        rescue_relation_max=rescue_relation_max,
        rescue_support_min=rescue_support_min,
    )
    if not mask.any():
        return weak_positive_logits.new_zeros(())
    pos_weight_tensor = weak_positive_logits.new_tensor(float(pos_weight))
    bce = F.binary_cross_entropy_with_logits(
        weak_positive_logits,
        target_adj.float(),
        pos_weight=pos_weight_tensor,
        reduction="none",
    )
    return (bce * mask.float()).sum() / (mask.float().sum() + eps)


def step30_recovery_loss(
    outputs: Dict[str, torch.Tensor],
    target_node_feats: torch.Tensor,
    target_adj: torch.Tensor,
    node_mask: torch.Tensor,
    relation_hints: torch.Tensor | None = None,
    pair_support_hints: torch.Tensor | None = None,
    pair_evidence_bundle: torch.Tensor | None = None,
    signed_pair_witness: torch.Tensor | None = None,
    type_loss_weight: float = 1.0,
    state_loss_weight: float = 1.0,
    edge_loss_weight: float = 1.0,
    edge_pos_weight: float = 1.0,
    hard_negative_edge_loss_weight: float = 0.0,
    hard_negative_hint_threshold: float = 0.5,
    edge_ranking_loss_weight: float = 0.0,
    edge_ranking_margin: float = 0.5,
    trust_aux_loss_weight: float = 0.0,
    trust_hint_threshold: float = 0.5,
    missed_edge_loss_weight: float = 0.0,
    missed_edge_hint_threshold: float = 0.5,
    rescue_safety_aux_loss_weight: float = 0.0,
    rescue_safety_relation_max: float = 0.5,
    rescue_safety_support_min: float = 0.55,
    rescue_safety_pos_weight: float = 1.0,
    signed_witness_correction_loss_weight: float = 0.0,
    signed_witness_relation_max: float = 0.5,
    signed_witness_support_min: float = 0.55,
    signed_witness_active_min: float = 0.25,
    signed_witness_ambiguous_weight: float = 0.25,
    false_admission_correction_loss_weight: float = 0.0,
    false_admission_relation_max: float = 0.5,
    false_admission_support_min: float = 0.55,
    rescue_residual_contrast_loss_weight: float = 0.0,
    rescue_residual_relation_max: float = 0.5,
    rescue_residual_support_min: float = 0.55,
    rescue_residual_margin: float = 0.25,
    safe_rescue_residual_preservation_loss_weight: float = 0.0,
    safe_rescue_residual_relation_max: float = 0.5,
    safe_rescue_residual_support_min: float = 0.55,
    safe_rescue_residual_floor: float = 0.45,
    rescue_candidate_latent_loss_weight: float = 0.0,
    rescue_candidate_relation_max: float = 0.5,
    rescue_candidate_support_min: float = 0.55,
    rescue_candidate_ambiguous_weight: float = 0.35,
    rescue_candidate_binary_calibration_loss_weight: float = 0.0,
    rescue_candidate_binary_pos_weight: float = 1.0,
    rescue_candidate_ambiguity_loss_weight: float = 0.0,
    rescue_candidate_ambiguity_pos_weight: float = 1.0,
    weak_positive_ambiguity_safety_loss_weight: float = 0.0,
    weak_positive_ambiguity_safety_pos_weight: float = 2.4,
) -> Dict[str, torch.Tensor]:
    target_type = target_node_feats[:, :, 0].long()
    target_state = target_node_feats[:, :, 1:]
    type_loss = masked_type_loss(outputs["type_logits"], target_type, node_mask)
    state_loss = masked_state_mse_loss(outputs["state_pred"], target_state, node_mask)
    edge_loss = masked_edge_bce_loss(
        outputs["edge_logits"],
        target_adj,
        node_mask,
        edge_pos_weight=edge_pos_weight,
    )
    hard_negative_edge_loss = outputs["edge_logits"].new_zeros(())
    edge_ranking_loss = outputs["edge_logits"].new_zeros(())
    trust_aux_loss = outputs["edge_logits"].new_zeros(())
    missed_edge_loss = outputs["edge_logits"].new_zeros(())
    rescue_safety_aux_loss = outputs["edge_logits"].new_zeros(())
    signed_witness_correction_loss = outputs["edge_logits"].new_zeros(())
    false_admission_correction_loss = outputs["edge_logits"].new_zeros(())
    rescue_residual_contrast_loss = outputs["edge_logits"].new_zeros(())
    safe_rescue_residual_preservation_loss = outputs["edge_logits"].new_zeros(())
    rescue_candidate_latent_loss = outputs["edge_logits"].new_zeros(())
    rescue_candidate_binary_calibration_loss = outputs["edge_logits"].new_zeros(())
    rescue_candidate_ambiguity_loss = outputs["edge_logits"].new_zeros(())
    weak_positive_ambiguity_safety_loss = outputs["edge_logits"].new_zeros(())
    if (
        relation_hints is not None
        and (hard_negative_edge_loss_weight > 0.0 or edge_ranking_loss_weight > 0.0)
    ):
        hard_negative_edge_loss = masked_hint_hard_negative_bce_loss(
            outputs["edge_logits"],
            target_adj,
            node_mask,
            relation_hints=relation_hints,
            hard_negative_hint_threshold=hard_negative_hint_threshold,
        )
        edge_ranking_loss = masked_edge_ranking_loss(
            outputs["edge_logits"],
            target_adj,
            node_mask,
            relation_hints=relation_hints,
            hard_negative_hint_threshold=hard_negative_hint_threshold,
            margin=edge_ranking_margin,
        )
    if relation_hints is not None and trust_aux_loss_weight > 0.0 and "edge_trust_logits" in outputs:
        trust_aux_loss = masked_hint_trust_loss(
            outputs["edge_trust_logits"],
            target_adj,
            node_mask,
            relation_hints=relation_hints,
            hint_threshold=trust_hint_threshold,
        )
    if relation_hints is not None and missed_edge_loss_weight > 0.0:
        missed_edge_loss = masked_missed_edge_recovery_loss(
            outputs["edge_logits"],
            target_adj,
            node_mask,
            relation_hints=relation_hints,
            missed_hint_threshold=missed_edge_hint_threshold,
        )
    if (
        relation_hints is not None
        and pair_support_hints is not None
        and rescue_safety_aux_loss_weight > 0.0
        and "rescue_safety_logits" in outputs
    ):
        rescue_safety_aux_loss = masked_rescue_safety_aux_loss(
            outputs["rescue_safety_logits"],
            target_adj,
            node_mask,
            relation_hints=relation_hints,
            pair_support_hints=pair_support_hints,
            rescue_relation_max=rescue_safety_relation_max,
            rescue_support_min=rescue_safety_support_min,
            rescue_pos_weight=rescue_safety_pos_weight,
        )
    if (
        relation_hints is not None
        and pair_support_hints is not None
        and "signed_pair_witness_correction_logits" in outputs
        and signed_witness_correction_loss_weight > 0.0
    ):
        signed_pair_witness = outputs.get("signed_pair_witness")
        if signed_pair_witness is None:
            raise ValueError(
                "step30_recovery_loss requires outputs['signed_pair_witness'] "
                "when signed witness correction supervision is enabled"
            )
        signed_witness_correction_loss = masked_signed_witness_correction_loss(
            outputs["signed_pair_witness_correction_logits"],
            target_adj,
            node_mask,
            relation_hints=relation_hints,
            pair_support_hints=pair_support_hints,
            signed_pair_witness=signed_pair_witness,
            rescue_relation_max=signed_witness_relation_max,
            rescue_support_min=signed_witness_support_min,
            witness_active_min=signed_witness_active_min,
            ambiguous_weight=signed_witness_ambiguous_weight,
        )
    if (
        relation_hints is not None
        and pair_support_hints is not None
        and "signed_pair_witness_correction_logits" in outputs
        and false_admission_correction_loss_weight > 0.0
    ):
        false_admission_correction_loss = masked_false_admission_correction_penalty(
            outputs["signed_pair_witness_correction_logits"],
            target_adj,
            node_mask,
            relation_hints=relation_hints,
            pair_support_hints=pair_support_hints,
            rescue_relation_max=false_admission_relation_max,
            rescue_support_min=false_admission_support_min,
        )
    if (
        relation_hints is not None
        and pair_support_hints is not None
        and "pair_evidence_rescue_residual" in outputs
        and rescue_residual_contrast_loss_weight > 0.0
    ):
        rescue_residual_contrast_loss = masked_rescue_residual_contrast_suppression_loss(
            outputs["pair_evidence_rescue_residual"],
            target_adj,
            node_mask,
            relation_hints=relation_hints,
            pair_support_hints=pair_support_hints,
            rescue_relation_max=rescue_residual_relation_max,
            rescue_support_min=rescue_residual_support_min,
            margin=rescue_residual_margin,
        )
    if (
        relation_hints is not None
        and pair_support_hints is not None
        and "pair_evidence_rescue_residual" in outputs
        and safe_rescue_residual_preservation_loss_weight > 0.0
    ):
        safe_rescue_residual_preservation_loss = masked_safe_rescue_residual_preservation_loss(
            outputs["pair_evidence_rescue_residual"],
            target_adj,
            node_mask,
            relation_hints=relation_hints,
            pair_support_hints=pair_support_hints,
            rescue_relation_max=safe_rescue_residual_relation_max,
            rescue_support_min=safe_rescue_residual_support_min,
            safe_floor=safe_rescue_residual_floor,
        )
    if (
        relation_hints is not None
        and pair_support_hints is not None
        and "rescue_candidate_logits" in outputs
        and rescue_candidate_latent_loss_weight > 0.0
    ):
        rescue_candidate_latent_loss = masked_rescue_candidate_latent_loss(
            outputs["rescue_candidate_logits"],
            target_adj,
            node_mask,
            relation_hints=relation_hints,
            pair_support_hints=pair_support_hints,
            pair_evidence_bundle=pair_evidence_bundle,
            rescue_relation_max=rescue_candidate_relation_max,
            rescue_support_min=rescue_candidate_support_min,
            ambiguous_weight=rescue_candidate_ambiguous_weight,
        )
    if (
        relation_hints is not None
        and pair_support_hints is not None
        and "rescue_candidate_binary_logits" in outputs
        and rescue_candidate_binary_calibration_loss_weight > 0.0
    ):
        rescue_candidate_binary_calibration_loss = masked_rescue_candidate_binary_calibration_loss(
            outputs["rescue_candidate_binary_logits"],
            target_adj,
            node_mask,
            relation_hints=relation_hints,
            pair_support_hints=pair_support_hints,
            pair_evidence_bundle=pair_evidence_bundle,
            rescue_relation_max=rescue_candidate_relation_max,
            rescue_support_min=rescue_candidate_support_min,
            pos_weight=rescue_candidate_binary_pos_weight,
        )
    if (
        relation_hints is not None
        and pair_support_hints is not None
        and "rescue_candidate_ambiguity_logits" in outputs
        and rescue_candidate_ambiguity_loss_weight > 0.0
    ):
        rescue_candidate_ambiguity_loss = masked_rescue_candidate_ambiguity_loss(
            outputs["rescue_candidate_ambiguity_logits"],
            target_adj,
            node_mask,
            relation_hints=relation_hints,
            pair_support_hints=pair_support_hints,
            pair_evidence_bundle=pair_evidence_bundle,
            rescue_relation_max=rescue_candidate_relation_max,
            rescue_support_min=rescue_candidate_support_min,
            pos_weight=rescue_candidate_ambiguity_pos_weight,
        )
    if (
        relation_hints is not None
        and pair_support_hints is not None
        and pair_evidence_bundle is not None
        and "weak_positive_ambiguity_safety_logits" in outputs
        and weak_positive_ambiguity_safety_loss_weight > 0.0
    ):
        weak_positive_ambiguity_safety_loss = masked_weak_positive_ambiguity_safety_loss(
            outputs["weak_positive_ambiguity_safety_logits"],
            target_adj,
            node_mask,
            relation_hints=relation_hints,
            pair_support_hints=pair_support_hints,
            pair_evidence_bundle=pair_evidence_bundle,
            signed_pair_witness=signed_pair_witness,
            rescue_relation_max=rescue_candidate_relation_max,
            rescue_support_min=rescue_candidate_support_min,
            pos_weight=weak_positive_ambiguity_safety_pos_weight,
        )
    total_loss = (
        type_loss_weight * type_loss
        + state_loss_weight * state_loss
        + edge_loss_weight * edge_loss
        + hard_negative_edge_loss_weight * hard_negative_edge_loss
        + edge_ranking_loss_weight * edge_ranking_loss
        + trust_aux_loss_weight * trust_aux_loss
        + missed_edge_loss_weight * missed_edge_loss
        + rescue_safety_aux_loss_weight * rescue_safety_aux_loss
        + signed_witness_correction_loss_weight * signed_witness_correction_loss
        + false_admission_correction_loss_weight * false_admission_correction_loss
        + rescue_residual_contrast_loss_weight * rescue_residual_contrast_loss
        + safe_rescue_residual_preservation_loss_weight * safe_rescue_residual_preservation_loss
        + rescue_candidate_latent_loss_weight * rescue_candidate_latent_loss
        + rescue_candidate_binary_calibration_loss_weight * rescue_candidate_binary_calibration_loss
        + rescue_candidate_ambiguity_loss_weight * rescue_candidate_ambiguity_loss
        + weak_positive_ambiguity_safety_loss_weight * weak_positive_ambiguity_safety_loss
    )
    return {
        "total_loss": total_loss,
        "type_loss": type_loss,
        "state_loss": state_loss,
        "edge_loss": edge_loss,
        "hard_negative_edge_loss": hard_negative_edge_loss,
        "edge_ranking_loss": edge_ranking_loss,
        "trust_aux_loss": trust_aux_loss,
        "missed_edge_loss": missed_edge_loss,
        "rescue_safety_aux_loss": rescue_safety_aux_loss,
        "signed_witness_correction_loss": signed_witness_correction_loss,
        "false_admission_correction_loss": false_admission_correction_loss,
        "rescue_residual_contrast_loss": rescue_residual_contrast_loss,
        "safe_rescue_residual_preservation_loss": safe_rescue_residual_preservation_loss,
        "rescue_candidate_latent_loss": rescue_candidate_latent_loss,
        "rescue_candidate_binary_calibration_loss": rescue_candidate_binary_calibration_loss,
        "rescue_candidate_ambiguity_loss": rescue_candidate_ambiguity_loss,
        "weak_positive_ambiguity_safety_loss": weak_positive_ambiguity_safety_loss,
    }


@torch.no_grad()
def step30_recovery_metrics(
    outputs: Dict[str, torch.Tensor],
    target_node_feats: torch.Tensor,
    target_adj: torch.Tensor,
    node_mask: torch.Tensor,
    edge_threshold: float = 0.5,
    edge_pred_override: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> Dict[str, float]:
    target_type = target_node_feats[:, :, 0].long()
    target_state = target_node_feats[:, :, 1:]
    type_pred = outputs["type_logits"].argmax(dim=-1)

    node_correct = ((type_pred == target_type).float() * node_mask).sum().item()
    node_total = node_mask.sum().item()
    state_abs = (outputs["state_pred"] - target_state).abs() * node_mask.unsqueeze(-1)
    state_sq = ((outputs["state_pred"] - target_state) ** 2) * node_mask.unsqueeze(-1)
    state_total = node_mask.sum().item() * target_state.shape[-1]

    pair_mask = build_pair_mask(node_mask)
    edge_prob = torch.sigmoid(outputs["edge_logits"])
    if edge_pred_override is None:
        edge_pred = (edge_prob >= edge_threshold).float()
    else:
        edge_pred = edge_pred_override.float()
    edge_target = target_adj.float()
    edge_correct = ((edge_pred == edge_target).float() * pair_mask).sum().item()
    edge_total = pair_mask.sum().item()
    tp = (edge_pred * edge_target * pair_mask).sum().item()
    pred_pos = (edge_pred * pair_mask).sum().item()
    true_pos = (edge_target * pair_mask).sum().item()
    precision = tp / pred_pos if pred_pos > 0 else 0.0
    recall = tp / true_pos if true_pos > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall + eps) if precision + recall > 0 else 0.0

    return {
        "node_type_accuracy": node_correct / node_total if node_total > 0 else 0.0,
        "node_state_mae": state_abs.sum().item() / state_total if state_total > 0 else 0.0,
        "node_state_mse": state_sq.sum().item() / state_total if state_total > 0 else 0.0,
        "edge_accuracy": edge_correct / edge_total if edge_total > 0 else 0.0,
        "edge_precision": precision,
        "edge_recall": recall,
        "edge_f1": f1,
        "edge_tp": tp,
        "edge_pred_pos": pred_pos,
        "edge_true_pos": true_pos,
    }
