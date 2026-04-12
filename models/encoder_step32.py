from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.baselines import build_mlp


@dataclass
class Step32RenderedBridgeConfig:
    image_channels: int = 4
    num_views: int = 2
    num_node_types: int = 3
    state_dim: int = 4
    hidden_dim: int = 64
    node_head_layers: int = 2
    edge_head_layers: int = 2
    line_samples: int = 7
    dropout: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Step32RenderedObservationEncoder(nn.Module):
    """Tiny image-like bridge from synthetic rendered views to graph recovery.

    The model uses a shared small CNN for each rendered view, samples node
    descriptors at known synthetic layout coordinates, and samples lightweight
    line descriptors between node pairs. It intentionally stays small so Step32
    tests the rendered evidence substrate rather than a large vision stack.
    """

    def __init__(self, config: Step32RenderedBridgeConfig):
        super().__init__()
        self.config = config
        h = int(config.hidden_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(config.image_channels, h // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(h // 2, h // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(h // 2, h, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(h, h, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.node_fusion = build_mlp(
            in_dim=h * 2 + 1,
            hidden_dim=h,
            out_dim=h,
            num_layers=2,
            dropout=config.dropout,
        )
        self.type_head = build_mlp(
            in_dim=h,
            hidden_dim=h,
            out_dim=config.num_node_types,
            num_layers=config.node_head_layers,
            dropout=config.dropout,
        )
        self.state_head = build_mlp(
            in_dim=h,
            hidden_dim=h,
            out_dim=config.state_dim,
            num_layers=config.node_head_layers,
            dropout=config.dropout,
        )
        edge_in_dim = h * 6 + 1
        self.edge_head = build_mlp(
            in_dim=edge_in_dim,
            hidden_dim=h,
            out_dim=1,
            num_layers=config.edge_head_layers,
            dropout=config.dropout,
        )

    @staticmethod
    def _to_grid(coords01: torch.Tensor) -> torch.Tensor:
        return coords01.mul(2.0).sub(1.0)

    def sample_node_features(self, feature_maps: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        batch_views, channels, _, _ = feature_maps.shape
        num_nodes = positions.shape[1]
        grid = self._to_grid(positions).view(batch_views, num_nodes, 1, 2)
        sampled = F.grid_sample(
            feature_maps,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return sampled.squeeze(-1).transpose(1, 2)

    def sample_line_features(self, feature_maps: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        batch_views, channels, _, _ = feature_maps.shape
        num_nodes = positions.shape[1]
        samples = int(self.config.line_samples)
        t = torch.linspace(0.0, 1.0, samples, device=positions.device, dtype=positions.dtype)
        p_i = positions.unsqueeze(2).unsqueeze(3)
        p_j = positions.unsqueeze(1).unsqueeze(3)
        line_points = p_i * (1.0 - t.view(1, 1, 1, samples, 1)) + p_j * t.view(1, 1, 1, samples, 1)
        grid = self._to_grid(line_points).reshape(batch_views, num_nodes * num_nodes, samples, 2)
        sampled = F.grid_sample(
            feature_maps,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        sampled = sampled.reshape(
            batch_views,
            channels,
            num_nodes,
            num_nodes,
            samples,
        )
        return sampled.mean(dim=-1).permute(0, 2, 3, 1)

    def forward(
        self,
        rendered_images: torch.Tensor,
        rendered_node_positions: torch.Tensor,
        rendered_visible_node_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size, num_views, channels, height, width = rendered_images.shape
        num_nodes = rendered_node_positions.shape[2]
        flat_images = rendered_images.reshape(batch_size * num_views, channels, height, width)
        flat_positions = rendered_node_positions.reshape(batch_size * num_views, num_nodes, 2)
        flat_visible = rendered_visible_node_mask.reshape(batch_size * num_views, num_nodes)

        feature_maps = self.cnn(flat_images)
        node_per_view = self.sample_node_features(feature_maps, flat_positions).reshape(
            batch_size,
            num_views,
            num_nodes,
            int(self.config.hidden_dim),
        )
        line_per_view = self.sample_line_features(feature_maps, flat_positions).reshape(
            batch_size,
            num_views,
            num_nodes,
            num_nodes,
            int(self.config.hidden_dim),
        )

        node_mean = node_per_view.mean(dim=1)
        node_std = node_per_view.std(dim=1, unbiased=False)
        visibility = flat_visible.reshape(batch_size, num_views, num_nodes).mean(dim=1).unsqueeze(-1)
        node_latents = self.node_fusion(torch.cat([node_mean, node_std, visibility], dim=-1))

        type_logits = self.type_head(node_latents)
        state_pred = self.state_head(node_latents)

        line_mean = line_per_view.mean(dim=1)
        line_std = line_per_view.std(dim=1, unbiased=False)
        h_i = node_latents.unsqueeze(2).expand(batch_size, num_nodes, num_nodes, int(self.config.hidden_dim))
        h_j = node_latents.unsqueeze(1).expand(batch_size, num_nodes, num_nodes, int(self.config.hidden_dim))
        p_i = rendered_node_positions[:, 0].unsqueeze(2)
        p_j = rendered_node_positions[:, 0].unsqueeze(1)
        distance = torch.linalg.vector_norm(p_i - p_j, dim=-1, keepdim=True)
        pair_features = torch.cat(
            [
                h_i,
                h_j,
                torch.abs(h_i - h_j),
                h_i * h_j,
                line_mean,
                line_std,
                distance,
            ],
            dim=-1,
        )
        edge_logits = self.edge_head(pair_features).squeeze(-1)
        edge_logits = 0.5 * (edge_logits + edge_logits.transpose(1, 2))
        diag_mask = torch.eye(num_nodes, device=edge_logits.device, dtype=torch.bool).unsqueeze(0)
        edge_logits = edge_logits.masked_fill(diag_mask, -1e9)

        return {
            "node_latents": node_latents,
            "type_logits": type_logits,
            "state_pred": state_pred,
            "edge_logits": edge_logits,
            "rendered_line_feature_mean": line_mean,
            "rendered_line_feature_std": line_std,
        }
