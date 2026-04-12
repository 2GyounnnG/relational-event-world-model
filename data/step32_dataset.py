from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from data.collate import pad_1d_mask, pad_adj, pad_node_features
from data.dataset import edge_pairs_to_dense_mask


class Step32RenderedObservationDataset(Dataset):
    def __init__(
        self,
        file_path: str | Path,
        undirected: bool = True,
        device: Optional[torch.device] = None,
    ):
        self.file_path = Path(file_path)
        self.undirected = undirected
        self.device = device

        with open(self.file_path, "rb") as f:
            self.samples: List[Dict[str, Any]] = pickle.load(f)

        if len(self.samples) == 0:
            raise ValueError(f"Dataset is empty: {self.file_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        raw = self.samples[idx]
        rendered = raw["rendered_observation"]
        views = rendered["views"]
        graph_t = raw["graph_t"]
        graph_t1 = raw.get("graph_t1", None)

        target_node_feats = self._to_float_tensor(graph_t["node_features"])
        target_adj = self._to_float_tensor(graph_t["adj"])
        num_nodes = target_node_feats.shape[0]

        item: Dict[str, Any] = {
            "rendered_images": torch.stack(
                [self._to_float_tensor(view["image"]) for view in views],
                dim=0,
            ),
            "rendered_node_positions": torch.stack(
                [self._to_float_tensor(view["node_positions"]) for view in views],
                dim=0,
            ),
            "rendered_visible_node_mask": torch.stack(
                [self._to_float_tensor(view["visible_node_mask"]) for view in views],
                dim=0,
            ),
            "rendered_trivial_relation_scores": torch.stack(
                [self._to_float_tensor(view["trivial_relation_scores"]) for view in views],
                dim=0,
            ),
            "rendered_trivial_support_scores": torch.stack(
                [self._to_float_tensor(view["trivial_support_scores"]) for view in views],
                dim=0,
            ),
            "target_node_feats": target_node_feats,
            "target_adj": target_adj,
            "step32_observation_variant": raw.get("step32_observation_variant", "unknown"),
            "step32_benchmark_version": raw.get("step32_benchmark_version", "unknown"),
            "step32_source_sample_index": raw.get("step32_source_sample_index", idx),
            "step32_view_profiles": rendered.get(
                "view_profiles",
                [view.get("view_profile", f"view_{i}") for i, view in enumerate(views)],
            ),
        }

        if graph_t1 is not None:
            item["next_node_feats"] = self._to_float_tensor(graph_t1["node_features"])
            item["next_adj"] = self._to_float_tensor(graph_t1["adj"])

        changed_nodes = raw.get("changed_nodes", None)
        if changed_nodes is not None:
            item["changed_nodes"] = self._to_float_tensor(changed_nodes).view(num_nodes)

        changed_edges = raw.get("changed_edges", None)
        if changed_edges is not None:
            item["changed_edges"] = edge_pairs_to_dense_mask(
                changed_edges,
                num_nodes=num_nodes,
                undirected=self.undirected,
            )

        scope_nodes = raw.get("event_scope_union_nodes", None)
        if scope_nodes is not None:
            scope_node_mask = torch.zeros(num_nodes, dtype=torch.float32)
            for node_idx in scope_nodes:
                scope_node_mask[int(node_idx)] = 1.0
            item["event_scope_union_nodes"] = scope_node_mask

        scope_edges = raw.get("event_scope_union_edges", None)
        if scope_edges is not None:
            item["event_scope_union_edges"] = edge_pairs_to_dense_mask(
                scope_edges,
                num_nodes=num_nodes,
                undirected=self.undirected,
            )

        for key in [
            "events",
            "step5_sample_id",
            "step5_dependency_bucket",
            "step6a_corruption_setting",
            "step22_corruption_setting",
        ]:
            if key in raw:
                item[key] = raw[key]

        if self.device is not None:
            item = self._move_to_device(item, self.device)

        return item

    @staticmethod
    def _to_float_tensor(x: Any) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.float()
        return torch.tensor(x, dtype=torch.float32)

    @staticmethod
    def _move_to_device(item: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        out = {}
        for key, value in item.items():
            if isinstance(value, torch.Tensor):
                out[key] = value.to(device)
            else:
                out[key] = value
        return out


def _pad_rendered_adj_stack(x: torch.Tensor, max_nodes: int) -> torch.Tensor:
    num_views = x.shape[0]
    out = torch.zeros((num_views, max_nodes, max_nodes), dtype=x.dtype)
    num_nodes = x.shape[1]
    out[:, :num_nodes, :num_nodes] = x
    return out


def _pad_positions(x: torch.Tensor, max_nodes: int) -> torch.Tensor:
    num_views = x.shape[0]
    out = torch.zeros((num_views, max_nodes, 2), dtype=x.dtype)
    num_nodes = x.shape[1]
    out[:, :num_nodes] = x
    return out


def _pad_visible(x: torch.Tensor, max_nodes: int) -> torch.Tensor:
    num_views = x.shape[0]
    out = torch.zeros((num_views, max_nodes), dtype=x.dtype)
    num_nodes = x.shape[1]
    out[:, :num_nodes] = x
    return out


def step32_rendered_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    max_nodes = max(b["target_node_feats"].shape[0] for b in batch)
    batch_size = len(batch)

    out["rendered_images"] = torch.stack([b["rendered_images"] for b in batch], dim=0)
    out["rendered_node_positions"] = torch.stack(
        [_pad_positions(b["rendered_node_positions"], max_nodes) for b in batch],
        dim=0,
    )
    out["rendered_visible_node_mask"] = torch.stack(
        [_pad_visible(b["rendered_visible_node_mask"], max_nodes) for b in batch],
        dim=0,
    )
    out["rendered_trivial_relation_scores"] = torch.stack(
        [_pad_rendered_adj_stack(b["rendered_trivial_relation_scores"], max_nodes) for b in batch],
        dim=0,
    )
    out["rendered_trivial_support_scores"] = torch.stack(
        [_pad_rendered_adj_stack(b["rendered_trivial_support_scores"], max_nodes) for b in batch],
        dim=0,
    )
    out["target_node_feats"] = torch.stack(
        [pad_node_features(b["target_node_feats"], max_nodes) for b in batch],
        dim=0,
    )
    out["target_adj"] = torch.stack([pad_adj(b["target_adj"], max_nodes) for b in batch], dim=0)

    if all("next_node_feats" in b for b in batch):
        out["next_node_feats"] = torch.stack(
            [pad_node_features(b["next_node_feats"], max_nodes) for b in batch],
            dim=0,
        )
    if all("next_adj" in b for b in batch):
        out["next_adj"] = torch.stack([pad_adj(b["next_adj"], max_nodes) for b in batch], dim=0)

    node_mask = torch.zeros((batch_size, max_nodes), dtype=torch.float32)
    for batch_idx, b in enumerate(batch):
        num_nodes = b["target_node_feats"].shape[0]
        node_mask[batch_idx, :num_nodes] = 1.0
    out["node_mask"] = node_mask

    for key in ["changed_nodes", "event_scope_union_nodes"]:
        if all(key in b for b in batch):
            out[key] = torch.stack([pad_1d_mask(b[key], max_nodes) for b in batch], dim=0)

    for key in ["changed_edges", "event_scope_union_edges"]:
        if all(key in b for b in batch):
            out[key] = torch.stack([pad_adj(b[key], max_nodes) for b in batch], dim=0)

    meta_keys = [
        "step32_observation_variant",
        "step32_benchmark_version",
        "step32_source_sample_index",
        "step32_view_profiles",
        "events",
        "step5_sample_id",
        "step5_dependency_bucket",
        "step6a_corruption_setting",
        "step22_corruption_setting",
    ]
    for key in meta_keys:
        if all(key in b for b in batch):
            out[key] = [b[key] for b in batch]

    return out
