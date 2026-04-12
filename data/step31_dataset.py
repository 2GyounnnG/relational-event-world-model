from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from data.collate import pad_1d_mask, pad_adj, pad_node_features
from data.dataset import edge_pairs_to_dense_mask


class Step31MultiViewObservationDataset(Dataset):
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
        mv_obs = raw["multi_view_observation"]
        views = mv_obs["views"]
        graph_t = raw["graph_t"]
        graph_t1 = raw.get("graph_t1", None)

        target_node_feats = self._to_float_tensor(graph_t["node_features"])
        target_adj = self._to_float_tensor(graph_t["adj"])
        num_nodes = target_node_feats.shape[0]

        item: Dict[str, Any] = {
            "multi_view_slot_features": torch.stack(
                [self._to_float_tensor(view["slot_features"]) for view in views],
                dim=0,
            ),
            "multi_view_relation_hints": torch.stack(
                [self._to_float_tensor(view["relation_hints"]) for view in views],
                dim=0,
            ),
            "multi_view_pair_support_hints": torch.stack(
                [self._to_float_tensor(view["pair_support_hints"]) for view in views],
                dim=0,
            ),
            "multi_view_signed_pair_witness": torch.stack(
                [self._to_float_tensor(view["signed_pair_witness"]) for view in views],
                dim=0,
            ),
            "multi_view_pair_evidence_bundle": torch.stack(
                [self._to_float_tensor(view["pair_evidence_bundle"]) for view in views],
                dim=0,
            ),
            "target_node_feats": target_node_feats,
            "target_adj": target_adj,
            "step31_observation_variant": raw.get(
                "step31_observation_variant",
                raw.get("step30_observation_variant", views[0].get("variant", "unknown")),
            ),
            "step31_benchmark_version": raw.get("step31_benchmark_version", "unknown"),
            "step31_source_sample_index": raw.get("step31_source_sample_index", idx),
            "step31_view_profiles": raw.get(
                "step31_view_profiles",
                [view.get("view_profile", f"view_{i}") for i, view in enumerate(views)],
            ),
        }

        # First view in Step30-compatible names for simple single-view baselines.
        first_view = views[0]
        item["weak_slot_features"] = self._to_float_tensor(first_view["slot_features"])
        item["weak_relation_hints"] = self._to_float_tensor(first_view["relation_hints"])
        item["weak_pair_support_hints"] = self._to_float_tensor(first_view["pair_support_hints"])
        item["weak_signed_pair_witness"] = self._to_float_tensor(first_view["signed_pair_witness"])
        item["weak_pair_evidence_bundle"] = self._to_float_tensor(first_view["pair_evidence_bundle"])

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


def _pad_multi_view_node_features(x: torch.Tensor, max_nodes: int) -> torch.Tensor:
    views = [pad_node_features(x[view_idx], max_nodes) for view_idx in range(x.shape[0])]
    return torch.stack(views, dim=0)


def _pad_multi_view_adj(x: torch.Tensor, max_nodes: int) -> torch.Tensor:
    views = [pad_adj(x[view_idx], max_nodes) for view_idx in range(x.shape[0])]
    return torch.stack(views, dim=0)


def _pad_multi_view_bundle(x: torch.Tensor, max_nodes: int) -> torch.Tensor:
    num_views = x.shape[0]
    bundle_dim = x.shape[-1]
    padded = torch.zeros((num_views, max_nodes, max_nodes, bundle_dim), dtype=x.dtype)
    num_nodes = x.shape[1]
    padded[:, :num_nodes, :num_nodes, :] = x
    return padded


def step31_multi_view_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    max_nodes = max(b["target_node_feats"].shape[0] for b in batch)
    batch_size = len(batch)

    out["multi_view_slot_features"] = torch.stack(
        [_pad_multi_view_node_features(b["multi_view_slot_features"], max_nodes) for b in batch],
        dim=0,
    )
    for key in [
        "multi_view_relation_hints",
        "multi_view_pair_support_hints",
        "multi_view_signed_pair_witness",
    ]:
        out[key] = torch.stack([_pad_multi_view_adj(b[key], max_nodes) for b in batch], dim=0)
    out["multi_view_pair_evidence_bundle"] = torch.stack(
        [_pad_multi_view_bundle(b["multi_view_pair_evidence_bundle"], max_nodes) for b in batch],
        dim=0,
    )

    out["weak_slot_features"] = torch.stack(
        [pad_node_features(b["weak_slot_features"], max_nodes) for b in batch],
        dim=0,
    )
    for key in [
        "weak_relation_hints",
        "weak_pair_support_hints",
        "weak_signed_pair_witness",
    ]:
        out[key] = torch.stack([pad_adj(b[key], max_nodes) for b in batch], dim=0)
    bundle_dim = int(batch[0]["weak_pair_evidence_bundle"].shape[-1])
    padded_bundles = []
    for b in batch:
        bundle = b["weak_pair_evidence_bundle"]
        padded = torch.zeros((max_nodes, max_nodes, bundle_dim), dtype=bundle.dtype)
        num_nodes = bundle.shape[0]
        padded[:num_nodes, :num_nodes, :] = bundle
        padded_bundles.append(padded)
    out["weak_pair_evidence_bundle"] = torch.stack(padded_bundles, dim=0)

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
        "step31_observation_variant",
        "step31_benchmark_version",
        "step31_source_sample_index",
        "step31_view_profiles",
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
