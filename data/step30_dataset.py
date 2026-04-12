from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from data.collate import pad_1d_mask, pad_adj, pad_node_features
from data.dataset import edge_pairs_to_dense_mask


class Step30WeakObservationDataset(Dataset):
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
        weak_obs = raw["weak_observation"]
        graph_t = raw["graph_t"]
        graph_t1 = raw.get("graph_t1", None)

        target_node_feats = self._to_float_tensor(graph_t["node_features"])
        target_adj = self._to_float_tensor(graph_t["adj"])
        num_nodes = target_node_feats.shape[0]

        item: Dict[str, Any] = {
            "weak_slot_features": self._to_float_tensor(weak_obs["slot_features"]),
            "weak_relation_hints": self._to_float_tensor(weak_obs["relation_hints"]),
            "target_node_feats": target_node_feats,
            "target_adj": target_adj,
            "step30_observation_variant": raw.get(
                "step30_observation_variant",
                weak_obs.get("variant", "unknown"),
            ),
            "step30_benchmark_version": raw.get("step30_benchmark_version", "unknown"),
            "step30_source_sample_index": raw.get("step30_source_sample_index", idx),
        }
        if "pair_support_hints" in weak_obs:
            item["weak_pair_support_hints"] = self._to_float_tensor(weak_obs["pair_support_hints"])
        if "signed_pair_witness" in weak_obs:
            item["weak_signed_pair_witness"] = self._to_float_tensor(weak_obs["signed_pair_witness"])
        if "pair_evidence_bundle" in weak_obs:
            item["weak_pair_evidence_bundle"] = self._to_float_tensor(weak_obs["pair_evidence_bundle"])
        if "positive_ambiguity_safety_hint" in weak_obs:
            item["weak_positive_ambiguity_safety_hint"] = self._to_float_tensor(
                weak_obs["positive_ambiguity_safety_hint"]
            )

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


def step30_weak_observation_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    max_nodes = max(b["target_node_feats"].shape[0] for b in batch)
    batch_size = len(batch)

    out["weak_slot_features"] = torch.stack(
        [pad_node_features(b["weak_slot_features"], max_nodes) for b in batch],
        dim=0,
    )
    out["weak_relation_hints"] = torch.stack(
        [pad_adj(b["weak_relation_hints"], max_nodes) for b in batch],
        dim=0,
    )
    if all("weak_pair_support_hints" in b for b in batch):
        out["weak_pair_support_hints"] = torch.stack(
            [pad_adj(b["weak_pair_support_hints"], max_nodes) for b in batch],
            dim=0,
        )
    if all("weak_signed_pair_witness" in b for b in batch):
        out["weak_signed_pair_witness"] = torch.stack(
            [pad_adj(b["weak_signed_pair_witness"], max_nodes) for b in batch],
            dim=0,
        )
    if all("weak_pair_evidence_bundle" in b for b in batch):
        bundle_dim = int(batch[0]["weak_pair_evidence_bundle"].shape[-1])
        padded_bundles = []
        for b in batch:
            bundle = b["weak_pair_evidence_bundle"]
            padded = torch.zeros((max_nodes, max_nodes, bundle_dim), dtype=bundle.dtype)
            num_nodes = bundle.shape[0]
            padded[:num_nodes, :num_nodes, :] = bundle
            padded_bundles.append(padded)
        out["weak_pair_evidence_bundle"] = torch.stack(padded_bundles, dim=0)
    if all("weak_positive_ambiguity_safety_hint" in b for b in batch):
        out["weak_positive_ambiguity_safety_hint"] = torch.stack(
            [pad_adj(b["weak_positive_ambiguity_safety_hint"], max_nodes) for b in batch],
            dim=0,
        )
    out["target_node_feats"] = torch.stack(
        [pad_node_features(b["target_node_feats"], max_nodes) for b in batch],
        dim=0,
    )
    out["target_adj"] = torch.stack(
        [pad_adj(b["target_adj"], max_nodes) for b in batch],
        dim=0,
    )

    if all("next_node_feats" in b for b in batch):
        out["next_node_feats"] = torch.stack(
            [pad_node_features(b["next_node_feats"], max_nodes) for b in batch],
            dim=0,
        )
    if all("next_adj" in b for b in batch):
        out["next_adj"] = torch.stack(
            [pad_adj(b["next_adj"], max_nodes) for b in batch],
            dim=0,
        )

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
        "step30_observation_variant",
        "step30_benchmark_version",
        "step30_source_sample_index",
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
