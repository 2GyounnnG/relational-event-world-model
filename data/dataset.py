from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset


def edge_pairs_to_dense_mask(
    edge_pairs: List[List[int]] | List[tuple[int, int]],
    num_nodes: int,
    undirected: bool = True,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    mask = torch.zeros((num_nodes, num_nodes), dtype=dtype)

    for u, v in edge_pairs:
        mask[u, v] = 1.0
        if undirected:
            mask[v, u] = 1.0

    return mask


class GraphEventDataset(Dataset):
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

        graph_t = raw["graph_t"]
        graph_t1 = raw["graph_t1"]

        node_feats = self._to_float_tensor(graph_t["node_features"])      # [N, F]
        next_node_feats = self._to_float_tensor(graph_t1["node_features"])  # [N, F]

        adj = self._to_float_tensor(graph_t["adj"])       # [N, N]
        next_adj = self._to_float_tensor(graph_t1["adj"])   # [N, N]

        num_nodes = node_feats.shape[0]

        item: Dict[str, Any] = {
            "node_feats": node_feats,
            "adj": adj,
            "next_node_feats": next_node_feats,
            "next_adj": next_adj,
        }

        obs_graph_t = raw.get("obs_graph_t", None)
        if obs_graph_t is not None:
            item["obs_node_feats"] = self._to_float_tensor(obs_graph_t["node_features"])
            item["obs_adj"] = self._to_float_tensor(obs_graph_t["adj"])

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
            for n in scope_nodes:
                scope_node_mask[n] = 1.0
            item["event_scope_union_nodes"] = scope_node_mask

        scope_edges = raw.get("event_scope_union_edges", None)
        if scope_edges is not None:
            item["event_scope_union_edges"] = edge_pairs_to_dense_mask(
                scope_edges,
                num_nodes=num_nodes,
                undirected=self.undirected,
            )

        # metadata
        if "events" in raw:
            item["events"] = raw["events"]
        if "independent_pairs" in raw:
            item["independent_pairs"] = raw["independent_pairs"]
        if "step3_pair_id" in raw:
            item["step3_pair_id"] = raw["step3_pair_id"]
        if "step3_ordered_variant" in raw:
            item["step3_ordered_variant"] = raw["step3_ordered_variant"]
        if "step3_ordered_signature" in raw:
            item["step3_ordered_signature"] = raw["step3_ordered_signature"]
        if "step3_unordered_signature" in raw:
            item["step3_unordered_signature"] = raw["step3_unordered_signature"]
        if "step3_base_graph_id" in raw:
            item["step3_base_graph_id"] = raw["step3_base_graph_id"]
        if "step3_event_specs" in raw:
            item["step3_event_specs"] = raw["step3_event_specs"]
        if "step3_pair_event_specs" in raw:
            item["step3_pair_event_specs"] = raw["step3_pair_event_specs"]
        if "step3_transition_role" in raw:
            item["step3_transition_role"] = raw["step3_transition_role"]
        if "step3_primary_event_index" in raw:
            item["step3_primary_event_index"] = raw["step3_primary_event_index"]
        if "step3_primary_event_type" in raw:
            item["step3_primary_event_type"] = raw["step3_primary_event_type"]
        if "step5_sample_id" in raw:
            item["step5_sample_id"] = raw["step5_sample_id"]
        if "step5_ordered_signature" in raw:
            item["step5_ordered_signature"] = raw["step5_ordered_signature"]
        if "step5_unordered_signature" in raw:
            item["step5_unordered_signature"] = raw["step5_unordered_signature"]
        if "step5_dependency_bucket" in raw:
            item["step5_dependency_bucket"] = raw["step5_dependency_bucket"]
        if "step5_dependency_reason" in raw:
            item["step5_dependency_reason"] = raw["step5_dependency_reason"]
        if "step5_pairwise_scope_overlaps" in raw:
            item["step5_pairwise_scope_overlaps"] = raw["step5_pairwise_scope_overlaps"]
        if "step5_event_valid_on_base" in raw:
            item["step5_event_valid_on_base"] = raw["step5_event_valid_on_base"]
        if "step6a_corruption_setting" in raw:
            item["step6a_corruption_setting"] = raw["step6a_corruption_setting"]
        if "step6a_corruption_config" in raw:
            item["step6a_corruption_config"] = raw["step6a_corruption_config"]
        if "step6a_source_sample_index" in raw:
            item["step6a_source_sample_index"] = raw["step6a_source_sample_index"]

        # optional convenience metadata
        item["num_events"] = len(raw["events"]) if "events" in raw else None

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
        for k, v in item.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(device)
            else:
                out[k] = v
        return out
