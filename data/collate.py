from __future__ import annotations

from typing import Any, Dict, List

import torch


def pad_node_features(x: torch.Tensor, max_nodes: int) -> torch.Tensor:
    """
    x: [N, F]
    return: [max_nodes, F]
    """
    n, f = x.shape
    out = torch.zeros((max_nodes, f), dtype=x.dtype)
    out[:n] = x
    return out


def pad_adj(adj: torch.Tensor, max_nodes: int) -> torch.Tensor:
    """
    adj: [N, N]
    return: [max_nodes, max_nodes]
    """
    n = adj.shape[0]
    out = torch.zeros((max_nodes, max_nodes), dtype=adj.dtype)
    out[:n, :n] = adj
    return out


def pad_1d_mask(mask: torch.Tensor, max_nodes: int) -> torch.Tensor:
    """
    mask: [N]
    return: [max_nodes]
    """
    n = mask.shape[0]
    out = torch.zeros((max_nodes,), dtype=mask.dtype)
    out[:n] = mask
    return out


def graph_event_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    max_nodes = max(b["node_feats"].shape[0] for b in batch)
    batch_size = len(batch)

    # always-present tensors
    out["node_feats"] = torch.stack(
        [pad_node_features(b["node_feats"], max_nodes) for b in batch], dim=0
    )
    out["adj"] = torch.stack(
        [pad_adj(b["adj"], max_nodes) for b in batch], dim=0
    )
    out["next_node_feats"] = torch.stack(
        [pad_node_features(b["next_node_feats"], max_nodes) for b in batch], dim=0
    )
    out["next_adj"] = torch.stack(
        [pad_adj(b["next_adj"], max_nodes) for b in batch], dim=0
    )
    if all("obs_node_feats" in b for b in batch):
        out["obs_node_feats"] = torch.stack(
            [pad_node_features(b["obs_node_feats"], max_nodes) for b in batch], dim=0
        )
    if all("obs_adj" in b for b in batch):
        out["obs_adj"] = torch.stack(
            [pad_adj(b["obs_adj"], max_nodes) for b in batch], dim=0
        )

    # node mask: 1 for real nodes, 0 for padded nodes
    node_mask = torch.zeros((batch_size, max_nodes), dtype=torch.float32)
    for i, b in enumerate(batch):
        n = b["node_feats"].shape[0]
        node_mask[i, :n] = 1.0
    out["node_mask"] = node_mask

    # optional tensor fields
    optional_1d_keys = [
        "changed_nodes",
        "event_scope_union_nodes",
    ]
    for key in optional_1d_keys:
        if all(key in b for b in batch):
            out[key] = torch.stack(
                [pad_1d_mask(b[key], max_nodes) for b in batch], dim=0
            )

    optional_2d_keys = [
        "changed_edges",
        "event_scope_union_edges",
    ]
    for key in optional_2d_keys:
        if all(key in b for b in batch):
            out[key] = torch.stack(
                [pad_adj(b[key], max_nodes) for b in batch], dim=0
            )

    # metadata
    meta_keys = [
        "events",
        "num_events",
        "independent_pairs",
        "step3_pair_id",
        "step3_ordered_variant",
        "step3_ordered_signature",
        "step3_unordered_signature",
        "step3_base_graph_id",
        "step3_event_specs",
        "step3_pair_event_specs",
        "step3_transition_role",
        "step3_primary_event_index",
        "step3_primary_event_type",
        "step5_sample_id",
        "step5_ordered_signature",
        "step5_unordered_signature",
        "step5_dependency_bucket",
        "step5_dependency_reason",
        "step5_pairwise_scope_overlaps",
        "step5_event_valid_on_base",
        "step6a_corruption_setting",
        "step6a_corruption_config",
        "step6a_source_sample_index",
    ]
    for key in meta_keys:
        if all(key in b for b in batch):
            out[key] = [b[key] for b in batch]

    return out
