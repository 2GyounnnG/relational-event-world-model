from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.oracle_local_delta import (
    OracleLocalDeltaRewriteConfig,
    OracleLocalDeltaRewriteModel,
    build_valid_edge_mask,
)
from models.proposal import ScopeProposalConfig, ScopeProposalModel
from train.eval_noisy_structured_observation import (
    build_loader,
    get_device,
    move_batch_to_device,
    require_keys,
    resolve_path,
    save_json,
)


EVENT_TYPE_ORDER = (
    "node_state_update",
    "edge_add",
    "edge_delete",
    "motif_type_flip",
)


def safe_div(num: float, den: float) -> Optional[float]:
    if den <= 0:
        return None
    return num / den


def safe_event_type_name(x: Any) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        for key in ["event_type", "type", "name", "kind", "event_name"]:
            if key in x:
                return str(x[key])
        if len(x) == 1:
            return str(next(iter(x.keys())))
    return str(x)


def extract_event_type_list(events_item: Any) -> list[str]:
    if events_item is None:
        return []
    if not isinstance(events_item, list):
        return [safe_event_type_name(events_item)]
    return [safe_event_type_name(e) for e in events_item]


def slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", text).strip("_")


def init_bucket() -> Dict[str, float]:
    return {
        "num_samples": 0.0,
        "node_changed_total": 0.0,
        "node_changed_covered": 0.0,
        "node_pred_total": 0.0,
        "node_pred_context_total": 0.0,
        "node_out_scope_changed_total": 0.0,
        "node_in_scope_changed_total": 0.0,
        "node_in_scope_changed_type_wrong": 0.0,
        "node_in_scope_changed_state_not_improved": 0.0,
        "node_in_scope_changed_state_pred_error_sum": 0.0,
        "node_in_scope_changed_state_copy_error_sum": 0.0,
        "node_in_scope_changed_state_count": 0.0,
        "node_in_scope_context_total": 0.0,
        "node_in_scope_context_type_wrong": 0.0,
        "node_in_scope_context_state_error_sum": 0.0,
        "node_in_scope_context_state_count": 0.0,
        "edge_changed_total": 0.0,
        "edge_changed_covered": 0.0,
        "edge_pred_total": 0.0,
        "edge_pred_context_total": 0.0,
        "edge_out_scope_changed_total": 0.0,
        "edge_in_scope_changed_total": 0.0,
        "edge_in_scope_changed_wrong": 0.0,
        "edge_in_scope_context_total": 0.0,
        "edge_in_scope_context_wrong": 0.0,
    }


def update_bucket(bucket: Dict[str, float], sample_stats: Dict[str, float]) -> None:
    for key, value in sample_stats.items():
        bucket[key] += value


def finalize_bucket(bucket: Dict[str, float]) -> Dict[str, Any]:
    return {
        "num_samples": int(bucket["num_samples"]),
        "proposal_changed_region_recall_node": safe_div(
            bucket["node_changed_covered"], bucket["node_changed_total"]
        ),
        "proposal_changed_region_recall_edge": safe_div(
            bucket["edge_changed_covered"], bucket["edge_changed_total"]
        ),
        "proposal_scope_excess_ratio_node": safe_div(
            bucket["node_pred_context_total"], bucket["node_pred_total"]
        ),
        "proposal_scope_excess_ratio_edge": safe_div(
            bucket["edge_pred_context_total"], bucket["edge_pred_total"]
        ),
        "out_of_scope_miss_node": safe_div(
            bucket["node_out_scope_changed_total"], bucket["node_changed_total"]
        ),
        "out_of_scope_miss_edge": safe_div(
            bucket["edge_out_scope_changed_total"], bucket["edge_changed_total"]
        ),
        "in_scope_under_edit_node_type": safe_div(
            bucket["node_in_scope_changed_type_wrong"], bucket["node_in_scope_changed_total"]
        ),
        # 节点状态是连续值，这里不用“完全正确”二值化，而是看是否至少比 copy/current 更接近目标。
        "in_scope_under_edit_node_state_not_improved": safe_div(
            bucket["node_in_scope_changed_state_not_improved"], bucket["node_in_scope_changed_state_count"]
        ),
        "in_scope_under_edit_node_state_pred_mae": safe_div(
            bucket["node_in_scope_changed_state_pred_error_sum"], bucket["node_in_scope_changed_state_count"]
        ),
        "in_scope_under_edit_node_state_copy_mae": safe_div(
            bucket["node_in_scope_changed_state_copy_error_sum"], bucket["node_in_scope_changed_state_count"]
        ),
        "in_scope_over_edit_node_type": safe_div(
            bucket["node_in_scope_context_type_wrong"], bucket["node_in_scope_context_total"]
        ),
        "in_scope_over_edit_node_state_mae": safe_div(
            bucket["node_in_scope_context_state_error_sum"], bucket["node_in_scope_context_state_count"]
        ),
        "in_scope_under_edit_edge": safe_div(
            bucket["edge_in_scope_changed_wrong"], bucket["edge_in_scope_changed_total"]
        ),
        "in_scope_over_edit_edge": safe_div(
            bucket["edge_in_scope_context_wrong"], bucket["edge_in_scope_context_total"]
        ),
        # 用 share_of_gt_changed 可以直接与 proposal miss 对比，帮助判断主导瓶颈。
        "in_scope_under_edit_node_type_share_of_gt_changed": safe_div(
            bucket["node_in_scope_changed_type_wrong"], bucket["node_changed_total"]
        ),
        "in_scope_under_edit_node_state_not_improved_share_of_gt_changed": safe_div(
            bucket["node_in_scope_changed_state_not_improved"], bucket["node_changed_total"]
        ),
        "in_scope_under_edit_edge_share_of_gt_changed": safe_div(
            bucket["edge_in_scope_changed_wrong"], bucket["edge_changed_total"]
        ),
        "node_changed_total": int(bucket["node_changed_total"]),
        "node_pred_total": int(bucket["node_pred_total"]),
        "node_in_scope_changed_total": int(bucket["node_in_scope_changed_total"]),
        "node_in_scope_context_total": int(bucket["node_in_scope_context_total"]),
        "edge_changed_total": int(bucket["edge_changed_total"]),
        "edge_pred_total": int(bucket["edge_pred_total"]),
        "edge_in_scope_changed_total": int(bucket["edge_in_scope_changed_total"]),
        "edge_in_scope_context_total": int(bucket["edge_in_scope_context_total"]),
    }


def build_sample_stats(
    batch: Dict[str, Any],
    rewrite_outputs: Dict[str, torch.Tensor],
    pred_scope_nodes: torch.Tensor,
    pred_scope_edges: torch.Tensor,
    sample_idx: int,
) -> Dict[str, float]:
    node_mask = batch["node_mask"][sample_idx].bool()
    valid_edge_mask = build_valid_edge_mask(batch["node_mask"][sample_idx : sample_idx + 1])[0].bool()

    changed_nodes = (batch["changed_nodes"][sample_idx] > 0.5) & node_mask
    changed_edges = (batch["changed_edges"][sample_idx] > 0.5) & valid_edge_mask
    context_nodes = (~changed_nodes) & node_mask
    context_edges = (~changed_edges) & valid_edge_mask

    pred_scope_node_mask = pred_scope_nodes[sample_idx] & node_mask
    pred_scope_edge_mask = pred_scope_edges[sample_idx] & valid_edge_mask

    in_scope_changed_nodes = changed_nodes & pred_scope_node_mask
    in_scope_context_nodes = context_nodes & pred_scope_node_mask
    out_of_scope_changed_nodes = changed_nodes & (~pred_scope_node_mask)

    in_scope_changed_edges = changed_edges & pred_scope_edge_mask
    in_scope_context_edges = context_edges & pred_scope_edge_mask
    out_of_scope_changed_edges = changed_edges & (~pred_scope_edge_mask)

    pred_type = rewrite_outputs["type_logits_full"][sample_idx].argmax(dim=-1)
    target_type = batch["next_node_feats"][sample_idx, :, 0].long()
    type_wrong = pred_type != target_type

    current_state = batch["node_feats"][sample_idx, :, 1:]
    target_state = batch["next_node_feats"][sample_idx, :, 1:]
    pred_state = rewrite_outputs["state_pred_full"][sample_idx]
    pred_state_err = (pred_state - target_state).abs().mean(dim=-1)
    copy_state_err = (current_state - target_state).abs().mean(dim=-1)
    state_not_improved = pred_state_err >= (copy_state_err - 1e-8)

    pred_adj = (torch.sigmoid(rewrite_outputs["edge_logits_full"][sample_idx]) >= 0.5)
    pred_adj = ((pred_adj | pred_adj.transpose(0, 1)) & valid_edge_mask).bool()
    target_adj = (batch["next_adj"][sample_idx] > 0.5) & valid_edge_mask
    edge_wrong = pred_adj != target_adj

    return {
        "num_samples": 1.0,
        "node_changed_total": changed_nodes.float().sum().item(),
        "node_changed_covered": in_scope_changed_nodes.float().sum().item(),
        "node_pred_total": pred_scope_node_mask.float().sum().item(),
        "node_pred_context_total": in_scope_context_nodes.float().sum().item(),
        "node_out_scope_changed_total": out_of_scope_changed_nodes.float().sum().item(),
        "node_in_scope_changed_total": in_scope_changed_nodes.float().sum().item(),
        "node_in_scope_changed_type_wrong": (type_wrong & in_scope_changed_nodes).float().sum().item(),
        "node_in_scope_changed_state_not_improved": (
            state_not_improved & in_scope_changed_nodes
        ).float().sum().item(),
        "node_in_scope_changed_state_pred_error_sum": pred_state_err[in_scope_changed_nodes].sum().item(),
        "node_in_scope_changed_state_copy_error_sum": copy_state_err[in_scope_changed_nodes].sum().item(),
        "node_in_scope_changed_state_count": in_scope_changed_nodes.float().sum().item(),
        "node_in_scope_context_total": in_scope_context_nodes.float().sum().item(),
        "node_in_scope_context_type_wrong": (type_wrong & in_scope_context_nodes).float().sum().item(),
        "node_in_scope_context_state_error_sum": pred_state_err[in_scope_context_nodes].sum().item(),
        "node_in_scope_context_state_count": in_scope_context_nodes.float().sum().item(),
        "edge_changed_total": changed_edges.float().sum().item(),
        "edge_changed_covered": in_scope_changed_edges.float().sum().item(),
        "edge_pred_total": pred_scope_edge_mask.float().sum().item(),
        "edge_pred_context_total": in_scope_context_edges.float().sum().item(),
        "edge_out_scope_changed_total": out_of_scope_changed_edges.float().sum().item(),
        "edge_in_scope_changed_total": in_scope_changed_edges.float().sum().item(),
        "edge_in_scope_changed_wrong": (edge_wrong & in_scope_changed_edges).float().sum().item(),
        "edge_in_scope_context_total": in_scope_context_edges.float().sum().item(),
        "edge_in_scope_context_wrong": (edge_wrong & in_scope_context_edges).float().sum().item(),
    }


def make_group_names(batch: Dict[str, Any], sample_idx: int) -> list[str]:
    names = ["overall"]

    events_item = batch.get("events", [None] * (sample_idx + 1))[sample_idx]
    event_types = sorted(set(extract_event_type_list(events_item)))
    for event_type in event_types:
        if event_type in EVENT_TYPE_ORDER:
            names.append(f"event_type::{event_type}")

    if "step6a_corruption_setting" in batch:
        setting = str(batch["step6a_corruption_setting"][sample_idx])
        names.append(f"corruption::{setting}")
        for event_type in event_types:
            if event_type in EVENT_TYPE_ORDER:
                names.append(f"corruption::{setting}::event_type::{event_type}")

    return names


@torch.no_grad()
def evaluate_localization_gap(
    proposal_model: ScopeProposalModel,
    rewrite_model: OracleLocalDeltaRewriteModel,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    node_threshold: float,
    edge_threshold: float,
    use_proposal_conditioning: bool,
) -> Dict[str, Any]:
    proposal_model.eval()
    rewrite_model.eval()

    buckets: dict[str, Dict[str, float]] = defaultdict(init_bucket)

    for batch in loader:
        require_keys(
            batch,
            [
                "node_feats",
                "adj",
                "next_node_feats",
                "next_adj",
                "node_mask",
                "changed_nodes",
                "changed_edges",
                "events",
            ],
        )
        batch = move_batch_to_device(batch, device)

        input_node_feats = batch.get("obs_node_feats", batch["node_feats"])
        input_adj = batch.get("obs_adj", batch["adj"])
        node_mask = batch["node_mask"].bool()
        valid_edge_mask = build_valid_edge_mask(batch["node_mask"]).bool()

        proposal_outputs = proposal_model(node_feats=input_node_feats, adj=input_adj)
        node_scope_logits = proposal_outputs.get("node_scope_logits", proposal_outputs["scope_logits"])
        proposal_node_probs = torch.sigmoid(node_scope_logits) * batch["node_mask"]
        pred_scope_nodes = (proposal_node_probs >= node_threshold) & node_mask

        if "edge_scope_logits" in proposal_outputs:
            proposal_edge_probs = torch.sigmoid(proposal_outputs["edge_scope_logits"]) * valid_edge_mask.float()
            pred_scope_edges = (proposal_edge_probs >= edge_threshold) & valid_edge_mask
        else:
            proposal_edge_probs = (
                proposal_node_probs.unsqueeze(2) * proposal_node_probs.unsqueeze(1) * valid_edge_mask.float()
            )
            pred_scope_edges = (
                pred_scope_nodes.unsqueeze(2) & pred_scope_nodes.unsqueeze(1) & valid_edge_mask
            )

        rewrite_outputs = rewrite_model(
            node_feats=input_node_feats,
            adj=input_adj,
            scope_node_mask=pred_scope_nodes.float(),
            scope_edge_mask=pred_scope_edges.float(),
            proposal_node_probs=proposal_node_probs if use_proposal_conditioning else None,
            proposal_edge_probs=proposal_edge_probs if use_proposal_conditioning else None,
        )

        batch_size = input_node_feats.shape[0]
        for sample_idx in range(batch_size):
            sample_stats = build_sample_stats(
                batch=batch,
                rewrite_outputs=rewrite_outputs,
                pred_scope_nodes=pred_scope_nodes,
                pred_scope_edges=pred_scope_edges,
                sample_idx=sample_idx,
            )
            for group_name in make_group_names(batch, sample_idx):
                update_bucket(buckets[group_name], sample_stats)

    overall = finalize_bucket(buckets["overall"])
    by_event_type = {
        event_type: finalize_bucket(buckets[f"event_type::{event_type}"])
        for event_type in EVENT_TYPE_ORDER
        if f"event_type::{event_type}" in buckets
    }

    by_corruption_setting: Dict[str, Any] = {}
    for group_name in sorted(buckets.keys()):
        if not group_name.startswith("corruption::"):
            continue
        parts = group_name.split("::")
        if len(parts) == 2:
            setting = parts[1]
            by_corruption_setting.setdefault(setting, {"overall": None, "by_event_type": {}})
            by_corruption_setting[setting]["overall"] = finalize_bucket(buckets[group_name])
        elif len(parts) == 4 and parts[2] == "event_type":
            setting = parts[1]
            event_type = parts[3]
            by_corruption_setting.setdefault(setting, {"overall": None, "by_event_type": {}})
            by_corruption_setting[setting]["by_event_type"][event_type] = finalize_bucket(buckets[group_name])

    return {
        "overall": overall,
        "by_event_type": by_event_type,
        "by_corruption_setting": by_corruption_setting,
    }


def build_payload(
    args: argparse.Namespace,
    proposal_checkpoint_path: Path,
    rewrite_checkpoint_path: Path,
    use_proposal_conditioning: bool,
    results: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "metadata": {
            "proposal_checkpoint_path": str(proposal_checkpoint_path),
            "rewrite_checkpoint_path": str(rewrite_checkpoint_path),
            "data_path": str(resolve_path(args.data_path)),
            "split_name": args.split_name,
            "run_name": args.run_name,
            "node_threshold": args.node_threshold,
            "edge_threshold": args.edge_threshold,
            "rewrite_uses_proposal_conditioning": use_proposal_conditioning,
            "model_input": "obs_graph_t when present; clean graph_t otherwise",
            "targets": "clean graph_t1 and GT changed region from clean graph delta",
            "evaluation_mode": "scope_edit_localization_gap",
        },
        "definitions": {
            "proposal_changed_region_recall": "GT changed items covered by predicted proposal scope / GT changed items",
            "proposal_scope_excess_ratio": "predicted scope items that are GT context / predicted scope items",
            "out_of_scope_miss": "GT changed items outside predicted scope / GT changed items",
            "in_scope_under_edit_edge": "GT changed edges inside scope that full rewrite still predicts incorrectly / in-scope GT changed edges",
            "in_scope_over_edit_edge": "GT context edges inside scope that full rewrite corrupts / in-scope GT context edges",
            "node_state_metrics_note_cn": "节点状态是连续值，因此不用“完全正确”硬二值，而是报告是否至少优于 copy/current，以及对应 MAE。",
        },
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal_checkpoint_path", type=str, required=True)
    parser.add_argument("--rewrite_checkpoint_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--split_name", type=str, default="eval")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--artifact_dir", type=str, default="artifacts/scope_edit_localization_gap")
    parser.add_argument("--node_threshold", type=float, default=0.20)
    parser.add_argument("--edge_threshold", type=float, default=0.15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    proposal_checkpoint_path = resolve_path(args.proposal_checkpoint_path)
    rewrite_checkpoint_path = resolve_path(args.rewrite_checkpoint_path)
    data_path = resolve_path(args.data_path)
    artifact_dir = resolve_path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)

    _, loader = build_loader(
        data_path=str(data_path),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    proposal_checkpoint = torch.load(proposal_checkpoint_path, map_location="cpu")
    proposal_model = ScopeProposalModel(ScopeProposalConfig(**proposal_checkpoint["model_config"])).to(device)
    proposal_model.load_state_dict(proposal_checkpoint["model_state_dict"])
    proposal_model.eval()

    rewrite_checkpoint = torch.load(rewrite_checkpoint_path, map_location="cpu")
    rewrite_model = OracleLocalDeltaRewriteModel(
        OracleLocalDeltaRewriteConfig(**rewrite_checkpoint["model_config"])
    ).to(device)
    rewrite_model.load_state_dict(rewrite_checkpoint["model_state_dict"])
    rewrite_model.eval()
    use_proposal_conditioning = bool(
        rewrite_checkpoint.get("model_config", {}).get("use_proposal_conditioning", False)
    )

    results = evaluate_localization_gap(
        proposal_model=proposal_model,
        rewrite_model=rewrite_model,
        loader=loader,
        device=device,
        node_threshold=args.node_threshold,
        edge_threshold=args.edge_threshold,
        use_proposal_conditioning=use_proposal_conditioning,
    )
    payload = build_payload(
        args=args,
        proposal_checkpoint_path=proposal_checkpoint_path,
        rewrite_checkpoint_path=rewrite_checkpoint_path,
        use_proposal_conditioning=use_proposal_conditioning,
        results=results,
    )

    if args.run_name is None:
        run_name = "__".join(
            [
                slugify(proposal_checkpoint_path.parent.name),
                slugify(rewrite_checkpoint_path.parent.name),
                slugify(data_path.stem),
            ]
        )
    else:
        run_name = slugify(args.run_name)

    out_path = artifact_dir / f"{run_name}.json"
    save_json(out_path, payload)

    print(f"device: {device}")
    print(f"proposal checkpoint: {proposal_checkpoint_path}")
    print(f"rewrite checkpoint: {rewrite_checkpoint_path}")
    print(f"data: {data_path}")
    print(f"saved json: {out_path}")
    print(json.dumps(results["overall"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
