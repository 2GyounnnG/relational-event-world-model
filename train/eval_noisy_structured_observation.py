from __future__ import annotations

import argparse
from collections import defaultdict
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.collate import graph_event_collate_fn
from data.dataset import GraphEventDataset
from models.oracle_local_delta import (
    EDGE_DELTA_ADD,
    EDGE_DELTA_DELETE,
    EDGE_DELTA_KEEP,
    OracleLocalDeltaRewriteConfig,
    OracleLocalDeltaRewriteModel,
    build_edge_delta_targets,
    build_valid_edge_mask,
)
from models.proposal import ScopeProposalConfig, ScopeProposalModel


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def get_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if has_mps:
        return torch.device("mps")
    return torch.device("cpu")


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def require_keys(batch: Dict[str, Any], required_keys: Iterable[str]) -> None:
    missing = [key for key in required_keys if key not in batch]
    if missing:
        raise KeyError(f"Missing required batch keys: {missing}")


def safe_div(num: float, den: float) -> Optional[float]:
    if den <= 0:
        return None
    return num / den


def summarize_group(records: list[Dict[str, Optional[float]]]) -> Dict[str, Optional[float]]:
    metric_names = [
        "full_type_acc",
        "full_state_mae",
        "full_edge_acc",
        "changed_type_acc",
        "flip_acc",
        "changed_edge_acc",
        "context_edge_acc",
        "delta_all",
        "keep",
        "add",
        "delete",
    ]
    summary: Dict[str, Optional[float]] = {"count": len(records)}
    for metric_name in metric_names:
        values = [record[metric_name] for record in records if record.get(metric_name) is not None]
        summary[metric_name] = (sum(values) / len(values)) if values else None

    node_tp = sum(float(record.get("proposal_node_tp", 0.0) or 0.0) for record in records)
    node_pred_pos = sum(float(record.get("proposal_node_pred_pos", 0.0) or 0.0) for record in records)
    node_true_pos = sum(float(record.get("proposal_node_true_pos", 0.0) or 0.0) for record in records)
    edge_tp = sum(float(record.get("proposal_edge_tp", 0.0) or 0.0) for record in records)
    edge_pred_pos = sum(float(record.get("proposal_edge_pred_pos", 0.0) or 0.0) for record in records)
    edge_true_pos = sum(float(record.get("proposal_edge_true_pos", 0.0) or 0.0) for record in records)

    node_precision = safe_div(node_tp, node_pred_pos)
    node_recall = safe_div(node_tp, node_true_pos)
    edge_precision = safe_div(edge_tp, edge_pred_pos)
    edge_recall = safe_div(edge_tp, edge_true_pos)

    node_f1 = None
    if node_precision is not None and node_recall is not None and (node_precision + node_recall) > 0:
        node_f1 = 2.0 * node_precision * node_recall / (node_precision + node_recall)
    edge_f1 = None
    if edge_precision is not None and edge_recall is not None and (edge_precision + edge_recall) > 0:
        edge_f1 = 2.0 * edge_precision * edge_recall / (edge_precision + edge_recall)

    summary["proposal_node_precision"] = node_precision
    summary["proposal_node_recall"] = node_recall
    summary["proposal_node_f1"] = node_f1
    summary["proposal_edge_precision"] = edge_precision
    summary["proposal_edge_recall"] = edge_recall
    summary["proposal_edge_f1"] = edge_f1
    return summary


def binary_precision_recall_f1(
    pred_mask: torch.Tensor,
    target_mask: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[Optional[float], Optional[float], Optional[float], float, float, float]:
    pred_bool = pred_mask.bool() & valid_mask.bool()
    target_bool = target_mask.bool() & valid_mask.bool()
    tp = (pred_bool & target_bool).float().sum().item()
    pred_pos = pred_bool.float().sum().item()
    true_pos = target_bool.float().sum().item()
    precision = safe_div(tp, pred_pos)
    recall = safe_div(tp, true_pos)
    if precision is None or recall is None or (precision + recall) <= 0:
        f1 = None
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1, tp, pred_pos, true_pos


def accuracy_from_mask(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> Optional[float]:
    mask_bool = mask.bool()
    total = mask_bool.float().sum().item()
    if total <= 0:
        return None
    correct = ((pred == target) & mask_bool).float().sum().item()
    return correct / total


def mae_from_mask(pred: torch.Tensor, target: torch.Tensor, node_mask: torch.Tensor) -> Optional[float]:
    mask = node_mask.unsqueeze(-1).float()
    total = (mask.sum().item() * pred.shape[-1])
    if total <= 0:
        return None
    abs_err = (pred - target).abs() * mask
    return abs_err.sum().item() / total


def build_loader(
    data_path: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple[GraphEventDataset, DataLoader]:
    dataset = GraphEventDataset(data_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=graph_event_collate_fn,
        pin_memory=pin_memory,
    )
    return dataset, loader


@torch.no_grad()
def evaluate(
    proposal_model: ScopeProposalModel,
    rewrite_model: OracleLocalDeltaRewriteModel,
    loader: DataLoader,
    device: torch.device,
    node_threshold: float,
    edge_threshold: float,
    use_proposal_conditioning: bool,
) -> Dict[str, Any]:
    proposal_model.eval()
    rewrite_model.eval()

    overall_records: list[Dict[str, Optional[float]]] = []
    setting_records: dict[str, list[Dict[str, Optional[float]]]] = defaultdict(list)

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
                "event_scope_union_nodes",
                "event_scope_union_edges",
                "step6a_corruption_setting",
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

        pred_type = rewrite_outputs["type_logits_full"].argmax(dim=-1)
        pred_state = rewrite_outputs["state_pred_full"]
        pred_adj = (torch.sigmoid(rewrite_outputs["edge_logits_full"]) >= 0.5).float()
        pred_adj = ((pred_adj + pred_adj.transpose(1, 2)) > 0.5).float()
        diag_mask = torch.eye(pred_adj.shape[1], device=pred_adj.device, dtype=torch.bool).unsqueeze(0)
        pred_adj = pred_adj.masked_fill(diag_mask, 0.0)

        target_type = batch["next_node_feats"][..., 0].long()
        current_type = batch["node_feats"][..., 0].long()
        target_state = batch["next_node_feats"][..., 1:]
        changed_nodes = batch["changed_nodes"].bool()
        flip_mask = current_type != target_type
        changed_edges = batch["changed_edges"].bool() & valid_edge_mask
        context_edges = (~changed_edges) & valid_edge_mask

        target_adj_bool = batch["next_adj"] > 0.5
        pred_adj_bool = pred_adj > 0.5
        target_delta = build_edge_delta_targets(batch["adj"], batch["next_adj"])
        pred_delta = build_edge_delta_targets(batch["adj"], pred_adj)

        oracle_node_scope = batch["event_scope_union_nodes"].bool() & node_mask
        oracle_edge_scope = batch["event_scope_union_edges"].bool() & valid_edge_mask

        batch_size = batch["node_feats"].shape[0]
        for idx in range(batch_size):
            setting_name = str(batch["step6a_corruption_setting"][idx])
            node_prec, node_rec, node_f1, node_tp, node_pred_pos, node_true_pos = binary_precision_recall_f1(
                pred_scope_nodes[idx],
                oracle_node_scope[idx],
                node_mask[idx],
            )
            edge_prec, edge_rec, edge_f1, edge_tp, edge_pred_pos, edge_true_pos = binary_precision_recall_f1(
                pred_scope_edges[idx],
                oracle_edge_scope[idx],
                valid_edge_mask[idx],
            )
            record = {
                "full_type_acc": accuracy_from_mask(pred_type[idx], target_type[idx], node_mask[idx]),
                "full_state_mae": mae_from_mask(pred_state[idx], target_state[idx], node_mask[idx]),
                "full_edge_acc": accuracy_from_mask(pred_adj_bool[idx], target_adj_bool[idx], valid_edge_mask[idx]),
                "changed_type_acc": accuracy_from_mask(pred_type[idx], target_type[idx], changed_nodes[idx]),
                "flip_acc": accuracy_from_mask(pred_type[idx], target_type[idx], flip_mask[idx]),
                "changed_edge_acc": accuracy_from_mask(pred_adj_bool[idx], target_adj_bool[idx], changed_edges[idx]),
                "context_edge_acc": accuracy_from_mask(pred_adj_bool[idx], target_adj_bool[idx], context_edges[idx]),
                "delta_all": accuracy_from_mask(pred_delta[idx], target_delta[idx], valid_edge_mask[idx]),
                "keep": accuracy_from_mask(
                    pred_delta[idx],
                    target_delta[idx],
                    valid_edge_mask[idx] & (target_delta[idx] == EDGE_DELTA_KEEP),
                ),
                "add": accuracy_from_mask(
                    pred_delta[idx],
                    target_delta[idx],
                    valid_edge_mask[idx] & (target_delta[idx] == EDGE_DELTA_ADD),
                ),
                "delete": accuracy_from_mask(
                    pred_delta[idx],
                    target_delta[idx],
                    valid_edge_mask[idx] & (target_delta[idx] == EDGE_DELTA_DELETE),
                ),
                "proposal_node_precision": node_prec,
                "proposal_node_recall": node_rec,
                "proposal_node_f1": node_f1,
                "proposal_node_tp": node_tp,
                "proposal_node_pred_pos": node_pred_pos,
                "proposal_node_true_pos": node_true_pos,
                "proposal_edge_precision": edge_prec,
                "proposal_edge_recall": edge_rec,
                "proposal_edge_f1": edge_f1,
                "proposal_edge_tp": edge_tp,
                "proposal_edge_pred_pos": edge_pred_pos,
                "proposal_edge_true_pos": edge_true_pos,
            }
            overall_records.append(record)
            setting_records[setting_name].append(record)

    return {
        "overall": summarize_group(overall_records),
        "by_corruption_setting": {
            setting_name: summarize_group(records)
            for setting_name, records in sorted(setting_records.items())
        },
    }


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal_checkpoint_path", type=str, required=True)
    parser.add_argument("--rewrite_checkpoint_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--split_name", type=str, default="eval")
    parser.add_argument("--node_threshold", type=float, default=0.20)
    parser.add_argument("--edge_threshold", type=float, default=0.15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    proposal_checkpoint_path = resolve_path(args.proposal_checkpoint_path)
    rewrite_checkpoint_path = resolve_path(args.rewrite_checkpoint_path)
    data_path = resolve_path(args.data_path)
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

    results = evaluate(
        proposal_model=proposal_model,
        rewrite_model=rewrite_model,
        loader=loader,
        device=device,
        node_threshold=args.node_threshold,
        edge_threshold=args.edge_threshold,
        use_proposal_conditioning=use_proposal_conditioning,
    )

    payload = {
        "metadata": {
            "proposal_checkpoint_path": str(proposal_checkpoint_path),
            "rewrite_checkpoint_path": str(rewrite_checkpoint_path),
            "data_path": str(data_path),
            "split_name": args.split_name,
            "node_threshold": args.node_threshold,
            "edge_threshold": args.edge_threshold,
            "rewrite_uses_proposal_conditioning": use_proposal_conditioning,
            "evaluation_mode": "noisy_structured_observation",
            "model_input": "corrupted obs_graph_t when present; clean graph_t otherwise",
            "targets": "clean graph_t1",
        },
        "results": results,
    }

    out_path = rewrite_checkpoint_path.parent / f"{args.split_name}_noisy_structured_observation.json"
    save_json(out_path, payload)

    print(f"device: {device}")
    print(f"proposal checkpoint: {proposal_checkpoint_path}")
    print(f"rewrite checkpoint: {rewrite_checkpoint_path}")
    print(f"data: {data_path}")
    print(f"saved json: {out_path}")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
