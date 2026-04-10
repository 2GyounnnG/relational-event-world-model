from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


def safe_div(num: float, den: float) -> Optional[float]:
    if den <= 0:
        return None
    return num / den


def load_rollout_dataset(path: Path) -> list[Dict[str, Any]]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"Rollout dataset is empty or malformed: {path}")
    return data


def inspect_rollout_dataset(samples: list[Dict[str, Any]]) -> Dict[str, Any]:
    horizon_counts: Counter[int] = Counter()
    support_keys = set(samples[0].keys()) if samples else set()
    for sample in samples:
        horizon_counts[int(sample["horizon"])] += 1
    return {
        "total_rollout_sample_count": len(samples),
        "horizon_distribution": {str(k): int(v) for k, v in sorted(horizon_counts.items())},
        "dataset_keys": sorted(support_keys),
    }


def graph_to_tensors(graph: Dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    node_feats = torch.tensor(graph["node_features"], dtype=torch.float32, device=device).unsqueeze(0)
    adj = torch.tensor(graph["adj"], dtype=torch.float32, device=device).unsqueeze(0)
    node_mask = torch.ones((1, node_feats.shape[1]), dtype=torch.float32, device=device)
    return node_feats, adj, node_mask


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


def summarize_metric_records(records: list[Dict[str, Optional[float]]]) -> Dict[str, Optional[float]]:
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
        vals = [record[metric_name] for record in records if record.get(metric_name) is not None]
        summary[metric_name] = (sum(vals) / len(vals)) if vals else None
    return summary


def evaluate_transition(
    current_gt_graph: Dict[str, Any],
    target_gt_graph: Dict[str, Any],
    pred_next_node_feats: torch.Tensor,
    pred_next_adj: torch.Tensor,
) -> Dict[str, Optional[float]]:
    current_node_feats = torch.tensor(current_gt_graph["node_features"], dtype=torch.float32)
    current_adj = torch.tensor(current_gt_graph["adj"], dtype=torch.float32)
    target_node_feats = torch.tensor(target_gt_graph["node_features"], dtype=torch.float32)
    target_adj = torch.tensor(target_gt_graph["adj"], dtype=torch.float32)

    num_nodes = current_node_feats.shape[0]
    node_mask = torch.ones(num_nodes, dtype=torch.float32)
    valid_edge_mask = build_valid_edge_mask(node_mask.unsqueeze(0))[0].bool()

    pred_type = pred_next_node_feats[:, 0].long()
    target_type = target_node_feats[:, 0].long()
    current_type = current_node_feats[:, 0].long()
    pred_state = pred_next_node_feats[:, 1:]
    target_state = target_node_feats[:, 1:]

    changed_nodes = torch.any((current_node_feats - target_node_feats).abs() > 1e-6, dim=-1)
    flip_mask = (current_type != target_type)

    changed_edges = (current_adj != target_adj) & valid_edge_mask
    context_edges = (~changed_edges) & valid_edge_mask

    pred_adj_bool = pred_next_adj > 0.5
    target_adj_bool = target_adj > 0.5

    target_delta = build_edge_delta_targets(current_adj, target_adj)
    pred_delta = build_edge_delta_targets(current_adj, pred_next_adj)

    return {
        "full_type_acc": accuracy_from_mask(pred_type, target_type, node_mask),
        "full_state_mae": mae_from_mask(pred_state, target_state, node_mask),
        "full_edge_acc": accuracy_from_mask(pred_adj_bool, target_adj_bool, valid_edge_mask),
        "changed_type_acc": accuracy_from_mask(pred_type, target_type, changed_nodes),
        "flip_acc": accuracy_from_mask(pred_type, target_type, flip_mask),
        "changed_edge_acc": accuracy_from_mask(pred_adj_bool, target_adj_bool, changed_edges),
        "context_edge_acc": accuracy_from_mask(pred_adj_bool, target_adj_bool, context_edges),
        "delta_all": accuracy_from_mask(pred_delta, target_delta, valid_edge_mask),
        "keep": accuracy_from_mask(pred_delta, target_delta, valid_edge_mask & (target_delta == EDGE_DELTA_KEEP)),
        "add": accuracy_from_mask(pred_delta, target_delta, valid_edge_mask & (target_delta == EDGE_DELTA_ADD)),
        "delete": accuracy_from_mask(pred_delta, target_delta, valid_edge_mask & (target_delta == EDGE_DELTA_DELETE)),
    }


def predict_next_graph(
    proposal_model: ScopeProposalModel,
    rewrite_model: OracleLocalDeltaRewriteModel,
    current_node_feats: torch.Tensor,
    current_adj: torch.Tensor,
    node_threshold: float,
    edge_threshold: float,
    use_proposal_conditioning: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    node_mask = torch.ones((1, current_node_feats.shape[1]), dtype=torch.float32, device=current_node_feats.device)
    valid_edge_mask = build_valid_edge_mask(node_mask).bool()

    proposal_outputs = proposal_model(node_feats=current_node_feats, adj=current_adj)
    node_scope_logits = proposal_outputs.get("node_scope_logits", proposal_outputs["scope_logits"])
    proposal_node_probs = torch.sigmoid(node_scope_logits) * node_mask
    pred_scope_nodes = (proposal_node_probs >= node_threshold) & node_mask.bool()

    if "edge_scope_logits" in proposal_outputs:
        proposal_edge_probs = torch.sigmoid(proposal_outputs["edge_scope_logits"]) * valid_edge_mask.float()
        pred_scope_edges = (proposal_edge_probs >= edge_threshold) & valid_edge_mask
    else:
        proposal_edge_probs = (
            proposal_node_probs.unsqueeze(2) * proposal_node_probs.unsqueeze(1) * valid_edge_mask.float()
        )
        pred_scope_edges = pred_scope_nodes.unsqueeze(2) & pred_scope_nodes.unsqueeze(1) & valid_edge_mask

    rewrite_outputs = rewrite_model(
        node_feats=current_node_feats,
        adj=current_adj,
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

    pred_next_node_feats = torch.cat([pred_type.float().unsqueeze(-1), pred_state], dim=-1)
    return pred_next_node_feats, pred_adj


@torch.no_grad()
def evaluate_rollouts(
    proposal_model: ScopeProposalModel,
    rewrite_model: OracleLocalDeltaRewriteModel,
    samples: list[Dict[str, Any]],
    device: torch.device,
    node_threshold: float,
    edge_threshold: float,
    use_proposal_conditioning: bool,
) -> Dict[str, Any]:
    final_records_overall: list[Dict[str, Optional[float]]] = []
    final_records_by_horizon: dict[int, list[Dict[str, Optional[float]]]] = defaultdict(list)
    step_records: dict[int, list[Dict[str, Optional[float]]]] = defaultdict(list)

    for sample in samples:
        horizon = int(sample["horizon"])
        graph_0 = sample["graph_0"]
        graph_steps = sample["graph_steps"]

        current_pred_node_feats, current_pred_adj, _ = graph_to_tensors(graph_0, device)
        current_gt_graph = graph_0

        for step_idx, target_gt_graph in enumerate(graph_steps, start=1):
            pred_next_node_feats, pred_next_adj = predict_next_graph(
                proposal_model=proposal_model,
                rewrite_model=rewrite_model,
                current_node_feats=current_pred_node_feats,
                current_adj=current_pred_adj,
                node_threshold=node_threshold,
                edge_threshold=edge_threshold,
                use_proposal_conditioning=use_proposal_conditioning,
            )

            metrics = evaluate_transition(
                current_gt_graph=current_gt_graph,
                target_gt_graph=target_gt_graph,
                pred_next_node_feats=pred_next_node_feats[0].detach().cpu(),
                pred_next_adj=pred_next_adj[0].detach().cpu(),
            )
            step_records[step_idx].append(metrics)
            if step_idx == horizon:
                final_records_overall.append(metrics)
                final_records_by_horizon[horizon].append(metrics)

            current_pred_node_feats = pred_next_node_feats
            current_pred_adj = pred_next_adj
            current_gt_graph = target_gt_graph

    return {
        "overall_final": summarize_metric_records(final_records_overall),
        "by_horizon": {
            str(horizon): summarize_metric_records(records)
            for horizon, records in sorted(final_records_by_horizon.items())
        },
        "by_step": [
            {
                "step_index": step_idx,
                **summarize_metric_records(records),
            }
            for step_idx, records in sorted(step_records.items())
        ],
    }


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    proposal_checkpoint_path = resolve_path(args.proposal_checkpoint_path)
    rewrite_checkpoint_path = resolve_path(args.rewrite_checkpoint_path)
    data_path = resolve_path(args.data_path)
    device = get_device(args.device)

    samples = load_rollout_dataset(data_path)
    dataset_support = inspect_rollout_dataset(samples)

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

    results = evaluate_rollouts(
        proposal_model=proposal_model,
        rewrite_model=rewrite_model,
        samples=samples,
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
            "rollout_mode": "autoregressive_predicted_state_feedback",
            "state_conversion": {
                "node_type": "argmax(type_logits_full)",
                "node_state": "state_pred_full",
                "edge_adj": "sigmoid(edge_logits_full)>=0.5 with undirected symmetrization",
            },
        },
        "dataset_support": dataset_support,
        "results": results,
    }

    out_path = rewrite_checkpoint_path.parent / f"{args.split_name}_rollout_stability.json"
    save_json(out_path, payload)

    print(f"device: {device}")
    print(f"proposal checkpoint: {proposal_checkpoint_path}")
    print(f"rewrite checkpoint: {rewrite_checkpoint_path}")
    print(f"data: {data_path}")
    print(f"total rollout samples: {dataset_support['total_rollout_sample_count']}")
    print(f"horizon distribution: {dataset_support['horizon_distribution']}")
    print(f"saved json: {out_path}")
    print(json.dumps(payload["results"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
