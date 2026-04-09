from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------
# Make project root importable when running:
#   python train/train_scope_proposal.py
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.collate import graph_event_collate_fn
from data.dataset import GraphEventDataset
from models.oracle_local import build_valid_edge_mask
from models.proposal import ScopeProposalConfig, ScopeProposalModel, scope_proposal_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)

    if torch.cuda.is_available():
        return torch.device("cuda")

    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if has_mps:
        return torch.device("mps")

    return torch.device("cpu")


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def build_dataloaders(
    train_path: str,
    val_path: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
):
    train_dataset = GraphEventDataset(train_path)
    val_dataset = GraphEventDataset(val_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=graph_event_collate_fn,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=graph_event_collate_fn,
        pin_memory=pin_memory,
    )
    return train_dataset, val_dataset, train_loader, val_loader


def require_scope_targets(batch: Dict) -> None:
    missing = [k for k in ["event_scope_union_nodes", "event_scope_union_edges"] if k not in batch]
    if missing:
        raise KeyError(f"Scope proposal training requires scope labels. Missing keys: {missing}")


@torch.no_grad()
def precision_recall_f1_from_logits(
    scope_logits: torch.Tensor,
    target_scope: torch.Tensor,
    node_mask: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    pred = (torch.sigmoid(scope_logits) >= threshold).float() * node_mask
    target = target_scope.float() * node_mask

    tp = (pred * target).sum().item()
    pred_pos = pred.sum().item()
    true_pos = target.sum().item()

    precision = tp / pred_pos if pred_pos > 0 else 0.0
    recall = tp / true_pos if true_pos > 0 else 0.0
    if precision + recall <= 0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)

    return {
        "tp": tp,
        "pred_pos": pred_pos,
        "true_pos": true_pos,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


@torch.no_grad()
def edge_precision_recall_f1_from_logits(
    edge_scope_logits: torch.Tensor,
    target_edge_scope: torch.Tensor,
    pair_mask: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    pred = (torch.sigmoid(edge_scope_logits) >= threshold).float() * pair_mask
    target = target_edge_scope.float() * pair_mask

    tp = (pred * target).sum().item()
    pred_pos = pred.sum().item()
    true_pos = target.sum().item()

    precision = tp / pred_pos if pred_pos > 0 else 0.0
    recall = tp / true_pos if true_pos > 0 else 0.0
    if precision + recall <= 0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)

    return {
        "tp": tp,
        "pred_pos": pred_pos,
        "true_pos": true_pos,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def average_metric_dict(metric_sums: Dict[str, float], count: int) -> Dict[str, float]:
    return {k: v / max(count, 1) for k, v in metric_sums.items()}


def run_epoch(
    model: ScopeProposalModel,
    loader: DataLoader,
    device: torch.device,
    node_scope_loss_weight: float,
    node_flip_weight: float,
    edge_scope_loss_weight: float,
    edge_scope_pos_weight: float,
    node_threshold: float,
    edge_threshold: float,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_clip: Optional[float] = None,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    metric_sums = {
        "total_loss": 0.0,
        "node_scope_loss": 0.0,
        "edge_scope_loss": 0.0,
        "node_precision": 0.0,
        "node_recall": 0.0,
        "node_f1": 0.0,
        "edge_precision": 0.0,
        "edge_recall": 0.0,
        "edge_f1": 0.0,
        "avg_predicted_node_scope_fraction": 0.0,
        "avg_oracle_node_scope_fraction": 0.0,
        "avg_predicted_edge_scope_fraction": 0.0,
        "avg_oracle_edge_scope_fraction": 0.0,
        "selection_score": 0.0,
    }
    num_batches = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in loader:
            require_scope_targets(batch)
            batch = move_batch_to_device(batch, device)

            outputs = model(
                node_feats=batch["node_feats"],
                adj=batch["adj"],
            )
            current_type = batch["node_feats"][:, :, 0].long()
            target_type = batch["next_node_feats"][:, :, 0].long()
            flip_target_mask = (current_type != target_type).to(batch["node_mask"].dtype)
            node_scope_weights = 1.0 + (node_flip_weight - 1.0) * flip_target_mask
            valid_edge_mask = build_valid_edge_mask(batch["node_mask"])
            loss_dict = scope_proposal_loss(
                outputs=outputs,
                target_node_scope=batch["event_scope_union_nodes"],
                target_edge_scope=batch["event_scope_union_edges"],
                node_mask=batch["node_mask"],
                pair_mask=valid_edge_mask,
                node_scope_loss_weight=node_scope_loss_weight,
                edge_scope_loss_weight=edge_scope_loss_weight,
                edge_scope_pos_weight=edge_scope_pos_weight,
                node_scope_weights=node_scope_weights,
            )

            if is_train:
                optimizer.zero_grad()
                loss_dict["total_loss"].backward()
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            node_prf = precision_recall_f1_from_logits(
                outputs["node_scope_logits"],
                batch["event_scope_union_nodes"],
                batch["node_mask"],
                threshold=node_threshold,
            )
            edge_prf = edge_precision_recall_f1_from_logits(
                outputs["edge_scope_logits"],
                batch["event_scope_union_edges"],
                valid_edge_mask,
                threshold=edge_threshold,
            )
            pred_node_mask = (torch.sigmoid(outputs["node_scope_logits"]) >= node_threshold).float() * batch["node_mask"]
            oracle_node_mask = batch["event_scope_union_nodes"] * batch["node_mask"]
            pred_edge_mask = (torch.sigmoid(outputs["edge_scope_logits"]) >= edge_threshold).float() * valid_edge_mask
            oracle_edge_mask = batch["event_scope_union_edges"] * valid_edge_mask

            metric_sums["total_loss"] += loss_dict["total_loss"].item()
            metric_sums["node_scope_loss"] += loss_dict["node_scope_loss"].item()
            metric_sums["edge_scope_loss"] += loss_dict["edge_scope_loss"].item()
            metric_sums["node_precision"] += node_prf["precision"]
            metric_sums["node_recall"] += node_prf["recall"]
            metric_sums["node_f1"] += node_prf["f1"]
            metric_sums["edge_precision"] += edge_prf["precision"]
            metric_sums["edge_recall"] += edge_prf["recall"]
            metric_sums["edge_f1"] += edge_prf["f1"]
            metric_sums["avg_predicted_node_scope_fraction"] += (
                pred_node_mask.sum() / batch["node_mask"].sum().clamp_min(1.0)
            ).item()
            metric_sums["avg_oracle_node_scope_fraction"] += (
                oracle_node_mask.sum() / batch["node_mask"].sum().clamp_min(1.0)
            ).item()
            metric_sums["avg_predicted_edge_scope_fraction"] += (
                pred_edge_mask.sum() / valid_edge_mask.sum().clamp_min(1.0)
            ).item()
            metric_sums["avg_oracle_edge_scope_fraction"] += (
                oracle_edge_mask.sum() / valid_edge_mask.sum().clamp_min(1.0)
            ).item()
            metric_sums["selection_score"] += 0.5 * (node_prf["f1"] + edge_prf["f1"])
            num_batches += 1

    return average_metric_dict(metric_sums, num_batches)


def save_json(path: Path, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="data/graph_event_train.pkl")
    parser.add_argument("--val_path", type=str, default="data/graph_event_val.pkl")
    parser.add_argument("--save_dir", type=str, default="checkpoints/scope_proposal")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--msg_pass_layers", type=int, default=3)
    parser.add_argument("--head_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--node_scope_loss_weight", type=float, default=1.0)
    parser.add_argument("--node_flip_weight", type=float, default=1.0)
    parser.add_argument("--edge_scope_loss_weight", type=float, default=1.0)
    parser.add_argument("--edge_scope_pos_weight", type=float, default=1.0)
    parser.add_argument("--node_threshold", type=float, default=0.5)
    parser.add_argument("--edge_threshold", type=float, default=0.5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    pin_memory = device.type == "cuda"

    save_dir = PROJECT_ROOT / args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset, train_loader, val_loader = build_dataloaders(
        train_path=args.train_path,
        val_path=args.val_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    sample = train_dataset[0]
    if "event_scope_union_nodes" not in sample or "event_scope_union_edges" not in sample:
        raise KeyError(
            "Scope proposal training requires event_scope_union_nodes and event_scope_union_edges in each dataset sample."
        )

    node_feat_dim = sample["node_feats"].shape[-1]
    model_config = ScopeProposalConfig(
        node_feat_dim=node_feat_dim,
        hidden_dim=args.hidden_dim,
        msg_pass_layers=args.msg_pass_layers,
        head_layers=args.head_layers,
        dropout=args.dropout,
    )
    model = ScopeProposalModel(model_config).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_score = float("-inf")
    best_ckpt_path = save_dir / "best.pt"
    last_ckpt_path = save_dir / "last.pt"
    best_metrics_path = save_dir / "best_metrics.json"

    print(f"device: {device}")
    print(f"train size: {len(train_dataset)}")
    print(f"val size: {len(val_dataset)}")
    print(f"node_feat_dim: {node_feat_dim}")
    print(model)

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            node_scope_loss_weight=args.node_scope_loss_weight,
            node_flip_weight=args.node_flip_weight,
            edge_scope_loss_weight=args.edge_scope_loss_weight,
            edge_scope_pos_weight=args.edge_scope_pos_weight,
            node_threshold=args.node_threshold,
            edge_threshold=args.edge_threshold,
            optimizer=optimizer,
            grad_clip=args.grad_clip,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            node_scope_loss_weight=args.node_scope_loss_weight,
            node_flip_weight=args.node_flip_weight,
            edge_scope_loss_weight=args.edge_scope_loss_weight,
            edge_scope_pos_weight=args.edge_scope_pos_weight,
            node_threshold=args.node_threshold,
            edge_threshold=args.edge_threshold,
            optimizer=None,
            grad_clip=None,
        )

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_metrics['total_loss']:.6f} "
            f"train_node_f1={train_metrics['node_f1']:.6f} "
            f"train_edge_f1={train_metrics['edge_f1']:.6f} | "
            f"val_loss={val_metrics['total_loss']:.6f} "
            f"val_node_f1={val_metrics['node_f1']:.6f} "
            f"val_edge_f1={val_metrics['edge_f1']:.6f} "
            f"val_score={val_metrics['selection_score']:.6f}"
        )

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": vars(model_config),
            "args": vars(args),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "selection_metric": "0.5 * (val_node_f1 + val_edge_f1)",
        }
        torch.save(ckpt, last_ckpt_path)

        if val_metrics["selection_score"] > best_val_score:
            best_val_score = val_metrics["selection_score"]
            torch.save(ckpt, best_ckpt_path)
            save_json(
                best_metrics_path,
                {
                    "epoch": epoch,
                    "best_val_selection_score": best_val_score,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "model_config": vars(model_config),
                    "args": vars(args),
                },
            )
            print(f"  saved new best checkpoint -> {best_ckpt_path}")

    print(f"training finished. best_val_selection_score={best_val_score:.6f}")
    print(f"best checkpoint: {best_ckpt_path}")
    print(f"last checkpoint: {last_ckpt_path}")
    print(f"best metrics json: {best_metrics_path}")


if __name__ == "__main__":
    main()
