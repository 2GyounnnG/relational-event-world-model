from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------
# Make project root importable when running:
#   python train/train_baseline_global.py
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import GraphEventDataset
from data.collate import graph_event_collate_fn
from models.baselines import (
    GlobalBaselineConfig,
    GlobalTransitionBaseline,
    global_baseline_loss,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


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
):
    train_dataset = GraphEventDataset(train_path)
    val_dataset = GraphEventDataset(val_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=graph_event_collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=graph_event_collate_fn,
        pin_memory=True,
    )

    return train_dataset, val_dataset, train_loader, val_loader


@torch.no_grad()
def compute_edge_accuracy(
    edge_logits: torch.Tensor,
    target_adj: torch.Tensor,
    node_mask: torch.Tensor,
) -> float:
    """
    edge_logits: [B, N, N]
    target_adj:  [B, N, N]
    node_mask:   [B, N]
    """
    pred_adj = (torch.sigmoid(edge_logits) >= 0.5).float()

    pair_mask = node_mask.unsqueeze(2) * node_mask.unsqueeze(1)  # [B, N, N]
    n = edge_logits.shape[1]
    diag = torch.eye(n, device=edge_logits.device).unsqueeze(0)
    pair_mask = pair_mask * (1.0 - diag)

    correct = ((pred_adj == target_adj.float()).float() * pair_mask).sum()
    total = pair_mask.sum().clamp_min(1.0)

    return (correct / total).item()


@torch.no_grad()
def compute_type_accuracy(
    type_logits: torch.Tensor,
    target_node_feats: torch.Tensor,
    node_mask: torch.Tensor,
) -> float:
    """
    type_logits:        [B, N, C]
    target_node_feats:  [B, N, F], first channel is type id
    node_mask:          [B, N]
    """
    pred_type = type_logits.argmax(dim=-1)          # [B, N]
    target_type = target_node_feats[:, :, 0].long() # [B, N]

    correct = ((pred_type == target_type).float() * node_mask).sum()
    total = node_mask.sum().clamp_min(1.0)
    return (correct / total).item()


@torch.no_grad()
def compute_state_mae(
    state_pred: torch.Tensor,
    target_node_feats: torch.Tensor,
    node_mask: torch.Tensor,
) -> float:
    """
    state_pred:         [B, N, state_dim]
    target_node_feats:  [B, N, F], channels 1: are continuous state
    node_mask:          [B, N]
    """
    target_state = target_node_feats[:, :, 1:]
    abs_err = torch.abs(state_pred - target_state)
    abs_err = abs_err * node_mask.unsqueeze(-1)
    denom = (node_mask.sum() * state_pred.shape[-1]).clamp_min(1.0)
    return (abs_err.sum() / denom).item()


def train_one_epoch(
    model: GlobalTransitionBaseline,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    edge_loss_weight: float,
    type_loss_weight: float,
    state_loss_weight: float,
    type_flip_weight: float,
    grad_clip: float | None,
) -> Dict[str, float]:
    model.train()

    total_loss_sum = 0.0
    type_loss_sum = 0.0
    state_loss_sum = 0.0
    edge_loss_sum = 0.0
    type_acc_sum = 0.0
    state_mae_sum = 0.0
    edge_acc_sum = 0.0
    num_batches = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        outputs = model(batch["node_feats"], batch["adj"])

        loss_dict = global_baseline_loss(
            outputs=outputs,
            current_node_feats=batch["node_feats"],
            target_node_feats=batch["next_node_feats"],
            target_adj=batch["next_adj"],
            node_mask=batch["node_mask"],
            edge_loss_weight=edge_loss_weight,
            type_loss_weight=type_loss_weight,
            state_loss_weight=state_loss_weight,
            type_flip_weight=type_flip_weight,
        )

        loss = loss_dict["total_loss"]

        optimizer.zero_grad()
        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss_sum += loss_dict["total_loss"].item()
        type_loss_sum += loss_dict["type_loss"].item()
        state_loss_sum += loss_dict["state_loss"].item()
        edge_loss_sum += loss_dict["edge_loss"].item()

        type_acc_sum += compute_type_accuracy(
            outputs["type_logits"],
            batch["next_node_feats"],
            batch["node_mask"],
        )
        state_mae_sum += compute_state_mae(
            outputs["state_pred"],
            batch["next_node_feats"],
            batch["node_mask"],
        )
        edge_acc_sum += compute_edge_accuracy(
            outputs["edge_logits"],
            batch["next_adj"],
            batch["node_mask"],
        )
        num_batches += 1

    return {
        "total_loss": total_loss_sum / max(num_batches, 1),
        "type_loss": type_loss_sum / max(num_batches, 1),
        "state_loss": state_loss_sum / max(num_batches, 1),
        "edge_loss": edge_loss_sum / max(num_batches, 1),
        "type_acc": type_acc_sum / max(num_batches, 1),
        "state_mae": state_mae_sum / max(num_batches, 1),
        "edge_acc": edge_acc_sum / max(num_batches, 1),
    }


@torch.no_grad()
def evaluate(
    model: GlobalTransitionBaseline,
    loader: DataLoader,
    device: torch.device,
    edge_loss_weight: float,
    type_loss_weight: float,
    state_loss_weight: float,
    type_flip_weight: float,
) -> Dict[str, float]:
    model.eval()

    total_loss_sum = 0.0
    type_loss_sum = 0.0
    state_loss_sum = 0.0
    edge_loss_sum = 0.0
    type_acc_sum = 0.0
    state_mae_sum = 0.0
    edge_acc_sum = 0.0
    num_batches = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        outputs = model(batch["node_feats"], batch["adj"])

        loss_dict = global_baseline_loss(
            outputs=outputs,
            current_node_feats=batch["node_feats"],
            target_node_feats=batch["next_node_feats"],
            target_adj=batch["next_adj"],
            node_mask=batch["node_mask"],
            edge_loss_weight=edge_loss_weight,
            type_loss_weight=type_loss_weight,
            state_loss_weight=state_loss_weight,
            type_flip_weight=type_flip_weight,
        )

        total_loss_sum += loss_dict["total_loss"].item()
        type_loss_sum += loss_dict["type_loss"].item()
        state_loss_sum += loss_dict["state_loss"].item()
        edge_loss_sum += loss_dict["edge_loss"].item()

        type_acc_sum += compute_type_accuracy(
            outputs["type_logits"],
            batch["next_node_feats"],
            batch["node_mask"],
        )
        state_mae_sum += compute_state_mae(
            outputs["state_pred"],
            batch["next_node_feats"],
            batch["node_mask"],
        )
        edge_acc_sum += compute_edge_accuracy(
            outputs["edge_logits"],
            batch["next_adj"],
            batch["node_mask"],
        )
        num_batches += 1

    return {
        "total_loss": total_loss_sum / max(num_batches, 1),
        "type_loss": type_loss_sum / max(num_batches, 1),
        "state_loss": state_loss_sum / max(num_batches, 1),
        "edge_loss": edge_loss_sum / max(num_batches, 1),
        "type_acc": type_acc_sum / max(num_batches, 1),
        "state_mae": state_mae_sum / max(num_batches, 1),
        "edge_acc": edge_acc_sum / max(num_batches, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="data/graph_event_train.pkl")
    parser.add_argument("--val_path", type=str, default="data/graph_event_val.pkl")
    parser.add_argument("--save_dir", type=str, default="checkpoints/global_baseline_typed")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--msg_pass_layers", type=int, default=3)
    parser.add_argument("--node_mlp_layers", type=int, default=2)
    parser.add_argument("--edge_mlp_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num_node_types", type=int, default=3)
    parser.add_argument("--edge_loss_weight", type=float, default=1.0)
    parser.add_argument("--type_loss_weight", type=float, default=1.0)
    parser.add_argument("--type_flip_weight", type=float, default=1.0)
    parser.add_argument("--state_loss_weight", type=float, default=1.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)

    save_dir = PROJECT_ROOT / args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset, train_loader, val_loader = build_dataloaders(
        train_path=args.train_path,
        val_path=args.val_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    sample = train_dataset[0]
    node_feat_dim = sample["node_feats"].shape[-1]
    state_dim = node_feat_dim - 1

    model_config = GlobalBaselineConfig(
        node_feat_dim=node_feat_dim,
        num_node_types=args.num_node_types,
        type_dim=1,
        state_dim=state_dim,
        hidden_dim=args.hidden_dim,
        msg_pass_layers=args.msg_pass_layers,
        node_mlp_layers=args.node_mlp_layers,
        edge_mlp_layers=args.edge_mlp_layers,
        dropout=args.dropout,
    )

    model = GlobalTransitionBaseline(model_config).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_loss = float("inf")
    best_ckpt_path = save_dir / "best.pt"
    last_ckpt_path = save_dir / "last.pt"

    print(f"device: {device}")
    print(f"train size: {len(train_dataset)}")
    print(f"val size: {len(val_dataset)}")
    print(f"node_feat_dim: {node_feat_dim}")
    print(f"state_dim: {state_dim}")
    print(model)

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            edge_loss_weight=args.edge_loss_weight,
            type_loss_weight=args.type_loss_weight,
            state_loss_weight=args.state_loss_weight,
            type_flip_weight=args.type_flip_weight,
            grad_clip=args.grad_clip,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            edge_loss_weight=args.edge_loss_weight,
            type_loss_weight=args.type_loss_weight,
            state_loss_weight=args.state_loss_weight,
            type_flip_weight=args.type_flip_weight,
        )

        print(
            f"[Epoch {epoch:03d}] "
            f"train_total={train_metrics['total_loss']:.6f} "
            f"train_type={train_metrics['type_loss']:.6f} "
            f"train_state={train_metrics['state_loss']:.6f} "
            f"train_edge={train_metrics['edge_loss']:.6f} "
            f"train_type_acc={train_metrics['type_acc']:.6f} "
            f"train_state_mae={train_metrics['state_mae']:.6f} "
            f"train_edge_acc={train_metrics['edge_acc']:.6f} | "
            f"val_total={val_metrics['total_loss']:.6f} "
            f"val_type={val_metrics['type_loss']:.6f} "
            f"val_state={val_metrics['state_loss']:.6f} "
            f"val_edge={val_metrics['edge_loss']:.6f} "
            f"val_type_acc={val_metrics['type_acc']:.6f} "
            f"val_state_mae={val_metrics['state_mae']:.6f} "
            f"val_edge_acc={val_metrics['edge_acc']:.6f}"
        )

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": vars(model_config),
            "args": vars(args),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }

        torch.save(ckpt, last_ckpt_path)

        if val_metrics["total_loss"] < best_val_loss:
            best_val_loss = val_metrics["total_loss"]
            torch.save(ckpt, best_ckpt_path)
            print(f"  saved new best checkpoint -> {best_ckpt_path}")

    print(f"training finished. best_val_loss={best_val_loss:.6f}")
    print(f"best checkpoint: {best_ckpt_path}")
    print(f"last checkpoint: {last_ckpt_path}")


if __name__ == "__main__":
    main()
