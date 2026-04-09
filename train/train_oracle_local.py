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
#   python train/train_oracle_local.py
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import GraphEventDataset
from data.collate import graph_event_collate_fn
from models.oracle_local import (
    OracleLocalRewriteConfig,
    OracleLocalRewriteModel,
    build_valid_edge_mask,
    oracle_full_prediction_loss,
    oracle_local_rewrite_loss,
)


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


@torch.no_grad()
def compute_type_accuracy_masked(
    type_logits: torch.Tensor,
    target_node_feats: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    pred_type = type_logits.argmax(dim=-1)
    target_type = target_node_feats[:, :, 0].long()

    correct = ((pred_type == target_type).float() * mask).sum()
    total = mask.sum().clamp_min(1.0)
    return (correct / total).item()


@torch.no_grad()
def compute_state_mae_masked(
    state_pred: torch.Tensor,
    target_node_feats: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    target_state = target_node_feats[:, :, 1:]
    abs_err = torch.abs(state_pred - target_state)
    abs_err = abs_err * mask.unsqueeze(-1)
    denom = (mask.sum() * state_pred.shape[-1]).clamp_min(1.0)
    return (abs_err.sum() / denom).item()


@torch.no_grad()
def compute_edge_accuracy_masked(
    edge_logits: torch.Tensor,
    target_adj: torch.Tensor,
    pair_mask: torch.Tensor,
) -> float:
    pred_adj = (torch.sigmoid(edge_logits) >= 0.5).float()
    correct = ((pred_adj == target_adj.float()).float() * pair_mask).sum()
    total = pair_mask.sum().clamp_min(1.0)
    return (correct / total).item()


def require_oracle_scope(batch: Dict) -> None:
    required_keys = ["event_scope_union_nodes", "event_scope_union_edges"]
    missing = [k for k in required_keys if k not in batch]
    if missing:
        raise KeyError(
            "Oracle local rewrite training requires scope annotations in the batch. "
            f"Missing keys: {missing}"
        )


def summarize_batch_metrics(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    local_loss_dict: Dict[str, torch.Tensor],
    full_loss_dict: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    node_mask = batch["node_mask"]
    scope_node_mask = batch["event_scope_union_nodes"] * node_mask
    scope_edge_mask = batch["event_scope_union_edges"] * build_valid_edge_mask(node_mask)

    return {
        "local_total_loss": local_loss_dict["total_loss"].item(),
        "local_type_loss": local_loss_dict["type_loss"].item(),
        "local_state_loss": local_loss_dict["state_loss"].item(),
        "local_edge_loss": local_loss_dict["edge_loss"].item(),
        "full_total_loss": full_loss_dict["total_loss"].item(),
        "full_type_loss": full_loss_dict["type_loss"].item(),
        "full_state_loss": full_loss_dict["state_loss"].item(),
        "full_edge_loss": full_loss_dict["edge_loss"].item(),
        "full_type_acc": compute_type_accuracy_masked(
            outputs["type_logits_full"],
            batch["next_node_feats"],
            node_mask,
        ),
        "full_state_mae": compute_state_mae_masked(
            outputs["state_pred_full"],
            batch["next_node_feats"],
            node_mask,
        ),
        "full_edge_acc": compute_edge_accuracy_masked(
            outputs["edge_logits_full"],
            batch["next_adj"],
            build_valid_edge_mask(node_mask),
        ),
        "scope_type_acc": compute_type_accuracy_masked(
            outputs["type_logits_local"],
            batch["next_node_feats"],
            scope_node_mask,
        ),
        "scope_state_mae": compute_state_mae_masked(
            outputs["state_pred_local"],
            batch["next_node_feats"],
            scope_node_mask,
        ),
        "scope_edge_acc": compute_edge_accuracy_masked(
            outputs["edge_logits_local"],
            batch["next_adj"],
            scope_edge_mask,
        ),
        "avg_scope_node_fraction": (scope_node_mask.sum() / node_mask.sum().clamp_min(1.0)).item(),
        "avg_scope_edge_fraction": (
            scope_edge_mask.sum() / build_valid_edge_mask(node_mask).sum().clamp_min(1.0)
        ).item(),
    }


def average_metric_dict(metric_sums: Dict[str, float], count: int) -> Dict[str, float]:
    return {k: v / max(count, 1) for k, v in metric_sums.items()}


def run_epoch(
    model: OracleLocalRewriteModel,
    loader: DataLoader,
    device: torch.device,
    edge_loss_weight: float,
    type_loss_weight: float,
    state_loss_weight: float,
    type_flip_weight: float,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_clip: Optional[float] = None,
) -> Dict[str, float]:
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    metric_sums: Dict[str, float] = {
        "local_total_loss": 0.0,
        "local_type_loss": 0.0,
        "local_state_loss": 0.0,
        "local_edge_loss": 0.0,
        "full_total_loss": 0.0,
        "full_type_loss": 0.0,
        "full_state_loss": 0.0,
        "full_edge_loss": 0.0,
        "full_type_acc": 0.0,
        "full_state_mae": 0.0,
        "full_edge_acc": 0.0,
        "scope_type_acc": 0.0,
        "scope_state_mae": 0.0,
        "scope_edge_acc": 0.0,
        "avg_scope_node_fraction": 0.0,
        "avg_scope_edge_fraction": 0.0,
    }
    num_batches = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in loader:
            require_oracle_scope(batch)
            batch = move_batch_to_device(batch, device)

            outputs = model(
                node_feats=batch["node_feats"],
                adj=batch["adj"],
                scope_node_mask=batch["event_scope_union_nodes"],
                scope_edge_mask=batch["event_scope_union_edges"],
            )

            local_loss_dict = oracle_local_rewrite_loss(
                outputs=outputs,
                current_node_feats=batch["node_feats"],
                target_node_feats=batch["next_node_feats"],
                target_adj=batch["next_adj"],
                node_mask=batch["node_mask"],
                scope_node_mask=batch["event_scope_union_nodes"],
                scope_edge_mask=batch["event_scope_union_edges"],
                edge_loss_weight=edge_loss_weight,
                type_loss_weight=type_loss_weight,
                state_loss_weight=state_loss_weight,
                type_flip_weight=type_flip_weight,
            )

            full_loss_dict = oracle_full_prediction_loss(
                outputs=outputs,
                target_node_feats=batch["next_node_feats"],
                target_adj=batch["next_adj"],
                node_mask=batch["node_mask"],
                edge_loss_weight=edge_loss_weight,
                type_loss_weight=type_loss_weight,
                state_loss_weight=state_loss_weight,
            )

            if is_train:
                optimizer.zero_grad()
                local_loss_dict["total_loss"].backward()
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            batch_metrics = summarize_batch_metrics(
                outputs=outputs,
                batch=batch,
                local_loss_dict=local_loss_dict,
                full_loss_dict=full_loss_dict,
            )
            for key, value in batch_metrics.items():
                metric_sums[key] += value
            num_batches += 1

    return average_metric_dict(metric_sums, num_batches)


def save_json(path: Path, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="data/graph_event_train.pkl")
    parser.add_argument("--val_path", type=str, default="data/graph_event_val.pkl")
    parser.add_argument("--save_dir", type=str, default="checkpoints/oracle_local_rewrite_typed")
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
    parser.add_argument("--copy_logit_value", type=float, default=10.0)
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
            "Oracle local rewrite training requires event_scope_union_nodes and "
            "event_scope_union_edges in each dataset sample."
        )

    node_feat_dim = sample["node_feats"].shape[-1]
    state_dim = node_feat_dim - 1

    model_config = OracleLocalRewriteConfig(
        node_feat_dim=node_feat_dim,
        num_node_types=args.num_node_types,
        type_dim=1,
        state_dim=state_dim,
        hidden_dim=args.hidden_dim,
        msg_pass_layers=args.msg_pass_layers,
        node_mlp_layers=args.node_mlp_layers,
        edge_mlp_layers=args.edge_mlp_layers,
        dropout=args.dropout,
        copy_logit_value=args.copy_logit_value,
    )

    model = OracleLocalRewriteModel(model_config).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_full_loss = float("inf")
    best_ckpt_path = save_dir / "best.pt"
    last_ckpt_path = save_dir / "last.pt"
    best_metrics_path = save_dir / "best_metrics.json"

    print(f"device: {device}")
    print(f"train size: {len(train_dataset)}")
    print(f"val size: {len(val_dataset)}")
    print(f"node_feat_dim: {node_feat_dim}")
    print(f"state_dim: {state_dim}")
    print(model)

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            edge_loss_weight=args.edge_loss_weight,
            type_loss_weight=args.type_loss_weight,
            state_loss_weight=args.state_loss_weight,
            type_flip_weight=args.type_flip_weight,
            optimizer=optimizer,
            grad_clip=args.grad_clip,
        )

        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            edge_loss_weight=args.edge_loss_weight,
            type_loss_weight=args.type_loss_weight,
            state_loss_weight=args.state_loss_weight,
            type_flip_weight=args.type_flip_weight,
            optimizer=None,
            grad_clip=None,
        )

        print(
            f"[Epoch {epoch:03d}] "
            f"train_local_total={train_metrics['local_total_loss']:.6f} "
            f"train_full_total={train_metrics['full_total_loss']:.6f} "
            f"train_scope_type_acc={train_metrics['scope_type_acc']:.6f} "
            f"train_scope_state_mae={train_metrics['scope_state_mae']:.6f} "
            f"train_scope_edge_acc={train_metrics['scope_edge_acc']:.6f} "
            f"train_full_type_acc={train_metrics['full_type_acc']:.6f} "
            f"train_full_state_mae={train_metrics['full_state_mae']:.6f} "
            f"train_full_edge_acc={train_metrics['full_edge_acc']:.6f} | "
            f"val_local_total={val_metrics['local_total_loss']:.6f} "
            f"val_full_total={val_metrics['full_total_loss']:.6f} "
            f"val_scope_type_acc={val_metrics['scope_type_acc']:.6f} "
            f"val_scope_state_mae={val_metrics['scope_state_mae']:.6f} "
            f"val_scope_edge_acc={val_metrics['scope_edge_acc']:.6f} "
            f"val_full_type_acc={val_metrics['full_type_acc']:.6f} "
            f"val_full_state_mae={val_metrics['full_state_mae']:.6f} "
            f"val_full_edge_acc={val_metrics['full_edge_acc']:.6f}"
        )

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": vars(model_config),
            "args": vars(args),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "selection_metric": "full_total_loss",
        }

        torch.save(ckpt, last_ckpt_path)

        if val_metrics["full_total_loss"] < best_val_full_loss:
            best_val_full_loss = val_metrics["full_total_loss"]
            torch.save(ckpt, best_ckpt_path)
            save_json(
                best_metrics_path,
                {
                    "epoch": epoch,
                    "best_val_full_total_loss": best_val_full_loss,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "model_config": vars(model_config),
                    "args": vars(args),
                },
            )
            print(f"  saved new best checkpoint -> {best_ckpt_path}")

    print(f"training finished. best_val_full_total_loss={best_val_full_loss:.6f}")
    print(f"best checkpoint: {best_ckpt_path}")
    print(f"last checkpoint: {last_ckpt_path}")
    print(f"best metrics json: {best_metrics_path}")


if __name__ == "__main__":
    main()
