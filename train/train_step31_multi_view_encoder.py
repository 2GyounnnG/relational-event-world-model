from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.step31_dataset import Step31MultiViewObservationDataset, step31_multi_view_collate_fn
from models.encoder_step30 import step30_recovery_loss, step30_recovery_metrics
from models.encoder_step31 import Step31MultiViewEncoderConfig, Step31MultiViewObservationEncoder


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


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def average_dict(metric_sums: Dict[str, float], count: int) -> Dict[str, float]:
    return {key: value / max(count, 1) for key, value in metric_sums.items()}


def selection_score(metrics: Dict[str, float]) -> float:
    return (
        float(metrics["node_type_accuracy"])
        + float(metrics["edge_f1"])
        - float(metrics["node_state_mae"])
    )


def build_loaders(
    train_path: str,
    val_path: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple[Step31MultiViewObservationDataset, Step31MultiViewObservationDataset, DataLoader, DataLoader]:
    train_dataset = Step31MultiViewObservationDataset(train_path)
    val_dataset = Step31MultiViewObservationDataset(val_path)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=step31_multi_view_collate_fn,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=step31_multi_view_collate_fn,
        pin_memory=pin_memory,
    )
    return train_dataset, val_dataset, train_loader, val_loader


def infer_config_from_dataset(
    dataset: Step31MultiViewObservationDataset,
    hidden_dim: int,
    msg_pass_layers: int,
    node_head_layers: int,
    edge_head_layers: int,
    dropout: float,
    num_node_types: int,
    use_relation_logit_residual: bool,
    relation_logit_residual_scale: float,
) -> Step31MultiViewEncoderConfig:
    first = dataset[0]
    return Step31MultiViewEncoderConfig(
        obs_slot_dim=int(first["multi_view_slot_features"].shape[-1]),
        num_views=int(first["multi_view_slot_features"].shape[0]),
        pair_evidence_bundle_dim=int(first["multi_view_pair_evidence_bundle"].shape[-1]),
        num_node_types=num_node_types,
        state_dim=int(first["target_node_feats"].shape[-1] - 1),
        hidden_dim=hidden_dim,
        msg_pass_layers=msg_pass_layers,
        node_head_layers=node_head_layers,
        edge_head_layers=edge_head_layers,
        dropout=dropout,
        use_relation_logit_residual=use_relation_logit_residual,
        relation_logit_residual_scale=relation_logit_residual_scale,
    )


def run_epoch(
    model: Step31MultiViewObservationEncoder,
    loader: DataLoader,
    device: torch.device,
    type_loss_weight: float,
    state_loss_weight: float,
    edge_loss_weight: float,
    edge_pos_weight: float,
    missed_edge_loss_weight: float,
    missed_edge_hint_threshold: float,
    edge_threshold: float,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_clip: Optional[float] = None,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    metric_sums: Dict[str, float] = {
        "total_loss": 0.0,
        "type_loss": 0.0,
        "state_loss": 0.0,
        "edge_loss": 0.0,
        "missed_edge_loss": 0.0,
        "node_type_accuracy": 0.0,
        "node_state_mae": 0.0,
        "node_state_mse": 0.0,
        "edge_accuracy": 0.0,
        "edge_precision": 0.0,
        "edge_recall": 0.0,
        "edge_f1": 0.0,
        "selection_score": 0.0,
    }
    num_batches = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            outputs = model(
                multi_view_slot_features=batch["multi_view_slot_features"],
                multi_view_relation_hints=batch["multi_view_relation_hints"],
                multi_view_pair_support_hints=batch["multi_view_pair_support_hints"],
                multi_view_signed_pair_witness=batch["multi_view_signed_pair_witness"],
                multi_view_pair_evidence_bundle=batch["multi_view_pair_evidence_bundle"],
            )
            relation_mean = batch["multi_view_relation_hints"].mean(dim=1)
            support_mean = batch["multi_view_pair_support_hints"].mean(dim=1)
            loss_dict = step30_recovery_loss(
                outputs=outputs,
                target_node_feats=batch["target_node_feats"],
                target_adj=batch["target_adj"],
                node_mask=batch["node_mask"],
                relation_hints=relation_mean,
                pair_support_hints=support_mean,
                type_loss_weight=type_loss_weight,
                state_loss_weight=state_loss_weight,
                edge_loss_weight=edge_loss_weight,
                edge_pos_weight=edge_pos_weight,
                missed_edge_loss_weight=missed_edge_loss_weight,
                missed_edge_hint_threshold=missed_edge_hint_threshold,
            )

            if is_train:
                optimizer.zero_grad()
                loss_dict["total_loss"].backward()
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            metrics = step30_recovery_metrics(
                outputs=outputs,
                target_node_feats=batch["target_node_feats"],
                target_adj=batch["target_adj"],
                node_mask=batch["node_mask"],
                edge_threshold=edge_threshold,
            )
            metrics["selection_score"] = selection_score(metrics)

            for key in ["total_loss", "type_loss", "state_loss", "edge_loss", "missed_edge_loss"]:
                metric_sums[key] += float(loss_dict[key].detach().item())
            for key in [
                "node_type_accuracy",
                "node_state_mae",
                "node_state_mse",
                "edge_accuracy",
                "edge_precision",
                "edge_recall",
                "edge_f1",
                "selection_score",
            ]:
                metric_sums[key] += float(metrics[key])
            num_batches += 1

    return average_dict(metric_sums, num_batches)


def save_checkpoint(
    path: Path,
    model: Step31MultiViewObservationEncoder,
    config: Step31MultiViewEncoderConfig,
    args: argparse.Namespace,
    epoch: int,
    val_metrics: Dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config.to_dict(),
            "args": vars(args),
            "epoch": epoch,
            "val_metrics": val_metrics,
            "best_validation_selection_score": val_metrics["selection_score"],
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--msg_pass_layers", type=int, default=3)
    parser.add_argument("--node_head_layers", type=int, default=2)
    parser.add_argument("--edge_head_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num_node_types", type=int, default=3)
    parser.add_argument("--type_loss_weight", type=float, default=1.0)
    parser.add_argument("--state_loss_weight", type=float, default=1.0)
    parser.add_argument("--edge_loss_weight", type=float, default=2.0)
    parser.add_argument("--edge_pos_weight", type=float, default=1.5)
    parser.add_argument("--missed_edge_loss_weight", type=float, default=0.5)
    parser.add_argument("--missed_edge_hint_threshold", type=float, default=0.5)
    parser.add_argument("--edge_threshold", type=float, default=0.5)
    parser.add_argument("--use_relation_logit_residual", action="store_true")
    parser.add_argument("--relation_logit_residual_scale", type=float, default=1.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    save_dir = Path(args.save_dir)
    pin_memory = device.type == "cuda"
    train_dataset, val_dataset, train_loader, val_loader = build_loaders(
        train_path=args.train_path,
        val_path=args.val_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    config = infer_config_from_dataset(
        train_dataset,
        hidden_dim=args.hidden_dim,
        msg_pass_layers=args.msg_pass_layers,
        node_head_layers=args.node_head_layers,
        edge_head_layers=args.edge_head_layers,
        dropout=args.dropout,
        num_node_types=args.num_node_types,
        use_relation_logit_residual=args.use_relation_logit_residual,
        relation_logit_residual_scale=args.relation_logit_residual_scale,
    )
    model = Step31MultiViewObservationEncoder(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_score = -float("inf")
    best_epoch = -1
    history: list[Dict[str, Any]] = []

    print(f"device: {device}")
    print(f"train samples: {len(train_dataset)}")
    print(f"val samples: {len(val_dataset)}")
    print(f"config: {config.to_dict()}")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            type_loss_weight=args.type_loss_weight,
            state_loss_weight=args.state_loss_weight,
            edge_loss_weight=args.edge_loss_weight,
            edge_pos_weight=args.edge_pos_weight,
            missed_edge_loss_weight=args.missed_edge_loss_weight,
            missed_edge_hint_threshold=args.missed_edge_hint_threshold,
            edge_threshold=args.edge_threshold,
            optimizer=optimizer,
            grad_clip=args.grad_clip,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            type_loss_weight=args.type_loss_weight,
            state_loss_weight=args.state_loss_weight,
            edge_loss_weight=args.edge_loss_weight,
            edge_pos_weight=args.edge_pos_weight,
            missed_edge_loss_weight=args.missed_edge_loss_weight,
            missed_edge_hint_threshold=args.missed_edge_hint_threshold,
            edge_threshold=args.edge_threshold,
            optimizer=None,
        )
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['total_loss']:.6f} "
            f"val_loss={val_metrics['total_loss']:.6f} "
            f"val_type_acc={val_metrics['node_type_accuracy']:.4f} "
            f"val_state_mae={val_metrics['node_state_mae']:.4f} "
            f"val_edge_f1={val_metrics['edge_f1']:.4f} "
            f"val_score={val_metrics['selection_score']:.4f}"
        )
        if val_metrics["selection_score"] > best_score:
            best_score = float(val_metrics["selection_score"])
            best_epoch = epoch
            save_checkpoint(save_dir / "best.pt", model, config, args, epoch, val_metrics)

    save_checkpoint(save_dir / "last.pt", model, config, args, args.epochs, val_metrics)
    summary = {
        "best_epoch": best_epoch,
        "best_validation_selection_score": best_score,
        "config": config.to_dict(),
        "args": vars(args),
        "history": history,
    }
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"best epoch: {best_epoch}")
    print(f"best validation selection score: {best_score:.6f}")
    print(f"saved best checkpoint: {save_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
