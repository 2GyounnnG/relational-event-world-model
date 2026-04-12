from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.step31_dataset import Step31MultiViewObservationDataset, step31_multi_view_collate_fn
from models.encoder_step30 import build_pair_mask, masked_edge_bce_loss, step30_recovery_metrics
from train.eval_step30_encoder_recovery import load_model as load_step30_model
from train.eval_step31_multi_view_bridge import (
    load_step31_model,
    simple_late_fusion_outputs,
)
from train.train_step31_multi_view_encoder import get_device, move_batch_to_device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loader(
    path: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    shuffle: bool,
) -> tuple[Step31MultiViewObservationDataset, DataLoader]:
    dataset = Step31MultiViewObservationDataset(path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=step31_multi_view_collate_fn,
        pin_memory=pin_memory,
    )
    return dataset, loader


def freeze_except_edge_head(model: torch.nn.Module) -> None:
    for _, param in model.named_parameters():
        param.requires_grad = False
    for _, param in model.edge_head.named_parameters():
        param.requires_grad = True


def late_fusion_positive_excess_loss(
    learned_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    node_mask: torch.Tensor,
    relation_std: torch.Tensor,
    support_std: torch.Tensor,
    disagreement_start: float,
    disagreement_width: float,
    margin: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Penalize learned-only positive over-admission in disagreement-rich pairs."""

    pair_mask = build_pair_mask(node_mask)
    max_std = torch.maximum(relation_std, support_std)
    gate = ((max_std - disagreement_start) / max(disagreement_width, 1e-6)).clamp(0.0, 1.0)
    over = torch.relu(learned_logits - teacher_logits - float(margin))
    loss = F.smooth_l1_loss(over, torch.zeros_like(over), reduction="none")
    weighted = loss * pair_mask * gate
    return weighted.sum() / ((pair_mask * gate).sum() + eps)


def loss_for_batch(
    model: torch.nn.Module,
    teacher_model: torch.nn.Module,
    batch: Dict[str, Any],
    edge_pos_weight: float,
    edge_loss_weight: float,
    teacher_loss_weight: float,
    disagreement_start: float,
    disagreement_width: float,
    teacher_margin: float,
) -> tuple[torch.Tensor, Dict[str, float]]:
    outputs = model(
        multi_view_slot_features=batch["multi_view_slot_features"],
        multi_view_relation_hints=batch["multi_view_relation_hints"],
        multi_view_pair_support_hints=batch["multi_view_pair_support_hints"],
        multi_view_signed_pair_witness=batch["multi_view_signed_pair_witness"],
        multi_view_pair_evidence_bundle=batch["multi_view_pair_evidence_bundle"],
    )
    with torch.no_grad():
        teacher_outputs = simple_late_fusion_outputs(teacher_model, batch)
    relation_std = batch["multi_view_relation_hints"].std(dim=1, unbiased=False)
    support_std = batch["multi_view_pair_support_hints"].std(dim=1, unbiased=False)
    edge_loss = masked_edge_bce_loss(
        outputs["edge_logits"],
        batch["target_adj"],
        batch["node_mask"],
        edge_pos_weight=edge_pos_weight,
    )
    teacher_loss = late_fusion_positive_excess_loss(
        learned_logits=outputs["edge_logits"],
        teacher_logits=teacher_outputs["edge_logits"].detach(),
        node_mask=batch["node_mask"],
        relation_std=relation_std,
        support_std=support_std,
        disagreement_start=disagreement_start,
        disagreement_width=disagreement_width,
        margin=teacher_margin,
    )
    total = float(edge_loss_weight) * edge_loss + float(teacher_loss_weight) * teacher_loss
    return total, {
        "total_loss": float(total.detach().item()),
        "edge_loss": float(edge_loss.detach().item()),
        "teacher_loss": float(teacher_loss.detach().item()),
    }


@torch.no_grad()
def evaluate_recovery(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    edge_threshold: float,
) -> Dict[str, float]:
    model.eval()
    sums = {
        "edge_precision": 0.0,
        "edge_recall": 0.0,
        "edge_f1": 0.0,
        "node_type_accuracy": 0.0,
        "node_state_mae": 0.0,
    }
    count = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        outputs = model(
            multi_view_slot_features=batch["multi_view_slot_features"],
            multi_view_relation_hints=batch["multi_view_relation_hints"],
            multi_view_pair_support_hints=batch["multi_view_pair_support_hints"],
            multi_view_signed_pair_witness=batch["multi_view_signed_pair_witness"],
            multi_view_pair_evidence_bundle=batch["multi_view_pair_evidence_bundle"],
        )
        metrics = step30_recovery_metrics(
            outputs=outputs,
            target_node_feats=batch["target_node_feats"],
            target_adj=batch["target_adj"],
            node_mask=batch["node_mask"],
            edge_threshold=edge_threshold,
        )
        for key in sums:
            sums[key] += float(metrics[key])
        count += 1
    return {key: value / max(count, 1) for key, value in sums.items()}


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    args: argparse.Namespace,
    epoch: int,
    val_metrics: Dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": model.config.to_dict(),
            "args": vars(args),
            "epoch": epoch,
            "val_metrics": val_metrics,
            "best_validation_selection_score": val_metrics["selection_score"],
        },
        path,
    )


def run_epoch(
    model: torch.nn.Module,
    teacher_model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    optimizer: Optional[torch.optim.Optimizer],
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    teacher_model.eval()
    total_loss = 0.0
    edge_loss = 0.0
    teacher_loss = 0.0
    count = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            loss, parts = loss_for_batch(
                model=model,
                teacher_model=teacher_model,
                batch=batch,
                edge_pos_weight=args.edge_pos_weight,
                edge_loss_weight=args.edge_loss_weight,
                teacher_loss_weight=args.teacher_loss_weight,
                disagreement_start=args.disagreement_start,
                disagreement_width=args.disagreement_width,
                teacher_margin=args.teacher_margin,
            )
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [param for param in model.parameters() if param.requires_grad],
                        args.grad_clip,
                    )
                optimizer.step()
            total_loss += parts["total_loss"]
            edge_loss += parts["edge_loss"]
            teacher_loss += parts["teacher_loss"]
            count += 1
    return {
        "total_loss": total_loss / max(count, 1),
        "edge_loss": edge_loss / max(count, 1),
        "teacher_loss": teacher_loss / max(count, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="data/graph_event_step31_multi_view_train.pkl")
    parser.add_argument("--val_path", default="data/graph_event_step31_multi_view_val.pkl")
    parser.add_argument("--base_checkpoint", default="checkpoints/step31_multi_view_encoder/best.pt")
    parser.add_argument("--single_view_checkpoint", default="checkpoints/step31_single_view_baseline/best.pt")
    parser.add_argument("--save_dir", default="checkpoints/step31d_late_fusion_distilled_encoder")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--edge_pos_weight", type=float, default=1.5)
    parser.add_argument("--edge_loss_weight", type=float, default=1.0)
    parser.add_argument("--teacher_loss_weight", type=float, default=0.8)
    parser.add_argument("--teacher_margin", type=float, default=0.0)
    parser.add_argument("--disagreement_start", type=float, default=0.08)
    parser.add_argument("--disagreement_width", type=float, default=0.07)
    parser.add_argument("--edge_threshold", type=float, default=0.5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    pin_memory = device.type == "cuda"
    _, train_loader = build_loader(
        args.train_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )
    _, val_loader = build_loader(
        args.val_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    model = load_step31_model(args.base_checkpoint, device)
    teacher_model = load_step30_model(args.single_view_checkpoint, device)
    freeze_except_edge_head(model)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_score = -float("inf")
    best_epoch = -1
    history = []
    save_dir = Path(args.save_dir)
    print(f"device: {device}")
    print(f"trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print("teacher: simple late fusion from frozen Step31 single-view checkpoint")

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, teacher_model, train_loader, device, args, optimizer)
        val_loss = run_epoch(model, teacher_model, val_loader, device, args, optimizer=None)
        val_recovery = evaluate_recovery(model, val_loader, device, args.edge_threshold)
        selection_score = (
            float(val_recovery["edge_f1"])
            + 0.25 * float(val_recovery["edge_precision"])
            - 0.05 * float(val_loss["teacher_loss"])
        )
        val_metrics = {
            **{f"train_{key}": value for key, value in train_loss.items()},
            **{f"val_{key}": value for key, value in val_loss.items()},
            **val_recovery,
            "selection_score": selection_score,
        }
        history.append({"epoch": epoch, "metrics": val_metrics})
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss['total_loss']:.6f} "
            f"val_loss={val_loss['total_loss']:.6f} "
            f"val_edge_p={val_recovery['edge_precision']:.4f} "
            f"val_edge_r={val_recovery['edge_recall']:.4f} "
            f"val_edge_f1={val_recovery['edge_f1']:.4f} "
            f"score={selection_score:.4f}"
        )
        if selection_score > best_score:
            best_score = float(selection_score)
            best_epoch = epoch
            save_checkpoint(save_dir / "best.pt", model, args, epoch, val_metrics)

    save_checkpoint(save_dir / "last.pt", model, args, args.epochs, val_metrics)
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_epoch": best_epoch,
                "best_validation_selection_score": best_score,
                "args": vars(args),
                "history": history,
            },
            f,
            indent=2,
        )
    print(f"best epoch: {best_epoch}")
    print(f"saved best checkpoint: {save_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
