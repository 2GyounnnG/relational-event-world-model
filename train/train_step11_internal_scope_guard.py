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

from data.collate import graph_event_collate_fn
from data.dataset import GraphEventDataset
from train.eval_noisy_structured_observation import move_batch_to_device, require_keys, resolve_path
from train.eval_step11_guarded_internal_completion import (
    EventScopeGuardConfig,
    EventScopeGuardHead,
    save_guard_checkpoint,
)
from train.eval_step9_gated_edge_completion import (
    get_base_proposal_outputs,
    load_completion_model,
    load_proposal_model,
)
from train.eval_step9_rescue_frontier import average_precision, auroc


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


def build_dataloaders(
    train_path: str,
    val_path: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple[GraphEventDataset, GraphEventDataset, DataLoader, DataLoader]:
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


def guard_loss(
    guard_logits: torch.Tensor,
    target_event_scope: torch.Tensor,
    candidate_mask: torch.Tensor,
    pos_weight: float,
) -> torch.Tensor:
    if candidate_mask.float().sum().item() <= 0:
        return guard_logits.sum() * 0.0
    pos_weight_tensor = torch.tensor(pos_weight, device=guard_logits.device, dtype=guard_logits.dtype)
    bce = F.binary_cross_entropy_with_logits(
        guard_logits,
        target_event_scope.float(),
        pos_weight=pos_weight_tensor,
        reduction="none",
    )
    bce = bce * candidate_mask.float()
    return bce.sum() / candidate_mask.float().sum().clamp_min(1.0)


def init_sums() -> Dict[str, float]:
    return {
        "loss_sum": 0.0,
        "loss_batches": 0.0,
        "candidate_total": 0.0,
        "event_scope_positive_total": 0.0,
        "changed_positive_total": 0.0,
        "guard_pred_pos_total": 0.0,
        "guard_event_scope_tp": 0.0,
    }


def finalize_metrics(sums: Dict[str, float], scores: list[torch.Tensor], labels: list[torch.Tensor]) -> Dict[str, Any]:
    precision = None
    recall = None
    f1 = None
    if sums["guard_pred_pos_total"] > 0:
        precision = sums["guard_event_scope_tp"] / sums["guard_pred_pos_total"]
    if sums["event_scope_positive_total"] > 0:
        recall = sums["guard_event_scope_tp"] / sums["event_scope_positive_total"]
    if precision is not None and recall is not None and precision + recall > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    ap = None
    auc = None
    if scores:
        score_tensor = torch.cat(scores).float()
        label_tensor = torch.cat(labels).bool()
        ap = average_precision(score_tensor, label_tensor)
        auc = auroc(score_tensor, label_tensor)
    return {
        "loss": sums["loss_sum"] / max(sums["loss_batches"], 1.0),
        "candidate_total": int(sums["candidate_total"]),
        "event_scope_positive_total": int(sums["event_scope_positive_total"]),
        "changed_positive_total": int(sums["changed_positive_total"]),
        "guard_precision_at_0.5": precision,
        "guard_recall_at_0.5": recall,
        "guard_f1_at_0.5": f1,
        "guard_event_scope_ap": ap,
        "guard_event_scope_auroc": auc,
        "selection_score": ap,
    }


def run_epoch(
    proposal_model: torch.nn.Module,
    completion_model: torch.nn.Module,
    guard_model: EventScopeGuardHead,
    loader: DataLoader,
    device: torch.device,
    node_threshold: float,
    edge_threshold: float,
    guard_pos_weight: float,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_clip: Optional[float] = None,
) -> Dict[str, Any]:
    is_train = optimizer is not None
    proposal_model.eval()
    completion_model.eval()
    guard_model.train() if is_train else guard_model.eval()
    sums = init_sums()
    all_scores: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for batch in loader:
        require_keys(
            batch,
            [
                "node_feats",
                "adj",
                "node_mask",
                "changed_edges",
                "event_scope_union_edges",
            ],
        )
        batch = move_batch_to_device(batch, device)
        with torch.no_grad():
            base_outputs = get_base_proposal_outputs(
                proposal_model=proposal_model,
                batch=batch,
                node_threshold=node_threshold,
                edge_threshold=edge_threshold,
            )
            completion_logits = completion_model(
                node_latents=base_outputs["node_latents"],
                base_edge_logits=base_outputs["edge_scope_logits"],
            )
            node_latents = base_outputs["node_latents"].detach()
            base_edge_logits = base_outputs["edge_scope_logits"].detach()
            completion_logits = completion_logits.detach()
            candidates = base_outputs["rescue_candidates"].detach()

        guard_logits = guard_model(
            node_latents=node_latents,
            base_edge_logits=base_edge_logits,
            completion_logits=completion_logits,
        )
        target = batch["event_scope_union_edges"]
        loss = guard_loss(
            guard_logits=guard_logits,
            target_event_scope=target,
            candidate_mask=candidates,
            pos_weight=guard_pos_weight,
        )

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(guard_model.parameters(), grad_clip)
            optimizer.step()

        with torch.no_grad():
            guard_probs = torch.sigmoid(guard_logits)
            pred = (guard_probs >= 0.5) & candidates
            target_bool = (target > 0.5) & candidates
            changed_bool = (batch["changed_edges"] > 0.5) & candidates
            sums["loss_sum"] += loss.item()
            sums["loss_batches"] += 1.0
            sums["candidate_total"] += candidates.float().sum().item()
            sums["event_scope_positive_total"] += target_bool.float().sum().item()
            sums["changed_positive_total"] += changed_bool.float().sum().item()
            sums["guard_pred_pos_total"] += pred.float().sum().item()
            sums["guard_event_scope_tp"] += (pred & target_bool).float().sum().item()
            if candidates.float().sum().item() > 0:
                all_scores.append(guard_probs[candidates].detach().float().cpu())
                all_labels.append(target_bool[candidates].detach().bool().cpu())

    return finalize_metrics(sums, all_scores, all_labels)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--proposal_checkpoint_path", type=str, required=True)
    parser.add_argument("--completion_checkpoint_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--node_threshold", type=float, default=0.20)
    parser.add_argument("--edge_threshold", type=float, default=0.15)
    parser.add_argument("--guard_pos_weight", type=float, default=10.0)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--head_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    proposal_checkpoint_path = resolve_path(args.proposal_checkpoint_path)
    completion_checkpoint_path = resolve_path(args.completion_checkpoint_path)
    save_dir = resolve_path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset, train_loader, val_loader = build_dataloaders(
        train_path=str(resolve_path(args.train_path)),
        val_path=str(resolve_path(args.val_path)),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    proposal_model = load_proposal_model(proposal_checkpoint_path, device)
    completion_model = load_completion_model(completion_checkpoint_path, device)
    proposal_model.requires_grad_(False)
    completion_model.requires_grad_(False)

    guard_config = EventScopeGuardConfig(
        proposal_hidden_dim=proposal_model.config.hidden_dim,
        hidden_dim=args.hidden_dim,
        head_layers=args.head_layers,
        dropout=args.dropout,
        include_base_edge_features=True,
        include_completion_features=True,
    )
    guard_model = EventScopeGuardHead(guard_config).to(device)
    optimizer = torch.optim.Adam(guard_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_score = float("-inf")
    best_epoch = -1
    epochs_without_improve = 0
    history: list[Dict[str, Any]] = []
    best_path = save_dir / "best.pt"
    last_path = save_dir / "last.pt"
    summary_path = save_dir / "training_summary.json"

    print(f"device: {device}")
    print(f"train size: {len(train_dataset)}")
    print(f"val size: {len(val_dataset)}")
    print(f"proposal checkpoint: {proposal_checkpoint_path}")
    print(f"completion checkpoint: {completion_checkpoint_path}")
    print("frozen proposal + frozen completion; training only event-scope guard")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            proposal_model=proposal_model,
            completion_model=completion_model,
            guard_model=guard_model,
            loader=train_loader,
            device=device,
            node_threshold=args.node_threshold,
            edge_threshold=args.edge_threshold,
            guard_pos_weight=args.guard_pos_weight,
            optimizer=optimizer,
            grad_clip=args.grad_clip,
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                proposal_model=proposal_model,
                completion_model=completion_model,
                guard_model=guard_model,
                loader=val_loader,
                device=device,
                node_threshold=args.node_threshold,
                edge_threshold=args.edge_threshold,
                guard_pos_weight=args.guard_pos_weight,
                optimizer=None,
                grad_clip=None,
            )
        val_score = val_metrics.get("selection_score")
        if val_score is None:
            val_score = -float(val_metrics.get("loss") or 0.0)
        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "selection_score": val_score,
            }
        )
        print(
            f"epoch {epoch:03d} | "
            f"train_loss={train_metrics.get('loss'):.6f} | "
            f"val_loss={val_metrics.get('loss'):.6f} | "
            f"val_ap={val_metrics.get('guard_event_scope_ap')} | "
            f"val_auc={val_metrics.get('guard_event_scope_auroc')} | "
            f"score={val_score}"
        )

        if val_score > best_score:
            best_score = float(val_score)
            best_epoch = epoch
            epochs_without_improve = 0
            save_guard_checkpoint(
                best_path,
                guard_model,
                args,
                extra={
                    "best_epoch": best_epoch,
                    "best_validation_selection_score": best_score,
                    "best_val_metrics": val_metrics,
                    "proposal_checkpoint_path": str(proposal_checkpoint_path),
                    "completion_checkpoint_path": str(completion_checkpoint_path),
                },
            )
        else:
            epochs_without_improve += 1

        save_guard_checkpoint(
            last_path,
            guard_model,
            args,
            extra={
                "epoch": epoch,
                "validation_selection_score": val_score,
                "val_metrics": val_metrics,
                "proposal_checkpoint_path": str(proposal_checkpoint_path),
                "completion_checkpoint_path": str(completion_checkpoint_path),
            },
        )
        save_json(
            summary_path,
            {
                "best_epoch": best_epoch,
                "best_validation_selection_score": best_score,
                "proposal_checkpoint_path": str(proposal_checkpoint_path),
                "completion_checkpoint_path": str(completion_checkpoint_path),
                "args": vars(args),
                "history": history,
            },
        )

        if args.patience > 0 and epochs_without_improve >= args.patience:
            print(f"early stopping after {epoch} epochs")
            break

    print(f"best epoch: {best_epoch}")
    print(f"best validation selection score: {best_score}")
    print(f"saved best checkpoint: {best_path}")
    print(f"saved summary: {summary_path}")


if __name__ == "__main__":
    main()
