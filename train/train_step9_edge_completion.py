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
from train.eval_step9_gated_edge_completion import (
    EdgeCompletionConfig,
    GatedInternalEdgeCompletionHead,
    get_base_proposal_outputs,
    load_proposal_model,
    safe_div,
    save_completion_checkpoint,
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


def init_metric_sums() -> Dict[str, float]:
    return {
        "loss_sum": 0.0,
        "loss_batches": 0.0,
        "candidate_total": 0.0,
        "candidate_oracle_edge_total": 0.0,
        "predicted_rescue_total": 0.0,
        "predicted_rescue_oracle_edge_total": 0.0,
        "changed_edge_total": 0.0,
        "base_changed_covered": 0.0,
        "naive_changed_covered": 0.0,
        "learned_changed_covered": 0.0,
        "base_edge_pred_total": 0.0,
        "naive_edge_pred_total": 0.0,
        "learned_edge_pred_total": 0.0,
    }


def finalize_metrics(metric_sums: Dict[str, float]) -> Dict[str, Optional[float]]:
    base_recall = safe_div(metric_sums["base_changed_covered"], metric_sums["changed_edge_total"])
    naive_recall = safe_div(metric_sums["naive_changed_covered"], metric_sums["changed_edge_total"])
    learned_recall = safe_div(metric_sums["learned_changed_covered"], metric_sums["changed_edge_total"])
    recall_recovery = None
    if None not in (base_recall, naive_recall, learned_recall):
        recall_recovery = safe_div(learned_recall - base_recall, naive_recall - base_recall)

    cost_fraction = safe_div(
        metric_sums["learned_edge_pred_total"] - metric_sums["base_edge_pred_total"],
        metric_sums["naive_edge_pred_total"] - metric_sums["base_edge_pred_total"],
    )
    selection_score = None
    if recall_recovery is not None and cost_fraction is not None:
        selection_score = recall_recovery - cost_fraction

    rescue_precision = safe_div(
        metric_sums["predicted_rescue_oracle_edge_total"],
        metric_sums["predicted_rescue_total"],
    )
    rescue_recall = safe_div(
        metric_sums["predicted_rescue_oracle_edge_total"],
        metric_sums["candidate_oracle_edge_total"],
    )

    return {
        "loss": safe_div(metric_sums["loss_sum"], metric_sums["loss_batches"]),
        "candidate_total": metric_sums["candidate_total"],
        "candidate_oracle_edge_total": metric_sums["candidate_oracle_edge_total"],
        "rescue_precision_vs_oracle_scope": rescue_precision,
        "rescue_recall_vs_oracle_scope": rescue_recall,
        "base_changed_edge_recall": base_recall,
        "naive_changed_edge_recall": naive_recall,
        "learned_changed_edge_recall": learned_recall,
        "recall_recovery_fraction_vs_naive": recall_recovery,
        "cost_fraction_vs_naive": cost_fraction,
        "selection_score": selection_score,
        "base_edge_pred_total": metric_sums["base_edge_pred_total"],
        "naive_edge_pred_total": metric_sums["naive_edge_pred_total"],
        "learned_edge_pred_total": metric_sums["learned_edge_pred_total"],
    }


def compute_completion_loss(
    completion_logits: torch.Tensor,
    target_edge_scope: torch.Tensor,
    candidate_mask: torch.Tensor,
    pos_weight: float,
) -> torch.Tensor:
    if candidate_mask.float().sum().item() <= 0:
        return completion_logits.sum() * 0.0
    pos_weight_tensor = torch.tensor(
        pos_weight,
        device=completion_logits.device,
        dtype=completion_logits.dtype,
    )
    bce = F.binary_cross_entropy_with_logits(
        completion_logits,
        target_edge_scope.float(),
        pos_weight=pos_weight_tensor,
        reduction="none",
    )
    bce = bce * candidate_mask.float()
    return bce.sum() / candidate_mask.float().sum().clamp_min(1.0)


def update_scope_metrics(
    metric_sums: Dict[str, float],
    batch: Dict[str, Any],
    base_outputs: Dict[str, torch.Tensor],
    completion_logits: torch.Tensor,
    completion_threshold: float,
) -> None:
    valid_edge_mask = base_outputs["valid_edge_mask"].bool()
    base_edges = base_outputs["pred_scope_edges"] & valid_edge_mask
    naive_edges = base_outputs["pred_scope_edges"] | base_outputs["node_induced_edges"]
    naive_edges = naive_edges & valid_edge_mask
    candidates = base_outputs["rescue_candidates"] & valid_edge_mask
    completion_probs = torch.sigmoid(completion_logits) * valid_edge_mask.float()
    learned_rescue = (completion_probs >= completion_threshold) & candidates
    learned_edges = base_edges | learned_rescue
    changed_edges = (batch["changed_edges"] > 0.5) & valid_edge_mask
    target_scope = (batch["event_scope_union_edges"] > 0.5) & valid_edge_mask

    metric_sums["candidate_total"] += candidates.float().sum().item()
    metric_sums["candidate_oracle_edge_total"] += (candidates & target_scope).float().sum().item()
    metric_sums["predicted_rescue_total"] += learned_rescue.float().sum().item()
    metric_sums["predicted_rescue_oracle_edge_total"] += (learned_rescue & target_scope).float().sum().item()

    metric_sums["changed_edge_total"] += changed_edges.float().sum().item()
    metric_sums["base_changed_covered"] += (base_edges & changed_edges).float().sum().item()
    metric_sums["naive_changed_covered"] += (naive_edges & changed_edges).float().sum().item()
    metric_sums["learned_changed_covered"] += (learned_edges & changed_edges).float().sum().item()
    metric_sums["base_edge_pred_total"] += base_edges.float().sum().item()
    metric_sums["naive_edge_pred_total"] += naive_edges.float().sum().item()
    metric_sums["learned_edge_pred_total"] += learned_edges.float().sum().item()


def run_epoch(
    proposal_model: torch.nn.Module,
    completion_model: GatedInternalEdgeCompletionHead,
    loader: DataLoader,
    device: torch.device,
    node_threshold: float,
    edge_threshold: float,
    completion_threshold: float,
    completion_pos_weight: float,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_clip: Optional[float] = None,
) -> Dict[str, Optional[float]]:
    is_train = optimizer is not None
    completion_model.train() if is_train else completion_model.eval()
    proposal_model.eval()
    metric_sums = init_metric_sums()

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
            # Freeze the proposal backbone. The residual head learns only from frozen node embeddings/logits.
            node_latents = base_outputs["node_latents"].detach()
            edge_scope_logits = base_outputs["edge_scope_logits"].detach()
            candidates = base_outputs["rescue_candidates"].detach()
            target_scope = batch["event_scope_union_edges"].detach()

        completion_logits = completion_model(
            node_latents=node_latents,
            base_edge_logits=edge_scope_logits,
        )
        loss = compute_completion_loss(
            completion_logits=completion_logits,
            target_edge_scope=target_scope,
            candidate_mask=candidates,
            pos_weight=completion_pos_weight,
        )

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(completion_model.parameters(), grad_clip)
            optimizer.step()

        metric_sums["loss_sum"] += loss.item()
        metric_sums["loss_batches"] += 1.0
        update_scope_metrics(
            metric_sums=metric_sums,
            batch=batch,
            base_outputs=base_outputs,
            completion_logits=completion_logits.detach(),
            completion_threshold=completion_threshold,
        )

    return finalize_metrics(metric_sums)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--proposal_checkpoint_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--node_threshold", type=float, default=0.20)
    parser.add_argument("--edge_threshold", type=float, default=0.15)
    parser.add_argument("--completion_threshold", type=float, default=0.50)
    parser.add_argument("--completion_pos_weight", type=float, default=10.0)
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
    proposal_model.requires_grad_(False)

    proposal_hidden_dim = proposal_model.config.hidden_dim
    completion_config = EdgeCompletionConfig(
        proposal_hidden_dim=proposal_hidden_dim,
        hidden_dim=args.hidden_dim,
        head_layers=args.head_layers,
        dropout=args.dropout,
        include_base_edge_features=True,
    )
    completion_model = GatedInternalEdgeCompletionHead(completion_config).to(device)
    optimizer = torch.optim.Adam(
        completion_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_score = float("-inf")
    best_epoch = -1
    epochs_without_improve = 0
    history: list[Dict[str, Any]] = []
    best_ckpt_path = save_dir / "best.pt"
    last_ckpt_path = save_dir / "last.pt"
    summary_path = save_dir / "training_summary.json"

    print(f"device: {device}")
    print(f"train size: {len(train_dataset)}")
    print(f"val size: {len(val_dataset)}")
    print(f"proposal checkpoint: {proposal_checkpoint_path}")
    print("node proposal unchanged; frozen proposal backbone; training only residual internal edge-completion head")
    print(
        f"thresholds: node={args.node_threshold}, edge={args.edge_threshold}, "
        f"completion={args.completion_threshold}"
    )

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            proposal_model=proposal_model,
            completion_model=completion_model,
            loader=train_loader,
            device=device,
            node_threshold=args.node_threshold,
            edge_threshold=args.edge_threshold,
            completion_threshold=args.completion_threshold,
            completion_pos_weight=args.completion_pos_weight,
            optimizer=optimizer,
            grad_clip=args.grad_clip,
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                proposal_model=proposal_model,
                completion_model=completion_model,
                loader=val_loader,
                device=device,
                node_threshold=args.node_threshold,
                edge_threshold=args.edge_threshold,
                completion_threshold=args.completion_threshold,
                completion_pos_weight=args.completion_pos_weight,
                optimizer=None,
                grad_clip=None,
            )

        val_score = val_metrics.get("selection_score")
        if val_score is None:
            val_score = -float(val_metrics.get("loss") or 0.0)
        history_entry = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "selection_score": val_score,
        }
        history.append(history_entry)
        print(
            f"epoch {epoch:03d} | "
            f"train_loss={train_metrics.get('loss'):.6f} | "
            f"val_loss={val_metrics.get('loss'):.6f} | "
            f"val_recovery={val_metrics.get('recall_recovery_fraction_vs_naive')} | "
            f"val_cost={val_metrics.get('cost_fraction_vs_naive')} | "
            f"score={val_score}"
        )

        if val_score > best_score:
            best_score = float(val_score)
            best_epoch = epoch
            epochs_without_improve = 0
            save_completion_checkpoint(
                best_ckpt_path,
                completion_model,
                args,
                extra={
                    "best_epoch": best_epoch,
                    "best_validation_selection_score": best_score,
                    "best_val_metrics": val_metrics,
                    "proposal_checkpoint_path": str(proposal_checkpoint_path),
                },
            )
        else:
            epochs_without_improve += 1

        save_completion_checkpoint(
            last_ckpt_path,
            completion_model,
            args,
            extra={
                "epoch": epoch,
                "validation_selection_score": val_score,
                "val_metrics": val_metrics,
                "proposal_checkpoint_path": str(proposal_checkpoint_path),
            },
        )
        save_json(
            summary_path,
            {
                "best_epoch": best_epoch,
                "best_validation_selection_score": best_score,
                "proposal_checkpoint_path": str(proposal_checkpoint_path),
                "args": vars(args),
                "history": history,
            },
        )

        if args.patience > 0 and epochs_without_improve >= args.patience:
            print(f"early stopping after {epoch} epochs")
            break

    print(f"best epoch: {best_epoch}")
    print(f"best validation selection score: {best_score}")
    print(f"saved best checkpoint: {best_ckpt_path}")
    print(f"saved summary: {summary_path}")


if __name__ == "__main__":
    main()
