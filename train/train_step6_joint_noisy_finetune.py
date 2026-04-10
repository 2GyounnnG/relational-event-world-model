from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.collate import graph_event_collate_fn
from data.dataset import GraphEventDataset
from models.oracle_local import build_valid_edge_mask as build_valid_edge_mask_proposal
from models.oracle_local_delta import (
    OracleLocalDeltaRewriteConfig,
    OracleLocalDeltaRewriteModel,
    build_valid_edge_mask,
    masked_edge_keep_regularization_loss_from_pair_mask,
    oracle_full_prediction_loss,
    oracle_local_delta_rewrite_loss,
)
from models.proposal import ScopeProposalConfig, ScopeProposalModel, scope_proposal_loss
from train.eval_noisy_structured_observation import evaluate
from train.train_oracle_local_delta_proposal_conditioned_fp_keep import (
    finalize_metric_accumulator,
    get_device,
    init_metric_accumulator,
    move_batch_to_device,
    require_oracle_scope,
    resolve_path,
    save_json,
    set_seed,
    update_metric_accumulator,
)
from train.train_step6_noisy_rewrite_finetune import build_rewrite_model, compute_noisy_selection_score, make_noisy_batch


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


def load_proposal_init(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[ScopeProposalModel, ScopeProposalConfig]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_config = ScopeProposalConfig(**checkpoint["model_config"])
    model = ScopeProposalModel(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"loaded init proposal checkpoint: {checkpoint_path}")
    return model, model_config


def get_joint_proposal_outputs(
    proposal_model: ScopeProposalModel,
    batch: Dict[str, torch.Tensor],
    node_threshold: float,
    edge_threshold: float,
) -> tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    proposal_outputs = proposal_model(
        node_feats=batch["node_feats"],
        adj=batch["adj"],
    )

    node_mask = batch["node_mask"].bool()
    valid_edge_mask = build_valid_edge_mask(batch["node_mask"]).bool()
    node_scope_logits = proposal_outputs.get("node_scope_logits", proposal_outputs["scope_logits"])
    proposal_node_probs = torch.sigmoid(node_scope_logits) * node_mask.float()

    if "edge_scope_logits" in proposal_outputs:
        proposal_edge_probs = torch.sigmoid(proposal_outputs["edge_scope_logits"]) * valid_edge_mask.float()
    else:
        proposal_edge_probs = (
            proposal_node_probs.unsqueeze(2) * proposal_node_probs.unsqueeze(1) * valid_edge_mask.float()
        )

    pred_scope_nodes = ((proposal_node_probs >= node_threshold) & node_mask).float()
    pred_scope_edges = ((proposal_edge_probs >= edge_threshold) & valid_edge_mask).float()
    return proposal_outputs, proposal_node_probs, proposal_edge_probs, pred_scope_nodes, pred_scope_edges


def run_joint_epoch(
    proposal_model: ScopeProposalModel,
    rewrite_model: OracleLocalDeltaRewriteModel,
    loader: DataLoader,
    device: torch.device,
    node_scope_loss_weight: float,
    node_flip_weight: float,
    edge_scope_loss_weight: float,
    edge_scope_pos_weight: float,
    edge_loss_weight: float,
    type_loss_weight: float,
    state_loss_weight: float,
    type_flip_weight: float,
    delta_keep_weight: float,
    delta_add_weight: float,
    delta_delete_weight: float,
    fp_edge_keep_loss_weight: float,
    node_threshold: float,
    edge_threshold: float,
    joint_proposal_loss_weight: float,
    use_proposal_conditioning: bool,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_clip: Optional[float] = None,
) -> Dict[str, float]:
    is_train = optimizer is not None
    proposal_model.train() if is_train else proposal_model.eval()
    rewrite_model.train() if is_train else rewrite_model.eval()

    acc = init_metric_accumulator()
    proposal_metric_sums = {
        "proposal_total_loss": 0.0,
        "proposal_node_scope_loss": 0.0,
        "proposal_edge_scope_loss": 0.0,
        "proposal_node_f1": 0.0,
        "proposal_edge_f1": 0.0,
    }
    num_batches = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in loader:
            require_oracle_scope(batch)
            batch = move_batch_to_device(batch, device)
            noisy_batch = make_noisy_batch(batch)

            proposal_outputs, proposal_node_probs, proposal_edge_probs, pred_scope_nodes, pred_scope_edges = (
                get_joint_proposal_outputs(
                    proposal_model=proposal_model,
                    batch=noisy_batch,
                    node_threshold=node_threshold,
                    edge_threshold=edge_threshold,
                )
            )

            current_type = batch["node_feats"][:, :, 0].long()
            target_type = batch["next_node_feats"][:, :, 0].long()
            flip_target_mask = (current_type != target_type).to(batch["node_mask"].dtype)
            node_scope_weights = 1.0 + (node_flip_weight - 1.0) * flip_target_mask
            valid_edge_mask_for_prop = build_valid_edge_mask_proposal(batch["node_mask"])
            proposal_loss_dict = scope_proposal_loss(
                outputs=proposal_outputs,
                target_node_scope=batch["event_scope_union_nodes"],
                target_edge_scope=batch["event_scope_union_edges"],
                node_mask=batch["node_mask"],
                pair_mask=valid_edge_mask_for_prop,
                node_scope_loss_weight=node_scope_loss_weight,
                edge_scope_loss_weight=edge_scope_loss_weight,
                edge_scope_pos_weight=edge_scope_pos_weight,
                node_scope_weights=node_scope_weights,
            )

            scope_node_mask = batch["event_scope_union_nodes"]
            scope_edge_mask = batch["event_scope_union_edges"]
            rewrite_outputs = rewrite_model(
                node_feats=noisy_batch["node_feats"],
                adj=noisy_batch["adj"],
                scope_node_mask=scope_node_mask,
                scope_edge_mask=scope_edge_mask,
                proposal_node_probs=proposal_node_probs if use_proposal_conditioning else None,
                proposal_edge_probs=proposal_edge_probs if use_proposal_conditioning else None,
            )

            rewrite_loss_dict = oracle_local_delta_rewrite_loss(
                outputs=rewrite_outputs,
                current_node_feats=noisy_batch["node_feats"],
                current_adj=noisy_batch["adj"],
                target_node_feats=batch["next_node_feats"],
                target_adj=batch["next_adj"],
                node_mask=batch["node_mask"],
                scope_node_mask=scope_node_mask,
                scope_edge_mask=scope_edge_mask,
                edge_loss_weight=edge_loss_weight,
                type_loss_weight=type_loss_weight,
                state_loss_weight=state_loss_weight,
                type_flip_weight=type_flip_weight,
                delta_keep_weight=delta_keep_weight,
                delta_add_weight=delta_add_weight,
                delta_delete_weight=delta_delete_weight,
            )

            valid_edge_mask = build_valid_edge_mask(batch["node_mask"])
            oracle_edge_scope_mask = batch["event_scope_union_edges"] * valid_edge_mask
            predicted_only_edge_mask = pred_scope_edges * valid_edge_mask * (1.0 - oracle_edge_scope_mask)
            fp_edge_keep_loss = masked_edge_keep_regularization_loss_from_pair_mask(
                rewrite_outputs["edge_delta_logits_local"],
                predicted_only_edge_mask,
            )
            rewrite_total_loss = rewrite_loss_dict["total_loss"] + fp_edge_keep_loss_weight * fp_edge_keep_loss
            total_loss = rewrite_total_loss + joint_proposal_loss_weight * proposal_loss_dict["total_loss"]

            rewrite_loss_dict = dict(rewrite_loss_dict)
            rewrite_loss_dict["main_total_loss"] = rewrite_loss_dict["total_loss"]
            rewrite_loss_dict["fp_edge_keep_loss"] = fp_edge_keep_loss
            rewrite_loss_dict["total_loss"] = rewrite_total_loss

            full_loss_dict = oracle_full_prediction_loss(
                outputs=rewrite_outputs,
                target_node_feats=batch["next_node_feats"],
                target_adj=batch["next_adj"],
                node_mask=batch["node_mask"],
                edge_loss_weight=edge_loss_weight,
                type_loss_weight=type_loss_weight,
                state_loss_weight=state_loss_weight,
            )

            if is_train:
                optimizer.zero_grad()
                total_loss.backward()
                if grad_clip is not None and grad_clip > 0:
                    params = list(proposal_model.parameters()) + list(rewrite_model.parameters())
                    torch.nn.utils.clip_grad_norm_(params, grad_clip)
                optimizer.step()

            update_metric_accumulator(
                acc=acc,
                outputs=rewrite_outputs,
                batch=noisy_batch,
                scope_node_mask=scope_node_mask,
                scope_edge_mask=scope_edge_mask,
                local_loss_dict=rewrite_loss_dict,
                full_loss_dict=full_loss_dict,
            )

            pred_node_mask = pred_scope_nodes * batch["node_mask"]
            oracle_node_mask = batch["event_scope_union_nodes"] * batch["node_mask"]
            pred_edge_mask = pred_scope_edges * valid_edge_mask_for_prop
            oracle_edge_mask = batch["event_scope_union_edges"] * valid_edge_mask_for_prop

            node_tp = (pred_node_mask * oracle_node_mask).sum().item()
            node_pred_pos = pred_node_mask.sum().item()
            node_true_pos = oracle_node_mask.sum().item()
            edge_tp = (pred_edge_mask * oracle_edge_mask).sum().item()
            edge_pred_pos = pred_edge_mask.sum().item()
            edge_true_pos = oracle_edge_mask.sum().item()
            node_precision = node_tp / node_pred_pos if node_pred_pos > 0 else 0.0
            node_recall = node_tp / node_true_pos if node_true_pos > 0 else 0.0
            edge_precision = edge_tp / edge_pred_pos if edge_pred_pos > 0 else 0.0
            edge_recall = edge_tp / edge_true_pos if edge_true_pos > 0 else 0.0
            node_f1 = (
                2.0 * node_precision * node_recall / (node_precision + node_recall)
                if (node_precision + node_recall) > 0
                else 0.0
            )
            edge_f1 = (
                2.0 * edge_precision * edge_recall / (edge_precision + edge_recall)
                if (edge_precision + edge_recall) > 0
                else 0.0
            )

            proposal_metric_sums["proposal_total_loss"] += proposal_loss_dict["total_loss"].item()
            proposal_metric_sums["proposal_node_scope_loss"] += proposal_loss_dict["node_scope_loss"].item()
            proposal_metric_sums["proposal_edge_scope_loss"] += proposal_loss_dict["edge_scope_loss"].item()
            proposal_metric_sums["proposal_node_f1"] += node_f1
            proposal_metric_sums["proposal_edge_f1"] += edge_f1
            num_batches += 1

    metrics = finalize_metric_accumulator(acc)
    denom = max(num_batches, 1)
    metrics["proposal_total_loss"] = proposal_metric_sums["proposal_total_loss"] / denom
    metrics["proposal_node_scope_loss"] = proposal_metric_sums["proposal_node_scope_loss"] / denom
    metrics["proposal_edge_scope_loss"] = proposal_metric_sums["proposal_edge_scope_loss"] / denom
    metrics["proposal_node_f1"] = proposal_metric_sums["proposal_node_f1"] / denom
    metrics["proposal_edge_f1"] = proposal_metric_sums["proposal_edge_f1"] / denom
    return metrics


def save_checkpoint(
    path: Path,
    epoch: int,
    model,
    model_config,
    optimizer_state_dict: Dict,
    args: Dict,
    train_metrics: Dict,
    val_metrics: Dict,
    val_noisy_results: Dict,
    selection_score: float,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer_state_dict,
            "model_config": vars(model_config),
            "args": args,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "val_noisy_results": val_noisy_results,
            "selection_score": selection_score,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--init_proposal_checkpoint", type=str, required=True)
    parser.add_argument("--init_rewrite_checkpoint", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--node_threshold", type=float, default=0.15)
    parser.add_argument("--edge_threshold", type=float, default=0.10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--edge_loss_weight", type=float, default=1.0)
    parser.add_argument("--type_loss_weight", type=float, default=1.0)
    parser.add_argument("--state_loss_weight", type=float, default=1.0)
    parser.add_argument("--type_flip_weight", type=float, default=1.0)
    parser.add_argument("--delta_keep_weight", type=float, default=1.10)
    parser.add_argument("--delta_add_weight", type=float, default=1.0)
    parser.add_argument("--delta_delete_weight", type=float, default=3.0)
    parser.add_argument("--fp_edge_keep_loss_weight", type=float, default=0.12)
    parser.add_argument("--node_scope_loss_weight", type=float, default=1.0)
    parser.add_argument("--node_flip_weight", type=float, default=2.0)
    parser.add_argument("--edge_scope_loss_weight", type=float, default=1.0)
    parser.add_argument("--edge_scope_pos_weight", type=float, default=2.0)
    parser.add_argument("--joint_proposal_loss_weight", type=float, default=1.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    pin_memory = device.type == "cuda"

    init_proposal_checkpoint_path = resolve_path(args.init_proposal_checkpoint)
    init_rewrite_checkpoint_path = resolve_path(args.init_rewrite_checkpoint)
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
    require_oracle_scope(sample)
    proposal_model, proposal_config = load_proposal_init(init_proposal_checkpoint_path, device)
    rewrite_model, rewrite_config, use_proposal_conditioning = build_rewrite_model(
        sample=sample,
        init_rewrite_checkpoint_path=init_rewrite_checkpoint_path,
        device=device,
    )
    optimizer = torch.optim.Adam(
        list(proposal_model.parameters()) + list(rewrite_model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    proposal_best_path = save_dir / "proposal_best.pt"
    proposal_last_path = save_dir / "proposal_last.pt"
    rewrite_best_path = save_dir / "rewrite_best.pt"
    rewrite_last_path = save_dir / "rewrite_last.pt"
    best_metrics_path = save_dir / "best_metrics.json"

    best_score = float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0

    print(f"device: {device}")
    print(f"train dataset size: {len(train_dataset)}")
    print(f"val dataset size: {len(val_dataset)}")
    print(f"proposal init checkpoint: {init_proposal_checkpoint_path}")
    print(f"rewrite init checkpoint: {init_rewrite_checkpoint_path}")
    print(f"proposal node threshold: {args.node_threshold}")
    print(f"proposal edge threshold: {args.edge_threshold}")
    print(f"proposal conditioning enabled: {use_proposal_conditioning}")
    print(f"joint proposal loss weight: {args.joint_proposal_loss_weight}")
    print(
        "validation selection metric: "
        "0.35 * full_edge_acc + 0.35 * context_edge_acc + 0.15 * changed_edge_acc + 0.15 * delete"
    )

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_joint_epoch(
            proposal_model=proposal_model,
            rewrite_model=rewrite_model,
            loader=train_loader,
            device=device,
            node_scope_loss_weight=args.node_scope_loss_weight,
            node_flip_weight=args.node_flip_weight,
            edge_scope_loss_weight=args.edge_scope_loss_weight,
            edge_scope_pos_weight=args.edge_scope_pos_weight,
            edge_loss_weight=args.edge_loss_weight,
            type_loss_weight=args.type_loss_weight,
            state_loss_weight=args.state_loss_weight,
            type_flip_weight=args.type_flip_weight,
            delta_keep_weight=args.delta_keep_weight,
            delta_add_weight=args.delta_add_weight,
            delta_delete_weight=args.delta_delete_weight,
            fp_edge_keep_loss_weight=args.fp_edge_keep_loss_weight,
            node_threshold=args.node_threshold,
            edge_threshold=args.edge_threshold,
            joint_proposal_loss_weight=args.joint_proposal_loss_weight,
            use_proposal_conditioning=use_proposal_conditioning,
            optimizer=optimizer,
            grad_clip=args.grad_clip,
        )

        val_metrics = run_joint_epoch(
            proposal_model=proposal_model,
            rewrite_model=rewrite_model,
            loader=val_loader,
            device=device,
            node_scope_loss_weight=args.node_scope_loss_weight,
            node_flip_weight=args.node_flip_weight,
            edge_scope_loss_weight=args.edge_scope_loss_weight,
            edge_scope_pos_weight=args.edge_scope_pos_weight,
            edge_loss_weight=args.edge_loss_weight,
            type_loss_weight=args.type_loss_weight,
            state_loss_weight=args.state_loss_weight,
            type_flip_weight=args.type_flip_weight,
            delta_keep_weight=args.delta_keep_weight,
            delta_add_weight=args.delta_add_weight,
            delta_delete_weight=args.delta_delete_weight,
            fp_edge_keep_loss_weight=args.fp_edge_keep_loss_weight,
            node_threshold=args.node_threshold,
            edge_threshold=args.edge_threshold,
            joint_proposal_loss_weight=args.joint_proposal_loss_weight,
            use_proposal_conditioning=use_proposal_conditioning,
            optimizer=None,
            grad_clip=None,
        )

        val_noisy_results = evaluate(
            proposal_model=proposal_model,
            rewrite_model=rewrite_model,
            loader=val_loader,
            device=device,
            node_threshold=args.node_threshold,
            edge_threshold=args.edge_threshold,
            use_proposal_conditioning=use_proposal_conditioning,
        )
        selection_score = compute_noisy_selection_score(val_noisy_results)
        noisy_overall = val_noisy_results["overall"]

        print(
            f"[Epoch {epoch:03d}] "
            f"train_rewrite_local_total={train_metrics['local_total_loss']:.6f} "
            f"train_proposal_total={train_metrics['proposal_total_loss']:.6f} "
            f"train_proposal_node_f1={train_metrics['proposal_node_f1']:.6f} "
            f"train_proposal_edge_f1={train_metrics['proposal_edge_f1']:.6f} | "
            f"val_rewrite_local_total={val_metrics['local_total_loss']:.6f} "
            f"val_proposal_total={val_metrics['proposal_total_loss']:.6f} "
            f"val_proposal_node_f1={val_metrics['proposal_node_f1']:.6f} "
            f"val_proposal_edge_f1={val_metrics['proposal_edge_f1']:.6f} | "
            f"noisy_full_edge={noisy_overall['full_edge_acc']:.6f} "
            f"noisy_changed_edge={noisy_overall['changed_edge_acc']:.6f} "
            f"noisy_context_edge={noisy_overall['context_edge_acc']:.6f} "
            f"noisy_delete={noisy_overall['delete']:.6f} "
            f"noisy_add={noisy_overall['add']:.6f} "
            f"selection_score={selection_score:.6f}"
        )

        save_checkpoint(
            path=proposal_last_path,
            epoch=epoch,
            model=proposal_model,
            model_config=proposal_config,
            optimizer_state_dict=optimizer.state_dict(),
            args=vars(args),
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            val_noisy_results=val_noisy_results,
            selection_score=selection_score,
        )
        save_checkpoint(
            path=rewrite_last_path,
            epoch=epoch,
            model=rewrite_model,
            model_config=rewrite_config,
            optimizer_state_dict=optimizer.state_dict(),
            args=vars(args),
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            val_noisy_results=val_noisy_results,
            selection_score=selection_score,
        )

        if selection_score > best_score:
            best_score = selection_score
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                path=proposal_best_path,
                epoch=epoch,
                model=proposal_model,
                model_config=proposal_config,
                optimizer_state_dict=optimizer.state_dict(),
                args=vars(args),
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                val_noisy_results=val_noisy_results,
                selection_score=selection_score,
            )
            save_checkpoint(
                path=rewrite_best_path,
                epoch=epoch,
                model=rewrite_model,
                model_config=rewrite_config,
                optimizer_state_dict=optimizer.state_dict(),
                args=vars(args),
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                val_noisy_results=val_noisy_results,
                selection_score=selection_score,
            )
            save_json(
                best_metrics_path,
                {
                    "epoch": epoch,
                    "best_selection_score": best_score,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "val_noisy_results": val_noisy_results,
                    "args": vars(args),
                    "proposal_init_checkpoint": str(init_proposal_checkpoint_path),
                    "rewrite_init_checkpoint": str(init_rewrite_checkpoint_path),
                    "proposal_best_checkpoint": str(proposal_best_path),
                    "rewrite_best_checkpoint": str(rewrite_best_path),
                    "proposal_conditioning_enabled": use_proposal_conditioning,
                    "selection_metric": (
                        "0.35*full_edge_acc + 0.35*context_edge_acc + "
                        "0.15*changed_edge_acc + 0.15*delete"
                    ),
                },
            )
            print(f"  saved new best checkpoints -> {proposal_best_path} and {rewrite_best_path}")
        else:
            epochs_without_improvement += 1
            print(f"  no improvement for {epochs_without_improvement} epoch(s)")

        if epochs_without_improvement >= args.patience:
            print(f"early stopping triggered at epoch {epoch} (best epoch: {best_epoch})")
            break

    print(f"training complete. best epoch={best_epoch}, best selection score={best_score:.6f}")


if __name__ == "__main__":
    main()
