from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.oracle_local import build_valid_edge_mask as build_valid_edge_mask_proposal
from models.oracle_local_delta import (
    OracleLocalDeltaRewriteModel,
    build_valid_edge_mask,
    masked_edge_keep_regularization_loss_from_pair_mask,
    oracle_full_prediction_loss,
    oracle_local_delta_rewrite_loss,
)
from models.proposal import ScopeProposalModel, scope_proposal_loss
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
from train.train_step6_joint_noisy_finetune import (
    build_dataloaders,
    get_joint_proposal_outputs,
    load_proposal_init,
    save_checkpoint,
)
from train.train_step6_noisy_rewrite_finetune import build_rewrite_model, make_noisy_batch
from train.train_step26_noisy_interaction_joint_deeper import (
    compute_coverage_emphasized_selection_score,
    safe_float,
)


def summarize_raw_samples(dataset) -> Dict[str, Any]:
    bucket_counts = Counter(str(sample.get("step5_dependency_bucket", "unknown")) for sample in dataset.samples)
    corruption_counts = Counter(str(sample.get("step6a_corruption_setting", "unknown")) for sample in dataset.samples)
    event_counts = Counter(
        str(sample.get("events", [{}])[0].get("event_type", "unknown"))
        for sample in dataset.samples
    )
    return {
        "sample_count": len(dataset),
        "dependency_bucket_counts": dict(sorted(bucket_counts.items())),
        "corruption_setting_counts": dict(sorted(corruption_counts.items())),
        "event_type_counts": dict(sorted(event_counts.items())),
    }


def write_training_summary(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def clone_rewrite_anchor_state(rewrite_model: OracleLocalDeltaRewriteModel) -> Dict[str, torch.Tensor]:
    return {
        name: param.detach().clone()
        for name, param in rewrite_model.named_parameters()
        if param.requires_grad
    }


def rewrite_parameter_anchor_loss(
    rewrite_model: OracleLocalDeltaRewriteModel,
    anchor_state: Dict[str, torch.Tensor],
) -> torch.Tensor:
    losses = []
    for name, param in rewrite_model.named_parameters():
        if not param.requires_grad:
            continue
        anchor = anchor_state[name].to(device=param.device, dtype=param.dtype)
        losses.append(torch.mean((param - anchor) ** 2))
    if not losses:
        return torch.zeros((), device=next(rewrite_model.parameters()).device)
    return torch.stack(losses).mean()


def run_joint_epoch_with_rewrite_anchor(
    proposal_model: ScopeProposalModel,
    rewrite_model: OracleLocalDeltaRewriteModel,
    loader,
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
    rewrite_anchor_weight: float,
    rewrite_anchor_state: Dict[str, torch.Tensor],
    use_proposal_conditioning: bool,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_clip: Optional[float] = None,
) -> Dict[str, float]:
    is_train = optimizer is not None
    proposal_model.train() if is_train else proposal_model.eval()
    rewrite_model.train() if is_train else rewrite_model.eval()

    acc = init_metric_accumulator()
    metric_sums = {
        "proposal_total_loss": 0.0,
        "proposal_node_scope_loss": 0.0,
        "proposal_edge_scope_loss": 0.0,
        "proposal_node_f1": 0.0,
        "proposal_edge_f1": 0.0,
        "rewrite_anchor_loss": 0.0,
        "rewrite_anchor_weighted_loss": 0.0,
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
            anchor_loss = rewrite_parameter_anchor_loss(rewrite_model, rewrite_anchor_state)
            rewrite_total_loss = (
                rewrite_loss_dict["total_loss"]
                + fp_edge_keep_loss_weight * fp_edge_keep_loss
                + rewrite_anchor_weight * anchor_loss
            )
            total_loss = rewrite_total_loss + joint_proposal_loss_weight * proposal_loss_dict["total_loss"]

            rewrite_loss_dict = dict(rewrite_loss_dict)
            rewrite_loss_dict["main_total_loss"] = rewrite_loss_dict["total_loss"]
            rewrite_loss_dict["fp_edge_keep_loss"] = fp_edge_keep_loss
            rewrite_loss_dict["rewrite_anchor_loss"] = anchor_loss
            rewrite_loss_dict["rewrite_anchor_weighted_loss"] = rewrite_anchor_weight * anchor_loss
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

            metric_sums["proposal_total_loss"] += proposal_loss_dict["total_loss"].item()
            metric_sums["proposal_node_scope_loss"] += proposal_loss_dict["node_scope_loss"].item()
            metric_sums["proposal_edge_scope_loss"] += proposal_loss_dict["edge_scope_loss"].item()
            metric_sums["proposal_node_f1"] += node_f1
            metric_sums["proposal_edge_f1"] += edge_f1
            metric_sums["rewrite_anchor_loss"] += anchor_loss.item()
            metric_sums["rewrite_anchor_weighted_loss"] += (rewrite_anchor_weight * anchor_loss).item()
            num_batches += 1

    metrics = finalize_metric_accumulator(acc)
    denom = max(num_batches, 1)
    for key, value in metric_sums.items():
        metrics[key] = value / denom
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="data/graph_event_step23_noisy_step5_train_transitions.pkl")
    parser.add_argument("--val_path", type=str, default="data/graph_event_step23_noisy_step5_val_transitions.pkl")
    parser.add_argument("--init_proposal_checkpoint", type=str, default="checkpoints/proposal_noisy_obs_p2/best.pt")
    parser.add_argument("--init_rewrite_checkpoint", type=str, default="checkpoints/step6_noisy_rewrite_rft1/best.pt")
    parser.add_argument("--save_dir", type=str, default="checkpoints/step28_rft1_anchored_joint")
    parser.add_argument("--node_threshold", type=float, default=0.15)
    parser.add_argument("--edge_threshold", type=float, default=0.10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
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
    parser.add_argument("--edge_scope_pos_weight", type=float, default=4.0)
    parser.add_argument("--joint_proposal_loss_weight", type=float, default=2.0)
    parser.add_argument("--rewrite_anchor_weight", type=float, default=25.0)
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
        train_path=str(resolve_path(args.train_path)),
        val_path=str(resolve_path(args.val_path)),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    sample = train_dataset[0]
    require_oracle_scope(sample)
    if "obs_node_feats" not in sample or "obs_adj" not in sample:
        raise KeyError("Step 28 expects noisy Step 22/23 transition samples with obs_node_feats/obs_adj.")

    proposal_model, proposal_config = load_proposal_init(init_proposal_checkpoint_path, device)
    rewrite_model, rewrite_config, use_proposal_conditioning = build_rewrite_model(
        sample=sample,
        init_rewrite_checkpoint_path=init_rewrite_checkpoint_path,
        device=device,
    )
    rewrite_anchor_state = clone_rewrite_anchor_state(rewrite_model)
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
    training_summary_path = save_dir / "training_summary.json"

    best_score = float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    selection_metric = (
        "0.25*full_edge + 0.25*context_edge + 0.20*changed_edge + "
        "0.10*add + 0.10*delete + 0.10*proposal_edge_recall"
    )
    recipe = {
        "rewrite_constraint_regime": "rft1_anchored_joint",
        "proposal_direction": "Step26-style coverage emphasized joint",
        "coverage_emphasis": {
            "joint_proposal_loss_weight": args.joint_proposal_loss_weight,
            "edge_scope_pos_weight": args.edge_scope_pos_weight,
            "selection_metric": selection_metric,
        },
        "rewrite_anchor": {
            "method": "parameter_l2_to_rft1_initialization",
            "rewrite_anchor_weight": args.rewrite_anchor_weight,
            "init_rewrite_checkpoint": str(init_rewrite_checkpoint_path),
        },
        "note": (
            "This keeps one recipe-level variable: Step26 coverage pressure plus an RFT1 "
            "rewrite parameter anchor. No threshold, architecture, or calibration change."
        ),
    }

    print(f"device: {device}")
    print(f"train dataset size: {len(train_dataset)}")
    print(f"val dataset size: {len(val_dataset)}")
    print(f"train summary: {summarize_raw_samples(train_dataset)}")
    print(f"val summary: {summarize_raw_samples(val_dataset)}")
    print(f"proposal init checkpoint: {init_proposal_checkpoint_path}")
    print(f"rewrite init checkpoint: {init_rewrite_checkpoint_path}")
    print(f"proposal node threshold: {args.node_threshold}")
    print(f"proposal edge threshold: {args.edge_threshold}")
    print(f"proposal conditioning enabled: {use_proposal_conditioning}")
    print(f"joint recipe: {recipe}")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_joint_epoch_with_rewrite_anchor(
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
            rewrite_anchor_weight=args.rewrite_anchor_weight,
            rewrite_anchor_state=rewrite_anchor_state,
            use_proposal_conditioning=use_proposal_conditioning,
            optimizer=optimizer,
            grad_clip=args.grad_clip,
        )
        val_metrics = run_joint_epoch_with_rewrite_anchor(
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
            rewrite_anchor_weight=args.rewrite_anchor_weight,
            rewrite_anchor_state=rewrite_anchor_state,
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
        selection_score = compute_coverage_emphasized_selection_score(val_noisy_results)
        noisy_overall = val_noisy_results["overall"]
        history.append(
            {
                "epoch": epoch,
                "selection_score": selection_score,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "val_noisy_overall": noisy_overall,
            }
        )

        print(
            f"[Epoch {epoch:03d}] "
            f"train_rewrite_local_total={train_metrics['local_total_loss']:.6f} "
            f"train_anchor={train_metrics['rewrite_anchor_weighted_loss']:.6f} "
            f"train_proposal_total={train_metrics['proposal_total_loss']:.6f} "
            f"val_rewrite_local_total={val_metrics['local_total_loss']:.6f} "
            f"val_anchor={val_metrics['rewrite_anchor_weighted_loss']:.6f} "
            f"val_proposal_total={val_metrics['proposal_total_loss']:.6f} "
            f"val_prop_edge_f1={val_metrics['proposal_edge_f1']:.6f} | "
            f"noisy_full_edge={safe_float(noisy_overall, 'full_edge_acc'):.6f} "
            f"noisy_changed_edge={safe_float(noisy_overall, 'changed_edge_acc'):.6f} "
            f"noisy_context_edge={safe_float(noisy_overall, 'context_edge_acc'):.6f} "
            f"noisy_delete={safe_float(noisy_overall, 'delete'):.6f} "
            f"noisy_add={safe_float(noisy_overall, 'add'):.6f} "
            f"noisy_prop_edge_recall={safe_float(noisy_overall, 'proposal_edge_recall'):.6f} "
            f"selection_score={selection_score:.6f}"
        )

        ckpt_args = vars(args)
        for path, model, model_config in (
            (proposal_last_path, proposal_model, proposal_config),
            (rewrite_last_path, rewrite_model, rewrite_config),
        ):
            save_checkpoint(
                path=path,
                epoch=epoch,
                model=model,
                model_config=model_config,
                optimizer_state_dict=optimizer.state_dict(),
                args=ckpt_args,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                val_noisy_results=val_noisy_results,
                selection_score=selection_score,
            )

        if selection_score > best_score:
            best_score = selection_score
            best_epoch = epoch
            epochs_without_improvement = 0
            for path, model, model_config in (
                (proposal_best_path, proposal_model, proposal_config),
                (rewrite_best_path, rewrite_model, rewrite_config),
            ):
                save_checkpoint(
                    path=path,
                    epoch=epoch,
                    model=model,
                    model_config=model_config,
                    optimizer_state_dict=optimizer.state_dict(),
                    args=ckpt_args,
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
                    "selection_metric": selection_metric,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "val_noisy_results": val_noisy_results,
                    "args": ckpt_args,
                    "recipe": recipe,
                    "proposal_init_checkpoint": str(init_proposal_checkpoint_path),
                    "rewrite_init_checkpoint": str(init_rewrite_checkpoint_path),
                    "proposal_best_checkpoint": str(proposal_best_path),
                    "rewrite_best_checkpoint": str(rewrite_best_path),
                    "proposal_conditioning_enabled": use_proposal_conditioning,
                    "train_dataset_summary": summarize_raw_samples(train_dataset),
                    "val_dataset_summary": summarize_raw_samples(val_dataset),
                },
            )
            print(f"  saved new best checkpoints -> {proposal_best_path} and {rewrite_best_path}")
        else:
            epochs_without_improvement += 1
            print(f"  no improvement for {epochs_without_improvement} epoch(s)")

        if epochs_without_improvement >= args.patience:
            print(f"early stopping triggered at epoch {epoch} (best epoch: {best_epoch})")
            break

    write_training_summary(
        training_summary_path,
        {
            "best_epoch": best_epoch,
            "best_selection_score": best_score,
            "selection_metric": selection_metric,
            "history": history,
            "args": vars(args),
            "recipe": recipe,
            "proposal_init_checkpoint": str(init_proposal_checkpoint_path),
            "rewrite_init_checkpoint": str(init_rewrite_checkpoint_path),
            "proposal_best_checkpoint": str(proposal_best_path),
            "rewrite_best_checkpoint": str(rewrite_best_path),
            "proposal_conditioning_enabled": use_proposal_conditioning,
            "train_dataset_summary": summarize_raw_samples(train_dataset),
            "val_dataset_summary": summarize_raw_samples(val_dataset),
        },
    )
    print(f"training complete. best epoch={best_epoch}, best selection score={best_score:.6f}")
    print(f"proposal best checkpoint: {proposal_best_path}")
    print(f"rewrite best checkpoint: {rewrite_best_path}")
    print(f"best metrics json: {best_metrics_path}")
    print(f"training summary json: {training_summary_path}")


if __name__ == "__main__":
    main()
