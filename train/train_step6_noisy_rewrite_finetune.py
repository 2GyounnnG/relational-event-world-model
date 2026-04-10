from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.oracle_local_delta import (
    OracleLocalDeltaRewriteConfig,
    OracleLocalDeltaRewriteModel,
    build_edge_delta_targets,
    build_valid_edge_mask,
    masked_edge_keep_regularization_loss_from_pair_mask,
    oracle_full_prediction_loss,
    oracle_local_delta_rewrite_loss,
)
from train.eval_noisy_structured_observation import build_loader, evaluate
from train.train_oracle_local_delta_proposal_conditioned_fp_keep import (
    build_dataloaders,
    finalize_metric_accumulator,
    get_device,
    init_metric_accumulator,
    load_frozen_proposal,
    move_batch_to_device,
    require_oracle_scope,
    resolve_path,
    save_json,
    set_seed,
    update_metric_accumulator,
)


def make_noisy_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    noisy_batch = dict(batch)
    noisy_batch["node_feats"] = batch.get("obs_node_feats", batch["node_feats"])
    noisy_batch["adj"] = batch.get("obs_adj", batch["adj"])
    return noisy_batch


def compute_noisy_selection_score(results: Dict[str, Dict[str, float]]) -> float:
    overall = results["overall"]
    return (
        0.35 * float(overall["full_edge_acc"])
        + 0.35 * float(overall["context_edge_acc"])
        + 0.15 * float(overall["changed_edge_acc"])
        + 0.15 * float(overall["delete"])
    )


def build_rewrite_model(
    sample: Dict[str, torch.Tensor],
    init_rewrite_checkpoint_path: Optional[Path],
    device: torch.device,
) -> tuple[OracleLocalDeltaRewriteModel, OracleLocalDeltaRewriteConfig, bool]:
    if init_rewrite_checkpoint_path is not None:
        checkpoint = torch.load(init_rewrite_checkpoint_path, map_location="cpu")
        model_config = OracleLocalDeltaRewriteConfig(**checkpoint["model_config"])
        model = OracleLocalDeltaRewriteModel(model_config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        use_proposal_conditioning = bool(
            checkpoint.get("model_config", {}).get("use_proposal_conditioning", False)
        )
        print(f"loaded init rewrite checkpoint: {init_rewrite_checkpoint_path}")
        return model, model_config, use_proposal_conditioning

    node_feat_dim = sample["node_feats"].shape[-1]
    state_dim = node_feat_dim - 1
    model_config = OracleLocalDeltaRewriteConfig(
        node_feat_dim=node_feat_dim,
        num_node_types=3,
        type_dim=1,
        state_dim=state_dim,
        hidden_dim=128,
        msg_pass_layers=3,
        node_mlp_layers=2,
        edge_mlp_layers=2,
        dropout=0.0,
        edge_dropout=0.0,
        copy_logit_value=10.0,
        use_proposal_conditioning=True,
    )
    model = OracleLocalDeltaRewriteModel(model_config).to(device)
    return model, model_config, bool(model_config.use_proposal_conditioning)


def get_noisy_proposal_predictions(
    proposal_model,
    batch: Dict[str, torch.Tensor],
    node_threshold: float,
    edge_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    return proposal_node_probs, proposal_edge_probs, pred_scope_nodes, pred_scope_edges


def run_noisy_epoch(
    model: OracleLocalDeltaRewriteModel,
    proposal_model,
    loader,
    device: torch.device,
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
    use_proposal_conditioning: bool,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_clip: Optional[float] = None,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    proposal_model.eval()

    acc = init_metric_accumulator()
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in loader:
            require_oracle_scope(batch)
            batch = move_batch_to_device(batch, device)
            noisy_batch = make_noisy_batch(batch)

            proposal_node_probs, proposal_edge_probs, pred_scope_nodes, pred_scope_edges = get_noisy_proposal_predictions(
                proposal_model=proposal_model,
                batch=noisy_batch,
                node_threshold=node_threshold,
                edge_threshold=edge_threshold,
            )

            scope_node_mask = batch["event_scope_union_nodes"]
            scope_edge_mask = batch["event_scope_union_edges"]
            outputs = model(
                node_feats=noisy_batch["node_feats"],
                adj=noisy_batch["adj"],
                scope_node_mask=scope_node_mask,
                scope_edge_mask=scope_edge_mask,
                proposal_node_probs=proposal_node_probs if use_proposal_conditioning else None,
                proposal_edge_probs=proposal_edge_probs if use_proposal_conditioning else None,
            )

            local_loss_dict = oracle_local_delta_rewrite_loss(
                outputs=outputs,
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
                outputs["edge_delta_logits_local"],
                predicted_only_edge_mask,
            )
            total_loss = local_loss_dict["total_loss"] + fp_edge_keep_loss_weight * fp_edge_keep_loss

            local_loss_dict = dict(local_loss_dict)
            local_loss_dict["main_total_loss"] = local_loss_dict["total_loss"]
            local_loss_dict["fp_edge_keep_loss"] = fp_edge_keep_loss
            local_loss_dict["total_loss"] = total_loss

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
                total_loss.backward()
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            update_metric_accumulator(
                acc=acc,
                outputs=outputs,
                batch=noisy_batch,
                scope_node_mask=scope_node_mask,
                scope_edge_mask=scope_edge_mask,
                local_loss_dict=local_loss_dict,
                full_loss_dict=full_loss_dict,
            )

    return finalize_metric_accumulator(acc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--proposal_checkpoint", type=str, required=True)
    parser.add_argument("--init_rewrite_checkpoint", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--node_threshold", type=float, default=0.15)
    parser.add_argument("--edge_threshold", type=float, default=0.10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--edge_loss_weight", type=float, default=1.0)
    parser.add_argument("--type_loss_weight", type=float, default=1.0)
    parser.add_argument("--state_loss_weight", type=float, default=1.0)
    parser.add_argument("--type_flip_weight", type=float, default=1.0)
    parser.add_argument("--delta_keep_weight", type=float, default=1.10)
    parser.add_argument("--delta_add_weight", type=float, default=1.0)
    parser.add_argument("--delta_delete_weight", type=float, default=3.0)
    parser.add_argument("--fp_edge_keep_loss_weight", type=float, default=0.12)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    pin_memory = device.type == "cuda"

    proposal_checkpoint_path = resolve_path(args.proposal_checkpoint)
    init_rewrite_checkpoint_path = resolve_path(args.init_rewrite_checkpoint)
    save_dir = PROJECT_ROOT / args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, _, train_loader, val_loader = build_dataloaders(
        train_path=args.train_path,
        val_path=args.val_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    sample = train_dataset[0]
    require_oracle_scope(sample)
    model, model_config, use_proposal_conditioning = build_rewrite_model(
        sample=sample,
        init_rewrite_checkpoint_path=init_rewrite_checkpoint_path,
        device=device,
    )
    proposal_model = load_frozen_proposal(proposal_checkpoint_path, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_ckpt_path = save_dir / "best.pt"
    last_ckpt_path = save_dir / "last.pt"
    best_metrics_path = save_dir / "best_metrics.json"

    best_score = float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0

    print(f"device: {device}")
    print(f"train dataset size: {len(train_dataset)}")
    print(f"val dataset size: {len(val_loader.dataset)}")
    print(f"frozen proposal checkpoint: {proposal_checkpoint_path}")
    print(f"rewrite init checkpoint: {init_rewrite_checkpoint_path}")
    print(f"proposal node threshold: {args.node_threshold}")
    print(f"proposal edge threshold: {args.edge_threshold}")
    print(
        "validation selection metric: "
        "0.35 * full_edge_acc + 0.35 * context_edge_acc + 0.15 * changed_edge_acc + 0.15 * delete"
    )
    print(f"proposal conditioning enabled: {use_proposal_conditioning}")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_noisy_epoch(
            model=model,
            proposal_model=proposal_model,
            loader=train_loader,
            device=device,
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
            use_proposal_conditioning=use_proposal_conditioning,
            optimizer=optimizer,
            grad_clip=args.grad_clip,
        )

        val_train_metrics = run_noisy_epoch(
            model=model,
            proposal_model=proposal_model,
            loader=val_loader,
            device=device,
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
            use_proposal_conditioning=use_proposal_conditioning,
            optimizer=None,
            grad_clip=None,
        )

        noisy_val_results = evaluate(
            proposal_model=proposal_model,
            rewrite_model=model,
            loader=val_loader,
            device=device,
            node_threshold=args.node_threshold,
            edge_threshold=args.edge_threshold,
            use_proposal_conditioning=use_proposal_conditioning,
        )
        selection_score = compute_noisy_selection_score(noisy_val_results)
        noisy_overall = noisy_val_results["overall"]

        print(
            f"[Epoch {epoch:03d}] "
            f"train_local_total={train_metrics['local_total_loss']:.6f} "
            f"train_full_total={train_metrics['full_total_loss']:.6f} "
            f"train_fp_edge_keep_loss={train_metrics['local_fp_edge_keep_loss']:.6f} | "
            f"val_local_total={val_train_metrics['local_total_loss']:.6f} "
            f"val_full_total={val_train_metrics['full_total_loss']:.6f} "
            f"val_fp_edge_keep_loss={val_train_metrics['local_fp_edge_keep_loss']:.6f} | "
            f"noisy_full_edge={noisy_overall['full_edge_acc']:.6f} "
            f"noisy_changed_edge={noisy_overall['changed_edge_acc']:.6f} "
            f"noisy_context_edge={noisy_overall['context_edge_acc']:.6f} "
            f"noisy_delete={noisy_overall['delete']:.6f} "
            f"noisy_add={noisy_overall['add']:.6f} "
            f"selection_score={selection_score:.6f}"
        )

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": vars(model_config),
            "args": vars(args),
            "train_metrics": train_metrics,
            "val_metrics": val_train_metrics,
            "val_noisy_results": noisy_val_results,
            "selection_metric": "0.35*full_edge_acc + 0.35*context_edge_acc + 0.15*changed_edge_acc + 0.15*delete",
            "selection_score": selection_score,
            "use_proposal_conditioning": use_proposal_conditioning,
            "fp_edge_keep_loss_weight": args.fp_edge_keep_loss_weight,
            "proposal_checkpoint": str(proposal_checkpoint_path),
            "init_rewrite_checkpoint": str(init_rewrite_checkpoint_path),
            "proposal_node_threshold": args.node_threshold,
            "proposal_edge_threshold": args.edge_threshold,
            "training_input_mode": "noisy_structured_observation",
            "training_target_mode": "clean_next_state",
        }
        torch.save(ckpt, last_ckpt_path)

        if selection_score > best_score:
            best_score = selection_score
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(ckpt, best_ckpt_path)
            save_json(
                best_metrics_path,
                {
                    "epoch": epoch,
                    "best_selection_score": best_score,
                    "selection_metric": ckpt["selection_metric"],
                    "train_metrics": train_metrics,
                    "val_metrics": val_train_metrics,
                    "val_noisy_results": noisy_val_results,
                    "model_config": vars(model_config),
                    "args": vars(args),
                    "use_proposal_conditioning": use_proposal_conditioning,
                    "fp_edge_keep_loss_weight": args.fp_edge_keep_loss_weight,
                    "proposal_checkpoint": str(proposal_checkpoint_path),
                    "init_rewrite_checkpoint": str(init_rewrite_checkpoint_path),
                    "proposal_node_threshold": args.node_threshold,
                    "proposal_edge_threshold": args.edge_threshold,
                    "training_input_mode": "noisy_structured_observation",
                    "training_target_mode": "clean_next_state",
                },
            )
            print(f"  saved new best checkpoint -> {best_ckpt_path}")
        else:
            epochs_without_improvement += 1
            print(f"  no improvement for {epochs_without_improvement} epoch(s)")

        if epochs_without_improvement >= args.patience:
            print(f"early stopping triggered at epoch {epoch} (best epoch: {best_epoch})")
            break

    print(f"training complete. best epoch={best_epoch}, best selection score={best_score:.6f}")


if __name__ == "__main__":
    main()
