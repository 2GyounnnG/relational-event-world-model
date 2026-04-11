from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train.eval_noisy_structured_observation import move_batch_to_device, require_keys, resolve_path
from train.eval_step12_guard_ranking_interaction import FIXED_RESCUE_BUDGET_FRACTION, build_ranked_outputs
from train.eval_step9_gated_edge_completion import (
    EDGE_COMPLETION_OFF,
    apply_edge_completion_mode,
    get_base_proposal_outputs,
    load_completion_model,
    load_proposal_model,
    load_rewrite_model,
)
from train.eval_step9_rescue_frontier import average_precision, auroc
from train.train_step17_rescue_fallback_gate import (
    RescueFallbackGate,
    RescueFallbackGateConfig,
    build_dataloaders,
    build_gate_feature_tensor,
    build_gate_targets,
    get_device,
    infer_feature_dim,
    save_gate_checkpoint,
    save_json,
    set_seed,
)


CHOOSER_OBJECTIVE = "pairwise_keep_ranking"


def pairwise_keep_ranking_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> Optional[torch.Tensor]:
    """Pairwise logistic loss over rescued edges.

    Positives are rescued edges where the Step9c path is strictly better than
    base. Negatives are all other rescued edges. We use all positive/negative
    pairs in the batch and no tuned margin.

    中文说明：这是排序目标，不是 0.5 阈值分类目标；目标是把应该保留
    Step9c 的 rescued edges 排到其它 rescued edges 前面。
    """
    rescued_logits = logits[mask]
    rescued_labels = labels[mask].bool()
    pos_logits = rescued_logits[rescued_labels]
    neg_logits = rescued_logits[~rescued_labels]
    if pos_logits.numel() <= 0 or neg_logits.numel() <= 0:
        return None
    diffs = pos_logits.unsqueeze(1) - neg_logits.unsqueeze(0)
    return F.softplus(-diffs).mean()


def init_sums() -> Dict[str, float]:
    return {
        "loss_sum": 0.0,
        "loss_batches": 0.0,
        "rescued_total": 0.0,
        "target_choose_step_total": 0.0,
        "pred_choose_step_total": 0.0,
        "chosen_correct_total": 0.0,
        "base_correct_total": 0.0,
        "step_correct_total": 0.0,
        "pairwise_batch_total": 0.0,
        "skipped_no_pair_batch_total": 0.0,
    }


def finalize_metrics(
    sums: Dict[str, float],
    scores: list[torch.Tensor],
    labels: list[torch.Tensor],
) -> Dict[str, Any]:
    ap = None
    auc = None
    if scores:
        score_tensor = torch.cat(scores).float()
        label_tensor = torch.cat(labels).bool()
        ap = average_precision(score_tensor, label_tensor)
        auc = auroc(score_tensor, label_tensor)
    selection_score = ap if ap is not None else -sums["loss_sum"] / max(sums["loss_batches"], 1.0)
    return {
        "loss": sums["loss_sum"] / max(sums["loss_batches"], 1.0),
        "rescued_total": int(sums["rescued_total"]),
        "target_choose_step_fraction": sums["target_choose_step_total"] / max(sums["rescued_total"], 1.0),
        "pred_choose_step_fraction": sums["pred_choose_step_total"] / max(sums["rescued_total"], 1.0),
        "chosen_correct_rate": sums["chosen_correct_total"] / max(sums["rescued_total"], 1.0),
        "base_path_correct_rate_on_rescued": sums["base_correct_total"] / max(sums["rescued_total"], 1.0),
        "step9c_path_correct_rate_on_rescued": sums["step_correct_total"] / max(sums["rescued_total"], 1.0),
        "choose_step_ap": ap,
        "choose_step_auroc": auc,
        "pairwise_batch_total": int(sums["pairwise_batch_total"]),
        "skipped_no_pair_batch_total": int(sums["skipped_no_pair_batch_total"]),
        "selection_score": selection_score,
    }


def run_epoch(
    proposal_model: torch.nn.Module,
    completion_model: torch.nn.Module,
    rewrite_model: torch.nn.Module,
    gate_model: RescueFallbackGate,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    node_threshold: float,
    edge_threshold: float,
    use_proposal_conditioning: bool,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_clip: Optional[float] = None,
) -> Dict[str, Any]:
    is_train = optimizer is not None
    proposal_model.eval()
    completion_model.eval()
    rewrite_model.eval()
    gate_model.train() if is_train else gate_model.eval()
    sums = init_sums()
    all_scores: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for batch in loader:
        require_keys(batch, ["node_feats", "adj", "next_adj", "node_mask", "changed_edges", "event_scope_union_edges"])
        batch = move_batch_to_device(batch, device)
        with torch.no_grad():
            base_outputs = get_base_proposal_outputs(
                proposal_model=proposal_model,
                batch=batch,
                node_threshold=node_threshold,
                edge_threshold=edge_threshold,
            )
            valid_edge_mask = base_outputs["valid_edge_mask"].bool()
            completion_logits = completion_model(
                node_latents=base_outputs["node_latents"],
                base_edge_logits=base_outputs["edge_scope_logits"],
            )
            completion_probs = torch.sigmoid(completion_logits) * valid_edge_mask.float()
            mode_base = apply_edge_completion_mode(
                base_outputs=base_outputs,
                edge_completion_mode=EDGE_COMPLETION_OFF,
                completion_model=None,
                completion_threshold=0.5,
            )
            mode_step9c = build_ranked_outputs(
                base_outputs=base_outputs,
                mode_name="step9c_completion_only",
                ranking_scores=completion_probs,
                completion_probs=completion_probs,
            )
            rewrite_base = rewrite_model(
                node_feats=mode_base["input_node_feats"],
                adj=mode_base["input_adj"],
                scope_node_mask=mode_base["pred_scope_nodes"].float(),
                scope_edge_mask=mode_base["final_pred_scope_edges"].float(),
                proposal_node_probs=mode_base["proposal_node_probs"] if use_proposal_conditioning else None,
                proposal_edge_probs=mode_base["final_proposal_edge_probs"] if use_proposal_conditioning else None,
            )
            rewrite_step9c = rewrite_model(
                node_feats=mode_step9c["input_node_feats"],
                adj=mode_step9c["input_adj"],
                scope_node_mask=mode_step9c["pred_scope_nodes"].float(),
                scope_edge_mask=mode_step9c["final_pred_scope_edges"].float(),
                proposal_node_probs=mode_step9c["proposal_node_probs"] if use_proposal_conditioning else None,
                proposal_edge_probs=mode_step9c["final_proposal_edge_probs"] if use_proposal_conditioning else None,
            )
            features = build_gate_feature_tensor(
                base_outputs=base_outputs,
                completion_logits=completion_logits,
                rewrite_base=rewrite_base,
                rewrite_step9c=rewrite_step9c,
            )
            target, rescued, base_correct, step_correct = build_gate_targets(
                batch=batch,
                valid_edge_mask=valid_edge_mask,
                rescued_edges=mode_step9c["rescued_edges"],
                rewrite_base=rewrite_base,
                rewrite_step9c=rewrite_step9c,
            )

        if rescued.float().sum().item() <= 0:
            continue
        gate_logits = gate_model(features)
        loss = pairwise_keep_ranking_loss(gate_logits, target, rescued)
        if loss is None:
            sums["skipped_no_pair_batch_total"] += 1.0
        else:
            sums["pairwise_batch_total"] += 1.0
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(gate_model.parameters(), grad_clip)
                optimizer.step()

        with torch.no_grad():
            probs = torch.sigmoid(gate_logits)
            choose_step = (probs >= 0.5) & rescued
            chosen_correct = torch.where(choose_step, step_correct, base_correct)
            if loss is not None:
                sums["loss_sum"] += loss.item()
                sums["loss_batches"] += 1.0
            sums["rescued_total"] += rescued.float().sum().item()
            sums["target_choose_step_total"] += target[rescued].sum().item()
            sums["pred_choose_step_total"] += choose_step.float().sum().item()
            sums["chosen_correct_total"] += (chosen_correct & rescued).float().sum().item()
            sums["base_correct_total"] += (base_correct & rescued).float().sum().item()
            sums["step_correct_total"] += (step_correct & rescued).float().sum().item()
            all_scores.append(probs[rescued].detach().float().cpu())
            all_labels.append(target[rescued].detach().bool().cpu())

    return finalize_metrics(sums, all_scores, all_labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--proposal_checkpoint_path", type=str, required=True)
    parser.add_argument("--completion_checkpoint_path", type=str, required=True)
    parser.add_argument("--rewrite_checkpoint_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--node_threshold", type=float, default=0.20)
    parser.add_argument("--edge_threshold", type=float, default=0.15)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
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
    rewrite_checkpoint_path = resolve_path(args.rewrite_checkpoint_path)
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
    rewrite_model, use_proposal_conditioning = load_rewrite_model(rewrite_checkpoint_path, device)
    proposal_model.requires_grad_(False)
    completion_model.requires_grad_(False)
    rewrite_model.requires_grad_(False)

    input_dim = infer_feature_dim(
        proposal_model=proposal_model,
        completion_model=completion_model,
        rewrite_model=rewrite_model,
        loader=train_loader,
        device=device,
        node_threshold=args.node_threshold,
        edge_threshold=args.edge_threshold,
        use_proposal_conditioning=use_proposal_conditioning,
    )
    gate_config = RescueFallbackGateConfig(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    gate_model = RescueFallbackGate(gate_config).to(device)
    optimizer = torch.optim.Adam(gate_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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
    print(f"feature dim: {input_dim}")
    print("frozen proposal + frozen completion + frozen rewrite; training compact pairwise ranking chooser only")
    print("chooser_objective: pairwise_keep_ranking")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            proposal_model,
            completion_model,
            rewrite_model,
            gate_model,
            train_loader,
            device,
            args.node_threshold,
            args.edge_threshold,
            use_proposal_conditioning,
            optimizer=optimizer,
            grad_clip=args.grad_clip,
        )
        val_metrics = run_epoch(
            proposal_model,
            completion_model,
            rewrite_model,
            gate_model,
            val_loader,
            device,
            args.node_threshold,
            args.edge_threshold,
            use_proposal_conditioning,
            optimizer=None,
            grad_clip=None,
        )
        val_score = val_metrics.get("selection_score")
        if val_score is None:
            val_score = -float(val_metrics.get("loss") or 0.0)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics, "selection_score": val_score})
        print(
            f"epoch {epoch:03d} | "
            f"train_loss={train_metrics.get('loss'):.6f} | "
            f"val_loss={val_metrics.get('loss'):.6f} | "
            f"val_ap={val_metrics.get('choose_step_ap')} | "
            f"val_auroc={val_metrics.get('choose_step_auroc')} | "
            f"val_target_frac={val_metrics.get('target_choose_step_fraction')} | "
            f"score={val_score}"
        )

        if val_score > best_score:
            best_score = float(val_score)
            best_epoch = epoch
            epochs_without_improve = 0
            save_gate_checkpoint(
                best_path,
                gate_model,
                args,
                extra={
                    "best_epoch": best_epoch,
                    "best_validation_selection_score": best_score,
                    "best_val_metrics": val_metrics,
                    "proposal_checkpoint_path": str(proposal_checkpoint_path),
                    "completion_checkpoint_path": str(completion_checkpoint_path),
                    "rewrite_checkpoint_path": str(rewrite_checkpoint_path),
                    "rescue_budget_fraction": FIXED_RESCUE_BUDGET_FRACTION,
                    "fallback_gate_mode": "learned_rescue_conditioned_gate",
                    "chooser_objective": CHOOSER_OBJECTIVE,
                    "tie_rule": "fallback_to_base_when_base_and_step9c_have_equal_correctness",
                },
            )
        else:
            epochs_without_improve += 1

        save_gate_checkpoint(
            last_path,
            gate_model,
            args,
            extra={
                "epoch": epoch,
                "validation_selection_score": val_score,
                "val_metrics": val_metrics,
                "proposal_checkpoint_path": str(proposal_checkpoint_path),
                "completion_checkpoint_path": str(completion_checkpoint_path),
                "rewrite_checkpoint_path": str(rewrite_checkpoint_path),
                "rescue_budget_fraction": FIXED_RESCUE_BUDGET_FRACTION,
                "fallback_gate_mode": "learned_rescue_conditioned_gate",
                "chooser_objective": CHOOSER_OBJECTIVE,
                "tie_rule": "fallback_to_base_when_base_and_step9c_have_equal_correctness",
            },
        )
        if epochs_without_improve >= args.patience:
            print(f"early stopping after {epoch} epochs")
            break

    summary = {
        "best_epoch": best_epoch,
        "best_validation_selection_score": best_score,
        "history": history,
        "args": vars(args),
        "model_config": asdict(gate_config),
        "proposal_checkpoint_path": str(proposal_checkpoint_path),
        "completion_checkpoint_path": str(completion_checkpoint_path),
        "rewrite_checkpoint_path": str(rewrite_checkpoint_path),
        "rescue_budget_fraction": FIXED_RESCUE_BUDGET_FRACTION,
        "fallback_gate_mode": "learned_rescue_conditioned_gate",
        "chooser_objective": CHOOSER_OBJECTIVE,
        "tie_rule": "fallback_to_base_when_base_and_step9c_have_equal_correctness",
    }
    save_json(summary_path, summary)
    print(f"best epoch: {best_epoch}")
    print(f"best validation selection score: {best_score}")
    print(f"saved best checkpoint: {best_path}")
    print(f"saved summary: {summary_path}")


if __name__ == "__main__":
    main()
