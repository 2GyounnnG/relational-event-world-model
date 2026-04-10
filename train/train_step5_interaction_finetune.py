from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train.eval_rollout_stability import load_rollout_dataset
from train.eval_step5_multievent_regime import evaluate_step5_regime
from train.train_step4_rollout_finetune import (
    compute_rewrite_loss_for_transition,
    decode_predicted_next_graph,
    get_device,
    load_frozen_proposal,
    load_rewrite_from_init,
    resolve_path,
    save_json,
    set_seed,
    transition_targets,
)


class Step5Dataset(Dataset):
    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        self.samples = [
            sample
            for sample in load_rollout_dataset(self.file_path)
            if len(sample.get("graph_steps", [])) >= 2 and len(sample.get("transition_samples", [])) >= 2
        ]
        if not self.samples:
            raise ValueError(f"No usable 2-step Step 5 samples found in {self.file_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


def rollout_collate_fn(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return batch


def build_dataloaders(
    train_path: str,
    val_path: str,
    batch_size: int,
    num_workers: int,
) -> tuple[Step5Dataset, Step5Dataset, DataLoader, DataLoader]:
    train_dataset = Step5Dataset(train_path)
    val_dataset = Step5Dataset(val_path)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=rollout_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=rollout_collate_fn,
    )
    return train_dataset, val_dataset, train_loader, val_loader


def dependency_weight_for_sample(
    raw_sample: Dict[str, Any],
    weight_fully_independent: float,
    weight_partially_dependent: float,
    weight_strongly_interacting: float,
) -> float:
    bucket = str(raw_sample.get("step5_dependency_bucket", "fully_independent"))
    if bucket == "strongly_interacting":
        return float(weight_strongly_interacting)
    if bucket == "partially_dependent":
        return float(weight_partially_dependent)
    return float(weight_fully_independent)


def init_train_accumulator() -> Dict[str, float]:
    return {
        "step1_total_loss_sum": 0.0,
        "step2_total_loss_sum": 0.0,
        "weighted_total_loss_sum": 0.0,
        "sample_weight_sum": 0.0,
        "num_batches": 0.0,
        "num_samples": 0.0,
    }


def finalize_train_accumulator(acc: Dict[str, float]) -> Dict[str, float]:
    num_batches = max(acc["num_batches"], 1.0)
    num_samples = max(acc["num_samples"], 1.0)
    weight_sum = max(acc["sample_weight_sum"], 1.0)
    return {
        "step1_total_loss": acc["step1_total_loss_sum"] / num_samples,
        "step2_total_loss": acc["step2_total_loss_sum"] / num_samples,
        "weighted_total_loss": acc["weighted_total_loss_sum"] / num_batches,
        "mean_sample_weight": acc["sample_weight_sum"] / num_samples,
        "num_samples": acc["num_samples"],
        "sample_weight_sum": weight_sum,
    }


def run_epoch(
    model: torch.nn.Module,
    proposal_model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    node_threshold: float,
    edge_threshold: float,
    edge_loss_weight: float,
    type_loss_weight: float,
    state_loss_weight: float,
    type_flip_weight: float,
    delta_keep_weight: float,
    delta_add_weight: float,
    delta_delete_weight: float,
    fp_edge_keep_loss_weight: float,
    rollout_loss_weight: float,
    weight_fully_independent: float,
    weight_partially_dependent: float,
    weight_strongly_interacting: float,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_clip: Optional[float] = None,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    proposal_model.eval()

    acc = init_train_accumulator()
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch_samples in loader:
            batch_total_loss = None
            batch_size = len(batch_samples)

            for raw_sample in batch_samples:
                transition_samples = raw_sample["transition_samples"]
                step1 = transition_targets(transition_samples[0], device)
                step2 = transition_targets(transition_samples[1], device)

                step1_outputs, step1_loss = compute_rewrite_loss_for_transition(
                    model=model,
                    proposal_model=proposal_model,
                    current_node_feats=step1["current_node_feats"],
                    current_adj=step1["current_adj"],
                    next_node_feats=step1["next_node_feats"],
                    next_adj=step1["next_adj"],
                    node_mask=step1["node_mask"],
                    scope_node_mask=step1["scope_node_mask"],
                    scope_edge_mask=step1["scope_edge_mask"],
                    node_threshold=node_threshold,
                    edge_threshold=edge_threshold,
                    edge_loss_weight=edge_loss_weight,
                    type_loss_weight=type_loss_weight,
                    state_loss_weight=state_loss_weight,
                    type_flip_weight=type_flip_weight,
                    delta_keep_weight=delta_keep_weight,
                    delta_add_weight=delta_add_weight,
                    delta_delete_weight=delta_delete_weight,
                    fp_edge_keep_loss_weight=fp_edge_keep_loss_weight,
                )

                # Match the evaluator exactly: decode the full predicted graph,
                # then feed that autoregressive state into the second transition.
                pred_step1_node_feats, pred_step1_adj = decode_predicted_next_graph(step1_outputs)

                _, step2_loss = compute_rewrite_loss_for_transition(
                    model=model,
                    proposal_model=proposal_model,
                    current_node_feats=pred_step1_node_feats,
                    current_adj=pred_step1_adj,
                    next_node_feats=step2["next_node_feats"],
                    next_adj=step2["next_adj"],
                    node_mask=step2["node_mask"],
                    scope_node_mask=step2["scope_node_mask"],
                    scope_edge_mask=step2["scope_edge_mask"],
                    node_threshold=node_threshold,
                    edge_threshold=edge_threshold,
                    edge_loss_weight=edge_loss_weight,
                    type_loss_weight=type_loss_weight,
                    state_loss_weight=state_loss_weight,
                    type_flip_weight=type_flip_weight,
                    delta_keep_weight=delta_keep_weight,
                    delta_add_weight=delta_add_weight,
                    delta_delete_weight=delta_delete_weight,
                    fp_edge_keep_loss_weight=fp_edge_keep_loss_weight,
                )

                sample_weight = dependency_weight_for_sample(
                    raw_sample,
                    weight_fully_independent=weight_fully_independent,
                    weight_partially_dependent=weight_partially_dependent,
                    weight_strongly_interacting=weight_strongly_interacting,
                )
                sample_total_loss = step1_loss["total_loss"] + rollout_loss_weight * step2_loss["total_loss"]
                weighted_sample_total_loss = sample_total_loss * sample_weight
                batch_total_loss = (
                    weighted_sample_total_loss
                    if batch_total_loss is None
                    else (batch_total_loss + weighted_sample_total_loss)
                )

                acc["step1_total_loss_sum"] += step1_loss["total_loss"].item()
                acc["step2_total_loss_sum"] += step2_loss["total_loss"].item()
                acc["weighted_total_loss_sum"] += weighted_sample_total_loss.item()
                acc["sample_weight_sum"] += sample_weight
                acc["num_samples"] += 1.0

            batch_total_loss = batch_total_loss / max(batch_size, 1)
            if is_train:
                optimizer.zero_grad()
                batch_total_loss.backward()
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            acc["num_batches"] += 1.0

    return finalize_train_accumulator(acc)


def compute_selection_score(val_results: Dict[str, Any]) -> float:
    overall = val_results["overall_final"]
    strongly_interacting = val_results["dependency_bucket_summary"].get("strongly_interacting", {})
    metrics = [
        overall.get("full_edge_acc"),
        overall.get("context_edge_acc"),
        overall.get("changed_edge_acc"),
        overall.get("delete"),
        strongly_interacting.get("changed_edge_acc"),
        strongly_interacting.get("delete"),
    ]
    values = [float(value) for value in metrics if value is not None]
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--proposal_checkpoint", type=str, required=True)
    parser.add_argument("--init_rewrite_checkpoint", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--node_threshold", type=float, default=0.20)
    parser.add_argument("--edge_threshold", type=float, default=0.15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=6)
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
    parser.add_argument("--rollout_loss_weight", type=float, default=0.50)
    parser.add_argument("--weight_fully_independent", type=float, required=True)
    parser.add_argument("--weight_partially_dependent", type=float, required=True)
    parser.add_argument("--weight_strongly_interacting", type=float, required=True)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    save_dir = resolve_path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = save_dir / "best.pt"
    last_ckpt_path = save_dir / "last.pt"
    best_metrics_path = save_dir / "best_metrics.json"

    train_path = resolve_path(args.train_path)
    val_path = resolve_path(args.val_path)
    proposal_checkpoint_path = resolve_path(args.proposal_checkpoint)
    init_rewrite_checkpoint_path = resolve_path(args.init_rewrite_checkpoint)

    train_dataset, val_dataset, train_loader, _ = build_dataloaders(
        train_path=str(train_path),
        val_path=str(val_path),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    val_samples = val_dataset.samples

    proposal_model = load_frozen_proposal(proposal_checkpoint_path, device)
    model, init_checkpoint = load_rewrite_from_init(init_rewrite_checkpoint_path, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_score = float("-inf")
    best_epoch = -1
    epochs_without_improve = 0
    proposal_conditioning_enabled = bool(
        init_checkpoint.get("model_config", {}).get("use_proposal_conditioning", False)
    )

    print(f"device: {device}")
    print(f"train samples: {len(train_dataset)}")
    print(f"val samples: {len(val_dataset)}")
    print(f"proposal checkpoint: {proposal_checkpoint_path}")
    print(f"init rewrite checkpoint: {init_rewrite_checkpoint_path}")
    print(f"save dir: {save_dir}")
    print(f"proposal conditioning enabled: {proposal_conditioning_enabled}")
    print(
        "autoregressive graph-state conversion: argmax(type_logits_full) + "
        "state_pred_full + thresholded/symmetrized edge_logits_full"
    )
    print(
        "dependency weights: "
        f"fully_independent={args.weight_fully_independent}, "
        f"partially_dependent={args.weight_partially_dependent}, "
        f"strongly_interacting={args.weight_strongly_interacting}"
    )

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            proposal_model=proposal_model,
            loader=train_loader,
            device=device,
            node_threshold=args.node_threshold,
            edge_threshold=args.edge_threshold,
            edge_loss_weight=args.edge_loss_weight,
            type_loss_weight=args.type_loss_weight,
            state_loss_weight=args.state_loss_weight,
            type_flip_weight=args.type_flip_weight,
            delta_keep_weight=args.delta_keep_weight,
            delta_add_weight=args.delta_add_weight,
            delta_delete_weight=args.delta_delete_weight,
            fp_edge_keep_loss_weight=args.fp_edge_keep_loss_weight,
            rollout_loss_weight=args.rollout_loss_weight,
            weight_fully_independent=args.weight_fully_independent,
            weight_partially_dependent=args.weight_partially_dependent,
            weight_strongly_interacting=args.weight_strongly_interacting,
            optimizer=optimizer,
            grad_clip=args.grad_clip,
        )

        val_results = evaluate_step5_regime(
            proposal_model=proposal_model,
            rewrite_model=model,
            samples=val_samples,
            device=device,
            node_threshold=args.node_threshold,
            edge_threshold=args.edge_threshold,
            use_proposal_conditioning=proposal_conditioning_enabled,
        )
        val_score = compute_selection_score(val_results)
        overall = val_results["overall_final"]
        strong = val_results["dependency_bucket_summary"].get("strongly_interacting", {})

        print(
            f"[epoch {epoch:02d}] "
            f"train_weighted_total={train_metrics['weighted_total_loss']:.6f} "
            f"train_step1={train_metrics['step1_total_loss']:.6f} "
            f"train_step2={train_metrics['step2_total_loss']:.6f} "
            f"mean_sample_weight={train_metrics['mean_sample_weight']:.4f} "
            f"val_full_edge_acc={overall['full_edge_acc']:.6f} "
            f"val_changed_edge_acc={overall['changed_edge_acc']:.6f} "
            f"val_context_edge_acc={overall['context_edge_acc']:.6f} "
            f"val_delete={overall['delete']:.6f} "
            f"val_strong_changed={strong.get('changed_edge_acc', 0.0):.6f} "
            f"val_strong_delete={strong.get('delete', 0.0):.6f} "
            f"val_selection_score={val_score:.6f}"
        )

        checkpoint_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": init_checkpoint["model_config"],
            "train_metrics": train_metrics,
            "val_step5_metrics": val_results,
            "val_selection_score": val_score,
            "proposal_checkpoint": str(proposal_checkpoint_path),
            "init_rewrite_checkpoint": str(init_rewrite_checkpoint_path),
            "rollout_loss_weight": args.rollout_loss_weight,
            "weight_fully_independent": args.weight_fully_independent,
            "weight_partially_dependent": args.weight_partially_dependent,
            "weight_strongly_interacting": args.weight_strongly_interacting,
            "selection_metric": "step5_balanced_overall_and_strong_interaction",
            "state_conversion": {
                "node_type": "argmax(type_logits_full)",
                "node_state": "state_pred_full",
                "edge_adj": "sigmoid(edge_logits_full)>=0.5 with undirected symmetrization",
            },
        }
        torch.save(checkpoint_payload, last_ckpt_path)

        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch
            epochs_without_improve = 0
            torch.save(checkpoint_payload, best_ckpt_path)
            save_json(
                best_metrics_path,
                {
                    "best_epoch": best_epoch,
                    "best_validation_selection_score": best_score,
                    "selection_metric": "step5_balanced_overall_and_strong_interaction",
                    "rollout_loss_weight": args.rollout_loss_weight,
                    "weight_fully_independent": args.weight_fully_independent,
                    "weight_partially_dependent": args.weight_partially_dependent,
                    "weight_strongly_interacting": args.weight_strongly_interacting,
                    "proposal_conditioning_enabled": proposal_conditioning_enabled,
                    "warm_start_used": True,
                    "train_metrics": train_metrics,
                    "val_step5_metrics": val_results,
                },
            )
            print(f"  new best checkpoint saved at epoch {epoch}")
        else:
            epochs_without_improve += 1
            print(
                f"  no improvement. patience {epochs_without_improve}/{args.patience} "
                f"(best epoch {best_epoch}, best score {best_score:.6f})"
            )

        if epochs_without_improve >= args.patience:
            print("early stopping triggered")
            break

    print(f"training finished. best epoch={best_epoch} best validation selection score={best_score:.6f}")
    print(f"best checkpoint: {best_ckpt_path}")
    print(f"last checkpoint: {last_ckpt_path}")
    print(f"best metrics json: {best_metrics_path}")


if __name__ == "__main__":
    main()
