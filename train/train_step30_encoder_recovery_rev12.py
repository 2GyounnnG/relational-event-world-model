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

from data.step30_dataset import Step30WeakObservationDataset, step30_weak_observation_collate_fn
from models.encoder_step30 import (
    Step30EncoderConfig,
    Step30WeakObservationEncoder,
    step30_recovery_loss,
    step30_recovery_metrics,
)
from train.eval_step30_encoder_recovery import get_device, move_batch_to_device
from train.utils_step30_decode import (
    hard_adj_from_scores,
    hard_adj_selective_rescue,
    threshold_tensor_for_variants,
    upper_pair_mask,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_rev6_initialized_model(checkpoint_path: str, device: torch.device) -> Step30WeakObservationEncoder:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config_dict = dict(checkpoint["config"])
    config_dict.setdefault("use_relation_hint_in_edge_head", True)
    config_dict.setdefault("use_relation_logit_residual", True)
    config_dict.setdefault("relation_logit_residual_scale", 1.0)
    config_dict.setdefault("use_trust_denoising_edge_decoder", False)
    config_dict.setdefault("use_pair_support_hints", True)
    config_dict["use_rescue_safety_aux_head"] = True
    config = Step30EncoderConfig(**config_dict)
    model = Step30WeakObservationEncoder(config).to(device)
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    allowed_missing = {
        "rescue_safety_head.0.weight",
        "rescue_safety_head.0.bias",
        "rescue_safety_head.2.weight",
        "rescue_safety_head.2.bias",
    }
    if unexpected:
        raise ValueError(f"Unexpected keys while loading rev6 checkpoint: {unexpected}")
    if not set(missing).issubset(allowed_missing):
        raise ValueError(f"Unexpected missing keys while loading rev6 checkpoint: {missing}")
    return model


def selection_score(metrics: Dict[str, float]) -> float:
    # Budget-aligned primary criterion, with a small recovery term as a guardrail.
    return float(metrics["rescue_accepted_precision"]) + 0.10 * float(metrics["noisy_edge_f1"])


def average(metrics: Dict[str, float], count: int) -> Dict[str, float]:
    return {key: value / max(count, 1) for key, value in metrics.items()}


def add_sample_metrics(target: Dict[str, float], metrics: Dict[str, float]) -> None:
    for key in [
        "node_type_accuracy",
        "node_state_mae",
        "node_state_mse",
        "edge_accuracy",
        "edge_precision",
        "edge_recall",
        "edge_f1",
    ]:
        target[key] += float(metrics.get(key, 0.0))


def empty_eval_sums() -> Dict[str, float]:
    return {
        "node_type_accuracy": 0.0,
        "node_state_mae": 0.0,
        "node_state_mse": 0.0,
        "edge_accuracy": 0.0,
        "edge_precision": 0.0,
        "edge_recall": 0.0,
        "edge_f1": 0.0,
    }


def rescue_acceptance_counts(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, Any],
    variants: list[str],
    edge_thresholds_by_variant: Dict[str, float],
    rescue_relation_max: float,
    rescue_support_min: float,
    rescue_budget_fraction: float,
) -> tuple[torch.Tensor, Dict[str, float]]:
    base_threshold = threshold_tensor_for_variants(
        variants=variants,
        default_threshold=float(edge_thresholds_by_variant.get("default", 0.5)),
        thresholds_by_variant=edge_thresholds_by_variant,
        device=batch["target_adj"].device,
    )
    pred_adj = hard_adj_selective_rescue(
        edge_logits=outputs["edge_logits"],
        relation_hints=batch["weak_relation_hints"],
        pair_support_hints=batch.get("weak_pair_support_hints"),
        node_mask=batch["node_mask"],
        variants=variants,
        base_threshold=base_threshold,
        rescue_variants={"noisy"},
        rescue_relation_max=rescue_relation_max,
        rescue_support_min=rescue_support_min,
        rescue_budget_fraction=rescue_budget_fraction,
        rescue_score_mode="aux",
        rescue_aux_logits=outputs.get("rescue_safety_logits"),
    )
    edge_scores = torch.sigmoid(outputs["edge_logits"])
    base_adj = hard_adj_from_scores(edge_scores, threshold=base_threshold).bool()
    upper = upper_pair_mask(batch["node_mask"])
    variant_mask = torch.tensor(
        [variant == "noisy" for variant in variants],
        device=batch["target_adj"].device,
        dtype=torch.bool,
    ).view(-1, 1, 1)
    pair_support = batch.get("weak_pair_support_hints")
    candidate = (
        upper
        & variant_mask
        & (~base_adj)
        & (batch["weak_relation_hints"] < float(rescue_relation_max))
        & (pair_support >= float(rescue_support_min))
    )
    accepted = candidate & (pred_adj > 0.5)
    target = batch["target_adj"].float() > 0.5
    accepted_count = float(accepted.sum().item())
    accepted_pos = float((accepted & target).sum().item())
    candidate_pos = float((candidate & target).sum().item())
    return pred_adj, {
        "rescue_candidate_count": float(candidate.sum().item()),
        "rescue_candidate_pos": candidate_pos,
        "rescue_accepted_count": accepted_count,
        "rescue_accepted_pos": accepted_pos,
        "rescue_accepted_fp": float((accepted & (~target)).sum().item()),
    }


def run_epoch(
    model: Step30WeakObservationEncoder,
    loader: DataLoader,
    device: torch.device,
    edge_thresholds_by_variant: Dict[str, float],
    rescue_safety_aux_loss_weight: float,
    rescue_safety_pos_weight: float,
    rescue_relation_max: float,
    rescue_support_min: float,
    rescue_budget_fraction: float,
    lr_optimizer: Optional[torch.optim.Optimizer] = None,
    grad_clip: Optional[float] = None,
) -> Dict[str, float]:
    is_train = lr_optimizer is not None
    model.train() if is_train else model.eval()
    loss_sums = {
        "total_loss": 0.0,
        "edge_loss": 0.0,
        "missed_edge_loss": 0.0,
        "rescue_safety_aux_loss": 0.0,
    }
    overall = empty_eval_sums()
    noisy = empty_eval_sums()
    counts = {"batches": 0, "overall_samples": 0, "noisy_samples": 0}
    rescue_counts = {
        "rescue_candidate_count": 0.0,
        "rescue_candidate_pos": 0.0,
        "rescue_accepted_count": 0.0,
        "rescue_accepted_pos": 0.0,
        "rescue_accepted_fp": 0.0,
    }

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            variants = [str(v) for v in batch.get("step30_observation_variant", [])]
            outputs = model(
                weak_slot_features=batch["weak_slot_features"],
                weak_relation_hints=batch["weak_relation_hints"],
                weak_pair_support_hints=batch.get("weak_pair_support_hints"),
            )
            loss_dict = step30_recovery_loss(
                outputs=outputs,
                target_node_feats=batch["target_node_feats"],
                target_adj=batch["target_adj"],
                node_mask=batch["node_mask"],
                relation_hints=batch["weak_relation_hints"],
                pair_support_hints=batch.get("weak_pair_support_hints"),
                type_loss_weight=1.0,
                state_loss_weight=1.0,
                edge_loss_weight=1.0,
                edge_pos_weight=1.0,
                missed_edge_loss_weight=0.20,
                missed_edge_hint_threshold=0.5,
                rescue_safety_aux_loss_weight=rescue_safety_aux_loss_weight,
                rescue_safety_relation_max=rescue_relation_max,
                rescue_safety_support_min=rescue_support_min,
                rescue_safety_pos_weight=rescue_safety_pos_weight,
            )
            if is_train:
                lr_optimizer.zero_grad()
                loss_dict["total_loss"].backward()
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                lr_optimizer.step()

            pred_adj, batch_rescue_counts = rescue_acceptance_counts(
                outputs=outputs,
                batch=batch,
                variants=variants,
                edge_thresholds_by_variant=edge_thresholds_by_variant,
                rescue_relation_max=rescue_relation_max,
                rescue_support_min=rescue_support_min,
                rescue_budget_fraction=rescue_budget_fraction,
            )
            for key in loss_sums:
                loss_sums[key] += float(loss_dict[key].detach().item())
            for key in rescue_counts:
                rescue_counts[key] += batch_rescue_counts[key]
            batch_size = int(batch["target_adj"].shape[0])
            for idx in range(batch_size):
                one_outputs = {key: value[idx : idx + 1] for key, value in outputs.items()}
                metrics = step30_recovery_metrics(
                    outputs=one_outputs,
                    target_node_feats=batch["target_node_feats"][idx : idx + 1],
                    target_adj=batch["target_adj"][idx : idx + 1],
                    node_mask=batch["node_mask"][idx : idx + 1],
                    edge_threshold=float(edge_thresholds_by_variant.get(variants[idx], edge_thresholds_by_variant["default"])),
                    edge_pred_override=pred_adj[idx : idx + 1],
                )
                add_sample_metrics(overall, metrics)
                counts["overall_samples"] += 1
                if variants[idx] == "noisy":
                    add_sample_metrics(noisy, metrics)
                    counts["noisy_samples"] += 1
            counts["batches"] += 1

    result = {
        **average(loss_sums, counts["batches"]),
        **{f"overall_{k}": v for k, v in average(overall, counts["overall_samples"]).items()},
        **{f"noisy_{k}": v for k, v in average(noisy, counts["noisy_samples"]).items()},
        **rescue_counts,
    }
    accepted = result["rescue_accepted_count"]
    accepted_pos = result["rescue_accepted_pos"]
    candidate_pos = result["rescue_candidate_pos"]
    result["rescue_accepted_precision"] = accepted_pos / accepted if accepted > 0 else 0.0
    result["rescue_candidate_recall"] = accepted_pos / candidate_pos if candidate_pos > 0 else 0.0
    result["selection_score"] = selection_score(result)
    return result


def save_checkpoint(
    path: Path,
    model: Step30WeakObservationEncoder,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="data/graph_event_step30_weak_obs_rev6_train.pkl")
    parser.add_argument("--val_path", type=str, default="data/graph_event_step30_weak_obs_rev6_val.pkl")
    parser.add_argument("--init_checkpoint_path", type=str, default="checkpoints/step30_encoder_recovery_rev6/best.pt")
    parser.add_argument("--save_dir", type=str, default="checkpoints/step30_encoder_recovery_rev12")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--rescue_safety_aux_loss_weight", type=float, default=0.20)
    parser.add_argument("--rescue_safety_pos_weight", type=float, default=5.0)
    parser.add_argument("--rescue_relation_max", type=float, default=0.5)
    parser.add_argument("--rescue_support_min", type=float, default=0.55)
    parser.add_argument("--rescue_budget_fraction", type=float, default=0.06)
    parser.add_argument("--edge_thresholds_by_variant", type=str, default="default:0.5,clean:0.5,noisy:0.55")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    thresholds = {}
    for part in args.edge_thresholds_by_variant.split(","):
        key, value = part.split(":", 1)
        thresholds[key.strip()] = float(value)

    train_dataset = Step30WeakObservationDataset(args.train_path)
    val_dataset = Step30WeakObservationDataset(args.val_path)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=step30_weak_observation_collate_fn,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=step30_weak_observation_collate_fn,
        pin_memory=device.type == "cuda",
    )
    model = load_rev6_initialized_model(args.init_checkpoint_path, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    save_dir = Path(args.save_dir)
    best_score = -float("inf")
    best_epoch = -1
    history = []

    print(f"device: {device}")
    print(f"train samples: {len(train_dataset)}")
    print(f"val samples: {len(val_dataset)}")
    print(f"config: {model.config.to_dict()}")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            edge_thresholds_by_variant=thresholds,
            rescue_safety_aux_loss_weight=args.rescue_safety_aux_loss_weight,
            rescue_safety_pos_weight=args.rescue_safety_pos_weight,
            rescue_relation_max=args.rescue_relation_max,
            rescue_support_min=args.rescue_support_min,
            rescue_budget_fraction=args.rescue_budget_fraction,
            lr_optimizer=optimizer,
            grad_clip=args.grad_clip,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            edge_thresholds_by_variant=thresholds,
            rescue_safety_aux_loss_weight=args.rescue_safety_aux_loss_weight,
            rescue_safety_pos_weight=args.rescue_safety_pos_weight,
            rescue_relation_max=args.rescue_relation_max,
            rescue_support_min=args.rescue_support_min,
            rescue_budget_fraction=args.rescue_budget_fraction,
            lr_optimizer=None,
        )
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(
            f"epoch={epoch:03d} train_loss={train_metrics['total_loss']:.6f} "
            f"val_loss={val_metrics['total_loss']:.6f} "
            f"val_noisy_edge_f1={val_metrics['noisy_edge_f1']:.4f} "
            f"val_rescue_prec={val_metrics['rescue_accepted_precision']:.4f} "
            f"val_rescue_recall={val_metrics['rescue_candidate_recall']:.4f} "
            f"val_score={val_metrics['selection_score']:.4f}"
        )
        if val_metrics["selection_score"] > best_score:
            best_score = float(val_metrics["selection_score"])
            best_epoch = epoch
            save_checkpoint(save_dir / "best.pt", model, args, epoch, val_metrics)

    save_checkpoint(save_dir / "last.pt", model, args, args.epochs, val_metrics)
    summary = {
        "best_epoch": best_epoch,
        "best_validation_selection_score": best_score,
        "config": model.config.to_dict(),
        "args": vars(args),
        "history": history,
    }
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"best epoch: {best_epoch}")
    print(f"best validation selection score: {best_score:.6f}")
    print(f"saved best checkpoint: {save_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
