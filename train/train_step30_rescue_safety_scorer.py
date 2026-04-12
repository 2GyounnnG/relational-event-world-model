from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.step30_dataset import Step30WeakObservationDataset, step30_weak_observation_collate_fn
from train.eval_step30_encoder_recovery import get_device, load_model, move_batch_to_device
from train.utils_step30_decode import (
    RESCUE_SAFETY_FEATURE_NAMES,
    hard_adj_from_scores,
    rescue_safety_feature_tensor,
    threshold_tensor_for_variants,
    upper_pair_mask,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def average_precision(scores: torch.Tensor, labels: torch.Tensor) -> float:
    labels = labels.float()
    positives = labels.sum().item()
    if positives <= 0:
        return 0.0
    order = torch.argsort(scores, descending=True)
    sorted_labels = labels[order]
    precision_at_rank = torch.cumsum(sorted_labels, dim=0) / torch.arange(
        1,
        sorted_labels.numel() + 1,
        device=sorted_labels.device,
        dtype=torch.float32,
    )
    return float((precision_at_rank * sorted_labels).sum().item() / positives)


def auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    labels = labels.float()
    pos_count = int((labels > 0.5).sum().item())
    neg_count = int((labels <= 0.5).sum().item())
    if pos_count == 0 or neg_count == 0:
        return 0.0
    order = torch.argsort(scores, descending=False)
    ranks = torch.arange(1, labels.numel() + 1, device=labels.device, dtype=torch.float32)
    pos_rank_sum = ranks[labels[order] > 0.5].sum()
    auc = (pos_rank_sum - pos_count * (pos_count + 1) / 2.0) / (pos_count * neg_count)
    return float(auc.item())


def precision_at_fraction(scores: torch.Tensor, labels: torch.Tensor, fraction: float) -> float:
    if scores.numel() == 0:
        return 0.0
    k = max(1, int(math.ceil(scores.numel() * float(fraction))))
    k = min(k, scores.numel())
    top = torch.topk(scores, k=k).indices
    return float(labels[top].float().mean().item())


@torch.no_grad()
def collect_rescue_candidates(
    data_path: str,
    encoder_checkpoint_path: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    edge_thresholds_by_variant: Dict[str, float],
    rescue_variants: set[str],
    rescue_relation_max: float,
    rescue_support_min: float,
) -> Dict[str, Any]:
    dataset = Step30WeakObservationDataset(data_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=step30_weak_observation_collate_fn,
        pin_memory=device.type == "cuda",
    )
    model = load_model(encoder_checkpoint_path, device)
    all_features: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    variant_counts: Dict[str, int] = {}
    positive_counts: Dict[str, int] = {}

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        variants = [str(v) for v in batch.get("step30_observation_variant", [])]
        outputs = model(
            weak_slot_features=batch["weak_slot_features"],
            weak_relation_hints=batch["weak_relation_hints"],
            weak_pair_support_hints=batch.get("weak_pair_support_hints"),
        )
        pair_support = batch.get("weak_pair_support_hints")
        if pair_support is None:
            raise ValueError("rev10 rescue-safety scoring requires weak_pair_support_hints")
        base_threshold = threshold_tensor_for_variants(
            variants=variants,
            default_threshold=float(edge_thresholds_by_variant.get("default", 0.5)),
            thresholds_by_variant=edge_thresholds_by_variant,
            device=batch["target_adj"].device,
        )
        edge_scores = torch.sigmoid(outputs["edge_logits"])
        base_adj = hard_adj_from_scores(edge_scores, threshold=base_threshold).bool()
        upper_mask = upper_pair_mask(batch["node_mask"])
        variant_mask = torch.tensor(
            [variant in rescue_variants for variant in variants],
            device=batch["target_adj"].device,
            dtype=torch.bool,
        ).view(-1, 1, 1)
        candidate_mask = (
            upper_mask
            & variant_mask
            & (~base_adj)
            & (batch["weak_relation_hints"] < float(rescue_relation_max))
            & (pair_support >= float(rescue_support_min))
        )
        if not candidate_mask.any():
            continue
        feature_tensor = rescue_safety_feature_tensor(
            edge_logits=outputs["edge_logits"],
            relation_hints=batch["weak_relation_hints"],
            pair_support_hints=pair_support,
            base_threshold=base_threshold,
            rescue_relation_max=rescue_relation_max,
            rescue_support_min=rescue_support_min,
        )
        labels = batch["target_adj"].float()
        all_features.append(feature_tensor[candidate_mask].detach().cpu())
        all_labels.append(labels[candidate_mask].detach().cpu())
        for batch_idx, variant in enumerate(variants):
            sample_mask = candidate_mask[batch_idx]
            count = int(sample_mask.sum().item())
            if count <= 0:
                continue
            variant_counts[variant] = variant_counts.get(variant, 0) + count
            positive_counts[variant] = positive_counts.get(variant, 0) + int(
                labels[batch_idx][sample_mask].sum().item()
            )

    if not all_features:
        raise ValueError(f"No rescue candidates found in {data_path}")
    features = torch.cat(all_features, dim=0).float()
    labels = torch.cat(all_labels, dim=0).float()
    return {
        "features": features,
        "labels": labels,
        "sample_count": len(dataset),
        "variant_counts": variant_counts,
        "positive_counts": positive_counts,
    }


def evaluate_linear(
    features: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    budget_fraction: float,
) -> Dict[str, float]:
    scores = ((features - mean) / std * weights).sum(dim=-1) + bias
    pred = (torch.sigmoid(scores) >= 0.5).float()
    tp = ((pred > 0.5) & (labels > 0.5)).sum().item()
    pred_pos = (pred > 0.5).sum().item()
    true_pos = (labels > 0.5).sum().item()
    precision = tp / pred_pos if pred_pos > 0 else 0.0
    recall = tp / true_pos if true_pos > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8) if precision + recall > 0 else 0.0
    return {
        "candidate_count": float(labels.numel()),
        "positive_count": float(true_pos),
        "positive_rate": float(labels.mean().item()) if labels.numel() else 0.0,
        "bce": float(F.binary_cross_entropy_with_logits(scores, labels).item()),
        "auroc": auroc(scores, labels),
        "ap": average_precision(scores, labels),
        "precision_at_rescue_fraction": precision_at_fraction(scores, labels, budget_fraction),
        "threshold_precision": precision,
        "threshold_recall": recall,
        "threshold_f1": f1,
        "avg_positive_score": float(torch.sigmoid(scores[labels > 0.5]).mean().item()) if true_pos > 0 else 0.0,
        "avg_negative_score": float(torch.sigmoid(scores[labels <= 0.5]).mean().item())
        if true_pos < labels.numel()
        else 0.0,
    }


def write_csv(path: Path, rows: list[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", type=str, default="data/graph_event_step30_weak_obs_rev6_train.pkl")
    parser.add_argument("--val_data_path", type=str, default="data/graph_event_step30_weak_obs_rev6_val.pkl")
    parser.add_argument("--encoder_checkpoint_path", type=str, default="checkpoints/step30_encoder_recovery_rev6/best.pt")
    parser.add_argument("--output_path", type=str, default="checkpoints/step30_rescue_safety_rev10/scorer.json")
    parser.add_argument("--summary_json", type=str, default="artifacts/step30_rescue_safety_rev10/train_summary.json")
    parser.add_argument("--summary_csv", type=str, default="artifacts/step30_rescue_safety_rev10/train_summary.csv")
    parser.add_argument("--edge_thresholds_by_variant", type=str, default="default:0.5,clean:0.5,noisy:0.55")
    parser.add_argument("--rescue_variants", type=str, default="noisy")
    parser.add_argument("--rescue_relation_max", type=float, default=0.5)
    parser.add_argument("--rescue_support_min", type=float, default=0.55)
    parser.add_argument("--rescue_budget_fraction", type=float, default=0.06)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--collect_batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    edge_thresholds = {}
    for part in args.edge_thresholds_by_variant.split(","):
        key, value = part.split(":", 1)
        edge_thresholds[key.strip()] = float(value)
    rescue_variants = {part.strip() for part in args.rescue_variants.split(",") if part.strip()}

    train = collect_rescue_candidates(
        data_path=args.train_data_path,
        encoder_checkpoint_path=args.encoder_checkpoint_path,
        device=device,
        batch_size=args.collect_batch_size,
        num_workers=args.num_workers,
        edge_thresholds_by_variant=edge_thresholds,
        rescue_variants=rescue_variants,
        rescue_relation_max=args.rescue_relation_max,
        rescue_support_min=args.rescue_support_min,
    )
    val = collect_rescue_candidates(
        data_path=args.val_data_path,
        encoder_checkpoint_path=args.encoder_checkpoint_path,
        device=device,
        batch_size=args.collect_batch_size,
        num_workers=args.num_workers,
        edge_thresholds_by_variant=edge_thresholds,
        rescue_variants=rescue_variants,
        rescue_relation_max=args.rescue_relation_max,
        rescue_support_min=args.rescue_support_min,
    )

    train_x = train["features"].to(device)
    train_y = train["labels"].to(device)
    val_x = val["features"].to(device)
    val_y = val["labels"].to(device)
    mean = train_x.mean(dim=0)
    std = train_x.std(dim=0).clamp_min(1e-6)
    model = torch.nn.Linear(train_x.shape[-1], 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pos = train_y.sum().item()
    neg = train_y.numel() - pos
    pos_weight = torch.tensor(neg / max(pos, 1.0), device=device, dtype=torch.float32)
    train_loader = DataLoader(
        TensorDataset((train_x - mean) / std, train_y),
        batch_size=args.batch_size,
        shuffle=True,
    )

    best_state: Dict[str, torch.Tensor] | None = None
    best_val_ap = -1.0
    best_epoch = -1
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for x, y in train_loader:
            logits = model(x).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            batches += 1

        model.eval()
        with torch.no_grad():
            weights = model.weight.detach().view(-1)
            bias = model.bias.detach().view(())
            train_metrics = evaluate_linear(
                train_x,
                train_y,
                weights,
                bias,
                mean,
                std,
                args.rescue_budget_fraction,
            )
            val_metrics = evaluate_linear(
                val_x,
                val_y,
                weights,
                bias,
                mean,
                std,
                args.rescue_budget_fraction,
            )
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(batches, 1),
                "train_ap": train_metrics["ap"],
                "val_ap": val_metrics["ap"],
                "val_auroc": val_metrics["auroc"],
                "val_precision_at_rescue_fraction": val_metrics["precision_at_rescue_fraction"],
            }
        )
        if val_metrics["ap"] > best_val_ap:
            best_val_ap = val_metrics["ap"]
            best_epoch = epoch
            best_state = {
                "weight": weights.detach().cpu().clone(),
                "bias": bias.detach().cpu().clone(),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            }
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    assert best_state is not None
    scorer = {
        "feature_names": RESCUE_SAFETY_FEATURE_NAMES,
        "feature_mean": mean.detach().cpu().tolist(),
        "feature_std": std.detach().cpu().tolist(),
        "weights": best_state["weight"].tolist(),
        "bias": float(best_state["bias"].item()),
        "metadata": {
            "step": "step30_rev10_rescue_safety",
            "target": "GT-positive edge among rev8-style rescue candidates",
            "train_data_path": args.train_data_path,
            "val_data_path": args.val_data_path,
            "encoder_checkpoint_path": args.encoder_checkpoint_path,
            "edge_thresholds_by_variant": edge_thresholds,
            "rescue_variants": sorted(rescue_variants),
            "rescue_relation_max": args.rescue_relation_max,
            "rescue_support_min": args.rescue_support_min,
            "rescue_budget_fraction": args.rescue_budget_fraction,
            "best_epoch": best_epoch,
            "selection_metric": "val_ap",
            "train_candidate_count": int(train_y.numel()),
            "val_candidate_count": int(val_y.numel()),
            "train_positive_count": int(train_y.sum().item()),
            "val_positive_count": int(val_y.sum().item()),
            "train_variant_counts": train["variant_counts"],
            "train_positive_counts": train["positive_counts"],
            "val_variant_counts": val["variant_counts"],
            "val_positive_counts": val["positive_counts"],
        },
        "train_metrics": best_state["train_metrics"],
        "val_metrics": best_state["val_metrics"],
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scorer, f, indent=2)

    summary = {
        "scorer_path": str(output_path),
        "best_epoch": best_epoch,
        "best_val_ap": best_val_ap,
        "train_metrics": best_state["train_metrics"],
        "val_metrics": best_state["val_metrics"],
        "metadata": scorer["metadata"],
        "history": history,
    }
    summary_json = Path(args.summary_json)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_csv(
        Path(args.summary_csv),
        [
            {"split": "train", **best_state["train_metrics"]},
            {"split": "val", **best_state["val_metrics"]},
        ],
    )
    print(json.dumps({k: summary[k] for k in ["scorer_path", "best_epoch", "best_val_ap"]}, indent=2))
    print(f"wrote scorer: {output_path}")
    print(f"wrote summary: {summary_json}")


if __name__ == "__main__":
    main()
