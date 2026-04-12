from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.step30_dataset import Step30WeakObservationDataset, step30_weak_observation_collate_fn
from models.encoder_step30 import build_pair_mask
from train.eval_step30_encoder_recovery import load_model, move_batch_to_device, threshold_for_variant


CLASS_NAMES = [
    "safe_missed_true_edge",
    "low_hint_pair_support_false_admission",
    "ambiguous_rescue_candidate",
]
SCALAR_FEATURE_NAMES = [
    "relation_hint",
    "pair_support_hint",
    "support_minus_relation",
    "signed_pair_witness",
    "bundle_positive_support",
    "bundle_false_admission_warning",
    "bundle_corroboration",
    "bundle_endpoint_compatibility",
    "bundle_margin_positive_minus_warning",
    "abs_bundle_margin",
    "rev6_score",
    "rev6_logit",
    "rev19_score",
    "rev19_residual",
    "rev21_score",
    "rev21_residual",
    "hint_endpoint_degree_sum",
    "support_endpoint_degree_sum",
    "hint_common_neighbors",
    "support_common_neighbors",
]


@dataclass
class ExtractedSplit:
    features: np.ndarray
    labels: np.ndarray
    binary_labels: np.ndarray
    scores: Dict[str, np.ndarray]
    metadata: Dict[str, Any]
    outside_noisy_counts: Dict[str, float]


class RescueCandidateProbe(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(latent_dim, len(CLASS_NAMES))

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.feature_net(features)
        logits = self.classifier(latent)
        return logits, latent


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


def parse_thresholds(value: str) -> Dict[str, float]:
    out: Dict[str, float] = {"default": 0.5}
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        key, raw_val = part.split(":", 1)
        out[key.strip()] = float(raw_val)
    return out


def sigmoid_np(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))


def logit_to_prob(value: float) -> float:
    return float(1.0 / (1.0 + math.exp(-max(min(value, 80.0), -80.0))))


def rescue_class_label(target: int, relation: float, support: float, bundle: np.ndarray) -> int:
    if target == 1:
        return 0
    ambiguous = (
        relation >= 0.45
        or support < 0.65
        or abs(float(bundle[0]) - float(bundle[1])) < 0.10
    )
    if ambiguous:
        return 2
    return 1


def binary_metrics(tp: float, fp: float, fn: float) -> Dict[str, float]:
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def auroc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores)
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = 0.5 * (start + 1 + end)
        ranks[order[start:end]] = avg_rank
        start = end
    rank_sum_pos = ranks[pos].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float | None:
    n_pos = int((labels == 1).sum())
    if n_pos == 0:
        return None
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    tp_cum = np.cumsum(sorted_labels == 1)
    ranks = np.arange(1, len(sorted_labels) + 1)
    precision_at_k = tp_cum / ranks
    return float((precision_at_k * (sorted_labels == 1)).sum() / n_pos)


def clean_json_numbers(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: clean_json_numbers(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [clean_json_numbers(value) for value in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def write_csv(path: Path, rows: list[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def add_counts(counts: Dict[str, float], pred: int, target: int) -> None:
    counts["tp"] += float(pred == 1 and target == 1)
    counts["fp"] += float(pred == 1 and target == 0)
    counts["fn"] += float(pred == 0 and target == 1)
    counts["tn"] += float(pred == 0 and target == 0)


@torch.no_grad()
def extract_split(
    data_path: str,
    models: Dict[str, Any],
    device: torch.device,
    thresholds: Dict[str, float],
    batch_size: int,
    num_workers: int,
    rescue_relation_max: float,
    rescue_support_min: float,
) -> ExtractedSplit:
    dataset = Step30WeakObservationDataset(data_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=step30_weak_observation_collate_fn,
        pin_memory=device.type == "cuda",
    )

    features: list[np.ndarray] = []
    labels: list[int] = []
    binary_labels: list[int] = []
    scores: Dict[str, list[float]] = {
        "rev6_score": [],
        "rev6_logit": [],
        "rev19_score": [],
        "rev19_residual": [],
        "rev21_score": [],
        "rev21_residual": [],
        "bundle_margin": [],
        "pair_support_hint": [],
    }
    class_counts = {name: 0 for name in CLASS_NAMES}
    outside_noisy_counts: Dict[str, float] = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}
    noisy_sample_count = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        variants = [str(v) for v in batch.get("step30_observation_variant", ["unknown"] * len(batch["target_adj"]))]
        target_adj = batch["target_adj"].float()
        node_mask = batch["node_mask"].float()
        relation_hints = batch["weak_relation_hints"].float()
        pair_support_hints = batch["weak_pair_support_hints"].float()
        signed_witness = batch["weak_signed_pair_witness"].float()
        pair_bundle = batch["weak_pair_evidence_bundle"].float()
        pair_mask = build_pair_mask(node_mask).bool()

        outputs_by_model: Dict[str, Dict[str, torch.Tensor]] = {}
        for model_name, model in models.items():
            outputs_by_model[model_name] = model(
                weak_slot_features=batch["weak_slot_features"],
                weak_relation_hints=relation_hints,
                weak_pair_support_hints=pair_support_hints,
                weak_signed_pair_witness=signed_witness,
                weak_pair_evidence_bundle=pair_bundle,
            )

        batch_size_i = int(target_adj.shape[0])
        for batch_idx in range(batch_size_i):
            if variants[batch_idx] != "noisy":
                continue
            noisy_sample_count += 1
            n = int(node_mask[batch_idx].sum().item())
            threshold = threshold_for_variant(thresholds, variants[batch_idx])
            target_np = target_adj[batch_idx, :n, :n].detach().cpu().numpy()
            relation_np = relation_hints[batch_idx, :n, :n].detach().cpu().numpy()
            support_np = pair_support_hints[batch_idx, :n, :n].detach().cpu().numpy()
            witness_np = signed_witness[batch_idx, :n, :n].detach().cpu().numpy()
            bundle_np = pair_bundle[batch_idx, :n, :n, :].detach().cpu().numpy()
            valid_np = pair_mask[batch_idx, :n, :n].detach().cpu().numpy().astype(bool)

            hint_adj = ((relation_np >= 0.5) & valid_np).astype(np.float32)
            support_adj = ((support_np >= rescue_support_min) & valid_np).astype(np.float32)
            hint_deg = hint_adj.sum(axis=1)
            support_deg = support_adj.sum(axis=1)

            rev6_logits = outputs_by_model["rev6"]["edge_logits"][batch_idx, :n, :n].detach().cpu().numpy()
            rev19_logits = outputs_by_model["rev19"]["edge_logits"][batch_idx, :n, :n].detach().cpu().numpy()
            rev21_logits = outputs_by_model["rev21"]["edge_logits"][batch_idx, :n, :n].detach().cpu().numpy()
            rev6_scores = sigmoid_np(rev6_logits)
            rev19_scores = sigmoid_np(rev19_logits)
            rev21_scores = sigmoid_np(rev21_logits)
            rev6_pred = (rev6_scores >= threshold).astype(np.int64)
            rev19_residual = (
                outputs_by_model["rev19"]["pair_evidence_rescue_residual"][batch_idx, :n, :n]
                .detach()
                .cpu()
                .numpy()
            )
            rev21_residual = (
                outputs_by_model["rev21"]["pair_evidence_rescue_residual"][batch_idx, :n, :n]
                .detach()
                .cpu()
                .numpy()
            )
            node_latents = outputs_by_model["rev6"]["node_latents"][batch_idx, :n, :].detach().cpu().numpy()

            for i in range(n):
                for j in range(i + 1, n):
                    if not bool(valid_np[i, j]):
                        continue
                    relation = float(relation_np[i, j])
                    support = float(support_np[i, j])
                    rescue_eligible = relation < rescue_relation_max and support >= rescue_support_min
                    target = int(target_np[i, j] >= 0.5)
                    if not rescue_eligible:
                        add_counts(outside_noisy_counts, int(rev6_pred[i, j]), target)
                        continue

                    bundle = bundle_np[i, j, :]
                    label = rescue_class_label(target, relation, support, bundle)
                    class_counts[CLASS_NAMES[label]] += 1
                    labels.append(label)
                    binary_labels.append(int(target == 1))

                    h_i = node_latents[i]
                    h_j = node_latents[j]
                    scalar = np.asarray(
                        [
                            relation,
                            support,
                            support - relation,
                            float(witness_np[i, j]),
                            float(bundle[0]),
                            float(bundle[1]),
                            float(bundle[2]),
                            float(bundle[3]),
                            float(bundle[0] - bundle[1]),
                            abs(float(bundle[0] - bundle[1])),
                            float(rev6_scores[i, j]),
                            float(rev6_logits[i, j]),
                            float(rev19_scores[i, j]),
                            float(rev19_residual[i, j]),
                            float(rev21_scores[i, j]),
                            float(rev21_residual[i, j]),
                            float(hint_deg[i] + hint_deg[j]),
                            float(support_deg[i] + support_deg[j]),
                            float(np.minimum(hint_adj[i], hint_adj[j]).sum()),
                            float(np.minimum(support_adj[i], support_adj[j]).sum()),
                        ],
                        dtype=np.float32,
                    )
                    pair_latent = np.concatenate(
                        [
                            h_i + h_j,
                            np.abs(h_i - h_j),
                            h_i * h_j,
                            scalar,
                        ],
                        axis=0,
                    ).astype(np.float32)
                    features.append(pair_latent)
                    scores["rev6_score"].append(float(rev6_scores[i, j]))
                    scores["rev6_logit"].append(float(rev6_logits[i, j]))
                    scores["rev19_score"].append(float(rev19_scores[i, j]))
                    scores["rev19_residual"].append(float(rev19_residual[i, j]))
                    scores["rev21_score"].append(float(rev21_scores[i, j]))
                    scores["rev21_residual"].append(float(rev21_residual[i, j]))
                    scores["bundle_margin"].append(float(bundle[0] - bundle[1]))
                    scores["pair_support_hint"].append(support)

    score_arrays = {name: np.asarray(values, dtype=np.float32) for name, values in scores.items()}
    return ExtractedSplit(
        features=np.stack(features, axis=0).astype(np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        binary_labels=np.asarray(binary_labels, dtype=np.int64),
        scores=score_arrays,
        metadata={
            "data_path": data_path,
            "sample_count": len(dataset),
            "noisy_sample_count": noisy_sample_count,
            "candidate_count": len(labels),
            "class_counts": class_counts,
            "feature_dim": int(features[0].shape[0]) if features else 0,
            "scalar_feature_names": SCALAR_FEATURE_NAMES,
            "endpoint_latent_source": "rev6_node_latents_sum_absdiff_product",
        },
        outside_noisy_counts=outside_noisy_counts,
    )


def standardize_splits(
    train: ExtractedSplit,
    val: ExtractedSplit,
    test: ExtractedSplit,
) -> tuple[ExtractedSplit, ExtractedSplit, ExtractedSplit, np.ndarray, np.ndarray]:
    mean = train.features.mean(axis=0, keepdims=True)
    std = train.features.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)

    def apply(split: ExtractedSplit) -> ExtractedSplit:
        return ExtractedSplit(
            features=((split.features - mean) / std).astype(np.float32),
            labels=split.labels,
            binary_labels=split.binary_labels,
            scores=split.scores,
            metadata=split.metadata,
            outside_noisy_counts=split.outside_noisy_counts,
        )

    return apply(train), apply(val), apply(test), mean.squeeze(0), std.squeeze(0)


def class_weight_tensor(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = np.bincount(labels, minlength=len(CLASS_NAMES)).astype(np.float64)
    weights = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def evaluate_probe_scores(
    model: RescueCandidateProbe,
    split: ExtractedSplit,
    device: torch.device,
    batch_size: int,
) -> Dict[str, Any]:
    dataset = TensorDataset(
        torch.tensor(split.features, dtype=torch.float32),
        torch.tensor(split.labels, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    logits_chunks = []
    latent_chunks = []
    model.eval()
    with torch.no_grad():
        for features, _ in loader:
            features = features.to(device)
            logits, latent = model(features)
            logits_chunks.append(logits.detach().cpu())
            latent_chunks.append(latent.detach().cpu())
    logits_np = torch.cat(logits_chunks, dim=0).numpy()
    probs_np = torch.softmax(torch.tensor(logits_np), dim=-1).numpy()
    latent_np = torch.cat(latent_chunks, dim=0).numpy()
    return {
        "logits": logits_np,
        "probs": probs_np,
        "safe_score": probs_np[:, 0],
        "latent": latent_np,
    }


def train_probe(
    train: ExtractedSplit,
    val: ExtractedSplit,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    latent_dim: int,
    hidden_dim: int,
) -> tuple[RescueCandidateProbe, Dict[str, Any]]:
    model = RescueCandidateProbe(
        input_dim=int(train.features.shape[1]),
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
    ).to(device)
    train_dataset = TensorDataset(
        torch.tensor(train.features, dtype=torch.float32),
        torch.tensor(train.labels, dtype=torch.long),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    weights = class_weight_tensor(train.labels, device)
    best_state: Dict[str, torch.Tensor] | None = None
    best_val_ap = -float("inf")
    best_epoch = -1
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            logits, _ = model(features)
            loss = F.cross_entropy(logits, labels, weight=weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().item())
            batches += 1

        train_scores = evaluate_probe_scores(model, train, device, batch_size=batch_size)
        val_scores = evaluate_probe_scores(model, val, device, batch_size=batch_size)
        train_ap = average_precision(train_scores["safe_score"], train.binary_labels)
        val_ap = average_precision(val_scores["safe_score"], val.binary_labels)
        val_auroc = auroc(val_scores["safe_score"], val.binary_labels)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(batches, 1),
            "train_safe_ap": train_ap,
            "val_safe_ap": val_ap,
            "val_safe_auroc": val_auroc,
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} train_loss={row['train_loss']:.6f} "
            f"train_ap={float(train_ap or 0.0):.4f} "
            f"val_ap={float(val_ap or 0.0):.4f} "
            f"val_auroc={float(val_auroc or 0.0):.4f}"
        )
        if val_ap is not None and val_ap > best_val_ap:
            best_val_ap = float(val_ap)
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "best_val_safe_ap": best_val_ap, "history": history}


def classification_rows(split: ExtractedSplit, safe_scores: Dict[str, np.ndarray]) -> list[Dict[str, Any]]:
    rows = []
    for name, score in safe_scores.items():
        rows.append(
            {
                "scorer": name,
                "safe_ap": average_precision(score, split.binary_labels),
                "safe_auroc": auroc(score, split.binary_labels),
                "safe_score_mean_positive": float(score[split.binary_labels == 1].mean()),
                "safe_score_mean_negative": float(score[split.binary_labels == 0].mean()),
            }
        )
    return rows


def class_breakdown_rows(labels: np.ndarray, pred_labels: np.ndarray) -> list[Dict[str, Any]]:
    rows = []
    for class_idx, class_name in enumerate(CLASS_NAMES):
        tp = float(((pred_labels == class_idx) & (labels == class_idx)).sum())
        fp = float(((pred_labels == class_idx) & (labels != class_idx)).sum())
        fn = float(((pred_labels != class_idx) & (labels == class_idx)).sum())
        rows.append(
            {
                "class": class_name,
                "count": int((labels == class_idx).sum()),
                "pred_count": int((pred_labels == class_idx).sum()),
                **binary_metrics(tp, fp, fn),
            }
        )
    return rows


def confusion_rows(labels: np.ndarray, pred_labels: np.ndarray) -> list[Dict[str, Any]]:
    rows = []
    for true_idx, true_name in enumerate(CLASS_NAMES):
        for pred_idx, pred_name in enumerate(CLASS_NAMES):
            rows.append(
                {
                    "true_class": true_name,
                    "pred_class": pred_name,
                    "count": int(((labels == true_idx) & (pred_labels == pred_idx)).sum()),
                }
            )
    return rows


def budget_specs(split: ExtractedSplit) -> Dict[str, int]:
    n = int(len(split.binary_labels))
    rev19_admitted = int((split.scores["rev19_score"] >= 0.55).sum())
    rev21_admitted = int((split.scores["rev21_score"] >= 0.55).sum())
    rev6_admitted = int((split.scores["rev6_score"] >= 0.55).sum())
    specs = {
        "top_05pct": max(1, int(round(0.05 * n))),
        "top_10pct": max(1, int(round(0.10 * n))),
        "top_20pct": max(1, int(round(0.20 * n))),
        "rev6_current_admitted": max(1, rev6_admitted),
        "rev21_current_admitted": max(1, rev21_admitted),
        "rev19_current_admitted": max(1, rev19_admitted),
    }
    return dict(sorted(specs.items(), key=lambda item: item[1]))


def admission_metrics_for_topk(score: np.ndarray, labels: np.ndarray, budget: int) -> Dict[str, float]:
    budget = min(max(int(budget), 1), len(labels))
    order = np.argsort(-score)
    chosen = np.zeros(len(labels), dtype=bool)
    chosen[order[:budget]] = True
    tp = float((chosen & (labels == 1)).sum())
    fp = float((chosen & (labels == 0)).sum())
    fn = float((~chosen & (labels == 1)).sum())
    return {
        "admitted": float(budget),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "unsafe_admission_rate": fp / max(float((labels == 0).sum()), 1.0),
        "safe_admission_rate": tp / max(float((labels == 1).sum()), 1.0),
        **binary_metrics(tp, fp, fn),
    }


def admission_rows(split: ExtractedSplit, safe_scores: Dict[str, np.ndarray]) -> list[Dict[str, Any]]:
    rows = []
    budgets = budget_specs(split)
    for scorer_name, score in safe_scores.items():
        for budget_name, budget in budgets.items():
            row = {
                "scorer": scorer_name,
                "budget_name": budget_name,
                "budget": budget,
            }
            row.update(admission_metrics_for_topk(score, split.binary_labels, budget))
            rows.append(row)

    # Keep the historical threshold operating points visible.
    for scorer_name, threshold in [("rev19_score_threshold", 0.55), ("rev21_score_threshold", 0.55)]:
        score_name = "rev19_score" if scorer_name.startswith("rev19") else "rev21_score"
        pred = split.scores[score_name] >= threshold
        tp = float((pred & (split.binary_labels == 1)).sum())
        fp = float((pred & (split.binary_labels == 0)).sum())
        fn = float((~pred & (split.binary_labels == 1)).sum())
        row = {
            "scorer": scorer_name,
            "budget_name": "current_threshold",
            "budget": int(pred.sum()),
            "admitted": float(pred.sum()),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "unsafe_admission_rate": fp / max(float((split.binary_labels == 0).sum()), 1.0),
            "safe_admission_rate": tp / max(float((split.binary_labels == 1).sum()), 1.0),
            **binary_metrics(tp, fp, fn),
        }
        rows.append(row)
    return rows


def recovery_simulation_rows(split: ExtractedSplit, safe_scores: Dict[str, np.ndarray]) -> list[Dict[str, Any]]:
    rows = []
    outside = split.outside_noisy_counts
    budgets = budget_specs(split)
    selected_budgets = {
        name: budget
        for name, budget in budgets.items()
        if name in {"rev6_current_admitted", "rev21_current_admitted", "rev19_current_admitted"}
    }

    def combine_row(name: str, pred: np.ndarray) -> Dict[str, Any]:
        tp = float(outside["tp"] + (pred & (split.binary_labels == 1)).sum())
        fp = float(outside["fp"] + (pred & (split.binary_labels == 0)).sum())
        fn = float(outside["fn"] + ((~pred) & (split.binary_labels == 1)).sum())
        return {
            "simulation": name,
            "scope": "noisy_upper_triangle_rev6_outside_rescue",
            "tp": tp,
            "fp": fp,
            "fn": fn,
            **binary_metrics(tp, fp, fn),
        }

    rows.append(combine_row("rev6_current_decode", split.scores["rev6_score"] >= 0.55))
    rows.append(combine_row("rev19_current_decode", split.scores["rev19_score"] >= 0.55))
    rows.append(combine_row("rev21_current_decode", split.scores["rev21_score"] >= 0.55))
    for scorer_name, score in safe_scores.items():
        order = np.argsort(-score)
        for budget_name, budget in selected_budgets.items():
            pred = np.zeros(len(split.binary_labels), dtype=bool)
            pred[order[:budget]] = True
            rows.append(combine_row(f"{scorer_name}_{budget_name}", pred))
    return rows


def save_probe_checkpoint(
    path: Path,
    model: RescueCandidateProbe,
    mean: np.ndarray,
    std: np.ndarray,
    args: argparse.Namespace,
    training_summary: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_mean": mean,
            "feature_std": std,
            "class_names": CLASS_NAMES,
            "scalar_feature_names": SCALAR_FEATURE_NAMES,
            "args": vars(args),
            "training_summary": training_summary,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--rev6_checkpoint", type=str, required=True)
    parser.add_argument("--rev19_checkpoint", type=str, required=True)
    parser.add_argument("--rev21_checkpoint", type=str, required=True)
    parser.add_argument("--thresholds", type=str, default="default:0.5,clean:0.5,noisy:0.55")
    parser.add_argument("--rescue_relation_max", type=float, default=0.5)
    parser.add_argument("--rescue_support_min", type=float, default=0.55)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--probe_batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts/step30_rescue_candidate_latent_probe_rev23",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints/step30_rescue_candidate_latent_probe_rev23/best.pt",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    thresholds = parse_thresholds(args.thresholds)
    models = {
        "rev6": load_model(args.rev6_checkpoint, device),
        "rev19": load_model(args.rev19_checkpoint, device),
        "rev21": load_model(args.rev21_checkpoint, device),
    }

    print("extracting train rescue candidates...")
    train = extract_split(
        args.train_path,
        models,
        device,
        thresholds,
        args.batch_size,
        args.num_workers,
        args.rescue_relation_max,
        args.rescue_support_min,
    )
    print("extracting val rescue candidates...")
    val = extract_split(
        args.val_path,
        models,
        device,
        thresholds,
        args.batch_size,
        args.num_workers,
        args.rescue_relation_max,
        args.rescue_support_min,
    )
    print("extracting test rescue candidates...")
    test = extract_split(
        args.test_path,
        models,
        device,
        thresholds,
        args.batch_size,
        args.num_workers,
        args.rescue_relation_max,
        args.rescue_support_min,
    )
    train_std, val_std, test_std, feature_mean, feature_std = standardize_splits(train, val, test)

    model, training_summary = train_probe(
        train_std,
        val_std,
        device=device,
        epochs=args.epochs,
        batch_size=args.probe_batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
    )
    save_probe_checkpoint(Path(args.checkpoint_path), model, feature_mean, feature_std, args, training_summary)

    val_probe = evaluate_probe_scores(model, val_std, device, batch_size=args.probe_batch_size)
    test_probe = evaluate_probe_scores(model, test_std, device, batch_size=args.probe_batch_size)
    test_pred_labels = test_probe["probs"].argmax(axis=1)
    safe_scores = {
        "rev23_latent_probe": test_probe["safe_score"],
        "rev21_residual": test.scores["rev21_residual"],
        "rev19_residual": test.scores["rev19_residual"],
        "rev6_score": test.scores["rev6_score"],
        "bundle_margin": test.scores["bundle_margin"],
        "pair_support_hint": test.scores["pair_support_hint"],
    }
    classification_table = classification_rows(test, safe_scores)
    class_breakdown_table = class_breakdown_rows(test.labels, test_pred_labels)
    confusion_table = confusion_rows(test.labels, test_pred_labels)
    admission_table = admission_rows(test, safe_scores)
    recovery_sim_table = recovery_simulation_rows(test, safe_scores)
    validation_table = classification_rows(
        val,
        {
            "rev23_latent_probe": val_probe["safe_score"],
            "rev21_residual": val.scores["rev21_residual"],
            "rev19_residual": val.scores["rev19_residual"],
            "rev6_score": val.scores["rev6_score"],
            "bundle_margin": val.scores["bundle_margin"],
        },
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "classification_table.csv", clean_json_numbers(classification_table))
    write_csv(output_dir / "validation_classification_table.csv", clean_json_numbers(validation_table))
    write_csv(output_dir / "class_breakdown_table.csv", class_breakdown_table)
    write_csv(output_dir / "confusion_matrix.csv", confusion_table)
    write_csv(output_dir / "admission_simulation_table.csv", admission_table)
    write_csv(output_dir / "recovery_simulation_table.csv", recovery_sim_table)
    write_csv(output_dir / "training_history.csv", clean_json_numbers(training_summary["history"]))

    summary = clean_json_numbers(
        {
            "metadata": {
                "train": train.metadata,
                "val": val.metadata,
                "test": test.metadata,
                "thresholds": thresholds,
                "rescue_relation_max": args.rescue_relation_max,
                "rescue_support_min": args.rescue_support_min,
                "checkpoint_path": args.checkpoint_path,
                "device": str(device),
            },
            "training_summary": training_summary,
            "classification_table": classification_table,
            "validation_classification_table": validation_table,
            "class_breakdown_table": class_breakdown_table,
            "confusion_matrix": confusion_table,
            "admission_simulation_table": admission_table,
            "recovery_simulation_table": recovery_sim_table,
        }
    )
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary["metadata"], indent=2))
    print(f"wrote rev23 rescue-candidate latent probe artifacts to: {output_dir}")
    print(f"saved probe checkpoint: {args.checkpoint_path}")


if __name__ == "__main__":
    main()
