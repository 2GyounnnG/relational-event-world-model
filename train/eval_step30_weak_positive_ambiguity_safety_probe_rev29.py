from __future__ import annotations

import argparse
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train.eval_step30_encoder_recovery import load_model
from train.eval_step30_rescue_ambiguity_subtype_probe_rev28 import (
    Rev28Rows,
    base_prediction,
    collect_rows,
    edge_metrics_from_arrays,
    parse_thresholds,
    prediction_with_additions,
    rev25_score,
    rev26_score,
    selected_additions,
    top20_budget,
    write_csv,
)
from train.eval_step30_rescue_candidate_latent_probe_rev23 import (
    average_precision,
    auroc,
    binary_metrics,
    clean_json_numbers,
)


FEATURE_NAMES = [
    "relation_hint",
    "pair_support_hint",
    "support_minus_relation",
    "signed_pair_witness",
    "bundle_positive",
    "bundle_warning",
    "bundle_corroboration",
    "bundle_endpoint_compat",
    "bundle_margin_positive_minus_warning",
    "bundle_total_evidence",
    "bundle_abs_margin",
    "rev6_score",
    "rev6_logit",
    "rev24_safe_prob",
    "rev24_false_prob",
    "rev24_ambiguous_prob",
    "rev24_safe_minus_reject",
    "rev26_binary_prob",
    "rev26_ambiguous_prob",
    "rev26_binary_minus_ambiguous",
]


@dataclass
class ProbeFit:
    model: nn.Linear
    mean: np.ndarray
    std: np.ndarray
    best_epoch: int
    val_ap: float | None
    val_auroc: float | None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def logit_np(prob: np.ndarray) -> np.ndarray:
    clipped = np.clip(prob, 1e-5, 1.0 - 1e-5)
    return np.log(clipped / (1.0 - clipped))


def weak_positive_mask(rows: Rev28Rows) -> np.ndarray:
    return rows.is_ambiguous_signal & (rows.subtype == "weak_positive_ambiguous")


def feature_matrix(rows: Rev28Rows) -> np.ndarray:
    rev24_margin = rows.rev24_safe_prob - np.maximum(rows.rev24_false_prob, rows.rev24_ambiguous_prob)
    rev26_margin = rev26_score(rows)
    features = np.stack(
        [
            rows.relation,
            rows.support,
            rows.support - rows.relation,
            rows.signed_witness,
            rows.bundle_positive,
            rows.bundle_warning,
            rows.bundle_corroboration,
            rows.bundle_endpoint_compat,
            rows.bundle_positive - rows.bundle_warning,
            rows.bundle_positive + rows.bundle_warning,
            np.abs(rows.bundle_positive - rows.bundle_warning),
            rows.rev6_score,
            logit_np(rows.rev6_score),
            rows.rev24_safe_prob,
            rows.rev24_false_prob,
            rows.rev24_ambiguous_prob,
            rev24_margin,
            rows.rev26_binary_prob,
            rows.rev26_ambiguous_prob,
            rev26_margin,
        ],
        axis=1,
    )
    return features.astype(np.float32)


def standardize(
    train_x: np.ndarray,
    *splits: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    train_z = (train_x - mean) / std
    split_z = [(x - mean) / std for x in splits]
    return train_z.astype(np.float32), [x.astype(np.float32) for x in split_z], mean, std


def fit_logistic_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> ProbeFit:
    set_seed(seed)
    train_z, [val_z], mean, std = standardize(train_x, val_x)
    model = nn.Linear(train_z.shape[1], 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    x_t = torch.tensor(train_z, dtype=torch.float32)
    y_t = torch.tensor(train_y.astype(np.float32), dtype=torch.float32)
    val_t = torch.tensor(val_z, dtype=torch.float32)
    pos = float((train_y == 1).sum())
    neg = float((train_y == 0).sum())
    pos_weight = torch.tensor(neg / max(pos, 1.0), dtype=torch.float32)
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_epoch = 0
    best_ap = -float("inf")
    best_auc: float | None = None
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x_t).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y_t, pos_weight=pos_weight)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_score = torch.sigmoid(model(val_t).squeeze(-1)).cpu().numpy()
        val_ap = average_precision(val_score, val_y)
        val_auc = auroc(val_score, val_y)
        score_for_selection = -float("inf") if val_ap is None else float(val_ap)
        if score_for_selection > best_ap:
            best_ap = score_for_selection
            best_auc = val_auc
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return ProbeFit(
        model=model,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        best_epoch=best_epoch,
        val_ap=None if best_ap == -float("inf") else float(best_ap),
        val_auroc=best_auc,
    )


def probe_scores(fit: ProbeFit, x: np.ndarray) -> np.ndarray:
    z = ((x - fit.mean) / fit.std).astype(np.float32)
    with torch.no_grad():
        score = torch.sigmoid(fit.model(torch.tensor(z)).squeeze(-1)).cpu().numpy()
    return score.astype(np.float32)


def precision_at_budget(score: np.ndarray, labels: np.ndarray, budget: int) -> Dict[str, float | int]:
    order = np.argsort(-score)
    k = min(max(int(budget), 1), len(order))
    selected = np.zeros(len(labels), dtype=bool)
    selected[order[:k]] = True
    tp = float((selected & (labels == 1)).sum())
    fp = float((selected & (labels == 0)).sum())
    fn = float((~selected & (labels == 1)).sum())
    out = binary_metrics(tp, fp, fn)
    return {
        "precision_budget": int(k),
        "precision_at_budget": out["precision"],
        "budget_recall": out["recall"],
        "budget_f1": out["f1"],
    }


def score_distribution_rows(
    score: np.ndarray,
    labels: np.ndarray,
    row_name: str,
) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for label, label_name in [(1, "gt_positive"), (0, "gt_negative")]:
        values = score[labels == label]
        row: Dict[str, Any] = {
            "row": row_name,
            "class": label_name,
            "count": int(len(values)),
        }
        if len(values) > 0:
            for q in [0.10, 0.25, 0.50, 0.75, 0.90]:
                row[f"q{int(q * 100):02d}"] = float(np.quantile(values, q))
            row["mean"] = float(values.mean())
        rows.append(row)
    return rows


def scorer_quality_row(
    score: np.ndarray,
    labels: np.ndarray,
    row_name: str,
    budget: int,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "row": row_name,
        "candidate_count": int(len(labels)),
        "positive_count": int((labels == 1).sum()),
        "negative_count": int((labels == 0).sum()),
        "ap": average_precision(score, labels),
        "auroc": auroc(score, labels),
        "score_mean_positive": float(score[labels == 1].mean()) if (labels == 1).any() else None,
        "score_mean_negative": float(score[labels == 0].mean()) if (labels == 0).any() else None,
    }
    row.update(precision_at_budget(score, labels, budget))
    return row


def one_feature_rows(
    test_x: np.ndarray,
    test_y: np.ndarray,
) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for idx, name in enumerate(FEATURE_NAMES):
        score = test_x[:, idx]
        forward_ap = average_precision(score, test_y)
        reverse_ap = average_precision(-score, test_y)
        if reverse_ap is not None and forward_ap is not None and reverse_ap > forward_ap:
            score = -score
            direction = "negative"
            ap = reverse_ap
        else:
            direction = "positive"
            ap = forward_ap
        rows.append(
            {
                "feature": name,
                "best_direction": direction,
                "ap": ap,
                "auroc": auroc(score, test_y),
                "positive_mean": float(test_x[test_y == 1, idx].mean()) if (test_y == 1).any() else None,
                "negative_mean": float(test_x[test_y == 0, idx].mean()) if (test_y == 0).any() else None,
            }
        )
    rows.sort(key=lambda row: -float(row["ap"] or 0.0))
    return rows


def coefficient_rows(fit: ProbeFit) -> list[Dict[str, Any]]:
    weights = fit.model.weight.detach().cpu().numpy().reshape(-1)
    rows = [
        {
            "feature": name,
            "coefficient": float(weights[idx]),
            "abs_coefficient": float(abs(weights[idx])),
        }
        for idx, name in enumerate(FEATURE_NAMES)
    ]
    rows.sort(key=lambda row: -float(row["abs_coefficient"]))
    return rows


def build_full_score(rows: Rev28Rows, mask: np.ndarray, local_score: np.ndarray) -> np.ndarray:
    full = np.full(rows.target.shape, -np.inf, dtype=np.float32)
    full[np.flatnonzero(mask)] = local_score
    return full


def replace_weak_positive_admissions(
    rows: Rev28Rows,
    base_additions: np.ndarray,
    local_score: np.ndarray,
    thresholds: Dict[str, float],
    keep_count: int,
) -> np.ndarray:
    weak_mask = weak_positive_mask(rows)
    base_pred = base_prediction(rows, thresholds)
    eligible = np.flatnonzero(weak_mask & (base_pred == 0))
    additions = base_additions.copy()
    additions[weak_mask] = False
    if keep_count <= 0 or len(eligible) == 0:
        return additions
    score_by_row = build_full_score(rows, weak_mask, local_score)
    order = np.argsort(-score_by_row[eligible])
    chosen = eligible[order[: min(int(keep_count), len(order))]]
    additions[chosen] = True
    return additions


def counterfactual_row(
    rows: Rev28Rows,
    additions: np.ndarray,
    thresholds: Dict[str, float],
    row_name: str,
) -> Dict[str, Any]:
    pred = prediction_with_additions(rows, additions, thresholds)
    noisy = rows.variant == "noisy"
    weak_mask = weak_positive_mask(rows)
    selected_weak = additions[weak_mask].astype(np.int64)
    weak_target = rows.target[weak_mask]
    weak_metrics = edge_metrics_from_arrays(selected_weak, weak_target)
    global_metrics = edge_metrics_from_arrays(pred[noisy], rows.target[noisy])
    return {
        "row": row_name,
        "selected_total": int(additions.sum()),
        "weak_positive_selected": int(selected_weak.sum()),
        "weak_positive_tp": int(((selected_weak == 1) & (weak_target == 1)).sum()),
        "weak_positive_fp": int(((selected_weak == 1) & (weak_target == 0)).sum()),
        "weak_positive_fn": int(((selected_weak == 0) & (weak_target == 1)).sum()),
        "weak_positive_precision": weak_metrics["precision"],
        "weak_positive_recall": weak_metrics["recall"],
        "weak_positive_f1": weak_metrics["f1"],
        "noisy_precision": global_metrics["precision"],
        "noisy_recall": global_metrics["recall"],
        "noisy_f1": global_metrics["f1"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="data/graph_event_step30_weak_obs_rev17_train.pkl")
    parser.add_argument("--val_path", default="data/graph_event_step30_weak_obs_rev17_val.pkl")
    parser.add_argument("--test_path", default="data/graph_event_step30_weak_obs_rev17_test.pkl")
    parser.add_argument("--rev6_checkpoint", default="checkpoints/step30_encoder_recovery_rev6/best.pt")
    parser.add_argument("--rev24_checkpoint", default="checkpoints/step30_encoder_recovery_rev24/best.pt")
    parser.add_argument("--rev26_checkpoint", default="checkpoints/step30_encoder_recovery_rev26/best.pt")
    parser.add_argument("--output_dir", default="artifacts/step30_weak_positive_ambiguity_safety_probe_rev29")
    parser.add_argument("--thresholds", default="clean:0.50,noisy:0.55")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rescue_relation_max", type=float, default=0.50)
    parser.add_argument("--rescue_support_min", type=float, default=0.55)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=129)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    thresholds = parse_thresholds(args.thresholds)
    models = {
        "rev6": load_model(args.rev6_checkpoint, device),
        "rev24": load_model(args.rev24_checkpoint, device),
        "rev26": load_model(args.rev26_checkpoint, device),
    }
    collect_kwargs = {
        "models": models,
        "device": device,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "rescue_relation_max": args.rescue_relation_max,
        "rescue_support_min": args.rescue_support_min,
    }
    train_rows = collect_rows(args.train_path, **collect_kwargs)
    val_rows = collect_rows(args.val_path, **collect_kwargs)
    test_rows = collect_rows(args.test_path, **collect_kwargs)

    train_mask = weak_positive_mask(train_rows)
    val_mask = weak_positive_mask(val_rows)
    test_mask = weak_positive_mask(test_rows)
    train_x = feature_matrix(train_rows)[train_mask]
    val_x = feature_matrix(val_rows)[val_mask]
    test_x = feature_matrix(test_rows)[test_mask]
    train_y = train_rows.target[train_mask].astype(np.int64)
    val_y = val_rows.target[val_mask].astype(np.int64)
    test_y = test_rows.target[test_mask].astype(np.int64)

    fit = fit_logistic_probe(
        train_x,
        train_y,
        val_x,
        val_y,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )
    probe_test = probe_scores(fit, test_x)
    rev26_add = selected_additions(
        test_rows,
        rev26_score(test_rows),
        thresholds,
        top20_budget(test_rows),
    )
    weak_selected_count = int((rev26_add & test_mask).sum())
    half_count = max(1, int(round(0.50 * weak_selected_count)))

    scorers = {
        "rev6_score": test_rows.rev6_score[test_mask],
        "rev24_safe_logit": test_rows.rev24_safe_logit[test_mask],
        "rev25_class_margin": rev25_score(test_rows)[test_mask],
        "rev26_binary_minus_ambiguous": rev26_score(test_rows)[test_mask],
        "rev29_logistic_probe": probe_test,
    }
    quality_rows = [
        scorer_quality_row(score, test_y, name, weak_selected_count)
        for name, score in scorers.items()
    ]
    distribution_rows: list[Dict[str, Any]] = []
    for name, score in scorers.items():
        distribution_rows.extend(score_distribution_rows(score, test_y, name))

    counterfactual_rows = [
        counterfactual_row(test_rows, rev26_add, thresholds, "rev26_baseline"),
    ]
    probe_same = replace_weak_positive_admissions(
        test_rows,
        rev26_add,
        probe_test,
        thresholds,
        weak_selected_count,
    )
    counterfactual_rows.append(
        counterfactual_row(
            test_rows,
            probe_same,
            thresholds,
            "rev29_probe_replace_same_weak_positive_count",
        )
    )
    probe_half = replace_weak_positive_admissions(
        test_rows,
        rev26_add,
        probe_test,
        thresholds,
        half_count,
    )
    counterfactual_rows.append(
        counterfactual_row(
            test_rows,
            probe_half,
            thresholds,
            "rev29_probe_replace_half_weak_positive_count",
        )
    )
    suppress = rev26_add.copy()
    suppress[test_mask] = False
    counterfactual_rows.append(
        counterfactual_row(
            test_rows,
            suppress,
            thresholds,
            "suppress_all_weak_positive_ambiguous",
        )
    )

    feature_rows = one_feature_rows(test_x, test_y)
    coef_rows = coefficient_rows(fit)
    summary = {
        "step": "step30_rev29_weak_positive_ambiguity_safety_probe",
        "backend_rerun": False,
        "backend_rerun_reason": "rev29 is offline diagnostic-only and does not trigger Step30c.",
        "scope": "weak_positive_ambiguous only",
        "probe": {
            "type": "standardized logistic regression",
            "best_epoch": fit.best_epoch,
            "val_ap": fit.val_ap,
            "val_auroc": fit.val_auroc,
            "train_count": int(len(train_y)),
            "train_positive_rate": float(train_y.mean()),
            "val_count": int(len(val_y)),
            "val_positive_rate": float(val_y.mean()),
            "test_count": int(len(test_y)),
            "test_positive_rate": float(test_y.mean()),
            "rev26_weak_positive_selected_count": weak_selected_count,
        },
        "quality_summary": quality_rows,
        "score_distributions": distribution_rows,
        "feature_ranking": feature_rows,
        "logistic_coefficients": coef_rows,
        "counterfactual_summary": counterfactual_rows,
        "args": vars(args),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "probe_quality_summary.csv", quality_rows)
    write_csv(output_dir / "score_distributions.csv", distribution_rows)
    write_csv(output_dir / "feature_ranking.csv", feature_rows)
    write_csv(output_dir / "logistic_coefficients.csv", coef_rows)
    write_csv(output_dir / "counterfactual_summary.csv", counterfactual_rows)
    with open(output_dir / "summary.json", "w") as f:
        json.dump(clean_json_numbers(summary), f, indent=2)
    print(json.dumps(clean_json_numbers(summary), indent=2))


if __name__ == "__main__":
    main()
