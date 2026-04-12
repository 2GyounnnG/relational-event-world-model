from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.step30_dataset import Step30WeakObservationDataset, step30_weak_observation_collate_fn
from train.eval_step30_encoder_recovery import (
    load_model,
    logit_from_prob,
    move_batch_to_device,
    threshold_for_variant,
)
from train.utils_step30_decode import (
    hard_adj_selective_rescue,
    load_rich_rescue_scorer,
    load_rescue_safety_scorer,
    threshold_tensor_for_variants,
)


HINT_BUCKETS = [
    ("very_low_hint", 0.0, 0.30),
    ("low_mid_hint", 0.30, 0.45),
    ("high_mid_hint", 0.45, 0.60),
    ("very_high_hint", 0.60, 1.000001),
]


def get_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if has_mps:
        return torch.device("mps")
    return torch.device("cpu")


def event_families(events_item: Any) -> list[str]:
    if not isinstance(events_item, list):
        return ["unknown"]
    families = sorted(
        {
            str(event.get("event_type", "unknown"))
            for event in events_item
            if isinstance(event, dict)
        }
    )
    return families or ["unknown"]


def hint_bucket(value: float) -> str:
    for name, lo, hi in HINT_BUCKETS:
        if lo <= value < hi:
            return name
    return "out_of_range"


def error_subtype(hint: float, target: int) -> str:
    high = hint >= 0.5
    ambiguous = 0.45 <= hint < 0.60
    if high and target == 0:
        return "hint_supported_false_positive"
    if not high and target == 1:
        return "hint_missed_true_edge"
    if ambiguous:
        return "ambiguous_mid_hint"
    return "easy_hint_agree"


def common_neighbor_bucket(value: float) -> str:
    if value < 0.5:
        return "common_neighbors_0"
    if value < 1.5:
        return "common_neighbors_1"
    return "common_neighbors_2plus"


def empty_binary_sums() -> Dict[str, float]:
    return {
        "count": 0.0,
        "target_pos": 0.0,
        "target_neg": 0.0,
        "pred_pos": 0.0,
        "tp": 0.0,
        "fp": 0.0,
        "fn": 0.0,
        "tn": 0.0,
        "wrong": 0.0,
        "score_sum": 0.0,
        "logit_sum": 0.0,
        "hint_sum": 0.0,
        "hint_degree_sum": 0.0,
        "hint_common_neighbor_sum": 0.0,
        "gt_degree_sum": 0.0,
        "gt_common_neighbor_sum": 0.0,
    }


def add_binary(
    sums: Dict[str, float],
    target: int,
    pred: int,
    score: float,
    logit: float,
    hint: float,
    hint_degree_sum: float,
    hint_common_neighbors: float,
    gt_degree_sum: float,
    gt_common_neighbors: float,
) -> None:
    sums["count"] += 1.0
    sums["target_pos"] += float(target == 1)
    sums["target_neg"] += float(target == 0)
    sums["pred_pos"] += float(pred == 1)
    sums["tp"] += float(pred == 1 and target == 1)
    sums["fp"] += float(pred == 1 and target == 0)
    sums["fn"] += float(pred == 0 and target == 1)
    sums["tn"] += float(pred == 0 and target == 0)
    sums["wrong"] += float(pred != target)
    sums["score_sum"] += score
    sums["logit_sum"] += logit
    sums["hint_sum"] += hint
    sums["hint_degree_sum"] += hint_degree_sum
    sums["hint_common_neighbor_sum"] += hint_common_neighbors
    sums["gt_degree_sum"] += gt_degree_sum
    sums["gt_common_neighbor_sum"] += gt_common_neighbors


def finalize_binary(sums: Dict[str, float]) -> Dict[str, float]:
    count = max(sums["count"], 1.0)
    tp = sums["tp"]
    fp = sums["fp"]
    fn = sums["fn"]
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        **sums,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "error_rate": sums["wrong"] / count,
        "avg_score": sums["score_sum"] / count,
        "avg_logit": sums["logit_sum"] / count,
        "avg_hint": sums["hint_sum"] / count,
        "avg_hint_degree_sum": sums["hint_degree_sum"] / count,
        "avg_hint_common_neighbors": sums["hint_common_neighbor_sum"] / count,
        "avg_gt_degree_sum": sums["gt_degree_sum"] / count,
        "avg_gt_common_neighbors": sums["gt_common_neighbor_sum"] / count,
    }


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


def pairwise_win_rate(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float | None:
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return None
    neg_sorted = np.sort(neg_scores)
    wins = np.searchsorted(neg_sorted, pos_scores, side="left").sum()
    ties = (
        np.searchsorted(neg_sorted, pos_scores, side="right")
        - np.searchsorted(neg_sorted, pos_scores, side="left")
    ).sum()
    return float((wins + 0.5 * ties) / (len(pos_scores) * len(neg_scores)))


def rows_from_table(table: Dict[tuple, Dict[str, float]], names: list[str]) -> list[Dict[str, Any]]:
    rows = []
    for key, sums in sorted(table.items()):
        row = {name: value for name, value in zip(names, key)}
        row.update(finalize_binary(sums))
        rows.append(row)
    return rows


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


def parse_thresholds(value: str | None) -> Any:
    if value is None:
        return 0.5
    value = value.strip()
    if "," not in value and ":" not in value:
        return float(value)
    out: Dict[str, float] = {"default": 0.5}
    for part in value.split(","):
        if not part:
            continue
        key, raw_val = part.split(":", 1)
        out[key.strip()] = float(raw_val)
    return out


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--rev2_checkpoint", type=str, default=None)
    parser.add_argument("--rev4_checkpoint", type=str, default=None)
    parser.add_argument("--rev6_checkpoint", type=str, default=None)
    parser.add_argument("--rev2_thresholds", type=str, default="default:0.5,clean:0.55,noisy:0.45")
    parser.add_argument("--rev4_thresholds", type=str, default="default:0.5,clean:0.4,noisy:0.4")
    parser.add_argument("--rev6_thresholds", type=str, default="default:0.5,clean:0.5,noisy:0.5")
    parser.add_argument("--trivial_thresholds", type=str, default="0.5")
    parser.add_argument("--rev6_decode_mode", choices=["threshold", "selective_rescue"], default="threshold")
    parser.add_argument("--rescue_variants", type=str, default="noisy")
    parser.add_argument("--rescue_relation_max", type=float, default=0.5)
    parser.add_argument("--rescue_support_min", type=float, default=0.55)
    parser.add_argument("--rescue_budget_fraction", type=float, default=0.0)
    parser.add_argument("--rescue_score_mode", choices=["raw", "guarded", "learned", "rich_learned", "aux"], default="raw")
    parser.add_argument("--rescue_support_weight", type=float, default=0.5)
    parser.add_argument("--rescue_relation_weight", type=float, default=0.25)
    parser.add_argument("--rescue_safety_scorer_path", type=str, default=None)
    parser.add_argument("--rich_rescue_scorer_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="artifacts/step30_edge_diagnostics_rev5")
    args = parser.parse_args()

    device = get_device(args.device)
    rescue_safety_scorer = load_rescue_safety_scorer(args.rescue_safety_scorer_path)
    rich_rescue_scorer = load_rich_rescue_scorer(args.rich_rescue_scorer_path)
    dataset = Step30WeakObservationDataset(args.data_path)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=step30_weak_observation_collate_fn,
        pin_memory=device.type == "cuda",
    )

    models: Dict[str, Any] = {"trivial": None}
    thresholds: Dict[str, Any] = {"trivial": parse_thresholds(args.trivial_thresholds)}
    if args.rev2_checkpoint is not None:
        models["rev2"] = load_model(args.rev2_checkpoint, device)
        thresholds["rev2"] = parse_thresholds(args.rev2_thresholds)
    if args.rev4_checkpoint is not None:
        models["rev4"] = load_model(args.rev4_checkpoint, device)
        thresholds["rev4"] = parse_thresholds(args.rev4_thresholds)
    if args.rev6_checkpoint is not None:
        models["rev6"] = load_model(args.rev6_checkpoint, device)
        thresholds["rev6"] = parse_thresholds(args.rev6_thresholds)

    bucket_table: Dict[tuple, Dict[str, float]] = defaultdict(empty_binary_sums)
    subtype_table: Dict[tuple, Dict[str, float]] = defaultdict(empty_binary_sums)
    local_table: Dict[tuple, Dict[str, float]] = defaultdict(empty_binary_sums)
    event_subtype_table: Dict[tuple, Dict[str, float]] = defaultdict(empty_binary_sums)
    ranking_store: Dict[tuple, Dict[str, list[float]]] = defaultdict(lambda: {"scores": [], "labels": [], "hints": []})

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        variants = [str(v) for v in batch.get("step30_observation_variant", ["unknown"] * len(batch["target_adj"]))]
        events_list = batch.get("events", [None] * int(batch["target_adj"].shape[0]))
        target_adj = batch["target_adj"].float()
        relation_hints = batch["weak_relation_hints"].float()
        pair_support_hints = batch.get("weak_pair_support_hints")
        if pair_support_hints is not None:
            pair_support_hints = pair_support_hints.float()
        signed_pair_witness = batch.get("weak_signed_pair_witness")
        if signed_pair_witness is not None:
            signed_pair_witness = signed_pair_witness.float()
        pair_evidence_bundle = batch.get("weak_pair_evidence_bundle")
        if pair_evidence_bundle is not None:
            pair_evidence_bundle = pair_evidence_bundle.float()
        node_mask = batch["node_mask"].float()

        trivial_hints = relation_hints
        if pair_support_hints is not None:
            trivial_hints = 0.5 * (relation_hints + pair_support_hints)
        if signed_pair_witness is not None:
            witness_hint = (0.5 + 0.5 * signed_pair_witness).clamp(0.0, 1.0)
            trivial_hints = 0.5 * (trivial_hints + witness_hint)
        if pair_evidence_bundle is not None:
            bundle_hint = (
                0.35 * pair_evidence_bundle[..., 0]
                + 0.30 * (1.0 - pair_evidence_bundle[..., 1])
                + 0.20 * pair_evidence_bundle[..., 2]
                + 0.15 * pair_evidence_bundle[..., 3]
            ).clamp(0.0, 1.0)
            trivial_hints = 0.5 * trivial_hints + 0.5 * bundle_hint
        logits_by_model: Dict[str, torch.Tensor] = {
            "trivial": logit_from_prob(trivial_hints),
        }
        latents_by_model: Dict[str, torch.Tensor] = {}
        aux_logits_by_model: Dict[str, torch.Tensor] = {}
        for model_name, model in models.items():
            if model is None:
                continue
            outputs = model(
                weak_slot_features=batch["weak_slot_features"],
                weak_relation_hints=relation_hints,
                weak_pair_support_hints=pair_support_hints,
                weak_signed_pair_witness=signed_pair_witness,
                weak_pair_evidence_bundle=pair_evidence_bundle,
            )
            logits_by_model[model_name] = outputs["edge_logits"].detach()
            if "node_latents" in outputs:
                latents_by_model[model_name] = outputs["node_latents"].detach()
            if "rescue_safety_logits" in outputs:
                aux_logits_by_model[model_name] = outputs["rescue_safety_logits"].detach()

        pred_adj_by_model: Dict[str, torch.Tensor] = {}
        if args.rev6_decode_mode == "selective_rescue" and "rev6" in logits_by_model:
            base_threshold = threshold_tensor_for_variants(
                variants=variants,
                default_threshold=0.5,
                thresholds_by_variant=thresholds["rev6"] if isinstance(thresholds["rev6"], dict) else None,
                device=target_adj.device,
            )
            pred_adj_by_model["rev6"] = hard_adj_selective_rescue(
                edge_logits=logits_by_model["rev6"],
                relation_hints=relation_hints,
                pair_support_hints=pair_support_hints,
                node_mask=node_mask,
                variants=variants,
                base_threshold=base_threshold,
                rescue_variants={part.strip() for part in args.rescue_variants.split(",") if part.strip()},
                rescue_relation_max=args.rescue_relation_max,
                rescue_support_min=args.rescue_support_min,
                rescue_budget_fraction=args.rescue_budget_fraction,
                rescue_score_mode=args.rescue_score_mode,
                rescue_support_weight=args.rescue_support_weight,
                rescue_relation_weight=args.rescue_relation_weight,
                rescue_safety_scorer=rescue_safety_scorer,
                node_latents=latents_by_model.get("rev6"),
                rescue_aux_logits=aux_logits_by_model.get("rev6"),
                rich_rescue_scorer=rich_rescue_scorer,
            )

        batch_size, num_nodes, _ = target_adj.shape
        for batch_idx in range(batch_size):
            n = int(node_mask[batch_idx].sum().item())
            variant = variants[batch_idx]
            families = event_families(events_list[batch_idx])
            adj_np = target_adj[batch_idx, :n, :n].detach().cpu().numpy()
            hint_np = relation_hints[batch_idx, :n, :n].detach().cpu().numpy()
            hint_binary = (hint_np >= 0.5).astype(np.float32)
            gt_degree = adj_np.sum(axis=1)
            hint_degree = hint_binary.sum(axis=1)
            gt_common = adj_np @ adj_np
            hint_common = hint_binary @ hint_binary

            for i in range(n):
                for j in range(i + 1, n):
                    target = int(adj_np[i, j] >= 0.5)
                    hint = float(hint_np[i, j])
                    bucket = hint_bucket(hint)
                    subtype = error_subtype(hint, target)
                    common_bucket = common_neighbor_bucket(float(hint_common[i, j]))
                    hint_degree_sum = float(hint_degree[i] + hint_degree[j])
                    gt_degree_sum = float(gt_degree[i] + gt_degree[j])
                    hint_common_neighbors = float(hint_common[i, j])
                    gt_common_neighbors = float(gt_common[i, j])

                    for model_name, logits in logits_by_model.items():
                        logit = float(logits[batch_idx, i, j].detach().cpu().item())
                        score = float(1.0 / (1.0 + math.exp(-logit)))
                        if model_name in pred_adj_by_model:
                            pred = int(pred_adj_by_model[model_name][batch_idx, i, j].item() >= 0.5)
                        else:
                            threshold = threshold_for_variant(thresholds[model_name], variant)
                            pred = int(score >= threshold)

                        for table, key in [
                            (bucket_table, (model_name, variant, bucket)),
                            (subtype_table, (model_name, variant, subtype)),
                            (local_table, (model_name, variant, subtype, common_bucket)),
                        ]:
                            add_binary(
                                table[key],
                                target=target,
                                pred=pred,
                                score=score,
                                logit=logit,
                                hint=hint,
                                hint_degree_sum=hint_degree_sum,
                                hint_common_neighbors=hint_common_neighbors,
                                gt_degree_sum=gt_degree_sum,
                                gt_common_neighbors=gt_common_neighbors,
                            )
                        for family in families:
                            add_binary(
                                event_subtype_table[(model_name, variant, family, subtype)],
                                target=target,
                                pred=pred,
                                score=score,
                                logit=logit,
                                hint=hint,
                                hint_degree_sum=hint_degree_sum,
                                hint_common_neighbors=hint_common_neighbors,
                                gt_degree_sum=gt_degree_sum,
                                gt_common_neighbors=gt_common_neighbors,
                            )

                        for ranking_variant in [variant, "all"]:
                            store = ranking_store[(model_name, ranking_variant)]
                            store["scores"].append(score)
                            store["labels"].append(float(target))
                            store["hints"].append(hint)

    ranking_rows: list[Dict[str, Any]] = []
    for (model_name, variant), values in sorted(ranking_store.items()):
        scores = np.asarray(values["scores"], dtype=np.float64)
        labels = np.asarray(values["labels"], dtype=np.int64)
        hints = np.asarray(values["hints"], dtype=np.float64)
        hard_neg = (labels == 0) & (hints >= 0.5)
        hint_missed_pos = (labels == 1) & (hints < 0.5)
        positives = labels == 1
        ranking_rows.append(
            {
                "model": model_name,
                "variant": variant,
                "pair_count": int(len(scores)),
                "positive_count": int(positives.sum()),
                "hard_negative_count": int(hard_neg.sum()),
                "hint_missed_positive_count": int(hint_missed_pos.sum()),
                "edge_auroc": auroc(scores, labels),
                "edge_ap": average_precision(scores, labels),
                "positive_vs_hard_negative_win_rate": pairwise_win_rate(scores[positives], scores[hard_neg]),
                "hint_missed_positive_vs_hard_negative_win_rate": pairwise_win_rate(
                    scores[hint_missed_pos],
                    scores[hard_neg],
                ),
                "avg_positive_score": float(scores[positives].mean()) if positives.any() else None,
                "avg_hard_negative_score": float(scores[hard_neg].mean()) if hard_neg.any() else None,
                "avg_hint_missed_positive_score": (
                    float(scores[hint_missed_pos].mean()) if hint_missed_pos.any() else None
                ),
            }
        )

    bucket_rows = rows_from_table(bucket_table, ["model", "variant", "hint_bucket"])
    subtype_rows = rows_from_table(subtype_table, ["model", "variant", "subtype"])
    local_rows = rows_from_table(local_table, ["model", "variant", "subtype", "hint_common_neighbor_bucket"])
    event_rows = rows_from_table(event_subtype_table, ["model", "variant", "event_family", "subtype"])

    result = clean_json_numbers(
        {
            "metadata": {
                "data_path": args.data_path,
                "num_samples": len(dataset),
                "thresholds": thresholds,
                "rev6_decode_mode": args.rev6_decode_mode,
                "rescue_variants": [part.strip() for part in args.rescue_variants.split(",") if part.strip()],
                "rescue_relation_max": args.rescue_relation_max,
                "rescue_support_min": args.rescue_support_min,
                "rescue_budget_fraction": args.rescue_budget_fraction,
                "rescue_score_mode": args.rescue_score_mode,
                "rescue_support_weight": args.rescue_support_weight,
                "rescue_relation_weight": args.rescue_relation_weight,
                "rescue_safety_scorer_path": args.rescue_safety_scorer_path,
                "rich_rescue_scorer_path": args.rich_rescue_scorer_path,
                "hint_buckets": HINT_BUCKETS,
                "pair_counting": "upper_triangle_undirected_pairs",
            },
            "bucket_rows": bucket_rows,
            "subtype_rows": subtype_rows,
            "ranking_rows": ranking_rows,
            "local_neighborhood_rows": local_rows,
            "event_subtype_rows": event_rows,
        }
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "summary.json", "w") as f:
        json.dump(result, f, indent=2)
    write_csv(output_dir / "bucket_table.csv", bucket_rows)
    write_csv(output_dir / "subtype_table.csv", subtype_rows)
    write_csv(output_dir / "ranking_table.csv", clean_json_numbers(ranking_rows))
    write_csv(output_dir / "local_neighborhood_table.csv", local_rows)
    write_csv(output_dir / "event_subtype_table.csv", event_rows)

    summary_rows = []
    for row in ranking_rows:
        summary_rows.append({"table": "ranking", **row})
    write_csv(output_dir / "summary.csv", clean_json_numbers(summary_rows))
    print(json.dumps(result["metadata"], indent=2))
    print(f"wrote diagnostics to: {output_dir}")


if __name__ == "__main__":
    main()
