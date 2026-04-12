from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.step30_dataset import Step30WeakObservationDataset, step30_weak_observation_collate_fn
from models.encoder_step30 import build_pair_mask
from train.eval_step30_encoder_recovery import (
    load_model,
    move_batch_to_device,
    threshold_for_variant,
)


CHANNEL_NAMES = [
    "positive_support",
    "false_admission_warning",
    "corroboration",
    "endpoint_compatibility",
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


def quantiles(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {
            "mean": 0.0,
            "q10": 0.0,
            "q25": 0.0,
            "q50": 0.0,
            "q75": 0.0,
            "q90": 0.0,
        }
    return {
        "mean": float(arr.mean()),
        "q10": float(np.quantile(arr, 0.10)),
        "q25": float(np.quantile(arr, 0.25)),
        "q50": float(np.quantile(arr, 0.50)),
        "q75": float(np.quantile(arr, 0.75)),
        "q90": float(np.quantile(arr, 0.90)),
    }


def binary_metrics(tp: float, fp: float, fn: float) -> Dict[str, float]:
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def safe_div(num: float, denom: float) -> float:
    return num / denom if denom > 0 else 0.0


def rescue_subtype(
    target: int,
    relation: float,
    support: float,
    bundle: np.ndarray,
) -> str:
    if target == 1:
        return "safe_missed_true_edge"
    ambiguous = (
        relation >= 0.45
        or support < 0.65
        or abs(float(bundle[0]) - float(bundle[1])) < 0.10
    )
    if ambiguous:
        return "ambiguous_rescue_candidate"
    return "low_hint_pair_support_false_admission"


def empty_binary_sums() -> Dict[str, float]:
    return {
        "count": 0.0,
        "target_pos": 0.0,
        "pred_pos": 0.0,
        "tp": 0.0,
        "fp": 0.0,
        "fn": 0.0,
        "score_sum": 0.0,
        "base_score_sum": 0.0,
        "residual_sum": 0.0,
    }


def add_binary(
    sums: Dict[str, float],
    *,
    target: int,
    pred: int,
    score: float,
    base_score: float,
    residual: float,
) -> None:
    sums["count"] += 1.0
    sums["target_pos"] += float(target == 1)
    sums["pred_pos"] += float(pred == 1)
    sums["tp"] += float(pred == 1 and target == 1)
    sums["fp"] += float(pred == 1 and target == 0)
    sums["fn"] += float(pred == 0 and target == 1)
    sums["score_sum"] += float(score)
    sums["base_score_sum"] += float(base_score)
    sums["residual_sum"] += float(residual)


def finalize_binary(sums: Dict[str, float]) -> Dict[str, float]:
    count = max(sums["count"], 1.0)
    return {
        **sums,
        "target_positive_rate": sums["target_pos"] / count,
        "admission_rate": sums["pred_pos"] / count,
        "avg_score": sums["score_sum"] / count,
        "avg_base_score": sums["base_score_sum"] / count,
        "avg_rescue_residual": sums["residual_sum"] / count,
        **binary_metrics(sums["tp"], sums["fp"], sums["fn"]),
    }


def summarize_feature_group(group_name: str, rows: list[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"group": group_name, "count": len(rows)}
    if not rows:
        return out
    feature_names = [
        "relation_hint",
        "pair_support_hint",
        "signed_pair_witness",
        "bundle_positive_support",
        "bundle_false_admission_warning",
        "bundle_corroboration",
        "bundle_endpoint_compatibility",
        "bundle_margin_positive_minus_warning",
        "rev6_score",
        "rev17_score",
        "rev19_base_score",
        "rev19_score",
        "rev19_rescue_residual",
        "gt_endpoint_degree_sum",
        "hint_endpoint_degree_sum",
        "support_endpoint_degree_sum",
        "hint_common_neighbors",
        "support_common_neighbors",
    ]
    for name in feature_names:
        stats = quantiles(float(row[name]) for row in rows)
        for stat_name, value in stats.items():
            out[f"{name}_{stat_name}"] = value
    return out


def write_csv(path: Path, rows: list[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def clean_json_numbers(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: clean_json_numbers(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [clean_json_numbers(value) for value in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def rename_candidate_label(obj: Any, candidate_name: str) -> Any:
    if candidate_name == "rev19":
        return obj
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            new_key = key
            if isinstance(key, str) and key.startswith("rev19_"):
                new_key = key.replace("rev19_", f"{candidate_name}_", 1)
            out[new_key] = rename_candidate_label(value, candidate_name)
        if out.get("model") == "rev19":
            out["model"] = candidate_name
        if isinstance(out.get("group"), str) and str(out["group"]).startswith("rev19_"):
            out["group"] = str(out["group"]).replace("rev19_", f"{candidate_name}_", 1)
        return out
    if isinstance(obj, list):
        return [rename_candidate_label(value, candidate_name) for value in obj]
    return obj


def build_model_pred_rows(
    records: list[Dict[str, Any]],
    model_names: list[str],
) -> list[Dict[str, Any]]:
    table: Dict[tuple[str, str], Dict[str, float]] = defaultdict(empty_binary_sums)
    for row in records:
        target = int(row["target"])
        subtype = str(row["rescue_subtype"])
        for model_name in model_names:
            add_binary(
                table[(model_name, subtype)],
                target=target,
                pred=int(row[f"{model_name}_pred"]),
                score=float(row[f"{model_name}_score"]),
                base_score=float(row.get("rev19_base_score", row[f"{model_name}_score"])),
                residual=float(row.get("rev19_rescue_residual", 0.0)),
            )

    out = []
    for (model_name, subtype), sums in sorted(table.items()):
        final = finalize_binary(sums)
        out.append({"model": model_name, "rescue_subtype": subtype, **final})
    return out


def build_event_family_rows(records: list[Dict[str, Any]], model_names: list[str]) -> list[Dict[str, Any]]:
    table: Dict[tuple[str, str], Dict[str, float]] = defaultdict(empty_binary_sums)
    group_table: Dict[tuple[str, str], list[Dict[str, Any]]] = defaultdict(list)
    for row in records:
        target = int(row["target"])
        target_group = "safe_true_edge" if target == 1 else "unsafe_false_candidate"
        for family in row["event_families"]:
            group_table[(family, target_group)].append(row)
            for model_name in model_names:
                add_binary(
                    table[(model_name, family)],
                    target=target,
                    pred=int(row[f"{model_name}_pred"]),
                    score=float(row[f"{model_name}_score"]),
                    base_score=float(row.get("rev19_base_score", row[f"{model_name}_score"])),
                    residual=float(row.get("rev19_rescue_residual", 0.0)),
                )

    rows = []
    for (model_name, family), sums in sorted(table.items()):
        rows.append({"model": model_name, "event_family": family, **finalize_binary(sums)})
    for (family, target_group), group_rows in sorted(group_table.items()):
        summary = summarize_feature_group(f"{family}:{target_group}", group_rows)
        rows.append(
            {
                "model": "rev19_feature_split",
                "event_family": family,
                "target_group": target_group,
                "count": summary["count"],
                "avg_score": summary.get("rev19_score_mean", 0.0),
                "avg_base_score": summary.get("rev19_base_score_mean", 0.0),
                "avg_rescue_residual": summary.get("rev19_rescue_residual_mean", 0.0),
                "admission_rate": safe_div(
                    sum(float(r["rev19_pred"]) for r in group_rows),
                    float(len(group_rows)),
                ),
                "avg_bundle_positive": summary.get("bundle_positive_support_mean", 0.0),
                "avg_bundle_warning": summary.get("bundle_false_admission_warning_mean", 0.0),
                "avg_pair_support": summary.get("pair_support_hint_mean", 0.0),
            }
        )
    return rows


def build_residual_behavior_rows(records: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    groups = {
        "safe_missed_true_edges": [row for row in records if int(row["target"]) == 1],
        "unsafe_false_admissions_all": [row for row in records if int(row["target"]) == 0],
        "rev19_admitted_safe_edges": [
            row for row in records if int(row["target"]) == 1 and int(row["rev19_pred"]) == 1
        ],
        "rev19_admitted_unsafe_false_edges": [
            row for row in records if int(row["target"]) == 0 and int(row["rev19_pred"]) == 1
        ],
        "rev19_rejected_safe_edges": [
            row for row in records if int(row["target"]) == 1 and int(row["rev19_pred"]) == 0
        ],
    }
    rows = []
    for group_name, group_rows in groups.items():
        residuals = [float(row["rev19_rescue_residual"]) for row in group_rows]
        stats = quantiles(residuals)
        count = len(group_rows)
        rows.append(
            {
                "group": group_name,
                "count": count,
                **{f"residual_{key}": value for key, value in stats.items()},
                "positive_residual_fraction": safe_div(
                    sum(float(value > 0.0) for value in residuals),
                    float(count),
                ),
                "negative_residual_fraction": safe_div(
                    sum(float(value < 0.0) for value in residuals),
                    float(count),
                ),
                "abs_residual_mean": safe_div(
                    sum(abs(value) for value in residuals),
                    float(count),
                ),
                "unsafe_positive_residual_fraction": (
                    safe_div(sum(float(value > 0.0) for value in residuals), float(count))
                    if "unsafe" in group_name
                    else 0.0
                ),
                "safe_too_small_residual_fraction": (
                    safe_div(sum(float(value <= 0.05) for value in residuals), float(count))
                    if "safe" in group_name
                    else 0.0
                ),
            }
        )
    return rows


def build_feature_separation_rows(records: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    groups = {
        "safe_missed_true_edges": [row for row in records if int(row["target"]) == 1],
        "unsafe_false_candidates": [row for row in records if int(row["target"]) == 0],
        "rev19_admitted_safe_edges": [
            row for row in records if int(row["target"]) == 1 and int(row["rev19_pred"]) == 1
        ],
        "rev19_admitted_unsafe_false_edges": [
            row for row in records if int(row["target"]) == 0 and int(row["rev19_pred"]) == 1
        ],
        "rev19_rejected_safe_edges": [
            row for row in records if int(row["target"]) == 1 and int(row["rev19_pred"]) == 0
        ],
        "rev19_rejected_unsafe_edges": [
            row for row in records if int(row["target"]) == 0 and int(row["rev19_pred"]) == 0
        ],
    }
    return [summarize_feature_group(group_name, group_rows) for group_name, group_rows in groups.items()]


def build_threshold_sensitivity_rows(
    records: list[Dict[str, Any]],
    thresholds: list[float],
) -> list[Dict[str, Any]]:
    target_pos = sum(float(row["target"] == 1) for row in records)
    rows = []
    for threshold in thresholds:
        tp = fp = fn = pred_pos = 0.0
        for row in records:
            pred = float(row["rev19_score"] >= threshold)
            target = float(row["target"])
            pred_pos += pred
            tp += float(pred == 1.0 and target == 1.0)
            fp += float(pred == 1.0 and target == 0.0)
            fn += float(pred == 0.0 and target == 1.0)
        rows.append(
            {
                "selection": "score_threshold",
                "value": threshold,
                "accepted_count": pred_pos,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "recall_within_rescue_scope": safe_div(tp, target_pos),
                **binary_metrics(tp, fp, fn),
            }
        )
    return rows


def build_topk_sensitivity_rows(records: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    if not records:
        return []
    target_pos = sum(float(row["target"] == 1) for row in records)
    current_accept_count = int(sum(int(row["rev19_pred"]) for row in records))
    budgets = sorted(
        {
            max(1, int(round(len(records) * frac)))
            for frac in [0.02, 0.05, 0.10, 0.20]
        }
        | {max(1, current_accept_count)}
    )
    rankers = {
        "rev19_final_score": lambda row: float(row["rev19_score"]),
        "rev19_rescue_residual": lambda row: float(row["rev19_rescue_residual"]),
        "bundle_margin": lambda row: float(row["bundle_margin_positive_minus_warning"]),
    }
    rows = []
    for ranker_name, key_fn in rankers.items():
        ranked = sorted(records, key=key_fn, reverse=True)
        for budget in budgets:
            chosen = ranked[: min(budget, len(ranked))]
            tp = sum(float(row["target"] == 1) for row in chosen)
            fp = sum(float(row["target"] == 0) for row in chosen)
            fn = max(target_pos - tp, 0.0)
            row = {
                "selection": "topk_budget",
                "ranker": ranker_name,
                "budget": budget,
                "budget_fraction": safe_div(float(budget), float(len(records))),
                "accepted_count": len(chosen),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "recall_within_rescue_scope": safe_div(tp, target_pos),
                **binary_metrics(tp, fp, fn),
            }
            if budget == current_accept_count:
                row["matches_current_accept_count"] = 1.0
            else:
                row["matches_current_accept_count"] = 0.0
            rows.append(row)
    return rows


@torch.no_grad()
def collect_records(args: argparse.Namespace) -> list[Dict[str, Any]]:
    device = get_device(args.device)
    thresholds = parse_thresholds(args.thresholds)
    dataset = Step30WeakObservationDataset(args.data_path)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=step30_weak_observation_collate_fn,
        pin_memory=device.type == "cuda",
    )
    models = {
        "rev6": load_model(args.rev6_checkpoint, device),
        "rev17": load_model(args.rev17_checkpoint, device),
        "rev19": load_model(args.rev19_checkpoint, device),
    }
    residual_scale = float(models["rev19"].config.rescue_scoped_bundle_residual_scale)
    records: list[Dict[str, Any]] = []
    source_counter: Counter[str] = Counter()

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        variants = [str(v) for v in batch.get("step30_observation_variant", ["unknown"] * len(batch["target_adj"]))]
        events_list = batch.get("events", [None] * int(batch["target_adj"].shape[0]))
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

        batch_size = int(target_adj.shape[0])
        for batch_idx in range(batch_size):
            if variants[batch_idx] != "noisy":
                continue
            n = int(node_mask[batch_idx].sum().item())
            threshold = threshold_for_variant(thresholds, variants[batch_idx])
            families = event_families(events_list[batch_idx])
            target_np = target_adj[batch_idx, :n, :n].detach().cpu().numpy()
            relation_np = relation_hints[batch_idx, :n, :n].detach().cpu().numpy()
            support_np = pair_support_hints[batch_idx, :n, :n].detach().cpu().numpy()
            witness_np = signed_witness[batch_idx, :n, :n].detach().cpu().numpy()
            bundle_np = pair_bundle[batch_idx, :n, :n, :].detach().cpu().numpy()
            valid_pair_np = pair_mask[batch_idx, :n, :n].detach().cpu().numpy().astype(bool)

            hint_adj = ((relation_np >= 0.5) & valid_pair_np).astype(np.float32)
            support_adj = ((support_np >= args.rescue_support_min) & valid_pair_np).astype(np.float32)
            gt_deg = (target_np * valid_pair_np.astype(np.float32)).sum(axis=1)
            hint_deg = hint_adj.sum(axis=1)
            support_deg = support_adj.sum(axis=1)

            scores_by_model: Dict[str, np.ndarray] = {}
            preds_by_model: Dict[str, np.ndarray] = {}
            for model_name, outputs in outputs_by_model.items():
                logits_np = outputs["edge_logits"][batch_idx, :n, :n].detach().cpu().numpy()
                scores = sigmoid_np(logits_np)
                scores_by_model[model_name] = scores
                preds_by_model[model_name] = (scores >= threshold).astype(np.int64)

            rev19_residual = outputs_by_model["rev19"].get("pair_evidence_rescue_residual")
            if rev19_residual is None:
                raise ValueError("rev19 checkpoint did not expose pair_evidence_rescue_residual")
            residual_np = rev19_residual[batch_idx, :n, :n].detach().cpu().numpy()
            rev19_logits_np = outputs_by_model["rev19"]["edge_logits"][batch_idx, :n, :n].detach().cpu().numpy()
            base_logits_np = rev19_logits_np - residual_scale * residual_np
            base_score_np = sigmoid_np(base_logits_np)

            for i in range(n):
                for j in range(i + 1, n):
                    if not bool(valid_pair_np[i, j]):
                        continue
                    relation = float(relation_np[i, j])
                    support = float(support_np[i, j])
                    rescue_eligible = relation < args.rescue_relation_max and support >= args.rescue_support_min
                    if not rescue_eligible:
                        continue
                    target = int(target_np[i, j] >= 0.5)
                    bundle = bundle_np[i, j, :]
                    source_counter["rescue_scope_pairs"] += 1
                    source_counter["rescue_scope_positives"] += int(target == 1)
                    record = {
                        "sample_index": int(batch.get("step30_source_sample_index", [batch_idx] * batch_size)[batch_idx]),
                        "i": i,
                        "j": j,
                        "event_families": families,
                        "target": target,
                        "rescue_subtype": rescue_subtype(target, relation, support, bundle),
                        "relation_hint": relation,
                        "pair_support_hint": support,
                        "signed_pair_witness": float(witness_np[i, j]),
                        "bundle_positive_support": float(bundle[0]),
                        "bundle_false_admission_warning": float(bundle[1]),
                        "bundle_corroboration": float(bundle[2]),
                        "bundle_endpoint_compatibility": float(bundle[3]),
                        "bundle_margin_positive_minus_warning": float(bundle[0] - bundle[1]),
                        "rev19_base_score": float(base_score_np[i, j]),
                        "rev19_rescue_residual": float(residual_np[i, j]),
                        "gt_endpoint_degree_sum": float(gt_deg[i] + gt_deg[j]),
                        "hint_endpoint_degree_sum": float(hint_deg[i] + hint_deg[j]),
                        "support_endpoint_degree_sum": float(support_deg[i] + support_deg[j]),
                        "hint_common_neighbors": float(np.minimum(hint_adj[i], hint_adj[j]).sum()),
                        "support_common_neighbors": float(np.minimum(support_adj[i], support_adj[j]).sum()),
                    }
                    for model_name in ["rev6", "rev17", "rev19"]:
                        record[f"{model_name}_score"] = float(scores_by_model[model_name][i, j])
                        record[f"{model_name}_pred"] = int(preds_by_model[model_name][i, j])
                    records.append(record)

    print(
        json.dumps(
            {
                "rescue_scope_pairs": int(source_counter["rescue_scope_pairs"]),
                "rescue_scope_positives": int(source_counter["rescue_scope_positives"]),
                "rescue_positive_rate": safe_div(
                    float(source_counter["rescue_scope_positives"]),
                    float(source_counter["rescue_scope_pairs"]),
                ),
                "thresholds": thresholds,
            },
            indent=2,
        )
    )
    return records


def write_artifacts(args: argparse.Namespace, records: list[Dict[str, Any]]) -> Dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_names = ["rev6", "rev17", "rev19"]
    subtype_rows = build_model_pred_rows(records, model_names)
    residual_rows = build_residual_behavior_rows(records)
    feature_rows = build_feature_separation_rows(records)
    threshold_rows = build_threshold_sensitivity_rows(records, [0.50, 0.55, 0.60, 0.65, 0.70])
    topk_rows = build_topk_sensitivity_rows(records)
    event_rows = build_event_family_rows(records, model_names)

    compact_rows = []
    for model_name in model_names:
        model_records = records
        tp = sum(float(row[f"{model_name}_pred"] == 1 and row["target"] == 1) for row in model_records)
        fp = sum(float(row[f"{model_name}_pred"] == 1 and row["target"] == 0) for row in model_records)
        fn = sum(float(row[f"{model_name}_pred"] == 0 and row["target"] == 1) for row in model_records)
        pred_pos = tp + fp
        compact_rows.append(
            {
                "model": model_name,
                "scope": "noisy_rescue_eligible",
                "count": len(model_records),
                "target_pos": sum(float(row["target"] == 1) for row in model_records),
                "pred_pos": pred_pos,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "admission_rate": safe_div(pred_pos, float(len(model_records))),
                "avg_score": safe_div(
                    sum(float(row[f"{model_name}_score"]) for row in model_records),
                    float(len(model_records)),
                ),
                **binary_metrics(tp, fp, fn),
            }
        )

    target_pos = sum(float(row["target"] == 1) for row in records)
    target_neg = len(records) - int(target_pos)
    rev19_unsafe_positive_residual_count = sum(
        float(row["target"] == 0 and row["rev19_rescue_residual"] > 0.0)
        for row in records
    )
    rev19_safe_too_small_count = sum(
        float(row["target"] == 1 and row["rev19_rescue_residual"] <= 0.05)
        for row in records
    )
    result = clean_json_numbers(
        {
            "metadata": {
                "data_path": args.data_path,
                "rev6_checkpoint": args.rev6_checkpoint,
                "rev17_checkpoint": args.rev17_checkpoint,
                "rev19_checkpoint": args.rev19_checkpoint,
                "candidate_name": args.candidate_name,
                "thresholds": parse_thresholds(args.thresholds),
                "noisy_only": True,
                "rescue_relation_max": args.rescue_relation_max,
                "rescue_support_min": args.rescue_support_min,
                "record_count": len(records),
                "target_positive_count": target_pos,
                "target_positive_rate": safe_div(target_pos, float(len(records))),
            },
            "key_findings": {
                "rev19_unsafe_positive_residual_fraction": safe_div(
                    rev19_unsafe_positive_residual_count,
                    float(target_neg),
                ),
                "rev19_safe_too_small_residual_fraction": safe_div(
                    rev19_safe_too_small_count,
                    target_pos,
                ),
            },
            "compact_rescue_scope_rows": compact_rows,
            "rescue_subtype_rows": subtype_rows,
            "residual_behavior_rows": residual_rows,
            "feature_separation_rows": feature_rows,
            "threshold_sensitivity_rows": threshold_rows,
            "topk_sensitivity_rows": topk_rows,
            "event_family_rescue_rows": event_rows,
        }
    )
    result = rename_candidate_label(result, args.candidate_name)

    with open(output_dir / "summary.json", "w") as f:
        json.dump(result, f, indent=2)
    write_csv(output_dir / "summary.csv", compact_rows)
    write_csv(output_dir / "rescue_subtype_table.csv", subtype_rows)
    write_csv(output_dir / "residual_behavior_table.csv", residual_rows)
    write_csv(output_dir / "feature_separation_table.csv", feature_rows)
    write_csv(output_dir / "threshold_sensitivity_table.csv", threshold_rows)
    write_csv(output_dir / "topk_sensitivity_table.csv", topk_rows)
    write_csv(output_dir / "event_family_rescue_table.csv", event_rows)
    print(f"wrote rev20 rescue-scope diagnostics to: {output_dir}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--rev6_checkpoint", type=str, required=True)
    parser.add_argument("--rev17_checkpoint", type=str, required=True)
    parser.add_argument("--rev19_checkpoint", type=str, required=True)
    parser.add_argument("--candidate_name", type=str, default="rev19")
    parser.add_argument("--thresholds", type=str, default="default:0.5,clean:0.5,noisy:0.55")
    parser.add_argument("--rescue_relation_max", type=float, default=0.5)
    parser.add_argument("--rescue_support_min", type=float, default=0.55)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts/step30_rescue_scope_diagnostics_rev20",
    )
    args = parser.parse_args()
    records = collect_records(args)
    write_artifacts(args, records)


if __name__ == "__main__":
    main()
