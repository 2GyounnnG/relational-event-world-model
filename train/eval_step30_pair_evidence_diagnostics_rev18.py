from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict

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
    logit_from_prob,
    move_batch_to_device,
    threshold_for_variant,
    trivial_baseline_outputs,
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


def relation_regime(value: float) -> str:
    if value < 0.30:
        return "very_low_relation"
    if value < 0.45:
        return "low_mid_relation"
    if value < 0.60:
        return "ambiguous_mid_relation"
    return "high_relation"


def support_regime(value: float) -> str:
    if value < 0.45:
        return "low_support"
    if value < 0.55:
        return "mid_support"
    return "high_support"


def witness_regime(value: float) -> str:
    if value <= -0.25:
        return "negative_witness"
    if value >= 0.25:
        return "positive_witness"
    return "neutral_witness"


def bundle_regime(bundle: np.ndarray) -> str:
    pos_hi = bool(bundle[0] >= 0.60)
    warning_hi = bool(bundle[1] >= 0.60)
    corr_hi = bool(bundle[2] >= 0.60)
    if pos_hi and not warning_hi and corr_hi:
        return "safe_like_bundle"
    if pos_hi and warning_hi:
        return "conflict_bundle"
    if warning_hi and not pos_hi:
        return "warning_bundle"
    if pos_hi or corr_hi:
        return "support_only_bundle"
    return "weak_bundle"


def fp_source(relation: float, support: float, bundle: np.ndarray) -> str:
    if relation >= 0.60:
        return "classic_hint_supported_fp"
    if 0.45 <= relation < 0.60:
        return "ambiguous_mid_hint_fp"
    if relation < 0.45 and (bundle[0] >= 0.60 or bundle[2] >= 0.60):
        return "low_hint_high_bundle_support_fp"
    if relation < 0.45 and support >= 0.55:
        return "low_hint_pair_support_rescue_fp"
    return "other_non_rescue_fp"


def empty_binary_sums() -> Dict[str, float]:
    return {
        "count": 0.0,
        "target_pos": 0.0,
        "pred_pos": 0.0,
        "tp": 0.0,
        "fp": 0.0,
        "fn": 0.0,
        "score_sum": 0.0,
        "relation_sum": 0.0,
        "support_sum": 0.0,
        "witness_sum": 0.0,
        "bundle_positive_sum": 0.0,
        "bundle_warning_sum": 0.0,
        "bundle_corroboration_sum": 0.0,
        "bundle_endpoint_sum": 0.0,
    }


def add_binary(
    sums: Dict[str, float],
    target: int,
    pred: int,
    score: float,
    relation: float,
    support: float,
    witness: float,
    bundle: np.ndarray,
) -> None:
    sums["count"] += 1.0
    sums["target_pos"] += float(target == 1)
    sums["pred_pos"] += float(pred == 1)
    sums["tp"] += float(pred == 1 and target == 1)
    sums["fp"] += float(pred == 1 and target == 0)
    sums["fn"] += float(pred == 0 and target == 1)
    sums["score_sum"] += score
    sums["relation_sum"] += relation
    sums["support_sum"] += support
    sums["witness_sum"] += witness
    sums["bundle_positive_sum"] += float(bundle[0])
    sums["bundle_warning_sum"] += float(bundle[1])
    sums["bundle_corroboration_sum"] += float(bundle[2])
    sums["bundle_endpoint_sum"] += float(bundle[3])


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
        "gt_positive_rate": sums["target_pos"] / count,
        "predicted_positive_rate": sums["pred_pos"] / count,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "avg_score": sums["score_sum"] / count,
        "avg_relation_hint": sums["relation_sum"] / count,
        "avg_pair_support": sums["support_sum"] / count,
        "avg_signed_witness": sums["witness_sum"] / count,
        "avg_bundle_positive": sums["bundle_positive_sum"] / count,
        "avg_bundle_warning": sums["bundle_warning_sum"] / count,
        "avg_bundle_corroboration": sums["bundle_corroboration_sum"] / count,
        "avg_bundle_endpoint": sums["bundle_endpoint_sum"] / count,
    }


def empty_fp_source_sums() -> Dict[str, float]:
    return {
        "fp_count": 0.0,
        "score_sum": 0.0,
        "relation_sum": 0.0,
        "support_sum": 0.0,
        "bundle_positive_sum": 0.0,
        "bundle_warning_sum": 0.0,
        "bundle_corroboration_sum": 0.0,
    }


def add_fp_source(
    sums: Dict[str, float],
    score: float,
    relation: float,
    support: float,
    bundle: np.ndarray,
) -> None:
    sums["fp_count"] += 1.0
    sums["score_sum"] += score
    sums["relation_sum"] += relation
    sums["support_sum"] += support
    sums["bundle_positive_sum"] += float(bundle[0])
    sums["bundle_warning_sum"] += float(bundle[1])
    sums["bundle_corroboration_sum"] += float(bundle[2])


def finalize_fp_source(sums: Dict[str, float], total_fp: float) -> Dict[str, float]:
    count = max(sums["fp_count"], 1.0)
    return {
        **sums,
        "fp_fraction": sums["fp_count"] / max(total_fp, 1.0),
        "avg_score": sums["score_sum"] / count,
        "avg_relation_hint": sums["relation_sum"] / count,
        "avg_pair_support": sums["support_sum"] / count,
        "avg_bundle_positive": sums["bundle_positive_sum"] / count,
        "avg_bundle_warning": sums["bundle_warning_sum"] / count,
        "avg_bundle_corroboration": sums["bundle_corroboration_sum"] / count,
    }


def binary_metrics_from_counts(tp: float, fp: float, fn: float) -> Dict[str, float]:
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def channel_stats(values: list[np.ndarray]) -> Dict[str, Any]:
    if not values:
        return {"count": 0}
    arr = np.stack(values, axis=0).astype(np.float64)
    row: Dict[str, Any] = {"count": int(arr.shape[0])}
    for idx, name in enumerate(CHANNEL_NAMES):
        row[f"{name}_mean"] = float(arr[:, idx].mean())
        row[f"{name}_q25"] = float(np.quantile(arr[:, idx], 0.25))
        row[f"{name}_q50"] = float(np.quantile(arr[:, idx], 0.50))
        row[f"{name}_q75"] = float(np.quantile(arr[:, idx], 0.75))
    return row


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


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--rev6_checkpoint", type=str, required=True)
    parser.add_argument("--rev13_checkpoint", type=str, required=True)
    parser.add_argument("--rev17_checkpoint", type=str, required=True)
    parser.add_argument("--candidate_name", type=str, default="rev17")
    parser.add_argument("--thresholds", type=str, default="default:0.5,clean:0.5,noisy:0.55")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="artifacts/step30_pair_evidence_diagnostics_rev18")
    parser.add_argument(
        "--mute_channels",
        type=str,
        default="positive_support,false_admission_warning",
        help="Comma-separated bundle channel names to mute to 0.5 for tiny diagnostic ablations.",
    )
    args = parser.parse_args()

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
        "rev13": load_model(args.rev13_checkpoint, device),
        args.candidate_name: load_model(args.rev17_checkpoint, device),
    }
    muted_channel_names = [part.strip() for part in args.mute_channels.split(",") if part.strip()]
    muted_channel_indices = {
        name: CHANNEL_NAMES.index(name)
        for name in muted_channel_names
        if name in CHANNEL_NAMES
    }

    pair_regime_table: Dict[tuple, Dict[str, float]] = defaultdict(empty_binary_sums)
    rescue_table: Dict[tuple, Dict[str, float]] = defaultdict(empty_binary_sums)
    event_fp_table: Dict[tuple, Dict[str, float]] = defaultdict(empty_binary_sums)
    fp_source_table: Dict[tuple, Dict[str, float]] = defaultdict(empty_fp_source_sums)
    fp_totals: Counter[str] = Counter()
    channel_groups: Dict[str, list[np.ndarray]] = defaultdict(list)
    mute_counts: Dict[str, Dict[str, float]] = defaultdict(lambda: {"tp": 0.0, "fp": 0.0, "fn": 0.0})

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

        logits_by_model: Dict[str, torch.Tensor] = {}
        for model_name, model in models.items():
            outputs = model(
                weak_slot_features=batch["weak_slot_features"],
                weak_relation_hints=relation_hints,
                weak_pair_support_hints=pair_support_hints,
                weak_signed_pair_witness=signed_witness,
                weak_pair_evidence_bundle=pair_bundle,
            )
            logits_by_model[model_name] = outputs["edge_logits"].detach()

        trivial_outputs = trivial_baseline_outputs(
            weak_slot_features=batch["weak_slot_features"],
            weak_relation_hints=relation_hints,
            num_node_types=models[args.candidate_name].config.num_node_types,
            state_dim=models[args.candidate_name].config.state_dim,
            weak_pair_support_hints=pair_support_hints,
            weak_signed_pair_witness=signed_witness,
            weak_pair_evidence_bundle=pair_bundle,
        )
        logits_by_model["trivial"] = trivial_outputs["edge_logits"].detach()

        for channel_name, channel_idx in muted_channel_indices.items():
            muted_bundle = pair_bundle.clone()
            muted_bundle[..., channel_idx] = 0.5
            muted_outputs = models[args.candidate_name](
                weak_slot_features=batch["weak_slot_features"],
                weak_relation_hints=relation_hints,
                weak_pair_support_hints=pair_support_hints,
                weak_signed_pair_witness=signed_witness,
                weak_pair_evidence_bundle=muted_bundle,
            )
            logits_by_model[f"{args.candidate_name}_mute_{channel_name}"] = muted_outputs["edge_logits"].detach()

        batch_size, num_nodes, _ = target_adj.shape
        pair_mask = build_pair_mask(node_mask).bool()
        for batch_idx in range(batch_size):
            if variants[batch_idx] != "noisy":
                continue
            n = int(node_mask[batch_idx].sum().item())
            families = event_families(events_list[batch_idx])
            relation_np = relation_hints[batch_idx, :n, :n].detach().cpu().numpy()
            support_np = pair_support_hints[batch_idx, :n, :n].detach().cpu().numpy()
            witness_np = signed_witness[batch_idx, :n, :n].detach().cpu().numpy()
            bundle_np = pair_bundle[batch_idx, :n, :n, :].detach().cpu().numpy()
            target_np = target_adj[batch_idx, :n, :n].detach().cpu().numpy()

            thresholds_i = {
                model_name: threshold_for_variant(thresholds, variants[batch_idx])
                for model_name in logits_by_model
            }
            score_by_model: Dict[str, np.ndarray] = {}
            pred_by_model: Dict[str, np.ndarray] = {}
            for model_name, logits in logits_by_model.items():
                logits_np = logits[batch_idx, :n, :n].detach().cpu().numpy()
                scores = 1.0 / (1.0 + np.exp(-np.clip(logits_np, -80.0, 80.0)))
                score_by_model[model_name] = scores
                pred_by_model[model_name] = (scores >= thresholds_i[model_name]).astype(np.int64)

            for i in range(n):
                for j in range(i + 1, n):
                    if not bool(pair_mask[batch_idx, i, j].item()):
                        continue
                    target = int(target_np[i, j] >= 0.5)
                    relation = float(relation_np[i, j])
                    support = float(support_np[i, j])
                    witness = float(witness_np[i, j])
                    bundle = bundle_np[i, j, :]
                    rescue_eligible = relation < 0.5 and support >= 0.55
                    regime_key_base = (
                        relation_regime(relation),
                        support_regime(support),
                        witness_regime(witness),
                        bundle_regime(bundle),
                    )

                    for model_name in ["rev6", "rev13", args.candidate_name, "trivial"]:
                        score = float(score_by_model[model_name][i, j])
                        pred = int(pred_by_model[model_name][i, j])
                        add_binary(
                            pair_regime_table[(model_name, *regime_key_base)],
                            target=target,
                            pred=pred,
                            score=score,
                            relation=relation,
                            support=support,
                            witness=witness,
                            bundle=bundle,
                        )
                        rescue_bucket = "rescue_eligible" if rescue_eligible else "ordinary_non_rescue"
                        add_binary(
                            rescue_table[(model_name, rescue_bucket)],
                            target=target,
                            pred=pred,
                            score=score,
                            relation=relation,
                            support=support,
                            witness=witness,
                            bundle=bundle,
                        )
                        if pred == 1 and target == 0:
                            source = fp_source(relation, support, bundle)
                            add_fp_source(
                                fp_source_table[(model_name, source)],
                                score=score,
                                relation=relation,
                                support=support,
                                bundle=bundle,
                            )
                            fp_totals[model_name] += 1
                        for family in families:
                            add_binary(
                                event_fp_table[(model_name, family)],
                                target=target,
                                pred=pred,
                                score=score,
                                relation=relation,
                                support=support,
                                witness=witness,
                                bundle=bundle,
                            )

                    candidate_pred = int(pred_by_model[args.candidate_name][i, j])
                    candidate_score = float(score_by_model[args.candidate_name][i, j])
                    _ = candidate_score
                    if target == 1 and relation < 0.5:
                        channel_groups["hint_missed_true_edges"].append(bundle)
                    if target == 0 and relation >= 0.5:
                        channel_groups["hard_negatives"].append(bundle)
                    if candidate_pred == 1 and target == 1 and relation < 0.5:
                        channel_groups["true_rescued_positives"].append(bundle)
                    if candidate_pred == 1 and target == 0 and rescue_eligible:
                        channel_groups["false_rescue_admissions"].append(bundle)
                    if candidate_pred == 1 and target == 0:
                        channel_groups["all_false_positives"].append(bundle)
                    if candidate_pred == 1 and target == 1:
                        channel_groups["all_true_positive_admissions"].append(bundle)

                    for channel_name in muted_channel_indices:
                        model_name = f"{args.candidate_name}_mute_{channel_name}"
                        pred = int(pred_by_model[model_name][i, j])
                        counts = mute_counts[model_name]
                        counts["tp"] += float(pred == 1 and target == 1)
                        counts["fp"] += float(pred == 1 and target == 0)
                        counts["fn"] += float(pred == 0 and target == 1)

    pair_regime_rows = []
    for key, sums in sorted(pair_regime_table.items()):
        row = {
            "model": key[0],
            "relation_regime": key[1],
            "support_regime": key[2],
            "witness_regime": key[3],
            "bundle_regime": key[4],
        }
        row.update(finalize_binary(sums))
        pair_regime_rows.append(row)

    fp_source_rows = []
    for key, sums in sorted(fp_source_table.items()):
        model_name, source = key
        row = {"model": model_name, "fp_source": source}
        row.update(finalize_fp_source(sums, float(fp_totals[model_name])))
        fp_source_rows.append(row)

    rescue_rows = []
    base_model_names = ["rev6", "rev13", args.candidate_name, "trivial"]
    total_target_pos_by_model = {
        model_name: sum(
            sums["target_pos"]
            for key, sums in rescue_table.items()
            if key[0] == model_name
        )
        for model_name in base_model_names
    }
    for key, sums in sorted(rescue_table.items()):
        model_name, rescue_bucket = key
        row = {"model": model_name, "admission_bucket": rescue_bucket}
        finalized = finalize_binary(sums)
        row.update(finalized)
        row["recall_contribution"] = finalized["tp"] / max(total_target_pos_by_model[model_name], 1.0)
        row["fp_fraction_of_model_fp"] = finalized["fp"] / max(float(fp_totals[model_name]), 1.0)
        rescue_rows.append(row)

    channel_rows = []
    for group, values in sorted(channel_groups.items()):
        row = {"group": group}
        row.update(channel_stats(values))
        channel_rows.append(row)

    event_rows = []
    for key, sums in sorted(event_fp_table.items()):
        model_name, family = key
        row = {"model": model_name, "event_family": family}
        row.update(finalize_binary(sums))
        event_rows.append(row)

    mute_rows = []
    for model_name, counts in sorted(mute_counts.items()):
        metrics = binary_metrics_from_counts(counts["tp"], counts["fp"], counts["fn"])
        mute_rows.append({"model": model_name, **counts, **metrics})

    top_candidate_fp_regimes = sorted(
        [row for row in pair_regime_rows if row["model"] == args.candidate_name],
        key=lambda row: float(row["fp"]),
        reverse=True,
    )[:12]

    result = clean_json_numbers(
        {
            "metadata": {
                "data_path": args.data_path,
                "num_samples": len(dataset),
                "thresholds": thresholds,
                "models": {
                    "rev6": args.rev6_checkpoint,
                    "rev13": args.rev13_checkpoint,
                    args.candidate_name: args.rev17_checkpoint,
                    "trivial": "decode_formula",
                },
                "noisy_only": True,
                "muted_channels": list(muted_channel_indices.keys()),
            },
            "top_candidate_fp_regimes": top_candidate_fp_regimes,
            "fp_source_rows": fp_source_rows,
            "rescue_vs_non_rescue_rows": rescue_rows,
            "bundle_channel_rows": channel_rows,
            "event_family_rows": event_rows,
            "mute_ablation_rows": mute_rows,
            "pair_regime_rows": pair_regime_rows,
        }
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "summary.json", "w") as f:
        json.dump(result, f, indent=2)
    write_csv(output_dir / "pair_regime_table.csv", pair_regime_rows)
    write_csv(output_dir / "fp_source_table.csv", fp_source_rows)
    write_csv(output_dir / "rescue_vs_non_rescue_table.csv", rescue_rows)
    write_csv(output_dir / "bundle_channel_table.csv", channel_rows)
    write_csv(output_dir / "event_family_fp_table.csv", event_rows)
    write_csv(output_dir / "mute_ablation_table.csv", mute_rows)
    write_csv(output_dir / f"top_{args.candidate_name}_fp_regimes.csv", top_candidate_fp_regimes)
    print(json.dumps(result["metadata"], indent=2))
    print(f"wrote rev18 pair-evidence diagnostics to: {output_dir}")


if __name__ == "__main__":
    main()
