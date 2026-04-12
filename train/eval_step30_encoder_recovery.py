from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.step30_dataset import Step30WeakObservationDataset, step30_weak_observation_collate_fn
from models.encoder_step30 import (
    Step30EncoderConfig,
    Step30WeakObservationEncoder,
    step30_recovery_metrics,
)
from train.utils_step30_decode import (
    hard_adj_selective_rescue,
    load_rich_rescue_scorer,
    load_rescue_safety_scorer,
    threshold_tensor_for_variants,
)


def get_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if has_mps:
        return torch.device("mps")
    return torch.device("cpu")


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def load_model(checkpoint_path: str, device: torch.device) -> Step30WeakObservationEncoder:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config_dict = dict(checkpoint["config"])
    config_dict.setdefault("use_relation_hint_in_edge_head", False)
    config_dict.setdefault("use_relation_logit_residual", False)
    config_dict.setdefault("relation_logit_residual_scale", 1.0)
    config_dict.setdefault("use_trust_denoising_edge_decoder", False)
    config_dict.setdefault("use_pair_support_hints", False)
    config_dict.setdefault("use_pair_evidence_bundle", False)
    config_dict.setdefault("pair_evidence_bundle_dim", 0)
    config_dict.setdefault("use_rescue_scoped_pair_evidence_bundle", False)
    config_dict.setdefault("rescue_scoped_bundle_relation_max", 0.5)
    config_dict.setdefault("rescue_scoped_bundle_residual_scale", 0.5)
    config_dict.setdefault("use_rescue_safety_aux_head", False)
    config_dict.setdefault("use_rescue_candidate_latent_head", False)
    config_dict.setdefault("rescue_candidate_latent_dim", 32)
    config_dict.setdefault("rescue_candidate_relation_max", 0.5)
    config_dict.setdefault("rescue_candidate_support_min", 0.55)
    config_dict.setdefault("use_rescue_candidate_binary_calibration_head", False)
    config_dict.setdefault("use_rescue_candidate_ambiguity_head", False)
    config_dict.setdefault("use_positive_ambiguity_safety_hint", False)
    config_dict.setdefault("positive_ambiguity_safety_projection_scale", 1.0)
    config_dict.setdefault("use_weak_positive_ambiguity_safety_head", False)
    config_dict.setdefault("use_signed_pair_witness", False)
    config_dict.setdefault("use_signed_pair_witness_in_edge_head", True)
    config_dict.setdefault("use_signed_pair_witness_correction", False)
    config_dict.setdefault("signed_pair_witness_correction_scale", 0.5)
    config = Step30EncoderConfig(**config_dict)
    model = Step30WeakObservationEncoder(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def empty_metric_sums() -> Dict[str, float]:
    return {
        "node_type_accuracy": 0.0,
        "node_state_mae": 0.0,
        "node_state_mse": 0.0,
        "edge_accuracy": 0.0,
        "edge_precision": 0.0,
        "edge_recall": 0.0,
        "edge_f1": 0.0,
        "edge_tp": 0.0,
        "edge_pred_pos": 0.0,
        "edge_true_pos": 0.0,
        "count": 0.0,
    }


def add_metrics(metric_sums: Dict[str, float], metrics: Dict[str, float]) -> None:
    for key in metric_sums:
        if key == "count":
            continue
        metric_sums[key] += float(metrics.get(key, 0.0))
    metric_sums["count"] += 1.0


def finalize(metric_sums: Dict[str, float]) -> Dict[str, float]:
    count = max(metric_sums["count"], 1.0)
    return {key: value / count for key, value in metric_sums.items() if key != "count"}


def logit_from_prob(prob: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    prob = prob.clamp(eps, 1.0 - eps)
    return torch.log(prob / (1.0 - prob))


def trivial_baseline_outputs(
    weak_slot_features: torch.Tensor,
    weak_relation_hints: torch.Tensor,
    num_node_types: int,
    state_dim: int,
    weak_pair_support_hints: torch.Tensor | None = None,
    weak_signed_pair_witness: torch.Tensor | None = None,
    weak_pair_evidence_bundle: torch.Tensor | None = None,
    weak_positive_ambiguity_safety_hint: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    type_hint = weak_slot_features[:, :, :num_node_types]
    state_start = num_node_types + 1
    state_hint = weak_slot_features[:, :, state_start : state_start + state_dim]
    edge_hint = weak_relation_hints
    if weak_pair_support_hints is not None:
        edge_hint = 0.5 * (weak_relation_hints + weak_pair_support_hints)
    if weak_signed_pair_witness is not None:
        witness_hint = (0.5 + 0.5 * weak_signed_pair_witness).clamp(0.0, 1.0)
        edge_hint = (edge_hint + witness_hint) / 2.0
    if weak_pair_evidence_bundle is not None:
        positive_support = weak_pair_evidence_bundle[..., 0]
        false_warning = weak_pair_evidence_bundle[..., 1]
        corroboration = weak_pair_evidence_bundle[..., 2]
        endpoint_compat = weak_pair_evidence_bundle[..., 3]
        bundle_hint = (
            0.35 * positive_support
            + 0.30 * (1.0 - false_warning)
            + 0.20 * corroboration
            + 0.15 * endpoint_compat
        ).clamp(0.0, 1.0)
        edge_hint = 0.5 * edge_hint + 0.5 * bundle_hint
    if weak_positive_ambiguity_safety_hint is not None:
        if weak_pair_support_hints is None:
            rescue_like = weak_relation_hints < 0.5
        else:
            rescue_like = (weak_relation_hints < 0.5) & (weak_pair_support_hints >= 0.55)
        safety_hint = weak_positive_ambiguity_safety_hint.clamp(0.0, 1.0)
        edge_hint = torch.where(
            rescue_like,
            0.75 * edge_hint + 0.25 * safety_hint,
            edge_hint,
        )
    return {
        "type_logits": type_hint * 10.0,
        "state_pred": state_hint,
        "edge_logits": logit_from_prob(edge_hint),
    }


def group_names(batch: Dict[str, Any]) -> Iterable[str]:
    if "step30_observation_variant" not in batch:
        return ["unknown"] * int(batch["weak_slot_features"].shape[0])
    return [str(v) for v in batch["step30_observation_variant"]]


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


def threshold_for_variant(edge_threshold: Any, variant: str) -> float:
    if isinstance(edge_threshold, dict):
        return float(edge_threshold.get(variant, edge_threshold.get("default", 0.5)))
    return float(edge_threshold)


def parse_variant_thresholds(spec: str) -> Dict[str, float]:
    thresholds: Dict[str, float] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError("--edge_thresholds_by_variant entries must look like clean:0.5,noisy:0.55")
        key, value = part.split(":", 1)
        thresholds[key.strip()] = float(value)
    if not thresholds:
        raise ValueError("--edge_thresholds_by_variant must contain at least one variant threshold")
    return thresholds


@torch.no_grad()
def evaluate(
    model: Step30WeakObservationEncoder,
    loader: DataLoader,
    device: torch.device,
    edge_threshold: Any,
    include_trivial_baseline: bool,
    decode_mode: str = "threshold",
    rescue_variants: set[str] | None = None,
    rescue_relation_max: float = 0.5,
    rescue_support_min: float = 0.55,
    rescue_budget_fraction: float = 0.0,
    rescue_score_mode: str = "raw",
    rescue_support_weight: float = 0.5,
    rescue_relation_weight: float = 0.25,
    rescue_safety_scorer: Dict[str, Any] | None = None,
    rich_rescue_scorer: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    overall = empty_metric_sums()
    by_variant: Dict[str, Dict[str, float]] = {}
    by_event_family: Dict[str, Dict[str, float]] = {}
    baseline_overall = empty_metric_sums()
    baseline_by_variant: Dict[str, Dict[str, float]] = {}
    baseline_by_event_family: Dict[str, Dict[str, float]] = {}

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        outputs = model(
            weak_slot_features=batch["weak_slot_features"],
            weak_relation_hints=batch["weak_relation_hints"],
            weak_pair_support_hints=batch.get("weak_pair_support_hints"),
            weak_signed_pair_witness=batch.get("weak_signed_pair_witness"),
            weak_pair_evidence_bundle=batch.get("weak_pair_evidence_bundle"),
            weak_positive_ambiguity_safety_hint=batch.get("weak_positive_ambiguity_safety_hint"),
        )
        baseline_outputs = None
        if include_trivial_baseline:
            baseline_outputs = trivial_baseline_outputs(
                weak_slot_features=batch["weak_slot_features"],
                weak_relation_hints=batch["weak_relation_hints"],
                num_node_types=model.config.num_node_types,
                state_dim=model.config.state_dim,
                weak_pair_support_hints=batch.get("weak_pair_support_hints"),
                weak_signed_pair_witness=batch.get("weak_signed_pair_witness"),
                weak_pair_evidence_bundle=batch.get("weak_pair_evidence_bundle"),
                weak_positive_ambiguity_safety_hint=batch.get(
                    "weak_positive_ambiguity_safety_hint"
                ),
            )

        variants = list(group_names(batch))
        model_edge_pred_override = None
        baseline_edge_pred_override = None
        if decode_mode == "selective_rescue":
            base_threshold = threshold_tensor_for_variants(
                variants=variants,
                default_threshold=0.5 if isinstance(edge_threshold, dict) else float(edge_threshold),
                thresholds_by_variant=edge_threshold if isinstance(edge_threshold, dict) else None,
                device=batch["target_adj"].device,
            )
            model_edge_pred_override = hard_adj_selective_rescue(
                edge_logits=outputs["edge_logits"],
                relation_hints=batch["weak_relation_hints"],
                pair_support_hints=batch.get("weak_pair_support_hints"),
                node_mask=batch["node_mask"],
                variants=variants,
                base_threshold=base_threshold,
                rescue_variants=rescue_variants or {"noisy"},
                rescue_relation_max=rescue_relation_max,
                rescue_support_min=rescue_support_min,
                rescue_budget_fraction=rescue_budget_fraction,
                rescue_score_mode=rescue_score_mode,
                rescue_support_weight=rescue_support_weight,
                rescue_relation_weight=rescue_relation_weight,
                rescue_safety_scorer=rescue_safety_scorer,
                node_latents=outputs.get("node_latents"),
                rescue_aux_logits=outputs.get("rescue_safety_logits"),
                rich_rescue_scorer=rich_rescue_scorer,
            )
            if baseline_outputs is not None:
                baseline_edge_pred_override = hard_adj_selective_rescue(
                    edge_logits=baseline_outputs["edge_logits"],
                    relation_hints=batch["weak_relation_hints"],
                    pair_support_hints=batch.get("weak_pair_support_hints"),
                    node_mask=batch["node_mask"],
                    variants=variants,
                    base_threshold=base_threshold,
                    rescue_variants=rescue_variants or {"noisy"},
                    rescue_relation_max=rescue_relation_max,
                    rescue_support_min=rescue_support_min,
                    rescue_budget_fraction=rescue_budget_fraction,
                    rescue_score_mode="raw" if rescue_score_mode in {"rich_learned", "aux"} else rescue_score_mode,
                    rescue_support_weight=rescue_support_weight,
                    rescue_relation_weight=rescue_relation_weight,
                    rescue_safety_scorer=rescue_safety_scorer,
                )
        events_list = batch.get("events", [None] * int(batch["weak_slot_features"].shape[0]))
        batch_size = int(batch["weak_slot_features"].shape[0])
        for idx in range(batch_size):
            one_outputs = {key: value[idx : idx + 1] for key, value in outputs.items()}
            one_metrics = step30_recovery_metrics(
                outputs=one_outputs,
                target_node_feats=batch["target_node_feats"][idx : idx + 1],
                target_adj=batch["target_adj"][idx : idx + 1],
                node_mask=batch["node_mask"][idx : idx + 1],
                edge_threshold=threshold_for_variant(edge_threshold, variants[idx]),
                edge_pred_override=(
                    model_edge_pred_override[idx : idx + 1]
                    if model_edge_pred_override is not None
                    else None
                ),
            )
            add_metrics(overall, one_metrics)
            variant = variants[idx]
            by_variant.setdefault(variant, empty_metric_sums())
            add_metrics(by_variant[variant], one_metrics)
            for family in event_families(events_list[idx]):
                by_event_family.setdefault(family, empty_metric_sums())
                add_metrics(by_event_family[family], one_metrics)

            if baseline_outputs is not None:
                one_baseline_outputs = {
                    key: value[idx : idx + 1] for key, value in baseline_outputs.items()
                }
                baseline_metrics = step30_recovery_metrics(
                    outputs=one_baseline_outputs,
                    target_node_feats=batch["target_node_feats"][idx : idx + 1],
                    target_adj=batch["target_adj"][idx : idx + 1],
                    node_mask=batch["node_mask"][idx : idx + 1],
                    edge_threshold=threshold_for_variant(edge_threshold, variants[idx]),
                    edge_pred_override=(
                        baseline_edge_pred_override[idx : idx + 1]
                        if baseline_edge_pred_override is not None
                        else None
                    ),
                )
                add_metrics(baseline_overall, baseline_metrics)
                baseline_by_variant.setdefault(variant, empty_metric_sums())
                add_metrics(baseline_by_variant[variant], baseline_metrics)
                for family in event_families(events_list[idx]):
                    baseline_by_event_family.setdefault(family, empty_metric_sums())
                    add_metrics(baseline_by_event_family[family], baseline_metrics)

    result: Dict[str, Any] = {
        "overall": finalize(overall),
        "by_observation_variant": {
            key: finalize(value) for key, value in sorted(by_variant.items())
        },
        "by_event_family": {
            key: finalize(value) for key, value in sorted(by_event_family.items())
        },
    }
    if include_trivial_baseline:
        result["trivial_baseline_overall"] = finalize(baseline_overall)
        result["trivial_baseline_by_observation_variant"] = {
            key: finalize(value) for key, value in sorted(baseline_by_variant.items())
        }
        result["trivial_baseline_by_event_family"] = {
            key: finalize(value) for key, value in sorted(baseline_by_event_family.items())
        }
    return result


def write_csv(path: Path, result: Dict[str, Any]) -> None:
    rows = []
    rows.append({"group": "overall", "name": "model", **result["overall"]})
    for variant, metrics in result["by_observation_variant"].items():
        rows.append({"group": f"variant:{variant}", "name": "model", **metrics})
    for family, metrics in result["by_event_family"].items():
        rows.append({"group": f"event_family:{family}", "name": "model", **metrics})
    if "trivial_baseline_overall" in result:
        rows.append({"group": "overall", "name": "trivial_baseline", **result["trivial_baseline_overall"]})
        for variant, metrics in result["trivial_baseline_by_observation_variant"].items():
            rows.append({"group": f"variant:{variant}", "name": "trivial_baseline", **metrics})
        for family, metrics in result["trivial_baseline_by_event_family"].items():
            rows.append({"group": f"event_family:{family}", "name": "trivial_baseline", **metrics})

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


def parse_threshold_grid(grid_arg: str) -> list[float]:
    values = []
    for part in grid_arg.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    if not values:
        raise ValueError("--threshold_grid must contain at least one numeric threshold")
    return values


def metric_for_calibration(result: Dict[str, Any], target: str) -> float:
    if target == "overall_edge_f1":
        value = result["overall"].get("edge_f1")
    elif target == "noisy_edge_f1":
        value = result["by_observation_variant"].get("noisy", {}).get("edge_f1")
    elif target == "clean_edge_f1":
        value = result["by_observation_variant"].get("clean", {}).get("edge_f1")
    else:
        raise ValueError(f"Unknown calibration target: {target}")
    return float(value or 0.0)


def best_threshold_for_variant(candidates: list[Dict[str, Any]], variant: str) -> Dict[str, Any]:
    metric_key = f"{variant}_edge_f1"
    return max(
        candidates,
        key=lambda item: (
            float(item.get(metric_key) or 0.0),
            -abs(float(item["threshold"]) - 0.5),
        ),
    )


@torch.no_grad()
def calibrate_edge_threshold(
    model: Step30WeakObservationEncoder,
    data_path: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    threshold_grid: list[float],
    calibration_target: str,
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
    candidates = []
    for threshold in threshold_grid:
        result = evaluate(
            model=model,
            loader=loader,
            device=device,
            edge_threshold=threshold,
            include_trivial_baseline=False,
        )
        score = metric_for_calibration(result, calibration_target)
        candidates.append(
            {
                "threshold": threshold,
                "score": score,
                "overall_edge_f1": result["overall"].get("edge_f1"),
                "clean_edge_f1": result["by_observation_variant"].get("clean", {}).get("edge_f1"),
                "noisy_edge_f1": result["by_observation_variant"].get("noisy", {}).get("edge_f1"),
            }
        )
    best = max(candidates, key=lambda item: (item["score"], -abs(item["threshold"] - 0.5)))
    return {
        "calibration_data_path": data_path,
        "calibration_target": calibration_target,
        "selected_edge_threshold": best["threshold"],
        "selected_score": best["score"],
        "candidates": candidates,
    }


@torch.no_grad()
def calibrate_variant_edge_thresholds(
    model: Step30WeakObservationEncoder,
    data_path: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    threshold_grid: list[float],
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
    candidates = []
    for threshold in threshold_grid:
        result = evaluate(
            model=model,
            loader=loader,
            device=device,
            edge_threshold=threshold,
            include_trivial_baseline=False,
        )
        candidates.append(
            {
                "threshold": threshold,
                "overall_edge_f1": result["overall"].get("edge_f1"),
                "clean_edge_f1": result["by_observation_variant"].get("clean", {}).get("edge_f1"),
                "noisy_edge_f1": result["by_observation_variant"].get("noisy", {}).get("edge_f1"),
            }
        )

    selected_by_variant: Dict[str, float] = {"default": 0.5}
    selected_scores_by_variant: Dict[str, float] = {}
    for variant in ["clean", "noisy"]:
        best = best_threshold_for_variant(candidates, variant)
        selected_by_variant[variant] = float(best["threshold"])
        selected_scores_by_variant[variant] = float(best.get(f"{variant}_edge_f1") or 0.0)

    return {
        "calibration_data_path": data_path,
        "calibration_target": "variant_edge_f1",
        "selected_edge_thresholds_by_variant": selected_by_variant,
        "selected_scores_by_variant": selected_scores_by_variant,
        "candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--edge_threshold", type=float, default=0.5)
    parser.add_argument(
        "--edge_thresholds_by_variant",
        type=str,
        default=None,
        help="Optional comma-separated thresholds such as default:0.5,clean:0.5,noisy:0.55.",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--no_trivial_baseline", action="store_true")
    parser.add_argument("--calibration_data_path", type=str, default=None)
    parser.add_argument("--calibrate_edge_threshold", action="store_true")
    parser.add_argument("--calibrate_variant_edge_thresholds", action="store_true")
    parser.add_argument("--decode_mode", choices=["threshold", "selective_rescue"], default="threshold")
    parser.add_argument("--rescue_variants", type=str, default="noisy")
    parser.add_argument("--rescue_relation_max", type=float, default=0.5)
    parser.add_argument("--rescue_support_min", type=float, default=0.55)
    parser.add_argument("--rescue_budget_fraction", type=float, default=0.0)
    parser.add_argument("--rescue_score_mode", choices=["raw", "guarded", "learned", "rich_learned", "aux"], default="raw")
    parser.add_argument("--rescue_support_weight", type=float, default=0.5)
    parser.add_argument("--rescue_relation_weight", type=float, default=0.25)
    parser.add_argument("--rescue_safety_scorer_path", type=str, default=None)
    parser.add_argument("--rich_rescue_scorer_path", type=str, default=None)
    parser.add_argument(
        "--calibration_target",
        choices=["overall_edge_f1", "clean_edge_f1", "noisy_edge_f1"],
        default="noisy_edge_f1",
    )
    parser.add_argument(
        "--threshold_grid",
        type=str,
        default="0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80",
    )
    args = parser.parse_args()

    device = get_device(args.device)
    dataset = Step30WeakObservationDataset(args.data_path)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=step30_weak_observation_collate_fn,
        pin_memory=device.type == "cuda",
    )
    model = load_model(args.checkpoint_path, device)
    calibration = None
    edge_threshold: Any = args.edge_threshold
    if args.edge_thresholds_by_variant is not None:
        if args.calibrate_edge_threshold or args.calibrate_variant_edge_thresholds:
            raise ValueError("Use explicit --edge_thresholds_by_variant or calibration, not both")
        edge_threshold = parse_variant_thresholds(args.edge_thresholds_by_variant)
    elif args.calibrate_edge_threshold:
        if args.calibrate_variant_edge_thresholds:
            raise ValueError("Use only one of --calibrate_edge_threshold or --calibrate_variant_edge_thresholds")
        if args.calibration_data_path is None:
            raise ValueError("--calibrate_edge_threshold requires --calibration_data_path")
        calibration = calibrate_edge_threshold(
            model=model,
            data_path=args.calibration_data_path,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            threshold_grid=parse_threshold_grid(args.threshold_grid),
            calibration_target=args.calibration_target,
        )
        edge_threshold = float(calibration["selected_edge_threshold"])
        print(f"selected calibrated edge threshold: {edge_threshold}")
    elif args.calibrate_variant_edge_thresholds:
        if args.calibration_data_path is None:
            raise ValueError("--calibrate_variant_edge_thresholds requires --calibration_data_path")
        calibration = calibrate_variant_edge_thresholds(
            model=model,
            data_path=args.calibration_data_path,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            threshold_grid=parse_threshold_grid(args.threshold_grid),
        )
        edge_threshold = calibration["selected_edge_thresholds_by_variant"]
        print(f"selected calibrated edge thresholds by variant: {edge_threshold}")

    result = evaluate(
        model=model,
        loader=loader,
        device=device,
        edge_threshold=edge_threshold,
        include_trivial_baseline=not args.no_trivial_baseline,
        decode_mode=args.decode_mode,
        rescue_variants={part.strip() for part in args.rescue_variants.split(",") if part.strip()},
        rescue_relation_max=args.rescue_relation_max,
        rescue_support_min=args.rescue_support_min,
        rescue_budget_fraction=args.rescue_budget_fraction,
        rescue_score_mode=args.rescue_score_mode,
        rescue_support_weight=args.rescue_support_weight,
        rescue_relation_weight=args.rescue_relation_weight,
        rescue_safety_scorer=load_rescue_safety_scorer(args.rescue_safety_scorer_path),
        rich_rescue_scorer=load_rich_rescue_scorer(args.rich_rescue_scorer_path),
    )
    result["metadata"] = {
        "data_path": args.data_path,
        "checkpoint_path": args.checkpoint_path,
        "num_samples": len(dataset),
        "edge_threshold": edge_threshold,
        "decode_mode": args.decode_mode,
        "rescue_variants": [part.strip() for part in args.rescue_variants.split(",") if part.strip()],
        "rescue_relation_max": args.rescue_relation_max,
        "rescue_support_min": args.rescue_support_min,
        "rescue_budget_fraction": args.rescue_budget_fraction,
        "rescue_score_mode": args.rescue_score_mode,
        "rescue_support_weight": args.rescue_support_weight,
        "rescue_relation_weight": args.rescue_relation_weight,
        "rescue_safety_scorer_path": args.rescue_safety_scorer_path,
        "rich_rescue_scorer_path": args.rich_rescue_scorer_path,
    }
    if calibration is not None:
        result["calibration"] = calibration

    result = clean_json_numbers(result)
    print(json.dumps(result, indent=2))
    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"wrote JSON: {output_path}")
    if args.output_csv is not None:
        write_csv(Path(args.output_csv), result)
        print(f"wrote CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
