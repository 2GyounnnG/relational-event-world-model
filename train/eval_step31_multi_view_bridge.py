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

from data.step31_dataset import Step31MultiViewObservationDataset, step31_multi_view_collate_fn
from models.encoder_step30 import Step30WeakObservationEncoder, build_pair_mask, logit_from_hint
from models.encoder_step31 import Step31MultiViewEncoderConfig, Step31MultiViewObservationEncoder
from train.eval_step30_encoder_recovery import load_model as load_step30_model
from train.eval_step30_encoder_recovery import get_device, move_batch_to_device


STEP30_REV6_REFERENCE = {
    "row": "step30_rev6_reference",
    "overall_precision": 0.7196309168140154,
    "overall_recall": 0.772661807707415,
    "overall_f1": 0.745204095591872,
    "clean_f1": 0.8281868226389127,
    "noisy_precision": 0.6306028424176168,
    "noisy_recall": 0.7048836904813861,
    "noisy_f1": 0.6656774858320454,
    "hint_missed_true_recall": 0.28170003512469266,
    "hint_supported_fp_error": 0.4688119566875095,
}


def load_step31_model(checkpoint_path: str, device: torch.device) -> Step31MultiViewObservationEncoder:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config_dict = dict(checkpoint["config"])
    config = Step31MultiViewEncoderConfig(**config_dict)
    model = Step31MultiViewObservationEncoder(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def clean_json_numbers(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: clean_json_numbers(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [clean_json_numbers(value) for value in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def parse_thresholds(spec: str) -> Dict[str, float]:
    out: Dict[str, float] = {"default": 0.5}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError("threshold entries must look like clean:0.50,noisy:0.55")
        key, value = part.split(":", 1)
        out[key.strip()] = float(value)
    return out


def threshold_tensor_for_variants(
    variants: Iterable[str],
    thresholds: Dict[str, float],
    device: torch.device,
) -> torch.Tensor:
    values = [float(thresholds.get(str(v), thresholds.get("default", 0.5))) for v in variants]
    return torch.tensor(values, dtype=torch.float32, device=device).view(-1, 1, 1)


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


def empty_counts() -> Dict[str, float]:
    return {
        "tp": 0.0,
        "pred": 0.0,
        "true": 0.0,
        "pairs": 0.0,
        "hint_missed_true": 0.0,
        "hint_missed_pred": 0.0,
        "hint_supported_fp_total": 0.0,
        "hint_supported_fp_pred": 0.0,
        "rescue_pred": 0.0,
        "rescue_tp": 0.0,
        "rescue_true": 0.0,
        "agreement_pred": 0.0,
        "agreement_tp": 0.0,
        "disagreement_pred": 0.0,
        "disagreement_tp": 0.0,
        "disagreement_fp": 0.0,
        "disagreement_neg": 0.0,
    }


def add_tensor_counts(dst: Dict[str, float], pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> None:
    dst["tp"] += float((pred * target * mask).sum().item())
    dst["pred"] += float((pred * mask).sum().item())
    dst["true"] += float((target * mask).sum().item())
    dst["pairs"] += float(mask.sum().item())


def add_diagnostics(
    dst: Dict[str, float],
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    relation_mean: torch.Tensor,
    support_mean: torch.Tensor,
    relation_std: torch.Tensor,
    support_std: torch.Tensor,
) -> None:
    hint_missed = mask * (target > 0.5).float() * (relation_mean < 0.5).float()
    hint_supported_fp = mask * (target < 0.5).float() * (relation_mean >= 0.5).float()
    rescue_scope = mask * (relation_mean < 0.5).float() * (support_mean >= 0.55).float()
    agreement = mask * (relation_std <= 0.08).float() * (support_std <= 0.08).float()
    disagreement = mask * ((relation_std >= 0.15) | (support_std >= 0.15)).float()

    dst["hint_missed_true"] += float(hint_missed.sum().item())
    dst["hint_missed_pred"] += float((pred * hint_missed).sum().item())
    dst["hint_supported_fp_total"] += float(hint_supported_fp.sum().item())
    dst["hint_supported_fp_pred"] += float((pred * hint_supported_fp).sum().item())
    dst["rescue_pred"] += float((pred * rescue_scope).sum().item())
    dst["rescue_tp"] += float((pred * target * rescue_scope).sum().item())
    dst["rescue_true"] += float((target * rescue_scope).sum().item())
    dst["agreement_pred"] += float((pred * agreement).sum().item())
    dst["agreement_tp"] += float((pred * target * agreement).sum().item())
    dst["disagreement_pred"] += float((pred * disagreement).sum().item())
    dst["disagreement_tp"] += float((pred * target * disagreement).sum().item())
    dst["disagreement_fp"] += float((pred * (target < 0.5).float() * disagreement).sum().item())
    dst["disagreement_neg"] += float(((target < 0.5).float() * disagreement).sum().item())


def finalize_counts(counts: Dict[str, float]) -> Dict[str, float]:
    precision = counts["tp"] / counts["pred"] if counts["pred"] > 0 else 0.0
    recall = counts["tp"] / counts["true"] if counts["true"] > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8) if precision + recall > 0 else 0.0
    rescue_precision = (
        counts["rescue_tp"] / counts["rescue_pred"] if counts["rescue_pred"] > 0 else 0.0
    )
    rescue_recall = (
        counts["rescue_tp"] / counts["rescue_true"] if counts["rescue_true"] > 0 else 0.0
    )
    rescue_f1 = (
        2.0 * rescue_precision * rescue_recall / (rescue_precision + rescue_recall + 1e-8)
        if rescue_precision + rescue_recall > 0
        else 0.0
    )
    agreement_precision = (
        counts["agreement_tp"] / counts["agreement_pred"] if counts["agreement_pred"] > 0 else 0.0
    )
    disagreement_precision = (
        counts["disagreement_tp"] / counts["disagreement_pred"]
        if counts["disagreement_pred"] > 0
        else 0.0
    )
    disagreement_fp_rate = (
        counts["disagreement_fp"] / counts["disagreement_neg"]
        if counts["disagreement_neg"] > 0
        else 0.0
    )
    return {
        "edge_precision": precision,
        "edge_recall": recall,
        "edge_f1": f1,
        "edge_tp": counts["tp"],
        "edge_pred_pos": counts["pred"],
        "edge_true_pos": counts["true"],
        "pair_count": counts["pairs"],
        "hint_missed_true_recall": (
            counts["hint_missed_pred"] / counts["hint_missed_true"]
            if counts["hint_missed_true"] > 0
            else 0.0
        ),
        "hint_supported_fp_error": (
            counts["hint_supported_fp_pred"] / counts["hint_supported_fp_total"]
            if counts["hint_supported_fp_total"] > 0
            else 0.0
        ),
        "rescue_scope_precision": rescue_precision,
        "rescue_scope_recall": rescue_recall,
        "rescue_scope_f1": rescue_f1,
        "agreement_pred_precision": agreement_precision,
        "disagreement_pred_precision": disagreement_precision,
        "disagreement_fp_rate": disagreement_fp_rate,
    }


def trivial_multi_view_outputs(batch: Dict[str, Any], num_node_types: int, state_dim: int) -> Dict[str, torch.Tensor]:
    slots = batch["multi_view_slot_features"]
    relation = batch["multi_view_relation_hints"]
    support = batch["multi_view_pair_support_hints"]
    witness = batch["multi_view_signed_pair_witness"]
    bundle = batch["multi_view_pair_evidence_bundle"]
    slot_mean = slots.mean(dim=1)
    type_hint = slot_mean[:, :, :num_node_types]
    state_start = num_node_types + 1
    state_hint = slot_mean[:, :, state_start : state_start + state_dim]

    relation_mean = relation.mean(dim=1)
    support_mean = support.mean(dim=1)
    relation_std = relation.std(dim=1, unbiased=False)
    support_std = support.std(dim=1, unbiased=False)
    witness_hint = (0.5 + 0.5 * witness.mean(dim=1)).clamp(0.0, 1.0)
    bundle_mean = bundle.mean(dim=1)
    bundle_hint = (
        0.35 * bundle_mean[..., 0]
        + 0.30 * (1.0 - bundle_mean[..., 1])
        + 0.20 * bundle_mean[..., 2]
        + 0.15 * bundle_mean[..., 3]
    ).clamp(0.0, 1.0)
    agreement = (1.0 - 2.0 * (relation_std + support_std).clamp(0.0, 0.5)).clamp(0.0, 1.0)
    edge_hint = (
        0.35 * relation_mean
        + 0.25 * support_mean
        + 0.15 * witness_hint
        + 0.20 * bundle_hint
        + 0.05 * agreement
    ).clamp(0.0, 1.0)
    return {
        "type_logits": type_hint * 10.0,
        "state_pred": state_hint,
        "edge_logits": logit_from_hint(edge_hint),
    }


def single_view_outputs_for_view(
    model: Step30WeakObservationEncoder,
    batch: Dict[str, Any],
    view_idx: int,
) -> Dict[str, torch.Tensor]:
    return model(
        weak_slot_features=batch["multi_view_slot_features"][:, view_idx],
        weak_relation_hints=batch["multi_view_relation_hints"][:, view_idx],
        weak_pair_support_hints=batch["multi_view_pair_support_hints"][:, view_idx],
        weak_signed_pair_witness=batch["multi_view_signed_pair_witness"][:, view_idx],
        weak_pair_evidence_bundle=batch["multi_view_pair_evidence_bundle"][:, view_idx],
    )


def simple_late_fusion_outputs(
    model: Step30WeakObservationEncoder,
    batch: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    view_outputs = [
        single_view_outputs_for_view(model, batch, view_idx)
        for view_idx in range(int(batch["multi_view_slot_features"].shape[1]))
    ]
    return {
        "type_logits": torch.stack([out["type_logits"] for out in view_outputs], dim=0).mean(dim=0),
        "state_pred": torch.stack([out["state_pred"] for out in view_outputs], dim=0).mean(dim=0),
        "edge_logits": torch.stack([out["edge_logits"] for out in view_outputs], dim=0).mean(dim=0),
    }


def flatten_row_name(row: str, group: str, name: str, metrics: Dict[str, float]) -> Dict[str, Any]:
    return {"row": row, "group": group, "name": name, **metrics}


@torch.no_grad()
def evaluate(
    step31_model: Step31MultiViewObservationEncoder,
    single_view_model: Step30WeakObservationEncoder,
    loader: DataLoader,
    device: torch.device,
    thresholds: Dict[str, float],
) -> Dict[str, Any]:
    row_counts: Dict[str, Dict[str, float]] = {
        "single_view_baseline": empty_counts(),
        "simple_late_fusion_baseline": empty_counts(),
        "trivial_multi_view": empty_counts(),
        "step31_multi_view_encoder": empty_counts(),
    }
    by_variant: Dict[str, Dict[str, Dict[str, float]]] = {row: {} for row in row_counts}
    by_event: Dict[str, Dict[str, Dict[str, float]]] = {row: {} for row in row_counts}

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        variants = [str(v) for v in batch["step31_observation_variant"]]
        events_list = batch.get("events", [None] * int(batch["target_adj"].shape[0]))
        threshold_tensor = threshold_tensor_for_variants(variants, thresholds, device)
        target = batch["target_adj"].float()
        pair_mask = build_pair_mask(batch["node_mask"])
        relation_mean = batch["multi_view_relation_hints"].mean(dim=1)
        support_mean = batch["multi_view_pair_support_hints"].mean(dim=1)
        relation_std = batch["multi_view_relation_hints"].std(dim=1, unbiased=False)
        support_std = batch["multi_view_pair_support_hints"].std(dim=1, unbiased=False)

        outputs_by_row = {
            "single_view_baseline": single_view_outputs_for_view(single_view_model, batch, 0),
            "simple_late_fusion_baseline": simple_late_fusion_outputs(single_view_model, batch),
            "trivial_multi_view": trivial_multi_view_outputs(
                batch,
                num_node_types=single_view_model.config.num_node_types,
                state_dim=single_view_model.config.state_dim,
            ),
            "step31_multi_view_encoder": step31_model(
                multi_view_slot_features=batch["multi_view_slot_features"],
                multi_view_relation_hints=batch["multi_view_relation_hints"],
                multi_view_pair_support_hints=batch["multi_view_pair_support_hints"],
                multi_view_signed_pair_witness=batch["multi_view_signed_pair_witness"],
                multi_view_pair_evidence_bundle=batch["multi_view_pair_evidence_bundle"],
            ),
        }

        for row_name, outputs in outputs_by_row.items():
            pred = (torch.sigmoid(outputs["edge_logits"]) >= threshold_tensor).float()
            add_tensor_counts(row_counts[row_name], pred, target, pair_mask)
            add_diagnostics(
                row_counts[row_name],
                pred,
                target,
                pair_mask,
                relation_mean=relation_mean,
                support_mean=support_mean,
                relation_std=relation_std,
                support_std=support_std,
            )
            for idx, variant in enumerate(variants):
                by_variant[row_name].setdefault(variant, empty_counts())
                sample_slice = slice(idx, idx + 1)
                add_tensor_counts(
                    by_variant[row_name][variant],
                    pred[sample_slice],
                    target[sample_slice],
                    pair_mask[sample_slice],
                )
                add_diagnostics(
                    by_variant[row_name][variant],
                    pred[sample_slice],
                    target[sample_slice],
                    pair_mask[sample_slice],
                    relation_mean=relation_mean[sample_slice],
                    support_mean=support_mean[sample_slice],
                    relation_std=relation_std[sample_slice],
                    support_std=support_std[sample_slice],
                )
                for family in event_families(events_list[idx]):
                    by_event[row_name].setdefault(family, empty_counts())
                    add_tensor_counts(
                        by_event[row_name][family],
                        pred[sample_slice],
                        target[sample_slice],
                        pair_mask[sample_slice],
                    )

    recovery_rows = []
    event_rows = []
    for row_name, counts in row_counts.items():
        overall = finalize_counts(counts)
        clean = finalize_counts(by_variant[row_name].get("clean", empty_counts()))
        noisy = finalize_counts(by_variant[row_name].get("noisy", empty_counts()))
        recovery_rows.append(
            {
                "row": row_name,
                "overall_precision": overall["edge_precision"],
                "overall_recall": overall["edge_recall"],
                "overall_f1": overall["edge_f1"],
                "clean_precision": clean["edge_precision"],
                "clean_recall": clean["edge_recall"],
                "clean_f1": clean["edge_f1"],
                "noisy_precision": noisy["edge_precision"],
                "noisy_recall": noisy["edge_recall"],
                "noisy_f1": noisy["edge_f1"],
                "hint_missed_true_recall": noisy["hint_missed_true_recall"],
                "hint_supported_fp_error": noisy["hint_supported_fp_error"],
                "rescue_scope_precision": noisy["rescue_scope_precision"],
                "rescue_scope_recall": noisy["rescue_scope_recall"],
                "agreement_pred_precision": noisy["agreement_pred_precision"],
                "disagreement_pred_precision": noisy["disagreement_pred_precision"],
                "disagreement_fp_rate": noisy["disagreement_fp_rate"],
            }
        )
        for family, family_counts in sorted(by_event[row_name].items()):
            event_metrics = finalize_counts(family_counts)
            event_rows.append(
                {
                    "row": row_name,
                    "event_family": family,
                    "edge_precision": event_metrics["edge_precision"],
                    "edge_recall": event_metrics["edge_recall"],
                    "edge_f1": event_metrics["edge_f1"],
                }
            )

    step31_row = next(row for row in recovery_rows if row["row"] == "step31_multi_view_encoder")
    single_row = next(row for row in recovery_rows if row["row"] == "single_view_baseline")
    trivial_row = next(row for row in recovery_rows if row["row"] == "trivial_multi_view")
    gate = {
        "beats_step30_rev6_noisy_f1": bool(
            step31_row["noisy_f1"] > STEP30_REV6_REFERENCE["noisy_f1"]
        ),
        "improves_over_single_view_noisy_f1": bool(
            step31_row["noisy_f1"] > single_row["noisy_f1"]
        ),
        "trivial_clearly_below_encoder": bool(
            trivial_row["noisy_f1"] + 0.05 < step31_row["noisy_f1"]
        ),
        "backend_rerun": False,
        "backend_rerun_reason": "Backend integration was not run; Step31 is recovery-first and gated.",
    }
    return {
        "step30_rev6_reference": STEP30_REV6_REFERENCE,
        "recovery_rows": recovery_rows,
        "event_family_rows": event_rows,
        "gate": gate,
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
    parser.add_argument("--data_path", default="data/graph_event_step31_multi_view_test.pkl")
    parser.add_argument("--step31_checkpoint", default="checkpoints/step31_multi_view_encoder/best.pt")
    parser.add_argument("--single_view_checkpoint", default="checkpoints/step31_single_view_baseline/best.pt")
    parser.add_argument("--output_dir", default="artifacts/step31_multi_view_bridge_probe")
    parser.add_argument("--thresholds", default="clean:0.50,noisy:0.55")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = get_device(args.device)
    dataset = Step31MultiViewObservationDataset(args.data_path)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=step31_multi_view_collate_fn,
        pin_memory=device.type == "cuda",
    )
    step31_model = load_step31_model(args.step31_checkpoint, device)
    single_view_model = load_step30_model(args.single_view_checkpoint, device)
    result = evaluate(
        step31_model=step31_model,
        single_view_model=single_view_model,
        loader=loader,
        device=device,
        thresholds=parse_thresholds(args.thresholds),
    )
    result["metadata"] = {
        "data_path": args.data_path,
        "step31_checkpoint": args.step31_checkpoint,
        "single_view_checkpoint": args.single_view_checkpoint,
        "num_samples": len(dataset),
        "thresholds": parse_thresholds(args.thresholds),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "recovery_summary.csv", result["recovery_rows"])
    write_csv(output_dir / "event_family_summary.csv", result["event_family_rows"])
    summary = clean_json_numbers(result)
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
