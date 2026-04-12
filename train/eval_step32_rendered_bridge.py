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

from data.step32_dataset import Step32RenderedObservationDataset, step32_rendered_collate_fn
from models.encoder_step30 import build_pair_mask, logit_from_hint
from models.encoder_step32 import Step32RenderedBridgeConfig, Step32RenderedObservationEncoder
from train.eval_step30_encoder_recovery import get_device, move_batch_to_device
from train.eval_step31_multi_view_bridge import STEP30_REV6_REFERENCE, event_families


STEP31_SIMPLE_LATE_FUSION_REFERENCE = {
    "row": "step31_simple_late_fusion_reference",
    "overall_precision": 0.9265511311453648,
    "overall_recall": 0.9131151129458384,
    "overall_f1": 0.9197840375169789,
    "clean_f1": 0.9702632432695112,
    "noisy_precision": 0.8821,
    "noisy_recall": 0.8501,
    "noisy_f1": 0.8658,
}


def load_step32_model(checkpoint_path: str, device: torch.device) -> Step32RenderedObservationEncoder:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = Step32RenderedBridgeConfig(**dict(checkpoint["config"]))
    model = Step32RenderedObservationEncoder(config).to(device)
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


def empty_counts() -> Dict[str, float]:
    return {
        "tp": 0.0,
        "pred": 0.0,
        "true": 0.0,
        "pairs": 0.0,
        "render_missed_true": 0.0,
        "render_missed_pred": 0.0,
        "render_supported_fp_total": 0.0,
        "render_supported_fp_pred": 0.0,
        "render_rescue_pred": 0.0,
        "render_rescue_tp": 0.0,
        "render_rescue_true": 0.0,
        "low_signal_true": 0.0,
        "low_signal_pred": 0.0,
    }


def add_tensor_counts(dst: Dict[str, float], pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> None:
    dst["tp"] += float((pred * target * mask).sum().item())
    dst["pred"] += float((pred * mask).sum().item())
    dst["true"] += float((target * mask).sum().item())
    dst["pairs"] += float(mask.sum().item())


def add_render_diagnostics(
    dst: Dict[str, float],
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    relation_score: torch.Tensor,
    support_score: torch.Tensor,
) -> None:
    missed = mask * (target > 0.5).float() * (relation_score < 0.35).float()
    supported_fp = mask * (target < 0.5).float() * (relation_score >= 0.48).float()
    rescue_scope = mask * (relation_score < 0.35).float() * (support_score >= 0.34).float()
    low_signal = mask * (relation_score < 0.25).float() * (support_score < 0.25).float()
    dst["render_missed_true"] += float(missed.sum().item())
    dst["render_missed_pred"] += float((pred * missed).sum().item())
    dst["render_supported_fp_total"] += float(supported_fp.sum().item())
    dst["render_supported_fp_pred"] += float((pred * supported_fp).sum().item())
    dst["render_rescue_pred"] += float((pred * rescue_scope).sum().item())
    dst["render_rescue_tp"] += float((pred * target * rescue_scope).sum().item())
    dst["render_rescue_true"] += float((target * rescue_scope).sum().item())
    dst["low_signal_true"] += float((target * low_signal).sum().item())
    dst["low_signal_pred"] += float((pred * low_signal).sum().item())


def finalize_counts(counts: Dict[str, float]) -> Dict[str, float]:
    precision = counts["tp"] / counts["pred"] if counts["pred"] > 0 else 0.0
    recall = counts["tp"] / counts["true"] if counts["true"] > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8) if precision + recall > 0 else 0.0
    rescue_precision = (
        counts["render_rescue_tp"] / counts["render_rescue_pred"]
        if counts["render_rescue_pred"] > 0
        else 0.0
    )
    rescue_recall = (
        counts["render_rescue_tp"] / counts["render_rescue_true"]
        if counts["render_rescue_true"] > 0
        else 0.0
    )
    rescue_f1 = (
        2.0 * rescue_precision * rescue_recall / (rescue_precision + rescue_recall + 1e-8)
        if rescue_precision + rescue_recall > 0
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
        "render_missed_true_recall": (
            counts["render_missed_pred"] / counts["render_missed_true"]
            if counts["render_missed_true"] > 0
            else 0.0
        ),
        "render_supported_fp_error": (
            counts["render_supported_fp_pred"] / counts["render_supported_fp_total"]
            if counts["render_supported_fp_total"] > 0
            else 0.0
        ),
        "render_rescue_scope_precision": rescue_precision,
        "render_rescue_scope_recall": rescue_recall,
        "render_rescue_scope_f1": rescue_f1,
        "low_signal_true_rate": (
            counts["low_signal_true"] / counts["pairs"] if counts["pairs"] > 0 else 0.0
        ),
        "low_signal_pred_rate": (
            counts["low_signal_pred"] / counts["pairs"] if counts["pairs"] > 0 else 0.0
        ),
    }


def trivial_relation_outputs(batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    relation = batch["rendered_trivial_relation_scores"].mean(dim=1).clamp(0.0, 1.0)
    num_types = 3
    state_dim = int(batch["target_node_feats"].shape[-1] - 1)
    return {
        "type_logits": batch["target_node_feats"].new_zeros(
            batch["target_node_feats"].shape[0],
            batch["target_node_feats"].shape[1],
            num_types,
        ),
        "state_pred": batch["target_node_feats"].new_zeros(
            batch["target_node_feats"].shape[0],
            batch["target_node_feats"].shape[1],
            state_dim,
        ),
        "edge_logits": logit_from_hint(relation),
    }


def rendered_projection_outputs(batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    relation = batch["rendered_trivial_relation_scores"].mean(dim=1)
    support = batch["rendered_trivial_support_scores"].mean(dim=1)
    score = (0.68 * relation + 0.32 * support).clamp(0.0, 1.0)
    num_types = 3
    state_dim = int(batch["target_node_feats"].shape[-1] - 1)
    return {
        "type_logits": batch["target_node_feats"].new_zeros(
            batch["target_node_feats"].shape[0],
            batch["target_node_feats"].shape[1],
            num_types,
        ),
        "state_pred": batch["target_node_feats"].new_zeros(
            batch["target_node_feats"].shape[0],
            batch["target_node_feats"].shape[1],
            state_dim,
        ),
        "edge_logits": logit_from_hint(score),
    }


@torch.no_grad()
def evaluate(
    model: Step32RenderedObservationEncoder,
    loader: DataLoader,
    device: torch.device,
    thresholds: Dict[str, float],
    baseline_thresholds: Dict[str, float],
) -> Dict[str, Any]:
    row_counts: Dict[str, Dict[str, float]] = {
        "trivial_rendered_relation": empty_counts(),
        "rendered_projection_baseline": empty_counts(),
        "step32_rendered_bridge": empty_counts(),
    }
    by_variant: Dict[str, Dict[str, Dict[str, float]]] = {row: {} for row in row_counts}
    by_event: Dict[str, Dict[str, Dict[str, float]]] = {row: {} for row in row_counts}

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        variants = [str(v) for v in batch["step32_observation_variant"]]
        events_list = batch.get("events", [None] * int(batch["target_adj"].shape[0]))
        model_threshold = threshold_tensor_for_variants(variants, thresholds, device)
        baseline_threshold = threshold_tensor_for_variants(variants, baseline_thresholds, device)
        target = batch["target_adj"].float()
        pair_mask = build_pair_mask(batch["node_mask"])
        relation_score = batch["rendered_trivial_relation_scores"].mean(dim=1)
        support_score = batch["rendered_trivial_support_scores"].mean(dim=1)

        outputs_by_row = {
            "trivial_rendered_relation": trivial_relation_outputs(batch),
            "rendered_projection_baseline": rendered_projection_outputs(batch),
            "step32_rendered_bridge": model(
                rendered_images=batch["rendered_images"],
                rendered_node_positions=batch["rendered_node_positions"],
                rendered_visible_node_mask=batch["rendered_visible_node_mask"],
            ),
        }

        for row_name, outputs in outputs_by_row.items():
            threshold = model_threshold if row_name == "step32_rendered_bridge" else baseline_threshold
            pred = (torch.sigmoid(outputs["edge_logits"]) >= threshold).float()
            add_tensor_counts(row_counts[row_name], pred, target, pair_mask)
            add_render_diagnostics(
                row_counts[row_name],
                pred,
                target,
                pair_mask,
                relation_score=relation_score,
                support_score=support_score,
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
                add_render_diagnostics(
                    by_variant[row_name][variant],
                    pred[sample_slice],
                    target[sample_slice],
                    pair_mask[sample_slice],
                    relation_score=relation_score[sample_slice],
                    support_score=support_score[sample_slice],
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
                "render_missed_true_recall": noisy["render_missed_true_recall"],
                "render_supported_fp_error": noisy["render_supported_fp_error"],
                "render_rescue_scope_precision": noisy["render_rescue_scope_precision"],
                "render_rescue_scope_recall": noisy["render_rescue_scope_recall"],
                "render_rescue_scope_f1": noisy["render_rescue_scope_f1"],
                "low_signal_true_rate": noisy["low_signal_true_rate"],
                "low_signal_pred_rate": noisy["low_signal_pred_rate"],
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

    rendered_row = next(row for row in recovery_rows if row["row"] == "step32_rendered_bridge")
    trivial_row = next(row for row in recovery_rows if row["row"] == "trivial_rendered_relation")
    gate = {
        "beats_step30_rev6_overall_f1": bool(
            rendered_row["overall_f1"] > STEP30_REV6_REFERENCE["overall_f1"]
        ),
        "beats_step30_rev6_noisy_f1": bool(
            rendered_row["noisy_f1"] > STEP30_REV6_REFERENCE["noisy_f1"]
        ),
        "trivial_clearly_below_learned": bool(
            trivial_row["noisy_f1"] + 0.05 < rendered_row["noisy_f1"]
        ),
        "backend_transfer_rerun": bool(
            rendered_row["overall_f1"] > STEP30_REV6_REFERENCE["overall_f1"]
            and rendered_row["noisy_f1"] > STEP30_REV6_REFERENCE["noisy_f1"]
            and trivial_row["noisy_f1"] + 0.05 < rendered_row["noisy_f1"]
        ),
        "backend_transfer_policy": (
            "Only run focused backend transfer if rendered recovery clearly beats Step30 rev6 "
            "and learned decoding is materially above trivial rendered decoding."
        ),
    }
    return {
        "step30_rev6_reference": STEP30_REV6_REFERENCE,
        "step31_simple_late_fusion_reference": STEP31_SIMPLE_LATE_FUSION_REFERENCE,
        "recovery_rows": recovery_rows,
        "event_family_rows": event_rows,
        "gate": gate,
    }


def write_csv(path: Path, rows: list[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/graph_event_step32_rendered_test.pkl")
    parser.add_argument("--step32_checkpoint", default="checkpoints/step32_rendered_bridge/best.pt")
    parser.add_argument("--output_dir", default="artifacts/step32_rendered_bridge_probe")
    parser.add_argument("--thresholds", default="clean:0.50,noisy:0.50")
    parser.add_argument("--baseline_thresholds", default="clean:0.42,noisy:0.42")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = get_device(args.device)
    dataset = Step32RenderedObservationDataset(args.data_path)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=step32_rendered_collate_fn,
        pin_memory=device.type == "cuda",
    )
    model = load_step32_model(args.step32_checkpoint, device)
    result = evaluate(
        model=model,
        loader=loader,
        device=device,
        thresholds=parse_thresholds(args.thresholds),
        baseline_thresholds=parse_thresholds(args.baseline_thresholds),
    )
    result["metadata"] = {
        "data_path": args.data_path,
        "step32_checkpoint": args.step32_checkpoint,
        "num_samples": len(dataset),
        "thresholds": parse_thresholds(args.thresholds),
        "baseline_thresholds": parse_thresholds(args.baseline_thresholds),
        "non_leakage_note": (
            "Rendered images are produced from weak corrupted observation cues; "
            "ground-truth adjacency is kept only as target/evaluation label."
        ),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "recovery_summary.csv", result["recovery_rows"])
    write_csv(output_dir / "event_family_summary.csv", result["event_family_rows"])
    summary = clean_json_numbers(result)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
