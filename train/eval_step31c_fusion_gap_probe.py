from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.step30_dataset import Step30WeakObservationDataset, step30_weak_observation_collate_fn
from data.step31_dataset import Step31MultiViewObservationDataset, step31_multi_view_collate_fn
from models.encoder_step30 import build_pair_mask
from train.eval_step30_encoder_recovery import load_model as load_step30_model
from train.eval_step30_frozen_backend_integration import (
    add_deltas,
    backend_forward_records,
    build_backend_specs,
    clean_json_numbers,
    decode_inputs,
    grouped_summaries,
    hard_adj_from_logits,
    load_encoder,
    move_batch_to_device,
    recovery_metrics_for_decoded,
    resolve_path,
    threshold_for_batch_variants,
    write_csv,
)
from train.eval_step31_multi_view_bridge import (
    add_diagnostics,
    add_tensor_counts,
    empty_counts,
    event_families,
    finalize_counts,
    load_step31_model,
    parse_thresholds,
    simple_late_fusion_outputs,
    single_view_outputs_for_view,
    threshold_tensor_for_variants,
    trivial_multi_view_outputs,
)


STEP31C_ROW = "step31c_agreement_damped_encoder"
STEP31D_ROW = "step31d_late_fusion_distilled_encoder"
STEP31E_ROW = "step31e_recall_preserving_teacher_encoder"


def decoded_from_outputs(
    outputs: Dict[str, torch.Tensor],
    edge_threshold: float | torch.Tensor,
) -> Dict[str, torch.Tensor]:
    pred_type = outputs["type_logits"].argmax(dim=-1).float().unsqueeze(-1)
    return {
        "node_feats": torch.cat([pred_type, outputs["state_pred"]], dim=-1),
        "adj": hard_adj_from_logits(outputs["edge_logits"], threshold=edge_threshold),
    }


def agreement_damped_outputs(
    learned_outputs: Dict[str, torch.Tensor],
    late_outputs: Dict[str, torch.Tensor],
    relation_std: torch.Tensor,
    support_std: torch.Tensor,
    disagreement_start: float,
    disagreement_width: float,
    damping_scale: float,
) -> Dict[str, torch.Tensor]:
    """Conservatively damp learned-only positive edge evidence in unstable pairs.

    The probe leaves the learned node path unchanged. It only moves learned edge
    logits toward late-fusion logits when views disagree and learned fusion is
    more positive than late fusion.
    """
    pair_disagreement = torch.maximum(relation_std, support_std)
    gate = ((pair_disagreement - disagreement_start) / max(disagreement_width, 1e-6)).clamp(
        0.0,
        1.0,
    )
    learned_edge_logits = learned_outputs["edge_logits"]
    late_edge_logits = late_outputs["edge_logits"]
    positive_excess = torch.relu(learned_edge_logits - late_edge_logits)
    damped_edge_logits = learned_edge_logits - float(damping_scale) * gate * positive_excess
    damped_edge_logits = 0.5 * (damped_edge_logits + damped_edge_logits.transpose(1, 2))
    diag_mask = torch.eye(
        damped_edge_logits.shape[1],
        device=damped_edge_logits.device,
        dtype=torch.bool,
    ).unsqueeze(0)
    damped_edge_logits = damped_edge_logits.masked_fill(diag_mask, -1e9)
    return {
        "type_logits": learned_outputs["type_logits"],
        "state_pred": learned_outputs["state_pred"],
        "edge_logits": damped_edge_logits,
    }


def bucket_name(relation_std: torch.Tensor, support_std: torch.Tensor) -> torch.Tensor:
    max_std = torch.maximum(relation_std, support_std)
    # 0=agreement, 1=mid, 2=disagreement
    bucket = torch.ones_like(max_std, dtype=torch.long)
    bucket = torch.where(max_std <= 0.08, torch.zeros_like(bucket), bucket)
    bucket = torch.where(max_std >= 0.15, torch.full_like(bucket, 2), bucket)
    return bucket


def empty_bucket_counts() -> Dict[str, float]:
    counts = empty_counts()
    counts.update(
        {
            "score_sum": 0.0,
            "score_count": 0.0,
            "pred_score_sum": 0.0,
            "fp_score_sum": 0.0,
            "fp_count": 0.0,
            "learned_excess_sum": 0.0,
            "learned_excess_fp_sum": 0.0,
        }
    )
    return counts


def add_bucket_counts(
    dst: Dict[str, float],
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    score: torch.Tensor,
    learned_logits: torch.Tensor,
    late_logits: torch.Tensor,
) -> None:
    add_tensor_counts(dst, pred, target, mask)
    masked_score = score * mask
    pred_mask = pred * mask
    fp_mask = pred * (target < 0.5).float() * mask
    learned_excess = torch.relu(learned_logits - late_logits)
    dst["score_sum"] += float(masked_score.sum().item())
    dst["score_count"] += float(mask.sum().item())
    dst["pred_score_sum"] += float((score * pred_mask).sum().item())
    dst["fp_score_sum"] += float((score * fp_mask).sum().item())
    dst["fp_count"] += float(fp_mask.sum().item())
    dst["learned_excess_sum"] += float((learned_excess * mask).sum().item())
    dst["learned_excess_fp_sum"] += float((learned_excess * fp_mask).sum().item())


def finalize_bucket_counts(counts: Dict[str, float]) -> Dict[str, float]:
    metrics = finalize_counts(counts)
    pred = counts["pred"]
    pairs = counts["pairs"]
    true = counts["true"]
    metrics.update(
        {
            "gt_positive_rate": true / pairs if pairs > 0 else 0.0,
            "pred_positive_rate": pred / pairs if pairs > 0 else 0.0,
            "avg_score": counts["score_sum"] / counts["score_count"]
            if counts["score_count"] > 0
            else 0.0,
            "avg_pred_score": counts["pred_score_sum"] / pred if pred > 0 else 0.0,
            "avg_fp_score": counts["fp_score_sum"] / counts["fp_count"]
            if counts["fp_count"] > 0
            else 0.0,
            "avg_learned_positive_excess": counts["learned_excess_sum"] / pairs
            if pairs > 0
            else 0.0,
            "avg_learned_positive_excess_on_fp": counts["learned_excess_fp_sum"] / counts["fp_count"]
            if counts["fp_count"] > 0
            else 0.0,
        }
    )
    return metrics


def write_rows(path: Path, rows: list[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def add_backend_records_for_modes(
    all_records: list[Dict[str, Any]],
    backends: list[Any],
    batch: Dict[str, Any],
    decoded_inputs: Dict[str, Dict[str, torch.Tensor]],
) -> None:
    recovery_by_mode: Dict[str, list[Dict[str, Any]]] = {}
    for input_mode, decoded in decoded_inputs.items():
        recovery_records, _ = recovery_metrics_for_decoded(
            decoded_node_feats=decoded["node_feats"],
            decoded_adj=decoded["adj"],
            target_node_feats=batch["target_node_feats"],
            target_adj=batch["target_adj"],
            node_mask=batch["node_mask"],
        )
        recovery_by_mode[input_mode] = recovery_records

    for backend in backends:
        for input_mode, decoded in decoded_inputs.items():
            records = backend_forward_records(
                backend=backend,
                input_node_feats=decoded["node_feats"],
                input_adj=decoded["adj"],
                batch=batch,
                recovery_records=recovery_by_mode[input_mode],
            )
            for record in records:
                record["backend"] = backend.name
                record["input_mode"] = input_mode
            all_records.extend(records)


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    device = torch.device(args.device) if args.device != "auto" else (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    thresholds = parse_thresholds(args.thresholds)
    edge_thresholds_by_variant = parse_thresholds(args.recovered_edge_thresholds_by_variant)
    backends = build_backend_specs(args, device)

    step30_rev6_model = load_encoder(resolve_path(args.step30_rev6_checkpoint), device)
    single_view_model = load_step30_model(args.step31_single_view_checkpoint, device)
    multi_view_model = load_step31_model(args.step31_multi_view_checkpoint, device)
    step31d_model = (
        load_step31_model(args.step31d_checkpoint, device)
        if args.step31d_checkpoint is not None
        else None
    )
    step31e_model = (
        load_step31_model(args.step31e_checkpoint, device)
        if args.step31e_checkpoint is not None
        else None
    )

    row_names = [
        "single_view_baseline",
        "simple_late_fusion_baseline",
        "step31_multi_view_encoder",
        STEP31C_ROW,
    ]
    if step31d_model is not None:
        row_names.append(STEP31D_ROW)
    if step31e_model is not None:
        row_names.append(STEP31E_ROW)
    row_counts: Dict[str, Dict[str, float]] = {row: empty_counts() for row in row_names}
    by_variant: Dict[str, Dict[str, Dict[str, float]]] = {row: {} for row in row_names}
    by_event: Dict[str, Dict[str, Dict[str, float]]] = {row: {} for row in row_names}
    bucket_counts: Dict[tuple[str, str, str], Dict[str, float]] = {}
    backend_records: list[Dict[str, Any]] = []

    step30_dataset = Step30WeakObservationDataset(args.step30_rev6_data_path)
    step30_loader = DataLoader(
        step30_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=step30_weak_observation_collate_fn,
        pin_memory=device.type == "cuda",
    )
    for batch_idx, batch in enumerate(step30_loader):
        if args.limit_batches is not None and batch_idx >= args.limit_batches:
            break
        batch = move_batch_to_device(batch, device)
        decoded = decode_inputs(
            batch=batch,
            encoder_model=step30_rev6_model,
            edge_threshold=args.recovered_edge_threshold,
            edge_thresholds_by_variant=edge_thresholds_by_variant,
        )
        add_backend_records_for_modes(
            backend_records,
            backends,
            batch,
            {
                "gt_structured": decoded["gt_structured"],
                "step30_rev6_reference": decoded["encoder_recovered"],
            },
        )

    step31_dataset = Step31MultiViewObservationDataset(args.step31_data_path)
    step31_loader = DataLoader(
        step31_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=step31_multi_view_collate_fn,
        pin_memory=device.type == "cuda",
    )

    for batch_idx, batch in enumerate(step31_loader):
        if args.limit_batches is not None and batch_idx >= args.limit_batches:
            break
        batch = move_batch_to_device(batch, device)
        batch["step30_observation_variant"] = batch["step31_observation_variant"]
        variants = [str(v) for v in batch["step31_observation_variant"]]
        events_list = batch.get("events", [None] * int(batch["target_adj"].shape[0]))
        recovery_threshold = threshold_tensor_for_variants(variants, thresholds, device)
        backend_threshold = threshold_for_batch_variants(
            variants=variants,
            default_threshold=args.recovered_edge_threshold,
            thresholds_by_variant=edge_thresholds_by_variant,
            device=batch["target_adj"].device,
        )
        target = batch["target_adj"].float()
        pair_mask = build_pair_mask(batch["node_mask"])
        relation_mean = batch["multi_view_relation_hints"].mean(dim=1)
        support_mean = batch["multi_view_pair_support_hints"].mean(dim=1)
        relation_std = batch["multi_view_relation_hints"].std(dim=1, unbiased=False)
        support_std = batch["multi_view_pair_support_hints"].std(dim=1, unbiased=False)
        buckets = bucket_name(relation_std, support_std)

        single_outputs = single_view_outputs_for_view(single_view_model, batch, 0)
        late_outputs = simple_late_fusion_outputs(single_view_model, batch)
        learned_outputs = multi_view_model(
            multi_view_slot_features=batch["multi_view_slot_features"],
            multi_view_relation_hints=batch["multi_view_relation_hints"],
            multi_view_pair_support_hints=batch["multi_view_pair_support_hints"],
            multi_view_signed_pair_witness=batch["multi_view_signed_pair_witness"],
            multi_view_pair_evidence_bundle=batch["multi_view_pair_evidence_bundle"],
        )
        probe_outputs = agreement_damped_outputs(
            learned_outputs=learned_outputs,
            late_outputs=late_outputs,
            relation_std=relation_std,
            support_std=support_std,
            disagreement_start=args.disagreement_start,
            disagreement_width=args.disagreement_width,
            damping_scale=args.damping_scale,
        )
        outputs_by_row = {
            "single_view_baseline": single_outputs,
            "simple_late_fusion_baseline": late_outputs,
            "step31_multi_view_encoder": learned_outputs,
            STEP31C_ROW: probe_outputs,
        }
        if step31d_model is not None:
            outputs_by_row[STEP31D_ROW] = step31d_model(
                multi_view_slot_features=batch["multi_view_slot_features"],
                multi_view_relation_hints=batch["multi_view_relation_hints"],
                multi_view_pair_support_hints=batch["multi_view_pair_support_hints"],
                multi_view_signed_pair_witness=batch["multi_view_signed_pair_witness"],
                multi_view_pair_evidence_bundle=batch["multi_view_pair_evidence_bundle"],
            )
        if step31e_model is not None:
            outputs_by_row[STEP31E_ROW] = step31e_model(
                multi_view_slot_features=batch["multi_view_slot_features"],
                multi_view_relation_hints=batch["multi_view_relation_hints"],
                multi_view_pair_support_hints=batch["multi_view_pair_support_hints"],
                multi_view_signed_pair_witness=batch["multi_view_signed_pair_witness"],
                multi_view_pair_evidence_bundle=batch["multi_view_pair_evidence_bundle"],
            )

        for row_name, outputs in outputs_by_row.items():
            pred = (torch.sigmoid(outputs["edge_logits"]) >= recovery_threshold).float()
            score = torch.sigmoid(outputs["edge_logits"])
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

            for variant_name, variant_mask in {
                "overall": torch.ones_like(pair_mask),
                "clean": torch.tensor(
                    [v == "clean" for v in variants],
                    device=device,
                    dtype=torch.float32,
                ).view(-1, 1, 1),
                "noisy": torch.tensor(
                    [v == "noisy" for v in variants],
                    device=device,
                    dtype=torch.float32,
                ).view(-1, 1, 1),
            }.items():
                valid_variant_mask = pair_mask * variant_mask
                for bucket_id, bucket_label in [
                    (0, "agreement"),
                    (1, "mid"),
                    (2, "disagreement"),
                ]:
                    bucket_mask = valid_variant_mask * (buckets == bucket_id).float()
                    key = (row_name, variant_name, bucket_label)
                    bucket_counts.setdefault(key, empty_bucket_counts())
                    add_bucket_counts(
                        bucket_counts[key],
                        pred,
                        target,
                        bucket_mask,
                        score,
                        learned_logits=learned_outputs["edge_logits"],
                        late_logits=late_outputs["edge_logits"],
                    )

        gt_decoded = {
            "node_feats": batch["target_node_feats"],
            "adj": batch["target_adj"],
        }
        decoded_inputs = {
            "gt_structured": gt_decoded,
            "step31_simple_late_fusion": decoded_from_outputs(late_outputs, backend_threshold),
            "step31_multi_view_encoder": decoded_from_outputs(learned_outputs, backend_threshold),
            STEP31C_ROW: decoded_from_outputs(probe_outputs, backend_threshold),
        }
        if step31d_model is not None:
            decoded_inputs[STEP31D_ROW] = decoded_from_outputs(
                outputs_by_row[STEP31D_ROW],
                backend_threshold,
            )
        if step31e_model is not None:
            decoded_inputs[STEP31E_ROW] = decoded_from_outputs(
                outputs_by_row[STEP31E_ROW],
                backend_threshold,
            )
        add_backend_records_for_modes(backend_records, backends, batch, decoded_inputs)

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
            metrics = finalize_counts(family_counts)
            event_rows.append(
                {
                    "row": row_name,
                    "event_family": family,
                    "edge_precision": metrics["edge_precision"],
                    "edge_recall": metrics["edge_recall"],
                    "edge_f1": metrics["edge_f1"],
                }
            )

    bucket_rows = []
    for (row_name, variant, bucket_label), counts in sorted(bucket_counts.items()):
        metrics = finalize_bucket_counts(counts)
        bucket_rows.append(
            {
                "row": row_name,
                "variant": variant,
                "bucket": bucket_label,
                "count": counts["pairs"],
                "edge_precision": metrics["edge_precision"],
                "edge_recall": metrics["edge_recall"],
                "edge_f1": metrics["edge_f1"],
                "gt_positive_rate": metrics["gt_positive_rate"],
                "pred_positive_rate": metrics["pred_positive_rate"],
                "avg_score": metrics["avg_score"],
                "avg_pred_score": metrics["avg_pred_score"],
                "avg_fp_score": metrics["avg_fp_score"],
                "avg_learned_positive_excess": metrics["avg_learned_positive_excess"],
                "avg_learned_positive_excess_on_fp": metrics["avg_learned_positive_excess_on_fp"],
            }
        )

    backend_rows = grouped_summaries(backend_records)
    add_deltas(backend_rows)

    return clean_json_numbers(
        {
            "metadata": {
                "step30_rev6_data_path": args.step30_rev6_data_path,
                "step31_data_path": args.step31_data_path,
                "step30_rev6_checkpoint": args.step30_rev6_checkpoint,
                "step31_single_view_checkpoint": args.step31_single_view_checkpoint,
                "step31_multi_view_checkpoint": args.step31_multi_view_checkpoint,
                "step31d_checkpoint": args.step31d_checkpoint,
                "step31e_checkpoint": args.step31e_checkpoint,
                "thresholds": thresholds,
                "recovered_edge_thresholds_by_variant": edge_thresholds_by_variant,
                "probe": {
                    "name": STEP31C_ROW,
                    "mechanism": "disagreement-gated damping of learned positive edge-logit excess toward late fusion",
                    "disagreement_start": args.disagreement_start,
                    "disagreement_width": args.disagreement_width,
                    "damping_scale": args.damping_scale,
                    "node_path": "unchanged learned multi-view encoder node predictions",
                },
                "notes": (
                    "Step31c is a diagnostic and narrow fusion probe only. It does not "
                    "train backend weights, add adapters, or introduce a new observation family."
                ),
            },
            "recovery_rows": recovery_rows,
            "event_family_rows": event_rows,
            "agreement_bucket_rows": bucket_rows,
            "backend_rows": backend_rows,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step30_rev6_data_path", default="data/graph_event_step30_weak_obs_rev6_test.pkl")
    parser.add_argument("--step31_data_path", default="data/graph_event_step31_multi_view_test.pkl")
    parser.add_argument("--step30_rev6_checkpoint", default="checkpoints/step30_encoder_recovery_rev6/best.pt")
    parser.add_argument("--step31_single_view_checkpoint", default="checkpoints/step31_single_view_baseline/best.pt")
    parser.add_argument("--step31_multi_view_checkpoint", default="checkpoints/step31_multi_view_encoder/best.pt")
    parser.add_argument("--step31d_checkpoint", default=None)
    parser.add_argument("--step31e_checkpoint", default=None)
    parser.add_argument("--output_dir", default="artifacts/step31c_fusion_gap_probe")
    parser.add_argument("--thresholds", default="clean:0.50,noisy:0.55")
    parser.add_argument("--recovered_edge_threshold", type=float, default=0.5)
    parser.add_argument("--recovered_edge_thresholds_by_variant", type=str, default="clean:0.50,noisy:0.55")
    parser.add_argument("--backends", type=str, default="w012,rft1_p2")
    parser.add_argument("--clean_proposal_checkpoint_path", type=str, default="checkpoints/scope_proposal_node_edge_flipw2/best.pt")
    parser.add_argument("--w012_checkpoint_path", type=str, default="checkpoints/fp_keep_w012/best.pt")
    parser.add_argument("--noisy_proposal_checkpoint_path", type=str, default="checkpoints/proposal_noisy_obs_p2/best.pt")
    parser.add_argument("--rft1_checkpoint_path", type=str, default="checkpoints/step6_noisy_rewrite_rft1/best.pt")
    parser.add_argument("--clean_node_threshold", type=float, default=0.20)
    parser.add_argument("--clean_edge_threshold", type=float, default=0.15)
    parser.add_argument("--noisy_node_threshold", type=float, default=0.15)
    parser.add_argument("--noisy_edge_threshold", type=float, default=0.10)
    parser.add_argument("--disagreement_start", type=float, default=0.08)
    parser.add_argument("--disagreement_width", type=float, default=0.07)
    parser.add_argument("--damping_scale", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--limit_batches", type=int, default=None)
    args = parser.parse_args()

    payload = evaluate(args)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    write_rows(output_dir / "recovery_summary.csv", payload["recovery_rows"])
    write_rows(output_dir / "event_family_summary.csv", payload["event_family_rows"])
    write_rows(output_dir / "agreement_bucket_summary.csv", payload["agreement_bucket_rows"])
    write_csv(output_dir / "backend_summary.csv", payload["backend_rows"])
    print(f"wrote JSON: {output_dir / 'summary.json'}")
    print(f"wrote recovery CSV: {output_dir / 'recovery_summary.csv'}")
    print(f"wrote backend CSV: {output_dir / 'backend_summary.csv'}")
    print(json.dumps(payload["recovery_rows"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
