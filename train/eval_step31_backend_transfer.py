from __future__ import annotations

import argparse
import json
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
    load_step31_model,
    simple_late_fusion_outputs,
    single_view_outputs_for_view,
    trivial_multi_view_outputs,
)
from train.eval_step30_encoder_recovery import load_model as load_step30_model


def parse_variant_thresholds(spec: str | None) -> Dict[str, float] | None:
    if not spec:
        return None
    thresholds: Dict[str, float] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError("threshold entries must look like clean:0.50,noisy:0.55")
        key, value = part.split(":", 1)
        thresholds[key.strip()] = float(value)
    return thresholds


def decoded_from_outputs(
    outputs: Dict[str, torch.Tensor],
    edge_threshold: float | torch.Tensor,
) -> Dict[str, torch.Tensor]:
    pred_type = outputs["type_logits"].argmax(dim=-1).float().unsqueeze(-1)
    return {
        "node_feats": torch.cat([pred_type, outputs["state_pred"]], dim=-1),
        "adj": hard_adj_from_logits(outputs["edge_logits"], threshold=edge_threshold),
    }


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
    edge_thresholds_by_variant = parse_variant_thresholds(args.recovered_edge_thresholds_by_variant)
    backends = build_backend_specs(args, device)

    step30_rev6_model = load_encoder(resolve_path(args.step30_rev6_checkpoint), device)
    step31_single_view_model = load_step30_model(args.step31_single_view_checkpoint, device)
    step31_multi_view_model = load_step31_model(args.step31_multi_view_checkpoint, device)

    all_records: list[Dict[str, Any]] = []

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
        decoded_inputs = {
            "gt_structured": decoded["gt_structured"],
            "step30_rev6_reference": decoded["encoder_recovered"],
        }
        add_backend_records_for_modes(all_records, backends, batch, decoded_inputs)

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
        variants = batch["step31_observation_variant"]
        edge_threshold_value = threshold_for_batch_variants(
            variants=variants,
            default_threshold=args.recovered_edge_threshold,
            thresholds_by_variant=edge_thresholds_by_variant,
            device=batch["target_adj"].device,
        )
        gt_decoded = {
            "node_feats": batch["target_node_feats"],
            "adj": batch["target_adj"],
        }
        single_outputs = single_view_outputs_for_view(step31_single_view_model, batch, 0)
        trivial_outputs = trivial_multi_view_outputs(
            batch,
            num_node_types=step31_single_view_model.config.num_node_types,
            state_dim=step31_single_view_model.config.state_dim,
        )
        late_fusion_outputs = simple_late_fusion_outputs(step31_single_view_model, batch)
        multi_view_outputs = step31_multi_view_model(
            multi_view_slot_features=batch["multi_view_slot_features"],
            multi_view_relation_hints=batch["multi_view_relation_hints"],
            multi_view_pair_support_hints=batch["multi_view_pair_support_hints"],
            multi_view_signed_pair_witness=batch["multi_view_signed_pair_witness"],
            multi_view_pair_evidence_bundle=batch["multi_view_pair_evidence_bundle"],
        )
        decoded_inputs = {
            "gt_structured": gt_decoded,
            "step31_single_view_baseline": decoded_from_outputs(
                single_outputs,
                edge_threshold_value,
            ),
            "step31_trivial_multi_view": decoded_from_outputs(
                trivial_outputs,
                edge_threshold_value,
            ),
            "step31_simple_late_fusion": decoded_from_outputs(
                late_fusion_outputs,
                edge_threshold_value,
            ),
            "step31_multi_view_encoder": decoded_from_outputs(
                multi_view_outputs,
                edge_threshold_value,
            ),
        }
        add_backend_records_for_modes(all_records, backends, batch, decoded_inputs)

    rows = grouped_summaries(all_records)
    add_deltas(rows)
    payload = {
        "metadata": {
            "step30_rev6_data_path": args.step30_rev6_data_path,
            "step31_data_path": args.step31_data_path,
            "step30_rev6_checkpoint": args.step30_rev6_checkpoint,
            "step31_single_view_checkpoint": args.step31_single_view_checkpoint,
            "step31_multi_view_checkpoint": args.step31_multi_view_checkpoint,
            "batch_size": args.batch_size,
            "limit_batches": args.limit_batches,
            "recovered_edge_threshold": args.recovered_edge_threshold,
            "recovered_edge_thresholds_by_variant": edge_thresholds_by_variant,
            "input_modes": [
                "gt_structured",
                "step30_rev6_reference",
                "step31_single_view_baseline",
                "step31_trivial_multi_view",
                "step31_simple_late_fusion",
                "step31_multi_view_encoder",
            ],
            "backend_specs": [
                {
                    "name": backend.name,
                    "proposal_checkpoint_path": str(backend.proposal_checkpoint_path),
                    "rewrite_checkpoint_path": str(backend.rewrite_checkpoint_path),
                    "node_threshold": backend.node_threshold,
                    "edge_threshold": backend.edge_threshold,
                    "rewrite_uses_proposal_conditioning": backend.use_proposal_conditioning,
                }
                for backend in backends
            ],
            "notes": (
                "Step31b is a frozen-backend transfer check only. It uses hard decoded "
                "graph_t_hat inputs and does not train adapters or backend weights."
            ),
        },
        "summary_rows": rows,
    }
    return clean_json_numbers(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step30_rev6_data_path", default="data/graph_event_step30_weak_obs_rev6_test.pkl")
    parser.add_argument("--step31_data_path", default="data/graph_event_step31_multi_view_test.pkl")
    parser.add_argument("--step30_rev6_checkpoint", default="checkpoints/step30_encoder_recovery_rev6/best.pt")
    parser.add_argument("--step31_single_view_checkpoint", default="checkpoints/step31_single_view_baseline/best.pt")
    parser.add_argument("--step31_multi_view_checkpoint", default="checkpoints/step31_multi_view_encoder/best.pt")
    parser.add_argument("--output_dir", default="artifacts/step31_backend_transfer_check")
    parser.add_argument("--backends", type=str, default="w012,rft1_p2")
    parser.add_argument("--clean_proposal_checkpoint_path", type=str, default="checkpoints/scope_proposal_node_edge_flipw2/best.pt")
    parser.add_argument("--w012_checkpoint_path", type=str, default="checkpoints/fp_keep_w012/best.pt")
    parser.add_argument("--noisy_proposal_checkpoint_path", type=str, default="checkpoints/proposal_noisy_obs_p2/best.pt")
    parser.add_argument("--rft1_checkpoint_path", type=str, default="checkpoints/step6_noisy_rewrite_rft1/best.pt")
    parser.add_argument("--clean_node_threshold", type=float, default=0.20)
    parser.add_argument("--clean_edge_threshold", type=float, default=0.15)
    parser.add_argument("--noisy_node_threshold", type=float, default=0.15)
    parser.add_argument("--noisy_edge_threshold", type=float, default=0.10)
    parser.add_argument("--recovered_edge_threshold", type=float, default=0.5)
    parser.add_argument("--recovered_edge_thresholds_by_variant", type=str, default="clean:0.50,noisy:0.55")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--limit_batches", type=int, default=None)
    args = parser.parse_args()

    payload = evaluate(args)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "summary.json"
    csv_path = output_dir / "summary.csv"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    write_csv(csv_path, payload["summary_rows"])
    print(f"wrote JSON: {json_path}")
    print(f"wrote CSV: {csv_path}")
    overall_rows = [
        row
        for row in payload["summary_rows"]
        if row["group_type"] == "overall" and row["group_name"] == "all"
    ]
    print(json.dumps(overall_rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
