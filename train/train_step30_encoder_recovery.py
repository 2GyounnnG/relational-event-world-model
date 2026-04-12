from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.step30_dataset import Step30WeakObservationDataset, step30_weak_observation_collate_fn
from models.encoder_step30 import (
    Step30EncoderConfig,
    Step30WeakObservationEncoder,
    step30_recovery_loss,
    step30_recovery_metrics,
)


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


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def average_dict(metric_sums: Dict[str, float], count: int) -> Dict[str, float]:
    return {key: value / max(count, 1) for key, value in metric_sums.items()}


def build_loaders(
    train_path: str,
    val_path: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple[Step30WeakObservationDataset, Step30WeakObservationDataset, DataLoader, DataLoader]:
    train_dataset = Step30WeakObservationDataset(train_path)
    val_dataset = Step30WeakObservationDataset(val_path)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=step30_weak_observation_collate_fn,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=step30_weak_observation_collate_fn,
        pin_memory=pin_memory,
    )
    return train_dataset, val_dataset, train_loader, val_loader


def infer_config_from_dataset(
    dataset: Step30WeakObservationDataset,
    hidden_dim: int,
    msg_pass_layers: int,
    node_head_layers: int,
    edge_head_layers: int,
    dropout: float,
    num_node_types: int,
    use_relation_logit_residual: bool,
    relation_logit_residual_scale: float,
    use_trust_denoising_edge_decoder: bool,
    use_pair_support_hints: bool,
    use_pair_evidence_bundle: bool,
    use_rescue_scoped_pair_evidence_bundle: bool,
    rescue_scoped_bundle_relation_max: float,
    rescue_scoped_bundle_residual_scale: float,
    use_rescue_candidate_latent_head: bool,
    rescue_candidate_latent_dim: int,
    rescue_candidate_relation_max: float,
    rescue_candidate_support_min: float,
    use_rescue_candidate_binary_calibration_head: bool,
    use_rescue_candidate_ambiguity_head: bool,
    use_positive_ambiguity_safety_hint: bool,
    positive_ambiguity_safety_projection_scale: float,
    use_weak_positive_ambiguity_safety_head: bool,
    use_signed_pair_witness: bool,
    use_signed_pair_witness_in_edge_head: bool,
    use_signed_pair_witness_correction: bool,
    signed_pair_witness_correction_scale: float,
) -> Step30EncoderConfig:
    first = dataset[0]
    obs_slot_dim = int(first["weak_slot_features"].shape[-1])
    state_dim = int(first["target_node_feats"].shape[-1] - 1)
    pair_evidence_bundle_dim = (
        int(first["weak_pair_evidence_bundle"].shape[-1])
        if (
            use_pair_evidence_bundle
            or use_rescue_scoped_pair_evidence_bundle
            or use_rescue_candidate_latent_head
        )
        and "weak_pair_evidence_bundle" in first
        else 0
    )
    return Step30EncoderConfig(
        obs_slot_dim=obs_slot_dim,
        num_node_types=num_node_types,
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        msg_pass_layers=msg_pass_layers,
        node_head_layers=node_head_layers,
        edge_head_layers=edge_head_layers,
        dropout=dropout,
        use_relation_hint_in_edge_head=True,
        use_relation_logit_residual=use_relation_logit_residual,
        relation_logit_residual_scale=relation_logit_residual_scale,
        use_trust_denoising_edge_decoder=use_trust_denoising_edge_decoder,
        use_pair_support_hints=use_pair_support_hints,
        use_pair_evidence_bundle=use_pair_evidence_bundle,
        pair_evidence_bundle_dim=pair_evidence_bundle_dim,
        use_rescue_scoped_pair_evidence_bundle=use_rescue_scoped_pair_evidence_bundle,
        rescue_scoped_bundle_relation_max=rescue_scoped_bundle_relation_max,
        rescue_scoped_bundle_residual_scale=rescue_scoped_bundle_residual_scale,
        use_rescue_candidate_latent_head=use_rescue_candidate_latent_head,
        rescue_candidate_latent_dim=rescue_candidate_latent_dim,
        rescue_candidate_relation_max=rescue_candidate_relation_max,
        rescue_candidate_support_min=rescue_candidate_support_min,
        use_rescue_candidate_binary_calibration_head=use_rescue_candidate_binary_calibration_head,
        use_rescue_candidate_ambiguity_head=use_rescue_candidate_ambiguity_head,
        use_positive_ambiguity_safety_hint=use_positive_ambiguity_safety_hint,
        positive_ambiguity_safety_projection_scale=positive_ambiguity_safety_projection_scale,
        use_weak_positive_ambiguity_safety_head=use_weak_positive_ambiguity_safety_head,
        use_signed_pair_witness=use_signed_pair_witness,
        use_signed_pair_witness_in_edge_head=use_signed_pair_witness_in_edge_head,
        use_signed_pair_witness_correction=use_signed_pair_witness_correction,
        signed_pair_witness_correction_scale=signed_pair_witness_correction_scale,
    )


def selection_score(metrics: Dict[str, float]) -> float:
    return (
        float(metrics["node_type_accuracy"])
        + float(metrics["edge_f1"])
        - float(metrics["node_state_mae"])
    )


def run_epoch(
    model: Step30WeakObservationEncoder,
    loader: DataLoader,
    device: torch.device,
    type_loss_weight: float,
    state_loss_weight: float,
    edge_loss_weight: float,
    edge_pos_weight: float,
    hard_negative_edge_loss_weight: float,
    hard_negative_hint_threshold: float,
    edge_ranking_loss_weight: float,
    edge_ranking_margin: float,
    trust_aux_loss_weight: float,
    trust_hint_threshold: float,
    missed_edge_loss_weight: float,
    missed_edge_hint_threshold: float,
    signed_witness_correction_loss_weight: float,
    signed_witness_relation_max: float,
    signed_witness_support_min: float,
    signed_witness_active_min: float,
    signed_witness_ambiguous_weight: float,
    false_admission_correction_loss_weight: float,
    false_admission_relation_max: float,
    false_admission_support_min: float,
    rescue_residual_contrast_loss_weight: float,
    rescue_residual_relation_max: float,
    rescue_residual_support_min: float,
    rescue_residual_margin: float,
    safe_rescue_residual_preservation_loss_weight: float,
    safe_rescue_residual_relation_max: float,
    safe_rescue_residual_support_min: float,
    safe_rescue_residual_floor: float,
    rescue_candidate_latent_loss_weight: float,
    rescue_candidate_relation_max: float,
    rescue_candidate_support_min: float,
    rescue_candidate_ambiguous_weight: float,
    rescue_candidate_binary_calibration_loss_weight: float,
    rescue_candidate_binary_pos_weight: float,
    rescue_candidate_ambiguity_loss_weight: float,
    rescue_candidate_ambiguity_pos_weight: float,
    weak_positive_ambiguity_safety_loss_weight: float,
    weak_positive_ambiguity_safety_pos_weight: float,
    edge_threshold: float,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_clip: Optional[float] = None,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    metric_sums: Dict[str, float] = {
        "total_loss": 0.0,
        "type_loss": 0.0,
        "state_loss": 0.0,
        "edge_loss": 0.0,
        "hard_negative_edge_loss": 0.0,
        "edge_ranking_loss": 0.0,
        "trust_aux_loss": 0.0,
        "missed_edge_loss": 0.0,
        "signed_witness_correction_loss": 0.0,
        "false_admission_correction_loss": 0.0,
        "rescue_residual_contrast_loss": 0.0,
        "safe_rescue_residual_preservation_loss": 0.0,
        "rescue_candidate_latent_loss": 0.0,
        "rescue_candidate_binary_calibration_loss": 0.0,
        "rescue_candidate_ambiguity_loss": 0.0,
        "weak_positive_ambiguity_safety_loss": 0.0,
        "node_type_accuracy": 0.0,
        "node_state_mae": 0.0,
        "node_state_mse": 0.0,
        "edge_accuracy": 0.0,
        "edge_precision": 0.0,
        "edge_recall": 0.0,
        "edge_f1": 0.0,
        "selection_score": 0.0,
    }
    num_batches = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
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
            loss_dict = step30_recovery_loss(
                outputs=outputs,
                target_node_feats=batch["target_node_feats"],
                target_adj=batch["target_adj"],
                node_mask=batch["node_mask"],
                relation_hints=batch["weak_relation_hints"],
                pair_support_hints=batch.get("weak_pair_support_hints"),
                pair_evidence_bundle=batch.get("weak_pair_evidence_bundle"),
                signed_pair_witness=batch.get("weak_signed_pair_witness"),
                type_loss_weight=type_loss_weight,
                state_loss_weight=state_loss_weight,
                edge_loss_weight=edge_loss_weight,
                edge_pos_weight=edge_pos_weight,
                hard_negative_edge_loss_weight=hard_negative_edge_loss_weight,
                hard_negative_hint_threshold=hard_negative_hint_threshold,
                edge_ranking_loss_weight=edge_ranking_loss_weight,
                edge_ranking_margin=edge_ranking_margin,
                trust_aux_loss_weight=trust_aux_loss_weight,
                trust_hint_threshold=trust_hint_threshold,
                missed_edge_loss_weight=missed_edge_loss_weight,
                missed_edge_hint_threshold=missed_edge_hint_threshold,
                signed_witness_correction_loss_weight=signed_witness_correction_loss_weight,
                signed_witness_relation_max=signed_witness_relation_max,
                signed_witness_support_min=signed_witness_support_min,
                signed_witness_active_min=signed_witness_active_min,
                signed_witness_ambiguous_weight=signed_witness_ambiguous_weight,
                false_admission_correction_loss_weight=false_admission_correction_loss_weight,
                false_admission_relation_max=false_admission_relation_max,
                false_admission_support_min=false_admission_support_min,
                rescue_residual_contrast_loss_weight=rescue_residual_contrast_loss_weight,
                rescue_residual_relation_max=rescue_residual_relation_max,
                rescue_residual_support_min=rescue_residual_support_min,
                rescue_residual_margin=rescue_residual_margin,
                safe_rescue_residual_preservation_loss_weight=safe_rescue_residual_preservation_loss_weight,
                safe_rescue_residual_relation_max=safe_rescue_residual_relation_max,
                safe_rescue_residual_support_min=safe_rescue_residual_support_min,
                safe_rescue_residual_floor=safe_rescue_residual_floor,
                rescue_candidate_latent_loss_weight=rescue_candidate_latent_loss_weight,
                rescue_candidate_relation_max=rescue_candidate_relation_max,
                rescue_candidate_support_min=rescue_candidate_support_min,
                rescue_candidate_ambiguous_weight=rescue_candidate_ambiguous_weight,
                rescue_candidate_binary_calibration_loss_weight=rescue_candidate_binary_calibration_loss_weight,
                rescue_candidate_binary_pos_weight=rescue_candidate_binary_pos_weight,
                rescue_candidate_ambiguity_loss_weight=rescue_candidate_ambiguity_loss_weight,
                rescue_candidate_ambiguity_pos_weight=rescue_candidate_ambiguity_pos_weight,
                weak_positive_ambiguity_safety_loss_weight=weak_positive_ambiguity_safety_loss_weight,
                weak_positive_ambiguity_safety_pos_weight=weak_positive_ambiguity_safety_pos_weight,
            )

            if is_train:
                optimizer.zero_grad()
                loss_dict["total_loss"].backward()
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            metrics = step30_recovery_metrics(
                outputs=outputs,
                target_node_feats=batch["target_node_feats"],
                target_adj=batch["target_adj"],
                node_mask=batch["node_mask"],
                edge_threshold=edge_threshold,
            )
            metrics["selection_score"] = selection_score(metrics)

            for key in [
                "total_loss",
                "type_loss",
                "state_loss",
                "edge_loss",
                "hard_negative_edge_loss",
                "edge_ranking_loss",
                "trust_aux_loss",
                "missed_edge_loss",
                "signed_witness_correction_loss",
                "false_admission_correction_loss",
                "rescue_residual_contrast_loss",
                "safe_rescue_residual_preservation_loss",
                "rescue_candidate_latent_loss",
                "rescue_candidate_binary_calibration_loss",
                "rescue_candidate_ambiguity_loss",
                "weak_positive_ambiguity_safety_loss",
            ]:
                metric_sums[key] += float(loss_dict[key].detach().item())
            for key in [
                "node_type_accuracy",
                "node_state_mae",
                "node_state_mse",
                "edge_accuracy",
                "edge_precision",
                "edge_recall",
                "edge_f1",
                "selection_score",
            ]:
                metric_sums[key] += float(metrics[key])
            num_batches += 1

    return average_dict(metric_sums, num_batches)


def save_checkpoint(
    path: Path,
    model: Step30WeakObservationEncoder,
    config: Step30EncoderConfig,
    args: argparse.Namespace,
    epoch: int,
    val_metrics: Dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config.to_dict(),
            "args": vars(args),
            "epoch": epoch,
            "val_metrics": val_metrics,
            "best_validation_selection_score": val_metrics["selection_score"],
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--msg_pass_layers", type=int, default=3)
    parser.add_argument("--node_head_layers", type=int, default=2)
    parser.add_argument("--edge_head_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num_node_types", type=int, default=3)
    parser.add_argument("--type_loss_weight", type=float, default=1.0)
    parser.add_argument("--state_loss_weight", type=float, default=1.0)
    parser.add_argument("--edge_loss_weight", type=float, default=1.0)
    parser.add_argument("--edge_pos_weight", type=float, default=1.0)
    parser.add_argument("--hard_negative_edge_loss_weight", type=float, default=0.0)
    parser.add_argument("--hard_negative_hint_threshold", type=float, default=0.5)
    parser.add_argument("--edge_ranking_loss_weight", type=float, default=0.0)
    parser.add_argument("--edge_ranking_margin", type=float, default=0.5)
    parser.add_argument("--edge_threshold", type=float, default=0.5)
    parser.add_argument("--use_relation_logit_residual", action="store_true")
    parser.add_argument("--relation_logit_residual_scale", type=float, default=1.0)
    parser.add_argument("--use_trust_denoising_edge_decoder", action="store_true")
    parser.add_argument("--use_pair_support_hints", action="store_true")
    parser.add_argument("--use_pair_evidence_bundle", action="store_true")
    parser.add_argument("--use_rescue_scoped_pair_evidence_bundle", action="store_true")
    parser.add_argument("--rescue_scoped_bundle_relation_max", type=float, default=0.5)
    parser.add_argument("--rescue_scoped_bundle_residual_scale", type=float, default=0.5)
    parser.add_argument("--freeze_non_rescue_scoped_bundle_parameters", action="store_true")
    parser.add_argument("--use_rescue_candidate_latent_head", action="store_true")
    parser.add_argument("--rescue_candidate_latent_dim", type=int, default=32)
    parser.add_argument("--rescue_candidate_relation_max", type=float, default=0.5)
    parser.add_argument("--rescue_candidate_support_min", type=float, default=0.55)
    parser.add_argument("--use_rescue_candidate_binary_calibration_head", action="store_true")
    parser.add_argument("--use_rescue_candidate_ambiguity_head", action="store_true")
    parser.add_argument("--use_positive_ambiguity_safety_hint", action="store_true")
    parser.add_argument("--positive_ambiguity_safety_projection_scale", type=float, default=1.0)
    parser.add_argument("--use_weak_positive_ambiguity_safety_head", action="store_true")
    parser.add_argument("--freeze_non_rescue_candidate_parameters", action="store_true")
    parser.add_argument("--use_signed_pair_witness", action="store_true")
    parser.add_argument("--disable_signed_pair_witness_in_edge_head", action="store_true")
    parser.add_argument("--use_signed_pair_witness_correction", action="store_true")
    parser.add_argument("--signed_pair_witness_correction_scale", type=float, default=0.5)
    parser.add_argument("--init_checkpoint_path", type=str, default=None)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--trust_aux_loss_weight", type=float, default=0.0)
    parser.add_argument("--trust_hint_threshold", type=float, default=0.5)
    parser.add_argument("--missed_edge_loss_weight", type=float, default=0.0)
    parser.add_argument("--missed_edge_hint_threshold", type=float, default=0.5)
    parser.add_argument("--signed_witness_correction_loss_weight", type=float, default=0.0)
    parser.add_argument("--signed_witness_relation_max", type=float, default=0.5)
    parser.add_argument("--signed_witness_support_min", type=float, default=0.55)
    parser.add_argument("--signed_witness_active_min", type=float, default=0.25)
    parser.add_argument("--signed_witness_ambiguous_weight", type=float, default=0.25)
    parser.add_argument("--false_admission_correction_loss_weight", type=float, default=0.0)
    parser.add_argument("--false_admission_relation_max", type=float, default=0.5)
    parser.add_argument("--false_admission_support_min", type=float, default=0.55)
    parser.add_argument("--rescue_residual_contrast_loss_weight", type=float, default=0.0)
    parser.add_argument("--rescue_residual_relation_max", type=float, default=0.5)
    parser.add_argument("--rescue_residual_support_min", type=float, default=0.55)
    parser.add_argument("--rescue_residual_margin", type=float, default=0.25)
    parser.add_argument("--safe_rescue_residual_preservation_loss_weight", type=float, default=0.0)
    parser.add_argument("--safe_rescue_residual_relation_max", type=float, default=0.5)
    parser.add_argument("--safe_rescue_residual_support_min", type=float, default=0.55)
    parser.add_argument("--safe_rescue_residual_floor", type=float, default=0.45)
    parser.add_argument("--rescue_candidate_latent_loss_weight", type=float, default=0.0)
    parser.add_argument("--rescue_candidate_ambiguous_weight", type=float, default=0.35)
    parser.add_argument("--rescue_candidate_binary_calibration_loss_weight", type=float, default=0.0)
    parser.add_argument("--rescue_candidate_binary_pos_weight", type=float, default=1.0)
    parser.add_argument("--rescue_candidate_ambiguity_loss_weight", type=float, default=0.0)
    parser.add_argument("--rescue_candidate_ambiguity_pos_weight", type=float, default=1.0)
    parser.add_argument("--weak_positive_ambiguity_safety_loss_weight", type=float, default=0.0)
    parser.add_argument("--weak_positive_ambiguity_safety_pos_weight", type=float, default=2.4)
    parser.add_argument(
        "--selection_metric",
        choices=[
            "default",
            "neg_rescue_candidate_latent_loss",
            "neg_rescue_candidate_binary_calibration_loss",
            "neg_rescue_candidate_ambiguity_loss",
            "neg_weak_positive_ambiguity_safety_loss",
        ],
        default="default",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    save_dir = Path(args.save_dir)
    pin_memory = device.type == "cuda"
    train_dataset, val_dataset, train_loader, val_loader = build_loaders(
        train_path=args.train_path,
        val_path=args.val_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    config = infer_config_from_dataset(
        train_dataset,
        hidden_dim=args.hidden_dim,
        msg_pass_layers=args.msg_pass_layers,
        node_head_layers=args.node_head_layers,
        edge_head_layers=args.edge_head_layers,
        dropout=args.dropout,
        num_node_types=args.num_node_types,
        use_relation_logit_residual=args.use_relation_logit_residual,
        relation_logit_residual_scale=args.relation_logit_residual_scale,
        use_trust_denoising_edge_decoder=args.use_trust_denoising_edge_decoder,
        use_pair_support_hints=args.use_pair_support_hints,
        use_pair_evidence_bundle=args.use_pair_evidence_bundle,
        use_rescue_scoped_pair_evidence_bundle=args.use_rescue_scoped_pair_evidence_bundle,
        rescue_scoped_bundle_relation_max=args.rescue_scoped_bundle_relation_max,
        rescue_scoped_bundle_residual_scale=args.rescue_scoped_bundle_residual_scale,
        use_rescue_candidate_latent_head=args.use_rescue_candidate_latent_head,
        rescue_candidate_latent_dim=args.rescue_candidate_latent_dim,
        rescue_candidate_relation_max=args.rescue_candidate_relation_max,
        rescue_candidate_support_min=args.rescue_candidate_support_min,
        use_rescue_candidate_binary_calibration_head=args.use_rescue_candidate_binary_calibration_head,
        use_rescue_candidate_ambiguity_head=args.use_rescue_candidate_ambiguity_head,
        use_positive_ambiguity_safety_hint=args.use_positive_ambiguity_safety_hint,
        positive_ambiguity_safety_projection_scale=args.positive_ambiguity_safety_projection_scale,
        use_weak_positive_ambiguity_safety_head=args.use_weak_positive_ambiguity_safety_head,
        use_signed_pair_witness=args.use_signed_pair_witness,
        use_signed_pair_witness_in_edge_head=not args.disable_signed_pair_witness_in_edge_head,
        use_signed_pair_witness_correction=args.use_signed_pair_witness_correction,
        signed_pair_witness_correction_scale=args.signed_pair_witness_correction_scale,
    )
    model = Step30WeakObservationEncoder(config).to(device)
    if args.init_checkpoint_path is not None:
        checkpoint = torch.load(args.init_checkpoint_path, map_location=device)
        incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        print(f"initialized from: {args.init_checkpoint_path}")
        print(f"missing keys: {list(incompatible.missing_keys)}")
        print(f"unexpected keys: {list(incompatible.unexpected_keys)}")
    if args.freeze_non_rescue_scoped_bundle_parameters:
        if not args.use_rescue_scoped_pair_evidence_bundle:
            raise ValueError(
                "--freeze_non_rescue_scoped_bundle_parameters requires "
                "--use_rescue_scoped_pair_evidence_bundle"
            )
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("pair_evidence_rescue_head.")
        trainable = [name for name, param in model.named_parameters() if param.requires_grad]
        print(f"trainable parameters after freeze: {trainable}")
    if args.freeze_non_rescue_candidate_parameters:
        if not args.use_rescue_candidate_latent_head:
            raise ValueError(
                "--freeze_non_rescue_candidate_parameters requires "
                "--use_rescue_candidate_latent_head"
            )
        for name, param in model.named_parameters():
            param.requires_grad = (
                name.startswith("rescue_candidate_latent.")
                or name.startswith("rescue_candidate_classifier.")
                or name.startswith("rescue_candidate_binary_calibration_head.")
                or name.startswith("rescue_candidate_ambiguity_head.")
                or name.startswith("positive_ambiguity_safety_projection.")
                or name.startswith("weak_positive_ambiguity_safety_head.")
            )
        trainable = [name for name, param in model.named_parameters() if param.requires_grad]
        print(f"trainable parameters after rescue-candidate freeze: {trainable}")
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters are available")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    best_score = -float("inf")
    best_epoch = -1
    history: list[Dict[str, Any]] = []

    print(f"device: {device}")
    print(f"train samples: {len(train_dataset)}")
    print(f"val samples: {len(val_dataset)}")
    print(f"config: {config.to_dict()}")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            type_loss_weight=args.type_loss_weight,
            state_loss_weight=args.state_loss_weight,
            edge_loss_weight=args.edge_loss_weight,
            edge_pos_weight=args.edge_pos_weight,
            hard_negative_edge_loss_weight=args.hard_negative_edge_loss_weight,
            hard_negative_hint_threshold=args.hard_negative_hint_threshold,
            edge_ranking_loss_weight=args.edge_ranking_loss_weight,
            edge_ranking_margin=args.edge_ranking_margin,
            trust_aux_loss_weight=args.trust_aux_loss_weight,
            trust_hint_threshold=args.trust_hint_threshold,
            missed_edge_loss_weight=args.missed_edge_loss_weight,
            missed_edge_hint_threshold=args.missed_edge_hint_threshold,
            signed_witness_correction_loss_weight=args.signed_witness_correction_loss_weight,
            signed_witness_relation_max=args.signed_witness_relation_max,
            signed_witness_support_min=args.signed_witness_support_min,
            signed_witness_active_min=args.signed_witness_active_min,
            signed_witness_ambiguous_weight=args.signed_witness_ambiguous_weight,
            false_admission_correction_loss_weight=args.false_admission_correction_loss_weight,
            false_admission_relation_max=args.false_admission_relation_max,
            false_admission_support_min=args.false_admission_support_min,
            rescue_residual_contrast_loss_weight=args.rescue_residual_contrast_loss_weight,
            rescue_residual_relation_max=args.rescue_residual_relation_max,
            rescue_residual_support_min=args.rescue_residual_support_min,
            rescue_residual_margin=args.rescue_residual_margin,
            safe_rescue_residual_preservation_loss_weight=args.safe_rescue_residual_preservation_loss_weight,
            safe_rescue_residual_relation_max=args.safe_rescue_residual_relation_max,
            safe_rescue_residual_support_min=args.safe_rescue_residual_support_min,
            safe_rescue_residual_floor=args.safe_rescue_residual_floor,
            rescue_candidate_latent_loss_weight=args.rescue_candidate_latent_loss_weight,
            rescue_candidate_relation_max=args.rescue_candidate_relation_max,
            rescue_candidate_support_min=args.rescue_candidate_support_min,
            rescue_candidate_ambiguous_weight=args.rescue_candidate_ambiguous_weight,
            rescue_candidate_binary_calibration_loss_weight=args.rescue_candidate_binary_calibration_loss_weight,
            rescue_candidate_binary_pos_weight=args.rescue_candidate_binary_pos_weight,
            rescue_candidate_ambiguity_loss_weight=args.rescue_candidate_ambiguity_loss_weight,
            rescue_candidate_ambiguity_pos_weight=args.rescue_candidate_ambiguity_pos_weight,
            weak_positive_ambiguity_safety_loss_weight=args.weak_positive_ambiguity_safety_loss_weight,
            weak_positive_ambiguity_safety_pos_weight=args.weak_positive_ambiguity_safety_pos_weight,
            edge_threshold=args.edge_threshold,
            optimizer=optimizer,
            grad_clip=args.grad_clip,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            type_loss_weight=args.type_loss_weight,
            state_loss_weight=args.state_loss_weight,
            edge_loss_weight=args.edge_loss_weight,
            edge_pos_weight=args.edge_pos_weight,
            hard_negative_edge_loss_weight=args.hard_negative_edge_loss_weight,
            hard_negative_hint_threshold=args.hard_negative_hint_threshold,
            edge_ranking_loss_weight=args.edge_ranking_loss_weight,
            edge_ranking_margin=args.edge_ranking_margin,
            trust_aux_loss_weight=args.trust_aux_loss_weight,
            trust_hint_threshold=args.trust_hint_threshold,
            missed_edge_loss_weight=args.missed_edge_loss_weight,
            missed_edge_hint_threshold=args.missed_edge_hint_threshold,
            signed_witness_correction_loss_weight=args.signed_witness_correction_loss_weight,
            signed_witness_relation_max=args.signed_witness_relation_max,
            signed_witness_support_min=args.signed_witness_support_min,
            signed_witness_active_min=args.signed_witness_active_min,
            signed_witness_ambiguous_weight=args.signed_witness_ambiguous_weight,
            false_admission_correction_loss_weight=args.false_admission_correction_loss_weight,
            false_admission_relation_max=args.false_admission_relation_max,
            false_admission_support_min=args.false_admission_support_min,
            rescue_residual_contrast_loss_weight=args.rescue_residual_contrast_loss_weight,
            rescue_residual_relation_max=args.rescue_residual_relation_max,
            rescue_residual_support_min=args.rescue_residual_support_min,
            rescue_residual_margin=args.rescue_residual_margin,
            safe_rescue_residual_preservation_loss_weight=args.safe_rescue_residual_preservation_loss_weight,
            safe_rescue_residual_relation_max=args.safe_rescue_residual_relation_max,
            safe_rescue_residual_support_min=args.safe_rescue_residual_support_min,
            safe_rescue_residual_floor=args.safe_rescue_residual_floor,
            rescue_candidate_latent_loss_weight=args.rescue_candidate_latent_loss_weight,
            rescue_candidate_relation_max=args.rescue_candidate_relation_max,
            rescue_candidate_support_min=args.rescue_candidate_support_min,
            rescue_candidate_ambiguous_weight=args.rescue_candidate_ambiguous_weight,
            rescue_candidate_binary_calibration_loss_weight=args.rescue_candidate_binary_calibration_loss_weight,
            rescue_candidate_binary_pos_weight=args.rescue_candidate_binary_pos_weight,
            rescue_candidate_ambiguity_loss_weight=args.rescue_candidate_ambiguity_loss_weight,
            rescue_candidate_ambiguity_pos_weight=args.rescue_candidate_ambiguity_pos_weight,
            weak_positive_ambiguity_safety_loss_weight=args.weak_positive_ambiguity_safety_loss_weight,
            weak_positive_ambiguity_safety_pos_weight=args.weak_positive_ambiguity_safety_pos_weight,
            edge_threshold=args.edge_threshold,
            optimizer=None,
        )
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['total_loss']:.6f} "
            f"val_loss={val_metrics['total_loss']:.6f} "
            f"val_type_acc={val_metrics['node_type_accuracy']:.4f} "
            f"val_state_mae={val_metrics['node_state_mae']:.4f} "
            f"val_edge_f1={val_metrics['edge_f1']:.4f} "
            f"val_rescue_candidate_loss={val_metrics['rescue_candidate_latent_loss']:.4f} "
            f"val_rescue_binary_loss={val_metrics['rescue_candidate_binary_calibration_loss']:.4f} "
            f"val_rescue_ambiguity_loss={val_metrics['rescue_candidate_ambiguity_loss']:.4f} "
            f"val_weak_positive_loss={val_metrics['weak_positive_ambiguity_safety_loss']:.4f} "
            f"val_score={val_metrics['selection_score']:.4f}"
        )
        checkpoint_score = float(val_metrics["selection_score"])
        if args.selection_metric == "neg_rescue_candidate_latent_loss":
            checkpoint_score = -float(val_metrics["rescue_candidate_latent_loss"])
        elif args.selection_metric == "neg_rescue_candidate_binary_calibration_loss":
            checkpoint_score = -float(val_metrics["rescue_candidate_binary_calibration_loss"])
        elif args.selection_metric == "neg_rescue_candidate_ambiguity_loss":
            checkpoint_score = -float(val_metrics["rescue_candidate_ambiguity_loss"])
        elif args.selection_metric == "neg_weak_positive_ambiguity_safety_loss":
            checkpoint_score = -float(val_metrics["weak_positive_ambiguity_safety_loss"])
        val_metrics_for_save = dict(val_metrics)
        val_metrics_for_save["selection_score"] = checkpoint_score
        if checkpoint_score > best_score:
            best_score = checkpoint_score
            best_epoch = epoch
            save_checkpoint(save_dir / "best.pt", model, config, args, epoch, val_metrics_for_save)

    save_checkpoint(save_dir / "last.pt", model, config, args, args.epochs, val_metrics)
    summary = {
        "best_epoch": best_epoch,
        "best_validation_selection_score": best_score,
        "config": config.to_dict(),
        "args": vars(args),
        "history": history,
    }
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"best epoch: {best_epoch}")
    print(f"best validation selection score: {best_score:.6f}")
    print(f"saved best checkpoint: {save_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
