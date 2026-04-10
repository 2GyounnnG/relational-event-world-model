from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------
# Make project root importable when running:
#   python train/train_oracle_local_delta_proposal_conditioned_oracle.py
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.collate import graph_event_collate_fn
from data.dataset import GraphEventDataset
from models.oracle_local_delta import (
    EDGE_DELTA_ADD,
    EDGE_DELTA_DELETE,
    EDGE_DELTA_KEEP,
    OracleLocalDeltaRewriteConfig,
    OracleLocalDeltaRewriteModel,
    build_edge_delta_targets,
    build_valid_edge_mask,
    oracle_full_prediction_loss,
    oracle_local_delta_rewrite_loss,
)
from models.proposal import ScopeProposalConfig, ScopeProposalModel


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


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


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def build_dataloaders(
    train_path: str,
    val_path: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
):
    train_dataset = GraphEventDataset(train_path)
    val_dataset = GraphEventDataset(val_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=graph_event_collate_fn,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=graph_event_collate_fn,
        pin_memory=pin_memory,
    )

    return train_dataset, val_dataset, train_loader, val_loader


@torch.no_grad()
def type_correct_and_total(
    type_logits: torch.Tensor,
    target_node_feats: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[float, float]:
    pred_type = type_logits.argmax(dim=-1)
    target_type = target_node_feats[:, :, 0].long()
    mask_f = mask.float()
    correct = ((pred_type == target_type).float() * mask_f).sum().item()
    total = mask_f.sum().item()
    return correct, total


@torch.no_grad()
def state_abs_sum_and_total_dims(
    state_pred: torch.Tensor,
    target_node_feats: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[float, float]:
    target_state = target_node_feats[:, :, 1:]
    abs_err = torch.abs(state_pred - target_state)
    abs_sum = (abs_err * mask.unsqueeze(-1).float()).sum().item()
    total_dims = (mask.float().sum() * state_pred.shape[-1]).item()
    return abs_sum, total_dims


@torch.no_grad()
def edge_correct_and_total(
    edge_logits: torch.Tensor,
    target_adj: torch.Tensor,
    pair_mask: torch.Tensor,
) -> Tuple[float, float]:
    pred_adj = (torch.sigmoid(edge_logits) >= 0.5).float()
    pair_mask_f = pair_mask.float()
    correct = ((pred_adj == target_adj.float()).float() * pair_mask_f).sum().item()
    total = pair_mask_f.sum().item()
    return correct, total


@torch.no_grad()
def edge_delta_correct_and_total(
    edge_delta_logits: torch.Tensor,
    current_adj: torch.Tensor,
    target_adj: torch.Tensor,
    pair_mask: torch.Tensor,
    label_id: Optional[int] = None,
) -> Tuple[float, float]:
    pred_delta = edge_delta_logits.argmax(dim=-1)
    target_delta = build_edge_delta_targets(current_adj, target_adj)

    mask = pair_mask.bool()
    if label_id is not None:
        mask = mask & (target_delta == label_id)

    correct = ((pred_delta == target_delta) & mask).float().sum().item()
    total = mask.float().sum().item()
    return correct, total


def safe_div(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return num / den


def require_oracle_scope(batch: Dict) -> None:
    required_keys = ["event_scope_union_nodes", "event_scope_union_edges"]
    missing = [k for k in required_keys if k not in batch]
    if missing:
        raise KeyError(
            "Proposal-conditioned oracle-supervised rewrite training requires scope annotations "
            f"in the batch. Missing keys: {missing}"
        )


def init_metric_accumulator() -> Dict[str, float]:
    return {
        "local_total_loss_sum": 0.0,
        "local_type_loss_sum": 0.0,
        "local_state_loss_sum": 0.0,
        "local_edge_loss_sum": 0.0,
        "full_total_loss_sum": 0.0,
        "full_type_loss_sum": 0.0,
        "full_state_loss_sum": 0.0,
        "full_edge_loss_sum": 0.0,
        "num_batches": 0.0,
        "full_type_correct": 0.0,
        "full_type_total": 0.0,
        "full_state_abs_sum": 0.0,
        "full_state_total_dims": 0.0,
        "full_edge_correct": 0.0,
        "full_edge_total": 0.0,
        "scope_type_correct": 0.0,
        "scope_type_total": 0.0,
        "scope_state_abs_sum": 0.0,
        "scope_state_total_dims": 0.0,
        "scope_edge_correct": 0.0,
        "scope_edge_total": 0.0,
        "scope_edge_delta_correct": 0.0,
        "scope_edge_delta_total": 0.0,
        "scope_keep_correct": 0.0,
        "scope_keep_total": 0.0,
        "scope_add_correct": 0.0,
        "scope_add_total": 0.0,
        "scope_delete_correct": 0.0,
        "scope_delete_total": 0.0,
        "scope_nodes": 0.0,
        "total_nodes": 0.0,
        "scope_edges": 0.0,
        "total_edges": 0.0,
    }


def finalize_metric_accumulator(acc: Dict[str, float]) -> Dict[str, float]:
    num_batches = max(acc["num_batches"], 1.0)
    return {
        "local_total_loss": acc["local_total_loss_sum"] / num_batches,
        "local_type_loss": acc["local_type_loss_sum"] / num_batches,
        "local_state_loss": acc["local_state_loss_sum"] / num_batches,
        "local_edge_loss": acc["local_edge_loss_sum"] / num_batches,
        "full_total_loss": acc["full_total_loss_sum"] / num_batches,
        "full_type_loss": acc["full_type_loss_sum"] / num_batches,
        "full_state_loss": acc["full_state_loss_sum"] / num_batches,
        "full_edge_loss": acc["full_edge_loss_sum"] / num_batches,
        "full_type_acc": safe_div(acc["full_type_correct"], acc["full_type_total"]),
        "full_state_mae": safe_div(acc["full_state_abs_sum"], acc["full_state_total_dims"]),
        "full_edge_acc": safe_div(acc["full_edge_correct"], acc["full_edge_total"]),
        "scope_type_acc": safe_div(acc["scope_type_correct"], acc["scope_type_total"]),
        "scope_state_mae": safe_div(acc["scope_state_abs_sum"], acc["scope_state_total_dims"]),
        "scope_edge_acc": safe_div(acc["scope_edge_correct"], acc["scope_edge_total"]),
        "scope_edge_delta_acc": safe_div(acc["scope_edge_delta_correct"], acc["scope_edge_delta_total"]),
        "scope_keep_acc": safe_div(acc["scope_keep_correct"], acc["scope_keep_total"]),
        "scope_add_acc": safe_div(acc["scope_add_correct"], acc["scope_add_total"]),
        "scope_delete_acc": safe_div(acc["scope_delete_correct"], acc["scope_delete_total"]),
        "avg_scope_node_fraction": safe_div(acc["scope_nodes"], acc["total_nodes"]),
        "avg_scope_edge_fraction": safe_div(acc["scope_edges"], acc["total_edges"]),
    }


def update_metric_accumulator(
    acc: Dict[str, float],
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    scope_node_mask: torch.Tensor,
    scope_edge_mask: torch.Tensor,
    local_loss_dict: Dict[str, torch.Tensor],
    full_loss_dict: Dict[str, torch.Tensor],
) -> None:
    node_mask = batch["node_mask"]
    valid_edge_mask = build_valid_edge_mask(node_mask)
    scope_node_mask = scope_node_mask * node_mask
    scope_edge_mask = scope_edge_mask * valid_edge_mask

    acc["local_total_loss_sum"] += local_loss_dict["total_loss"].item()
    acc["local_type_loss_sum"] += local_loss_dict["type_loss"].item()
    acc["local_state_loss_sum"] += local_loss_dict["state_loss"].item()
    acc["local_edge_loss_sum"] += local_loss_dict["edge_loss"].item()
    acc["full_total_loss_sum"] += full_loss_dict["total_loss"].item()
    acc["full_type_loss_sum"] += full_loss_dict["type_loss"].item()
    acc["full_state_loss_sum"] += full_loss_dict["state_loss"].item()
    acc["full_edge_loss_sum"] += full_loss_dict["edge_loss"].item()
    acc["num_batches"] += 1.0

    c, t = type_correct_and_total(outputs["type_logits_full"], batch["next_node_feats"], node_mask)
    acc["full_type_correct"] += c
    acc["full_type_total"] += t

    s, d = state_abs_sum_and_total_dims(outputs["state_pred_full"], batch["next_node_feats"], node_mask)
    acc["full_state_abs_sum"] += s
    acc["full_state_total_dims"] += d

    c, t = edge_correct_and_total(outputs["edge_logits_full"], batch["next_adj"], valid_edge_mask)
    acc["full_edge_correct"] += c
    acc["full_edge_total"] += t

    c, t = type_correct_and_total(outputs["type_logits_local"], batch["next_node_feats"], scope_node_mask)
    acc["scope_type_correct"] += c
    acc["scope_type_total"] += t

    s, d = state_abs_sum_and_total_dims(outputs["state_pred_local"], batch["next_node_feats"], scope_node_mask)
    acc["scope_state_abs_sum"] += s
    acc["scope_state_total_dims"] += d

    c, t = edge_correct_and_total(outputs["edge_logits_local"], batch["next_adj"], scope_edge_mask)
    acc["scope_edge_correct"] += c
    acc["scope_edge_total"] += t

    c, t = edge_delta_correct_and_total(
        outputs["edge_delta_logits_local"], batch["adj"], batch["next_adj"], scope_edge_mask, label_id=None
    )
    acc["scope_edge_delta_correct"] += c
    acc["scope_edge_delta_total"] += t

    c, t = edge_delta_correct_and_total(
        outputs["edge_delta_logits_local"], batch["adj"], batch["next_adj"], scope_edge_mask, label_id=EDGE_DELTA_KEEP
    )
    acc["scope_keep_correct"] += c
    acc["scope_keep_total"] += t

    c, t = edge_delta_correct_and_total(
        outputs["edge_delta_logits_local"], batch["adj"], batch["next_adj"], scope_edge_mask, label_id=EDGE_DELTA_ADD
    )
    acc["scope_add_correct"] += c
    acc["scope_add_total"] += t

    c, t = edge_delta_correct_and_total(
        outputs["edge_delta_logits_local"], batch["adj"], batch["next_adj"], scope_edge_mask, label_id=EDGE_DELTA_DELETE
    )
    acc["scope_delete_correct"] += c
    acc["scope_delete_total"] += t

    acc["scope_nodes"] += scope_node_mask.sum().item()
    acc["total_nodes"] += node_mask.sum().item()
    acc["scope_edges"] += scope_edge_mask.sum().item()
    acc["total_edges"] += valid_edge_mask.sum().item()


def init_bridge_metric_accumulator() -> Dict[str, float]:
    return {
        "scope_type_correct": 0.0,
        "scope_type_total": 0.0,
        "full_type_correct": 0.0,
        "full_type_total": 0.0,
        "full_state_abs_sum": 0.0,
        "full_state_total_dims": 0.0,
        "full_edge_correct": 0.0,
        "full_edge_total": 0.0,
        "scope_edge_correct": 0.0,
        "scope_edge_total": 0.0,
        "scope_edge_delta_correct": 0.0,
        "scope_edge_delta_total": 0.0,
        "scope_keep_correct": 0.0,
        "scope_keep_total": 0.0,
        "scope_add_correct": 0.0,
        "scope_add_total": 0.0,
        "scope_delete_correct": 0.0,
        "scope_delete_total": 0.0,
        "changed_edge_correct": 0.0,
        "changed_edge_total": 0.0,
        "context_edge_correct": 0.0,
        "context_edge_total": 0.0,
        "scope_nodes": 0.0,
        "total_nodes": 0.0,
        "scope_edges": 0.0,
        "total_edges": 0.0,
    }


def finalize_bridge_metric_accumulator(acc: Dict[str, float]) -> Dict[str, float]:
    return {
        "full_type_acc": safe_div(acc["full_type_correct"], acc["full_type_total"]),
        "full_state_mae": safe_div(acc["full_state_abs_sum"], acc["full_state_total_dims"]),
        "full_edge_acc": safe_div(acc["full_edge_correct"], acc["full_edge_total"]),
        "scope_type_acc": safe_div(acc["scope_type_correct"], acc["scope_type_total"]),
        "scope_edge_acc": safe_div(acc["scope_edge_correct"], acc["scope_edge_total"]),
        "scope_edge_delta_acc": safe_div(acc["scope_edge_delta_correct"], acc["scope_edge_delta_total"]),
        "scope_keep_acc": safe_div(acc["scope_keep_correct"], acc["scope_keep_total"]),
        "scope_add_acc": safe_div(acc["scope_add_correct"], acc["scope_add_total"]),
        "scope_delete_acc": safe_div(acc["scope_delete_correct"], acc["scope_delete_total"]),
        "changed_edge_acc": safe_div(acc["changed_edge_correct"], acc["changed_edge_total"]),
        "context_edge_acc": safe_div(acc["context_edge_correct"], acc["context_edge_total"]),
        "avg_scope_node_fraction": safe_div(acc["scope_nodes"], acc["total_nodes"]),
        "avg_scope_edge_fraction": safe_div(acc["scope_edges"], acc["total_edges"]),
    }


@torch.no_grad()
def update_bridge_metric_accumulator(
    acc: Dict[str, float],
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    scope_node_mask: torch.Tensor,
    scope_edge_mask: torch.Tensor,
) -> None:
    node_mask = batch["node_mask"]
    valid_edge_mask = build_valid_edge_mask(node_mask)
    scope_node_mask = scope_node_mask * node_mask
    scope_edge_mask = scope_edge_mask * valid_edge_mask
    changed_edge_mask = (batch["changed_edges"] > 0.5).float() * valid_edge_mask
    context_edge_mask = scope_edge_mask * (1.0 - changed_edge_mask)

    c, t = type_correct_and_total(outputs["type_logits_full"], batch["next_node_feats"], node_mask)
    acc["full_type_correct"] += c
    acc["full_type_total"] += t

    s, d = state_abs_sum_and_total_dims(outputs["state_pred_full"], batch["next_node_feats"], node_mask)
    acc["full_state_abs_sum"] += s
    acc["full_state_total_dims"] += d

    c, t = edge_correct_and_total(outputs["edge_logits_full"], batch["next_adj"], valid_edge_mask)
    acc["full_edge_correct"] += c
    acc["full_edge_total"] += t

    c, t = type_correct_and_total(outputs["type_logits_local"], batch["next_node_feats"], scope_node_mask)
    acc["scope_type_correct"] += c
    acc["scope_type_total"] += t

    c, t = edge_correct_and_total(outputs["edge_logits_local"], batch["next_adj"], scope_edge_mask)
    acc["scope_edge_correct"] += c
    acc["scope_edge_total"] += t

    c, t = edge_delta_correct_and_total(
        outputs["edge_delta_logits_local"], batch["adj"], batch["next_adj"], scope_edge_mask, label_id=None
    )
    acc["scope_edge_delta_correct"] += c
    acc["scope_edge_delta_total"] += t

    c, t = edge_delta_correct_and_total(
        outputs["edge_delta_logits_local"], batch["adj"], batch["next_adj"], scope_edge_mask, label_id=EDGE_DELTA_KEEP
    )
    acc["scope_keep_correct"] += c
    acc["scope_keep_total"] += t

    c, t = edge_delta_correct_and_total(
        outputs["edge_delta_logits_local"], batch["adj"], batch["next_adj"], scope_edge_mask, label_id=EDGE_DELTA_ADD
    )
    acc["scope_add_correct"] += c
    acc["scope_add_total"] += t

    c, t = edge_delta_correct_and_total(
        outputs["edge_delta_logits_local"], batch["adj"], batch["next_adj"], scope_edge_mask, label_id=EDGE_DELTA_DELETE
    )
    acc["scope_delete_correct"] += c
    acc["scope_delete_total"] += t

    c, t = edge_correct_and_total(outputs["edge_logits_local"], batch["next_adj"], changed_edge_mask)
    acc["changed_edge_correct"] += c
    acc["changed_edge_total"] += t

    c, t = edge_correct_and_total(outputs["edge_logits_local"], batch["next_adj"], context_edge_mask)
    acc["context_edge_correct"] += c
    acc["context_edge_total"] += t

    acc["scope_nodes"] += scope_node_mask.sum().item()
    acc["total_nodes"] += node_mask.sum().item()
    acc["scope_edges"] += scope_edge_mask.sum().item()
    acc["total_edges"] += valid_edge_mask.sum().item()


def save_json(path: Path, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def compute_selection_score(
    val_metrics: Dict[str, float],
    keep_weight: float,
    add_weight: float,
    delete_weight: float,
) -> float:
    denom = keep_weight + add_weight + delete_weight
    if denom <= 0:
        raise ValueError("selection weights must sum to a positive value")
    return (
        keep_weight * val_metrics["scope_keep_acc"]
        + add_weight * val_metrics["scope_add_acc"]
        + delete_weight * val_metrics["scope_delete_acc"]
    ) / denom


def load_frozen_proposal(proposal_checkpoint_path: Path, device: torch.device) -> ScopeProposalModel:
    checkpoint = torch.load(proposal_checkpoint_path, map_location="cpu")
    model = ScopeProposalModel(ScopeProposalConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def maybe_load_rewrite_init(
    model: OracleLocalDeltaRewriteModel,
    init_checkpoint_path: Optional[Path],
) -> None:
    if init_checkpoint_path is None:
        return

    checkpoint = torch.load(init_checkpoint_path, map_location="cpu")
    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=False,
    )
    print(f"loaded init rewrite checkpoint: {init_checkpoint_path}")
    if missing_keys:
        print(f"  missing keys on load: {missing_keys}")
    if unexpected_keys:
        print(f"  unexpected keys on load: {unexpected_keys}")


@torch.no_grad()
def get_proposal_predictions(
    proposal_model: ScopeProposalModel,
    batch: Dict[str, torch.Tensor],
    node_threshold: float,
    edge_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    proposal_outputs = proposal_model(
        node_feats=batch["node_feats"],
        adj=batch["adj"],
    )

    node_mask = batch["node_mask"].bool()
    valid_edge_mask = build_valid_edge_mask(batch["node_mask"]).bool()

    node_scope_logits = proposal_outputs.get("node_scope_logits", proposal_outputs["scope_logits"])
    proposal_node_probs = torch.sigmoid(node_scope_logits) * node_mask.float()

    if "edge_scope_logits" in proposal_outputs:
        proposal_edge_probs = torch.sigmoid(proposal_outputs["edge_scope_logits"]) * valid_edge_mask.float()
    else:
        proposal_edge_probs = (
            proposal_node_probs.unsqueeze(2) * proposal_node_probs.unsqueeze(1) * valid_edge_mask.float()
        )

    pred_scope_nodes = ((proposal_node_probs >= node_threshold) & node_mask).float()
    pred_scope_edges = ((proposal_edge_probs >= edge_threshold) & valid_edge_mask).float()
    return proposal_node_probs, proposal_edge_probs, pred_scope_nodes, pred_scope_edges


def run_oracle_epoch(
    model: OracleLocalDeltaRewriteModel,
    proposal_model: ScopeProposalModel,
    loader: DataLoader,
    device: torch.device,
    edge_loss_weight: float,
    type_loss_weight: float,
    state_loss_weight: float,
    type_flip_weight: float,
    delta_keep_weight: float,
    delta_add_weight: float,
    delta_delete_weight: float,
    node_threshold: float,
    edge_threshold: float,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_clip: Optional[float] = None,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    proposal_model.eval()

    acc = init_metric_accumulator()
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in loader:
            require_oracle_scope(batch)
            batch = move_batch_to_device(batch, device)
            proposal_node_probs, proposal_edge_probs, _, _ = get_proposal_predictions(
                proposal_model=proposal_model,
                batch=batch,
                node_threshold=node_threshold,
                edge_threshold=edge_threshold,
            )

            scope_node_mask = batch["event_scope_union_nodes"]
            scope_edge_mask = batch["event_scope_union_edges"]
            outputs = model(
                node_feats=batch["node_feats"],
                adj=batch["adj"],
                scope_node_mask=scope_node_mask,
                scope_edge_mask=scope_edge_mask,
                proposal_node_probs=proposal_node_probs,
                proposal_edge_probs=proposal_edge_probs,
            )

            # Oracle scope still defines where rewrite supervision is applied.
            local_loss_dict = oracle_local_delta_rewrite_loss(
                outputs=outputs,
                current_node_feats=batch["node_feats"],
                current_adj=batch["adj"],
                target_node_feats=batch["next_node_feats"],
                target_adj=batch["next_adj"],
                node_mask=batch["node_mask"],
                scope_node_mask=scope_node_mask,
                scope_edge_mask=scope_edge_mask,
                edge_loss_weight=edge_loss_weight,
                type_loss_weight=type_loss_weight,
                state_loss_weight=state_loss_weight,
                type_flip_weight=type_flip_weight,
                delta_keep_weight=delta_keep_weight,
                delta_add_weight=delta_add_weight,
                delta_delete_weight=delta_delete_weight,
            )

            full_loss_dict = oracle_full_prediction_loss(
                outputs=outputs,
                target_node_feats=batch["next_node_feats"],
                target_adj=batch["next_adj"],
                node_mask=batch["node_mask"],
                edge_loss_weight=edge_loss_weight,
                type_loss_weight=type_loss_weight,
                state_loss_weight=state_loss_weight,
            )

            if is_train:
                optimizer.zero_grad()
                local_loss_dict["total_loss"].backward()
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            update_metric_accumulator(
                acc,
                outputs,
                batch,
                scope_node_mask,
                scope_edge_mask,
                local_loss_dict,
                full_loss_dict,
            )

    return finalize_metric_accumulator(acc)


@torch.no_grad()
def run_bridge_validation(
    model: OracleLocalDeltaRewriteModel,
    proposal_model: ScopeProposalModel,
    loader: DataLoader,
    device: torch.device,
    node_threshold: float,
    edge_threshold: float,
) -> Dict[str, float]:
    model.eval()
    proposal_model.eval()

    acc = init_bridge_metric_accumulator()
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        proposal_node_probs, proposal_edge_probs, pred_scope_nodes, pred_scope_edges = get_proposal_predictions(
            proposal_model=proposal_model,
            batch=batch,
            node_threshold=node_threshold,
            edge_threshold=edge_threshold,
        )

        outputs = model(
            node_feats=batch["node_feats"],
            adj=batch["adj"],
            scope_node_mask=pred_scope_nodes,
            scope_edge_mask=pred_scope_edges,
            proposal_node_probs=proposal_node_probs,
            proposal_edge_probs=proposal_edge_probs,
        )
        update_bridge_metric_accumulator(
            acc,
            outputs,
            batch,
            pred_scope_nodes,
            pred_scope_edges,
        )

    return finalize_bridge_metric_accumulator(acc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="data/graph_event_train.pkl")
    parser.add_argument("--val_path", type=str, default="data/graph_event_val.pkl")
    parser.add_argument("--proposal_checkpoint", type=str, required=True)
    parser.add_argument("--init_rewrite_checkpoint", type=str, default=None)
    parser.add_argument("--node_threshold", type=float, default=0.20)
    parser.add_argument("--edge_threshold", type=float, default=0.15)
    parser.add_argument(
        "--use_proposal_conditioning",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints/oracle_local_delta_proposal_conditioned_oracle",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--msg_pass_layers", type=int, default=3)
    parser.add_argument("--node_mlp_layers", type=int, default=2)
    parser.add_argument("--edge_mlp_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--edge_dropout", type=float, default=0.0)
    parser.add_argument("--num_node_types", type=int, default=3)
    parser.add_argument("--copy_logit_value", type=float, default=10.0)
    parser.add_argument("--edge_loss_weight", type=float, default=1.0)
    parser.add_argument("--type_loss_weight", type=float, default=1.0)
    parser.add_argument("--state_loss_weight", type=float, default=1.0)
    parser.add_argument("--type_flip_weight", type=float, default=1.0)
    parser.add_argument("--delta_keep_weight", type=float, default=1.10)
    parser.add_argument("--delta_add_weight", type=float, default=1.0)
    parser.add_argument("--delta_delete_weight", type=float, default=3.0)
    parser.add_argument("--selection_keep_weight", type=float, default=0.5)
    parser.add_argument("--selection_add_weight", type=float, default=0.5)
    parser.add_argument("--selection_delete_weight", type=float, default=0.5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    pin_memory = device.type == "cuda"

    proposal_checkpoint_path = resolve_path(args.proposal_checkpoint)
    init_rewrite_checkpoint_path = (
        resolve_path(args.init_rewrite_checkpoint) if args.init_rewrite_checkpoint is not None else None
    )
    save_dir = PROJECT_ROOT / args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset, train_loader, val_loader = build_dataloaders(
        train_path=args.train_path,
        val_path=args.val_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    sample = train_dataset[0]
    require_oracle_scope(sample)
    node_feat_dim = sample["node_feats"].shape[-1]
    state_dim = node_feat_dim - 1

    model_config = OracleLocalDeltaRewriteConfig(
        node_feat_dim=node_feat_dim,
        num_node_types=args.num_node_types,
        type_dim=1,
        state_dim=state_dim,
        hidden_dim=args.hidden_dim,
        msg_pass_layers=args.msg_pass_layers,
        node_mlp_layers=args.node_mlp_layers,
        edge_mlp_layers=args.edge_mlp_layers,
        dropout=args.dropout,
        edge_dropout=args.edge_dropout,
        copy_logit_value=args.copy_logit_value,
        use_proposal_conditioning=args.use_proposal_conditioning,
    )
    model = OracleLocalDeltaRewriteModel(model_config).to(device)
    maybe_load_rewrite_init(model, init_rewrite_checkpoint_path)
    proposal_model = load_frozen_proposal(proposal_checkpoint_path, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_score = float("-inf")
    best_epoch = -1
    epochs_without_improvement = 0

    best_ckpt_path = save_dir / "best.pt"
    last_ckpt_path = save_dir / "last.pt"
    best_metrics_path = save_dir / "best_metrics.json"

    print(f"device: {device}")
    print(f"train size: {len(train_dataset)}")
    print(f"val size: {len(val_dataset)}")
    print(f"proposal checkpoint: {proposal_checkpoint_path}")
    print(f"init rewrite checkpoint: {init_rewrite_checkpoint_path}")
    print(f"use proposal conditioning: {args.use_proposal_conditioning}")
    print(f"bridge node threshold: {args.node_threshold}")
    print(f"bridge edge threshold: {args.edge_threshold}")
    print(f"node_feat_dim: {node_feat_dim}")
    print(f"state_dim: {state_dim}")
    print(
        f"bridge selection metric: {args.selection_keep_weight:.3f} * bridge_scope_keep_acc + "
        f"{args.selection_add_weight:.3f} * bridge_scope_add_acc + "
        f"{args.selection_delete_weight:.3f} * bridge_scope_delete_acc"
    )
    print(f"early stopping patience: {args.patience}")
    print(model)

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_oracle_epoch(
            model=model,
            proposal_model=proposal_model,
            loader=train_loader,
            device=device,
            edge_loss_weight=args.edge_loss_weight,
            type_loss_weight=args.type_loss_weight,
            state_loss_weight=args.state_loss_weight,
            type_flip_weight=args.type_flip_weight,
            delta_keep_weight=args.delta_keep_weight,
            delta_add_weight=args.delta_add_weight,
            delta_delete_weight=args.delta_delete_weight,
            node_threshold=args.node_threshold,
            edge_threshold=args.edge_threshold,
            optimizer=optimizer,
            grad_clip=args.grad_clip,
        )

        val_metrics = run_oracle_epoch(
            model=model,
            proposal_model=proposal_model,
            loader=val_loader,
            device=device,
            edge_loss_weight=args.edge_loss_weight,
            type_loss_weight=args.type_loss_weight,
            state_loss_weight=args.state_loss_weight,
            type_flip_weight=args.type_flip_weight,
            delta_keep_weight=args.delta_keep_weight,
            delta_add_weight=args.delta_add_weight,
            delta_delete_weight=args.delta_delete_weight,
            node_threshold=args.node_threshold,
            edge_threshold=args.edge_threshold,
            optimizer=None,
            grad_clip=None,
        )

        bridge_val_metrics = run_bridge_validation(
            model=model,
            proposal_model=proposal_model,
            loader=val_loader,
            device=device,
            node_threshold=args.node_threshold,
            edge_threshold=args.edge_threshold,
        )

        selection_score = compute_selection_score(
            bridge_val_metrics,
            keep_weight=args.selection_keep_weight,
            add_weight=args.selection_add_weight,
            delete_weight=args.selection_delete_weight,
        )

        print(
            f"[Epoch {epoch:03d}] "
            f"train_local_total={train_metrics['local_total_loss']:.6f} "
            f"train_full_total={train_metrics['full_total_loss']:.6f} "
            f"train_scope_type_acc={train_metrics['scope_type_acc']:.6f} "
            f"train_scope_edge_acc={train_metrics['scope_edge_acc']:.6f} "
            f"train_scope_keep_acc={train_metrics['scope_keep_acc']:.6f} "
            f"train_scope_add_acc={train_metrics['scope_add_acc']:.6f} "
            f"train_scope_delete_acc={train_metrics['scope_delete_acc']:.6f} | "
            f"val_local_total={val_metrics['local_total_loss']:.6f} "
            f"val_full_total={val_metrics['full_total_loss']:.6f} "
            f"val_scope_type_acc={val_metrics['scope_type_acc']:.6f} "
            f"val_scope_edge_acc={val_metrics['scope_edge_acc']:.6f} "
            f"val_scope_keep_acc={val_metrics['scope_keep_acc']:.6f} "
            f"val_scope_add_acc={val_metrics['scope_add_acc']:.6f} "
            f"val_scope_delete_acc={val_metrics['scope_delete_acc']:.6f} | "
            f"bridge_scope_edge_acc={bridge_val_metrics['scope_edge_acc']:.6f} "
            f"bridge_keep_acc={bridge_val_metrics['scope_keep_acc']:.6f} "
            f"bridge_add_acc={bridge_val_metrics['scope_add_acc']:.6f} "
            f"bridge_delete_acc={bridge_val_metrics['scope_delete_acc']:.6f} "
            f"bridge_context_edge_acc={bridge_val_metrics['context_edge_acc']:.6f} "
            f"selection_score={selection_score:.6f}"
        )

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": vars(model_config),
            "args": vars(args),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "bridge_val_metrics": bridge_val_metrics,
            "selection_metric": "bridge_balanced_scope_keep_add_delete_acc",
            "selection_score": selection_score,
            "selection_formula": {
                "scope_keep_acc_weight": args.selection_keep_weight,
                "scope_add_acc_weight": args.selection_add_weight,
                "scope_delete_acc_weight": args.selection_delete_weight,
            },
            "edge_prediction_mode": "delta_3class",
            "edge_delta_label_map": {"0": "keep", "1": "add", "2": "delete"},
            "use_proposal_conditioning": args.use_proposal_conditioning,
            "proposal_checkpoint": str(proposal_checkpoint_path),
            "init_rewrite_checkpoint": str(init_rewrite_checkpoint_path) if init_rewrite_checkpoint_path else None,
            "scope_source": "oracle_supervision_with_proposal_conditioning",
            "proposal_node_threshold": args.node_threshold,
            "proposal_edge_threshold": args.edge_threshold,
        }

        torch.save(ckpt, last_ckpt_path)

        if selection_score > best_score:
            best_score = selection_score
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(ckpt, best_ckpt_path)
            save_json(
                best_metrics_path,
                {
                    "epoch": epoch,
                    "best_selection_score": best_score,
                    "selection_metric": "bridge_balanced_scope_keep_add_delete_acc",
                    "selection_formula": {
                        "scope_keep_acc_weight": args.selection_keep_weight,
                        "scope_add_acc_weight": args.selection_add_weight,
                        "scope_delete_acc_weight": args.selection_delete_weight,
                    },
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "bridge_val_metrics": bridge_val_metrics,
                    "model_config": vars(model_config),
                    "args": vars(args),
                    "edge_prediction_mode": "delta_3class",
                    "edge_delta_label_map": {"0": "keep", "1": "add", "2": "delete"},
                    "use_proposal_conditioning": args.use_proposal_conditioning,
                    "proposal_checkpoint": str(proposal_checkpoint_path),
                    "init_rewrite_checkpoint": (
                        str(init_rewrite_checkpoint_path) if init_rewrite_checkpoint_path else None
                    ),
                    "scope_source": "oracle_supervision_with_proposal_conditioning",
                    "proposal_node_threshold": args.node_threshold,
                    "proposal_edge_threshold": args.edge_threshold,
                },
            )
            print(f"  saved new best checkpoint -> {best_ckpt_path}")
        else:
            epochs_without_improvement += 1
            print(f"  no improvement for {epochs_without_improvement} epoch(s)")

        if epochs_without_improvement >= args.patience:
            print(
                f"early stopping triggered at epoch {epoch} "
                f"(best epoch: {best_epoch}, best selection score: {best_score:.6f})"
            )
            break

    print(f"training finished. best epoch={best_epoch} best selection score={best_score:.6f}")
    print(f"best checkpoint: {best_ckpt_path}")
    print(f"last checkpoint: {last_ckpt_path}")
    print(f"best metrics json: {best_metrics_path}")


if __name__ == "__main__":
    main()
