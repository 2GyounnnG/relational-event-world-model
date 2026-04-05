from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------
# Make project root importable when running:
#   python train/oracle_sanity_check.py
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.collate import graph_event_collate_fn
from data.dataset import GraphEventDataset


REQUIRED_SCOPE_KEYS = (
    "event_scope_union_nodes",
    "event_scope_union_edges",
)

OPTIONAL_CHANGED_KEYS = (
    "changed_nodes",
    "changed_edges",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def build_loader(
    data_path: str,
    batch_size: int,
    num_workers: int,
) -> tuple[GraphEventDataset, DataLoader]:
    dataset = GraphEventDataset(data_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=graph_event_collate_fn,
        pin_memory=True,
    )
    return dataset, loader


@torch.no_grad()
def oracle_merge_back(batch: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    for key in REQUIRED_SCOPE_KEYS:
        if key not in batch:
            raise KeyError(
                f"Batch is missing required oracle scope field: {key}. "
                "Make sure the dataset contains union event scope annotations."
            )

    scope_node_mask = batch["event_scope_union_nodes"].float()      # [B, N]
    scope_edge_mask = batch["event_scope_union_edges"].float()      # [B, N, N]

    node_recon = (
        scope_node_mask.unsqueeze(-1) * batch["next_node_feats"]
        + (1.0 - scope_node_mask.unsqueeze(-1)) * batch["node_feats"]
    )
    adj_recon = (
        scope_edge_mask * batch["next_adj"]
        + (1.0 - scope_edge_mask) * batch["adj"]
    )

    return node_recon, adj_recon


@torch.no_grad()
def compute_type_accuracy_from_node_feats(
    pred_node_feats: torch.Tensor,
    target_node_feats: torch.Tensor,
    node_mask: torch.Tensor,
) -> float:
    pred_type = pred_node_feats[:, :, 0].round().long()
    target_type = target_node_feats[:, :, 0].round().long()

    correct = ((pred_type == target_type).float() * node_mask).sum()
    total = node_mask.sum().clamp_min(1.0)
    return (correct / total).item()


@torch.no_grad()
def compute_state_mae_from_node_feats(
    pred_node_feats: torch.Tensor,
    target_node_feats: torch.Tensor,
    node_mask: torch.Tensor,
) -> float:
    pred_state = pred_node_feats[:, :, 1:]
    target_state = target_node_feats[:, :, 1:]

    abs_err = torch.abs(pred_state - target_state)
    abs_err = abs_err * node_mask.unsqueeze(-1)
    denom = (node_mask.sum() * pred_state.shape[-1]).clamp_min(1.0)
    return (abs_err.sum() / denom).item()


@torch.no_grad()
def compute_edge_accuracy_from_adj(
    pred_adj: torch.Tensor,
    target_adj: torch.Tensor,
    node_mask: torch.Tensor,
) -> float:
    pred_adj_bin = pred_adj.round().float()
    target_adj_bin = target_adj.round().float()

    pair_mask = node_mask.unsqueeze(2) * node_mask.unsqueeze(1)
    n = pred_adj.shape[1]
    diag = torch.eye(n, device=pred_adj.device).unsqueeze(0)
    pair_mask = pair_mask * (1.0 - diag)

    correct = ((pred_adj_bin == target_adj_bin).float() * pair_mask).sum()
    total = pair_mask.sum().clamp_min(1.0)
    return (correct / total).item()


@torch.no_grad()
def count_changed_outside_scope(
    batch: Dict[str, torch.Tensor],
) -> dict[str, Optional[float]]:
    out: dict[str, Optional[float]] = {
        "changed_nodes_outside_scope": None,
        "changed_edges_outside_scope": None,
    }

    node_mask = batch["node_mask"].float()

    if "changed_nodes" in batch:
        changed_nodes = batch["changed_nodes"].float()
        scope_nodes = batch["event_scope_union_nodes"].float()
        uncovered_nodes = (changed_nodes * (1.0 - scope_nodes) * node_mask).sum()
        out["changed_nodes_outside_scope"] = uncovered_nodes.item()

    if "changed_edges" in batch:
        changed_edges = batch["changed_edges"].float()
        scope_edges = batch["event_scope_union_edges"].float()

        pair_mask = node_mask.unsqueeze(2) * node_mask.unsqueeze(1)
        n = changed_edges.shape[1]
        diag = torch.eye(n, device=changed_edges.device).unsqueeze(0)
        pair_mask = pair_mask * (1.0 - diag)

        uncovered_edges = (changed_edges * (1.0 - scope_edges) * pair_mask).sum()
        out["changed_edges_outside_scope"] = uncovered_edges.item()

    return out


@torch.no_grad()
def evaluate_oracle_reconstruction(
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> Dict[str, float]:
    type_acc_sum = 0.0
    state_mae_sum = 0.0
    edge_acc_sum = 0.0
    node_scope_frac_sum = 0.0
    edge_scope_frac_sum = 0.0
    changed_nodes_outside_scope_sum = 0.0
    changed_edges_outside_scope_sum = 0.0
    changed_nodes_outside_scope_seen = False
    changed_edges_outside_scope_seen = False
    num_batches = 0
    num_samples = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        batch = move_batch_to_device(batch, device)
        node_recon, adj_recon = oracle_merge_back(batch)

        type_acc_sum += compute_type_accuracy_from_node_feats(
            pred_node_feats=node_recon,
            target_node_feats=batch["next_node_feats"],
            node_mask=batch["node_mask"],
        )
        state_mae_sum += compute_state_mae_from_node_feats(
            pred_node_feats=node_recon,
            target_node_feats=batch["next_node_feats"],
            node_mask=batch["node_mask"],
        )
        edge_acc_sum += compute_edge_accuracy_from_adj(
            pred_adj=adj_recon,
            target_adj=batch["next_adj"],
            node_mask=batch["node_mask"],
        )

        real_nodes = batch["node_mask"].sum().clamp_min(1.0)
        real_pairs = (
            batch["node_mask"].unsqueeze(2) * batch["node_mask"].unsqueeze(1)
        )
        n = batch["node_mask"].shape[1]
        diag = torch.eye(n, device=device).unsqueeze(0)
        real_pairs = (real_pairs * (1.0 - diag)).sum().clamp_min(1.0)

        node_scope_frac_sum += (
            batch["event_scope_union_nodes"].sum() / real_nodes
        ).item()
        edge_scope_frac_sum += (
            batch["event_scope_union_edges"].sum() / real_pairs
        ).item()

        changed_scope_stats = count_changed_outside_scope(batch)
        if changed_scope_stats["changed_nodes_outside_scope"] is not None:
            changed_nodes_outside_scope_sum += changed_scope_stats["changed_nodes_outside_scope"]
            changed_nodes_outside_scope_seen = True
        if changed_scope_stats["changed_edges_outside_scope"] is not None:
            changed_edges_outside_scope_sum += changed_scope_stats["changed_edges_outside_scope"]
            changed_edges_outside_scope_seen = True

        num_batches += 1
        num_samples += batch["node_feats"].shape[0]

    if num_batches == 0:
        raise RuntimeError("No batches were evaluated. Check --max_batches and dataset size.")

    metrics: Dict[str, float] = {
        "type_acc": type_acc_sum / num_batches,
        "state_mae": state_mae_sum / num_batches,
        "edge_acc": edge_acc_sum / num_batches,
        "avg_node_scope_fraction": node_scope_frac_sum / num_batches,
        "avg_edge_scope_fraction": edge_scope_frac_sum / num_batches,
        "num_batches": float(num_batches),
        "num_samples": float(num_samples),
    }

    if changed_nodes_outside_scope_seen:
        metrics["changed_nodes_outside_scope"] = changed_nodes_outside_scope_sum
    if changed_edges_outside_scope_seen:
        metrics["changed_edges_outside_scope"] = changed_edges_outside_scope_sum

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/graph_event_val.pkl")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--state_tol", type=float, default=1e-8)
    parser.add_argument("--acc_tol", type=float, default=1e-12)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)

    dataset, loader = build_loader(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print(f"device: {device}")
    print(f"data path: {PROJECT_ROOT / args.data_path}")
    print(f"dataset size: {len(dataset)}")

    sample = dataset[0]
    print(f"node feat dim: {sample['node_feats'].shape[-1]}")
    print(
        "available optional keys:",
        sorted(k for k in sample.keys() if k not in {"node_feats", "adj", "next_node_feats", "next_adj"}),
    )

    metrics = evaluate_oracle_reconstruction(
        loader=loader,
        device=device,
        max_batches=args.max_batches,
    )

    print("\n[Oracle merge-back sanity check]")
    print(f"type_acc={metrics['type_acc']:.12f}")
    print(f"state_mae={metrics['state_mae']:.12e}")
    print(f"edge_acc={metrics['edge_acc']:.12f}")
    print(f"avg_node_scope_fraction={metrics['avg_node_scope_fraction']:.6f}")
    print(f"avg_edge_scope_fraction={metrics['avg_edge_scope_fraction']:.6f}")

    if "changed_nodes_outside_scope" in metrics:
        print(f"changed_nodes_outside_scope={metrics['changed_nodes_outside_scope']:.1f}")
    if "changed_edges_outside_scope" in metrics:
        print(f"changed_edges_outside_scope={metrics['changed_edges_outside_scope']:.1f}")

    passed = True
    failure_reasons: list[str] = []

    if metrics["type_acc"] < 1.0 - args.acc_tol:
        passed = False
        failure_reasons.append("type reconstruction is not perfect")

    if metrics["state_mae"] > args.state_tol:
        passed = False
        failure_reasons.append(
            f"state reconstruction MAE exceeds tolerance ({metrics['state_mae']:.3e} > {args.state_tol:.3e})"
        )

    if metrics["edge_acc"] < 1.0 - args.acc_tol:
        passed = False
        failure_reasons.append("edge reconstruction is not perfect")

    if metrics.get("changed_nodes_outside_scope", 0.0) > 0.0:
        passed = False
        failure_reasons.append("some changed nodes fall outside oracle scope")

    if metrics.get("changed_edges_outside_scope", 0.0) > 0.0:
        passed = False
        failure_reasons.append("some changed edges fall outside oracle scope")

    if passed:
        print("\nPASS: oracle union scope + merge-back perfectly reconstructs the next graph.")
    else:
        print("\nFAIL:")
        for reason in failure_reasons:
            print(f"  - {reason}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
