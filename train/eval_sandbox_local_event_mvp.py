from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.sandbox_local_event_dataset import (  # noqa: E402
    SandboxLocalEventDataset,
    sandbox_local_event_collate_fn,
)
from models.sandbox_local_event_operator import SandboxLocalEventOperator  # noqa: E402
from models.sandbox_monolithic_baseline import SandboxMonolithicBaseline  # noqa: E402


MODEL_TYPES = ("local_operator", "monolithic_baseline")
EVENT_TYPES = ("node_impulse", "spring_break")

# Keep these aligned with train/train_sandbox_local_event_mvp.py.
# Node features are [x, y, vx, vy, mass, radius, pinned]: dynamics dimensions
# get full weight, mostly constant physical attributes get a small weight.
NODE_LOSS_WEIGHTS = [1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1]
NODE_MUTABLE_DIMS = [0, 1, 2, 3]

# Edge features are [spring_active, rest_length, stiffness, current_distance]:
# spring_active/current_distance are the main event/rollout targets, while
# rest_length/stiffness are mostly constant in this single-event MVP.
EDGE_LOSS_WEIGHTS = [1.0, 0.1, 0.1, 1.0]
EDGE_MUTABLE_DIMS = [0, 3]


def resolve_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate sandbox local-event MVP one-step models.")
    parser.add_argument("--model_type", choices=MODEL_TYPES, required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--test_path", default="data/sandbox_local_event_mvp_test.pkl")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--output_dir", default="artifacts/sandbox_local_event_mvp_eval")
    return parser.parse_args()


def default_checkpoint(model_type: str) -> Path:
    return resolve_path(f"checkpoints/sandbox_local_event_mvp_{model_type}/best.pt")


def get_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


def make_loader(path: str, batch_size: int, num_workers: int) -> DataLoader:
    dataset = SandboxLocalEventDataset(resolve_path(path))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=sandbox_local_event_collate_fn,
    )


def build_model(model_type: str, sample: Dict[str, Any], hidden_dim: int) -> torch.nn.Module:
    node_dim = int(sample["node_features_t"].shape[-1])
    edge_dim = int(sample["edge_features_t"].shape[-1])
    event_param_dim = int(sample["event_params"].shape[-1])
    if model_type == "local_operator":
        return SandboxLocalEventOperator(
            node_feature_dim=node_dim,
            edge_feature_dim=edge_dim,
            event_param_dim=event_param_dim,
            hidden_dim=hidden_dim,
        )
    if model_type == "monolithic_baseline":
        return SandboxMonolithicBaseline(
            node_feature_dim=node_dim,
            edge_feature_dim=edge_dim,
            event_param_dim=event_param_dim,
            hidden_dim=hidden_dim,
        )
    raise ValueError(f"unknown model_type: {model_type}")


def load_model(model_type: str, checkpoint_path: Path, sample: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_model_type = checkpoint.get("model_type")
    if checkpoint_model_type != model_type:
        raise ValueError(
            f"checkpoint model_type {checkpoint_model_type!r} does not match requested {model_type!r}"
        )
    checkpoint_args = checkpoint.get("args", {})
    hidden_dim = int(checkpoint_args.get("hidden_dim", 64))
    model = build_model(model_type, sample, hidden_dim=hidden_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def forward_model(model: torch.nn.Module, model_type: str, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    common = {
        "node_features_t": batch["node_features_t"],
        "edge_index": batch["edge_index"],
        "edge_features_t": batch["edge_features_t"],
        "event_type_id": batch["event_type_id"],
        "event_params": batch["event_params"],
        "node_batch_index": batch["node_batch_index"],
        "edge_batch_index": batch["edge_batch_index"],
        "num_nodes_per_graph": batch["num_nodes_per_graph"],
        "num_edges_per_graph": batch["num_edges_per_graph"],
    }
    if model_type == "local_operator":
        return model(
            **common,
            event_node_mask=batch["event_node_mask"],
            event_edge_mask=batch["event_edge_mask"],
            event_scope_node_mask=batch["event_scope_node_mask"],
            event_scope_edge_mask=batch["event_scope_edge_mask"],
        )
    return model(**common)


def make_bucket() -> Dict[str, float]:
    return defaultdict(float)


def add_weighted_sse(
    bucket: Dict[str, float],
    prefix: str,
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    row_mask: torch.Tensor,
) -> None:
    if not bool(row_mask.any()):
        return
    pred_sel = pred[row_mask]
    target_sel = target[row_mask]
    weights = weights.to(device=pred.device, dtype=pred.dtype)
    bucket[f"{prefix}_weighted_sse"] += float(((pred_sel - target_sel).pow(2) * weights).sum().detach().cpu())
    bucket[f"{prefix}_weighted_count"] += float(pred_sel.shape[0]) * float(weights.sum().detach().cpu())


def add_masked_mae(
    bucket: Dict[str, float],
    name: str,
    pred: torch.Tensor,
    target: torch.Tensor,
    changed_mask: torch.Tensor,
    row_mask: torch.Tensor,
    dims: Iterable[int],
) -> None:
    active = row_mask & changed_mask
    if not bool(active.any()):
        return
    dim_index = torch.tensor(list(dims), device=pred.device, dtype=torch.long)
    selected = (pred[active].index_select(-1, dim_index) - target[active].index_select(-1, dim_index)).abs()
    bucket[f"{name}_sum"] += float(selected.sum().detach().cpu())
    bucket[f"{name}_count"] += float(selected.numel())


def update_bucket(
    bucket: Dict[str, float],
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, Any],
    graph_mask: torch.Tensor,
) -> None:
    node_mask = graph_mask[batch["node_batch_index"]]
    edge_mask = graph_mask[batch["edge_batch_index"]]

    node_weights = torch.tensor(NODE_LOSS_WEIGHTS, device=outputs["node_features_pred"].device)
    edge_weights = torch.tensor(EDGE_LOSS_WEIGHTS, device=outputs["edge_features_pred"].device)
    add_weighted_sse(
        bucket,
        "node",
        outputs["node_features_pred"],
        batch["node_features_next"],
        node_weights,
        node_mask,
    )
    add_weighted_sse(
        bucket,
        "edge",
        outputs["edge_features_pred"],
        batch["edge_features_next"],
        edge_weights,
        edge_mask,
    )

    changed_nodes = batch["changed_node_mask"] > 0.5
    changed_edges = batch["changed_edge_mask"] > 0.5
    add_masked_mae(
        bucket,
        "changed_node_error",
        outputs["node_features_pred"],
        batch["node_features_next"],
        changed_nodes,
        node_mask,
        NODE_MUTABLE_DIMS,
    )
    add_masked_mae(
        bucket,
        "unchanged_node_preservation_error",
        outputs["node_features_pred"],
        batch["copy_node_features_next"],
        ~changed_nodes,
        node_mask,
        NODE_MUTABLE_DIMS,
    )
    add_masked_mae(
        bucket,
        "changed_edge_error",
        outputs["edge_features_pred"],
        batch["edge_features_next"],
        changed_edges,
        edge_mask,
        EDGE_MUTABLE_DIMS,
    )
    add_masked_mae(
        bucket,
        "unchanged_edge_preservation_error",
        outputs["edge_features_pred"],
        batch["copy_edge_features_next"],
        ~changed_edges,
        edge_mask,
        EDGE_MUTABLE_DIMS,
    )

    bucket["num_samples"] += float(graph_mask.sum().detach().cpu())
    bucket["total_changed_node_count"] += float((changed_nodes & node_mask).sum().detach().cpu())
    bucket["total_changed_edge_count"] += float((changed_edges & edge_mask).sum().detach().cpu())


def finalize_bucket(bucket: Dict[str, float]) -> Dict[str, float]:
    node_loss = bucket["node_weighted_sse"] / max(bucket["node_weighted_count"], 1.0)
    edge_loss = bucket["edge_weighted_sse"] / max(bucket["edge_weighted_count"], 1.0)
    out = {
        "num_samples": int(bucket["num_samples"]),
        "total_changed_node_count": int(bucket["total_changed_node_count"]),
        "total_changed_edge_count": int(bucket["total_changed_edge_count"]),
        "total_loss": node_loss + edge_loss,
    }
    for name in (
        "changed_node_error",
        "unchanged_node_preservation_error",
        "changed_edge_error",
        "unchanged_edge_preservation_error",
    ):
        out[name] = bucket[f"{name}_sum"] / max(bucket[f"{name}_count"], 1.0)
    return out


def evaluate(model: torch.nn.Module, model_type: str, loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    overall = make_bucket()
    by_event = {event_type: make_bucket() for event_type in EVENT_TYPES}
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            outputs = forward_model(model, model_type, batch)

            graph_count = int(batch["event_type_id"].shape[0])
            all_graphs = torch.ones(graph_count, device=device, dtype=torch.bool)
            update_bucket(overall, outputs, batch, all_graphs)

            for event_type in EVENT_TYPES:
                event_mask = torch.tensor(
                    [name == event_type for name in batch["event_type"]],
                    device=device,
                    dtype=torch.bool,
                )
                update_bucket(by_event[event_type], outputs, batch, event_mask)

    overall_metrics = finalize_bucket(overall)
    breakdown = {event_type: finalize_bucket(bucket) for event_type, bucket in by_event.items()}
    overall_metrics["num_node_impulse_samples"] = breakdown["node_impulse"]["num_samples"]
    overall_metrics["num_spring_break_samples"] = breakdown["spring_break"]["num_samples"]
    return {
        "overall": overall_metrics,
        "event_type_breakdown": breakdown,
    }


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def save_breakdown_csv(path: Path, breakdown: Dict[str, Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "event_type",
        "num_samples",
        "total_loss",
        "changed_node_error",
        "unchanged_node_preservation_error",
        "changed_edge_error",
        "unchanged_edge_preservation_error",
        "total_changed_node_count",
        "total_changed_edge_count",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for event_type in EVENT_TYPES:
            row = {"event_type": event_type}
            row.update(breakdown[event_type])
            writer.writerow(row)


def print_summary(results: Dict[str, Any]) -> None:
    overall = results["overall"]
    print("overall:")
    print(
        f"  num_samples={overall['num_samples']} "
        f"node_impulse={overall['num_node_impulse_samples']} "
        f"spring_break={overall['num_spring_break_samples']}"
    )
    print(
        f"  total_loss={overall['total_loss']:.6f} "
        f"changed_node_error={overall['changed_node_error']:.6f} "
        f"unchanged_node_preservation_error={overall['unchanged_node_preservation_error']:.6f}"
    )
    print(
        f"  changed_edge_error={overall['changed_edge_error']:.6f} "
        f"unchanged_edge_preservation_error={overall['unchanged_edge_preservation_error']:.6f} "
        f"changed_nodes={overall['total_changed_node_count']} "
        f"changed_edges={overall['total_changed_edge_count']}"
    )
    print("by_event_type:")
    for event_type, row in results["event_type_breakdown"].items():
        print(
            f"  {event_type}: "
            f"n={row['num_samples']} "
            f"loss={row['total_loss']:.6f} "
            f"changed_node={row['changed_node_error']:.6f} "
            f"unchanged_node={row['unchanged_node_preservation_error']:.6f} "
            f"changed_edge={row['changed_edge_error']:.6f} "
            f"unchanged_edge={row['unchanged_edge_preservation_error']:.6f}"
        )


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    checkpoint_path = resolve_path(args.checkpoint) if args.checkpoint else default_checkpoint(args.model_type)
    output_dir = resolve_path(args.output_dir)

    loader = make_loader(args.test_path, args.batch_size, args.num_workers)
    sample = loader.dataset[0]
    model = load_model(args.model_type, checkpoint_path, sample, device)
    results = evaluate(model, args.model_type, loader, device)

    summary = {
        "model_type": args.model_type,
        "checkpoint": str(checkpoint_path),
        "test_path": str(resolve_path(args.test_path)),
        **results,
    }
    summary_path = output_dir / f"{args.model_type}_test_summary.json"
    breakdown_path = output_dir / f"{args.model_type}_event_type_breakdown.csv"
    save_json(summary_path, summary)
    save_breakdown_csv(breakdown_path, results["event_type_breakdown"])

    print_summary(results)
    print(f"summary_json={summary_path}")
    print(f"event_type_breakdown_csv={breakdown_path}")


if __name__ == "__main__":
    main()
