from __future__ import annotations

import argparse
import json
import random
import sys
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

# Node features are [x, y, vx, vy, mass, radius, pinned].
# The first four are the one-step mutable dynamics target. The last three are
# mostly constant physical attributes, so they are kept in the prediction target
# with a lower conservative weight rather than ignored.
NODE_LOSS_WEIGHTS = [1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1]
NODE_MUTABLE_DIMS = [0, 1, 2, 3]

# Edge features are [spring_active, rest_length, stiffness, current_distance].
# Event edits and rollout mainly affect spring_active/current_distance in this
# clean MVP. rest_length/stiffness are mostly constant and get lower weight.
EDGE_LOSS_WEIGHTS = [1.0, 0.1, 0.1, 1.0]
EDGE_MUTABLE_DIMS = [0, 3]


def resolve_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train sandbox local-event MVP one-step models.")
    parser.add_argument("--model_type", choices=MODEL_TYPES, required=True)
    parser.add_argument("--train_path", default="data/sandbox_local_event_mvp_train.pkl")
    parser.add_argument("--val_path", default="data/sandbox_local_event_mvp_val.pkl")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--save_dir", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_val_batches", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="Run only a couple batches for quick plumbing checks.")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def make_loader(path: str, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    dataset = SandboxLocalEventDataset(resolve_path(path))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
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


def weighted_mse(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.to(device=pred.device, dtype=pred.dtype)
    return ((pred - target).pow(2) * weights).sum() / (pred.shape[0] * weights.sum()).clamp_min(1.0)


def compute_loss(outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> torch.Tensor:
    node_weights = torch.tensor(NODE_LOSS_WEIGHTS, device=outputs["node_features_pred"].device)
    edge_weights = torch.tensor(EDGE_LOSS_WEIGHTS, device=outputs["edge_features_pred"].device)
    node_loss = weighted_mse(outputs["node_features_pred"], batch["node_features_next"], node_weights)
    edge_loss = weighted_mse(outputs["edge_features_pred"], batch["edge_features_next"], edge_weights)
    return node_loss + edge_loss


def add_masked_mae(
    metrics: Dict[str, float],
    name: str,
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    dims: Iterable[int],
) -> None:
    dim_index = torch.tensor(list(dims), device=pred.device, dtype=torch.long)
    selected = (pred.index_select(-1, dim_index) - target.index_select(-1, dim_index)).abs()
    mask_f = mask.to(device=pred.device, dtype=pred.dtype).unsqueeze(-1)
    metrics[f"{name}_sum"] += float((selected * mask_f).sum().detach().cpu())
    metrics[f"{name}_count"] += float(mask_f.sum().detach().cpu()) * float(len(dim_index))


def init_metrics() -> Dict[str, float]:
    return {
        "loss_sum": 0.0,
        "loss_count": 0.0,
        "changed_node_error_sum": 0.0,
        "changed_node_error_count": 0.0,
        "unchanged_node_preservation_error_sum": 0.0,
        "unchanged_node_preservation_error_count": 0.0,
        "changed_edge_error_sum": 0.0,
        "changed_edge_error_count": 0.0,
        "unchanged_edge_preservation_error_sum": 0.0,
        "unchanged_edge_preservation_error_count": 0.0,
    }


def finalize_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    out = {"total_loss": metrics["loss_sum"] / max(metrics["loss_count"], 1.0)}
    for name in (
        "changed_node_error",
        "unchanged_node_preservation_error",
        "changed_edge_error",
        "unchanged_edge_preservation_error",
    ):
        out[name] = metrics[f"{name}_sum"] / max(metrics[f"{name}_count"], 1.0)
    return out


def update_metrics(metrics: Dict[str, float], loss: torch.Tensor, outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> None:
    graph_count = int(batch["event_type_id"].shape[0])
    metrics["loss_sum"] += float(loss.detach().cpu()) * graph_count
    metrics["loss_count"] += float(graph_count)

    changed_nodes = batch["changed_node_mask"] > 0.5
    changed_edges = batch["changed_edge_mask"] > 0.5
    add_masked_mae(
        metrics,
        "changed_node_error",
        outputs["node_features_pred"],
        batch["node_features_next"],
        changed_nodes,
        NODE_MUTABLE_DIMS,
    )
    add_masked_mae(
        metrics,
        "unchanged_node_preservation_error",
        outputs["node_features_pred"],
        batch["copy_node_features_next"],
        ~changed_nodes,
        NODE_MUTABLE_DIMS,
    )
    add_masked_mae(
        metrics,
        "changed_edge_error",
        outputs["edge_features_pred"],
        batch["edge_features_next"],
        changed_edges,
        EDGE_MUTABLE_DIMS,
    )
    add_masked_mae(
        metrics,
        "unchanged_edge_preservation_error",
        outputs["edge_features_pred"],
        batch["copy_edge_features_next"],
        ~changed_edges,
        EDGE_MUTABLE_DIMS,
    )


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    model_type: str,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    metrics = init_metrics()
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch_index, batch in enumerate(loader):
            if max_batches and batch_index >= max_batches:
                break
            batch = move_batch_to_device(batch, device)
            outputs = forward_model(model, model_type, batch)
            loss = compute_loss(outputs, batch)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            update_metrics(metrics, loss, outputs, batch)
    return finalize_metrics(metrics)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.epochs = min(args.epochs, 1)
        args.max_train_batches = args.max_train_batches or 2
        args.max_val_batches = args.max_val_batches or 2

    seed_everything(args.seed)
    device = get_device(args.device)
    save_dir = resolve_path(args.save_dir or f"checkpoints/sandbox_local_event_mvp_{args.model_type}")
    save_dir.mkdir(parents=True, exist_ok=True)

    train_loader = make_loader(args.train_path, args.batch_size, args.num_workers, shuffle=True)
    val_loader = make_loader(args.val_path, args.batch_size, args.num_workers, shuffle=False)

    sample = train_loader.dataset[0]
    model = build_model(args.model_type, sample, args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = float("inf")
    history: List[Dict[str, Any]] = []
    best_path = save_dir / "best.pt"
    history_path = save_dir / "history.json"

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            args.model_type,
            device,
            optimizer,
            max_batches=args.max_train_batches,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            args.model_type,
            device,
            optimizer=None,
            max_batches=args.max_val_batches,
        )

        row = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(row)

        if val_metrics["total_loss"] < best_val_loss:
            best_val_loss = val_metrics["total_loss"]
            torch.save(
                {
                    "model_type": args.model_type,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "args": vars(args),
                    "node_loss_weights": NODE_LOSS_WEIGHTS,
                    "edge_loss_weights": EDGE_LOSS_WEIGHTS,
                },
                best_path,
            )

        save_json(
            history_path,
            {
                "model_type": args.model_type,
                "best_val_loss": best_val_loss,
                "best_checkpoint": str(best_path),
                "history": history,
            },
        )

        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics['total_loss']:.6f} "
            f"val_loss={val_metrics['total_loss']:.6f} "
            f"changed_node_error={val_metrics['changed_node_error']:.6f} "
            f"unchanged_node_preservation_error={val_metrics['unchanged_node_preservation_error']:.6f} "
            f"changed_edge_error={val_metrics['changed_edge_error']:.6f} "
            f"unchanged_edge_preservation_error={val_metrics['unchanged_edge_preservation_error']:.6f}"
        )

    print(f"best_checkpoint={best_path}")
    print(f"history_json={history_path}")


if __name__ == "__main__":
    main()
