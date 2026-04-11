from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.collate import graph_event_collate_fn
from data.dataset import GraphEventDataset
from models.oracle_local import build_valid_edge_mask
from models.proposal import ScopeProposalConfig, ScopeProposalModel, scope_proposal_loss


INTERACTION_SAMPLE_WEIGHTS = {
    "fully_independent": 1.0,
    "partially_dependent": 1.5,
    "strongly_interacting": 2.0,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


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
        out[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return out


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def require_scope_targets(batch: Dict[str, Any]) -> None:
    missing = [key for key in ["event_scope_union_nodes", "event_scope_union_edges"] if key not in batch]
    if missing:
        raise KeyError(f"Step 23 proposal training requires clean oracle scope labels. Missing: {missing}")


def get_model_inputs(batch: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    return batch.get("obs_node_feats", batch["node_feats"]), batch.get("obs_adj", batch["adj"])


def bucket_for_raw_sample(raw_sample: Dict[str, Any]) -> str:
    return str(raw_sample.get("step5_dependency_bucket", "unknown"))


def build_interaction_sampler(dataset: GraphEventDataset, seed: int) -> WeightedRandomSampler:
    weights = [
        INTERACTION_SAMPLE_WEIGHTS.get(bucket_for_raw_sample(sample), 1.0)
        for sample in dataset.samples
    ]
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )


def build_dataloaders(
    train_path: str,
    val_path: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    seed: int,
) -> tuple[GraphEventDataset, GraphEventDataset, DataLoader, DataLoader]:
    train_dataset = GraphEventDataset(train_path)
    val_dataset = GraphEventDataset(val_path)
    train_sampler = build_interaction_sampler(train_dataset, seed=seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
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


def infer_event_type(batch: Dict[str, Any], index: int) -> str:
    events = batch.get("events")
    if not events:
        return "unknown"
    sample_events = events[index]
    if not sample_events:
        return "unknown"
    return str(sample_events[0].get("event_type", "unknown"))


class ScopeMetricAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.node_tp = 0.0
        self.node_pred = 0.0
        self.node_true = 0.0
        self.node_excess = 0.0
        self.edge_tp = 0.0
        self.edge_pred = 0.0
        self.edge_true = 0.0
        self.edge_excess = 0.0
        self.changed_edge_tp = 0.0
        self.changed_edge_true = 0.0

    def update(
        self,
        pred_node: torch.Tensor,
        true_node: torch.Tensor,
        node_mask: torch.Tensor,
        pred_edge: torch.Tensor,
        true_edge: torch.Tensor,
        valid_edge: torch.Tensor,
        changed_edge: Optional[torch.Tensor],
    ) -> None:
        pred_node = pred_node.bool() & node_mask.bool()
        true_node = true_node.bool() & node_mask.bool()
        pred_edge = pred_edge.bool() & valid_edge.bool()
        true_edge = true_edge.bool() & valid_edge.bool()

        self.count += 1
        self.node_tp += (pred_node & true_node).float().sum().item()
        self.node_pred += pred_node.float().sum().item()
        self.node_true += true_node.float().sum().item()
        self.node_excess += (pred_node & (~true_node)).float().sum().item()
        self.edge_tp += (pred_edge & true_edge).float().sum().item()
        self.edge_pred += pred_edge.float().sum().item()
        self.edge_true += true_edge.float().sum().item()
        self.edge_excess += (pred_edge & (~true_edge)).float().sum().item()
        if changed_edge is not None:
            changed = changed_edge.bool() & valid_edge.bool()
            self.changed_edge_tp += (pred_edge & changed).float().sum().item()
            self.changed_edge_true += changed.float().sum().item()

    @staticmethod
    def _safe_div(num: float, den: float) -> Optional[float]:
        return None if den <= 0 else num / den

    @staticmethod
    def _f1(precision: Optional[float], recall: Optional[float]) -> Optional[float]:
        if precision is None or recall is None or precision + recall <= 0:
            return None
        return 2.0 * precision * recall / (precision + recall)

    def finalize(self) -> Dict[str, Any]:
        node_precision = self._safe_div(self.node_tp, self.node_pred)
        node_recall = self._safe_div(self.node_tp, self.node_true)
        edge_precision = self._safe_div(self.edge_tp, self.edge_pred)
        edge_recall = self._safe_div(self.edge_tp, self.edge_true)
        changed_recall = self._safe_div(self.changed_edge_tp, self.changed_edge_true)
        return {
            "count": self.count,
            "proposal_node_scope_precision": node_precision,
            "proposal_node_scope_recall": node_recall,
            "proposal_node_scope_f1": self._f1(node_precision, node_recall),
            "proposal_node_scope_excess_ratio": self._safe_div(self.node_excess, self.node_pred),
            "proposal_edge_scope_precision": edge_precision,
            "proposal_edge_scope_recall": edge_recall,
            "proposal_edge_scope_f1": self._f1(edge_precision, edge_recall),
            "proposal_edge_scope_excess_ratio": self._safe_div(self.edge_excess, self.edge_pred),
            "proposal_changed_edge_recall": changed_recall,
            "proposal_out_of_scope_miss_edge": None if changed_recall is None else 1.0 - changed_recall,
            "proposal_counts": {
                "node_tp": self.node_tp,
                "node_pred_pos": self.node_pred,
                "node_true_pos": self.node_true,
                "edge_tp": self.edge_tp,
                "edge_pred_pos": self.edge_pred,
                "edge_true_pos": self.edge_true,
                "changed_edge_tp": self.changed_edge_tp,
                "changed_edge_true_pos": self.changed_edge_true,
            },
        }


def value_or_zero(metrics: Dict[str, Any], key: str) -> float:
    value = metrics.get(key)
    return float(value) if value is not None else 0.0


def selection_score(overall: Dict[str, Any], strong: Dict[str, Any]) -> float:
    # 固定选择规则：保留整体 proposal 质量，同时给 strongly_interacting changed-edge coverage 明确压力。
    return (
        0.40 * value_or_zero(overall, "proposal_edge_scope_f1")
        + 0.20 * value_or_zero(overall, "proposal_node_scope_f1")
        + 0.25 * value_or_zero(strong, "proposal_changed_edge_recall")
        + 0.15 * value_or_zero(strong, "proposal_edge_scope_recall")
    )


def empty_group_accumulators() -> Dict[str, Dict[str, ScopeMetricAccumulator]]:
    return {
        "by_dependency_bucket": defaultdict(ScopeMetricAccumulator),
        "by_corruption_setting": defaultdict(ScopeMetricAccumulator),
        "by_event_type": defaultdict(ScopeMetricAccumulator),
    }


def finalize_grouped(grouped: Dict[str, ScopeMetricAccumulator]) -> Dict[str, Any]:
    return {group: acc.finalize() for group, acc in sorted(grouped.items())}


def run_epoch(
    model: ScopeProposalModel,
    loader: DataLoader,
    device: torch.device,
    node_scope_loss_weight: float,
    node_flip_weight: float,
    edge_scope_loss_weight: float,
    edge_scope_pos_weight: float,
    node_threshold: float,
    edge_threshold: float,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_clip: Optional[float] = None,
) -> Dict[str, Any]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    overall_acc = ScopeMetricAccumulator()
    grouped = empty_group_accumulators()
    loss_sums = {"total_loss": 0.0, "node_scope_loss": 0.0, "edge_scope_loss": 0.0}
    num_batches = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in loader:
            require_scope_targets(batch)
            batch = move_batch_to_device(batch, device)
            node_feats, adj = get_model_inputs(batch)
            outputs = model(node_feats=node_feats, adj=adj)
            valid_edge_mask = build_valid_edge_mask(batch["node_mask"])

            current_type = batch["node_feats"][:, :, 0].long()
            target_type = batch["next_node_feats"][:, :, 0].long()
            flip_target_mask = (current_type != target_type).to(batch["node_mask"].dtype)
            node_scope_weights = 1.0 + (node_flip_weight - 1.0) * flip_target_mask
            loss_dict = scope_proposal_loss(
                outputs=outputs,
                target_node_scope=batch["event_scope_union_nodes"],
                target_edge_scope=batch["event_scope_union_edges"],
                node_mask=batch["node_mask"],
                pair_mask=valid_edge_mask,
                node_scope_loss_weight=node_scope_loss_weight,
                edge_scope_loss_weight=edge_scope_loss_weight,
                edge_scope_pos_weight=edge_scope_pos_weight,
                node_scope_weights=node_scope_weights,
            )

            if is_train:
                optimizer.zero_grad()
                loss_dict["total_loss"].backward()
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            pred_node = (torch.sigmoid(outputs["node_scope_logits"]) >= node_threshold) & batch["node_mask"].bool()
            pred_edge = (torch.sigmoid(outputs["edge_scope_logits"]) >= edge_threshold) & valid_edge_mask.bool()
            changed_edges = batch.get("changed_edges")

            bsz = batch["node_mask"].shape[0]
            for idx in range(bsz):
                sample_changed = changed_edges[idx] if changed_edges is not None else None
                overall_acc.update(
                    pred_node=pred_node[idx],
                    true_node=batch["event_scope_union_nodes"][idx],
                    node_mask=batch["node_mask"][idx],
                    pred_edge=pred_edge[idx],
                    true_edge=batch["event_scope_union_edges"][idx],
                    valid_edge=valid_edge_mask[idx],
                    changed_edge=sample_changed,
                )
                bucket = str(batch.get("step5_dependency_bucket", ["unknown"] * bsz)[idx])
                corruption = str(batch.get("step6a_corruption_setting", ["unknown"] * bsz)[idx])
                event_type = infer_event_type(batch, idx)
                for acc in (
                    grouped["by_dependency_bucket"][bucket],
                    grouped["by_corruption_setting"][corruption],
                    grouped["by_event_type"][event_type],
                ):
                    acc.update(
                        pred_node=pred_node[idx],
                        true_node=batch["event_scope_union_nodes"][idx],
                        node_mask=batch["node_mask"][idx],
                        pred_edge=pred_edge[idx],
                        true_edge=batch["event_scope_union_edges"][idx],
                        valid_edge=valid_edge_mask[idx],
                        changed_edge=sample_changed,
                    )

            for key in loss_sums:
                loss_sums[key] += float(loss_dict[key].item())
            num_batches += 1

    losses = {key: value / max(num_batches, 1) for key, value in loss_sums.items()}
    overall = overall_acc.finalize()
    by_bucket = finalize_grouped(grouped["by_dependency_bucket"])
    strong_metrics = by_bucket.get("strongly_interacting", ScopeMetricAccumulator().finalize())
    score = selection_score(overall, strong_metrics)
    return {
        **losses,
        "selection_score": score,
        "selection_metric": (
            "0.40*overall_edge_f1 + 0.20*overall_node_f1 + "
            "0.25*strong_changed_edge_recall + 0.15*strong_edge_scope_recall"
        ),
        "overall": overall,
        "by_dependency_bucket": by_bucket,
        "by_corruption_setting": finalize_grouped(grouped["by_corruption_setting"]),
        "by_event_type": finalize_grouped(grouped["by_event_type"]),
    }


def summarize_raw_samples(dataset: GraphEventDataset) -> Dict[str, Any]:
    bucket_counts = Counter(str(sample.get("step5_dependency_bucket", "unknown")) for sample in dataset.samples)
    corruption_counts = Counter(str(sample.get("step6a_corruption_setting", "unknown")) for sample in dataset.samples)
    event_counts = Counter(
        str(sample.get("events", [{}])[0].get("event_type", "unknown"))
        for sample in dataset.samples
    )
    return {
        "sample_count": len(dataset),
        "dependency_bucket_counts": dict(sorted(bucket_counts.items())),
        "corruption_setting_counts": dict(sorted(corruption_counts.items())),
        "event_type_counts": dict(sorted(event_counts.items())),
    }


def load_initialized_model(checkpoint_path: Path, device: torch.device) -> tuple[ScopeProposalModel, ScopeProposalConfig]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = ScopeProposalConfig(**checkpoint["model_config"])
    model = ScopeProposalModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="data/graph_event_step23_noisy_step5_train_transitions.pkl")
    parser.add_argument("--val_path", type=str, default="data/graph_event_step23_noisy_step5_val_transitions.pkl")
    parser.add_argument("--init_proposal_checkpoint", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--node_scope_loss_weight", type=float, default=1.0)
    parser.add_argument("--node_flip_weight", type=float, default=2.0)
    parser.add_argument("--edge_scope_loss_weight", type=float, default=1.0)
    parser.add_argument("--edge_scope_pos_weight", type=float, default=2.0)
    parser.add_argument("--node_threshold", type=float, default=0.15)
    parser.add_argument("--edge_threshold", type=float, default=0.10)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    pin_memory = device.type == "cuda"
    save_dir = resolve_path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset, train_loader, val_loader = build_dataloaders(
        train_path=str(resolve_path(args.train_path)),
        val_path=str(resolve_path(args.val_path)),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        seed=args.seed,
    )
    sample = train_dataset[0]
    if "obs_node_feats" not in sample or "obs_adj" not in sample:
        raise KeyError("Step 23 training expects noisy Step 22/23 transition samples with obs_node_feats/obs_adj.")
    if "event_scope_union_nodes" not in sample or "event_scope_union_edges" not in sample:
        raise KeyError("Step 23 training requires clean oracle scope labels.")

    init_path = resolve_path(args.init_proposal_checkpoint)
    model, model_config = load_initialized_model(init_path, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_ckpt_path = save_dir / "best.pt"
    last_ckpt_path = save_dir / "last.pt"
    best_metrics_path = save_dir / "best_metrics.json"
    summary_path = save_dir / "training_summary.json"
    best_val_score = float("-inf")
    best_epoch = -1
    epochs_without_improve = 0

    print(f"device: {device}")
    print(f"train dataset: {args.train_path} size={len(train_dataset)}")
    print(f"val dataset: {args.val_path} size={len(val_dataset)}")
    print(f"init proposal checkpoint: {init_path}")
    print(f"thresholds: node={args.node_threshold} edge={args.edge_threshold}")
    print(f"fixed interaction sampler weights: {INTERACTION_SAMPLE_WEIGHTS}")
    print(f"train summary: {summarize_raw_samples(train_dataset)}")
    print(f"val summary: {summarize_raw_samples(val_dataset)}")

    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            node_scope_loss_weight=args.node_scope_loss_weight,
            node_flip_weight=args.node_flip_weight,
            edge_scope_loss_weight=args.edge_scope_loss_weight,
            edge_scope_pos_weight=args.edge_scope_pos_weight,
            node_threshold=args.node_threshold,
            edge_threshold=args.edge_threshold,
            optimizer=optimizer,
            grad_clip=args.grad_clip,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            node_scope_loss_weight=args.node_scope_loss_weight,
            node_flip_weight=args.node_flip_weight,
            edge_scope_loss_weight=args.edge_scope_loss_weight,
            edge_scope_pos_weight=args.edge_scope_pos_weight,
            node_threshold=args.node_threshold,
            edge_threshold=args.edge_threshold,
            optimizer=None,
            grad_clip=None,
        )
        strong_val = val_metrics["by_dependency_bucket"].get("strongly_interacting", {})
        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_metrics['total_loss']:.6f} "
            f"val_loss={val_metrics['total_loss']:.6f} "
            f"val_edge_f1={value_or_zero(val_metrics['overall'], 'proposal_edge_scope_f1'):.6f} "
            f"val_edge_recall={value_or_zero(val_metrics['overall'], 'proposal_edge_scope_recall'):.6f} "
            f"strong_changed_recall={value_or_zero(strong_val, 'proposal_changed_edge_recall'):.6f} "
            f"strong_out_miss={value_or_zero(strong_val, 'proposal_out_of_scope_miss_edge'):.6f} "
            f"val_score={val_metrics['selection_score']:.6f}"
        )

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": vars(model_config),
            "args": vars(args),
            "init_proposal_checkpoint": str(init_path),
            "proposal_training_regime": "noisy_interaction_aware_P2",
            "interaction_sampler_weights": INTERACTION_SAMPLE_WEIGHTS,
            "input_mode": "obs_node_feats/obs_adj",
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "selection_metric": val_metrics["selection_metric"],
            "selection_score": val_metrics["selection_score"],
        }
        torch.save(ckpt, last_ckpt_path)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["total_loss"],
                "val_loss": val_metrics["total_loss"],
                "val_selection_score": val_metrics["selection_score"],
                "val_overall": val_metrics["overall"],
                "val_strongly_interacting": strong_val,
            }
        )

        if val_metrics["selection_score"] > best_val_score:
            best_val_score = val_metrics["selection_score"]
            best_epoch = epoch
            epochs_without_improve = 0
            torch.save(ckpt, best_ckpt_path)
            save_json(
                best_metrics_path,
                {
                    "epoch": epoch,
                    "best_val_selection_score": best_val_score,
                    "selection_metric": val_metrics["selection_metric"],
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "model_config": vars(model_config),
                    "args": vars(args),
                    "init_proposal_checkpoint": str(init_path),
                    "proposal_training_regime": "noisy_interaction_aware_P2",
                    "interaction_sampler_weights": INTERACTION_SAMPLE_WEIGHTS,
                },
            )
            print(f"  saved new best checkpoint -> {best_ckpt_path}")
        else:
            epochs_without_improve += 1
            print(f"  no improvement. patience {epochs_without_improve}/{args.patience}")
            if epochs_without_improve >= args.patience:
                print("early stopping triggered")
                break

    save_json(
        summary_path,
        {
            "best_epoch": best_epoch,
            "best_val_selection_score": best_val_score,
            "selection_metric": (
                "0.40*overall_edge_f1 + 0.20*overall_node_f1 + "
                "0.25*strong_changed_edge_recall + 0.15*strong_edge_scope_recall"
            ),
            "history": history,
            "train_dataset_summary": summarize_raw_samples(train_dataset),
            "val_dataset_summary": summarize_raw_samples(val_dataset),
            "args": vars(args),
            "init_proposal_checkpoint": str(init_path),
            "proposal_training_regime": "noisy_interaction_aware_P2",
            "interaction_sampler_weights": INTERACTION_SAMPLE_WEIGHTS,
        },
    )
    print(f"training finished. best_epoch={best_epoch} best_val_selection_score={best_val_score:.6f}")
    print(f"best checkpoint: {best_ckpt_path}")
    print(f"last checkpoint: {last_ckpt_path}")
    print(f"best metrics json: {best_metrics_path}")
    print(f"training summary json: {summary_path}")


if __name__ == "__main__":
    main()
