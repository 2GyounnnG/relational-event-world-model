from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.collate import graph_event_collate_fn
from data.dataset import GraphEventDataset
from models.oracle_local_delta import (
    OracleLocalDeltaRewriteConfig,
    OracleLocalDeltaRewriteModel,
    build_valid_edge_mask,
    masked_edge_keep_regularization_loss_from_pair_mask,
    oracle_local_delta_rewrite_loss,
)
from train.eval_sequential_composition_consistency import (
    build_pair_summary,
    evaluate_step3b,
    inspect_step3b_dataset,
)
from train.train_oracle_local_delta_proposal_conditioned_fp_keep import (
    get_device,
    get_proposal_predictions,
    load_frozen_proposal,
    maybe_load_rewrite_init,
    move_batch_to_device,
    resolve_path,
    save_json,
    set_seed,
)


REQUIRED_ROLES = ("base_to_A", "base_to_B", "A_to_AB", "B_to_AB")


class SequentialPairBatchSampler(BatchSampler):
    def __init__(self, pair_indices: List[List[int]], pairs_per_batch: int, shuffle: bool, seed: int):
        self.pair_indices = pair_indices
        self.pairs_per_batch = max(1, pairs_per_batch)
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self) -> Iterator[List[int]]:
        order = list(range(len(self.pair_indices)))
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(order)
        for start in range(0, len(order), self.pairs_per_batch):
            batch_pair_ids = order[start : start + self.pairs_per_batch]
            batch: List[int] = []
            for pair_idx in batch_pair_ids:
                batch.extend(self.pair_indices[pair_idx])
            yield batch

    def __len__(self) -> int:
        return math.ceil(len(self.pair_indices) / self.pairs_per_batch)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


def build_pair_index_groups(dataset: GraphEventDataset) -> List[List[int]]:
    by_pair_id: Dict[str, Dict[str, int]] = {}
    for idx, raw in enumerate(dataset.samples):
        pair_id = raw.get("step3_pair_id")
        role = raw.get("step3_transition_role")
        if pair_id is None or role is None:
            continue
        pair_id = str(pair_id)
        by_pair_id.setdefault(pair_id, {})
        by_pair_id[pair_id][str(role)] = idx

    groups: List[List[int]] = []
    for _, role_map in sorted(by_pair_id.items()):
        if set(role_map.keys()) != set(REQUIRED_ROLES):
            continue
        groups.append([role_map[role] for role in REQUIRED_ROLES])
    return groups


def build_dataloaders(
    train_path: str,
    val_path: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    seed: int,
) -> tuple[GraphEventDataset, GraphEventDataset, DataLoader, DataLoader, SequentialPairBatchSampler]:
    train_dataset = GraphEventDataset(train_path)
    val_dataset = GraphEventDataset(val_path)

    train_pair_groups = build_pair_index_groups(train_dataset)
    if not train_pair_groups:
        raise ValueError("No complete Step 3 sequential pair groups found in train dataset")

    pairs_per_batch = max(1, batch_size // 4)
    batch_sampler = SequentialPairBatchSampler(
        pair_indices=train_pair_groups,
        pairs_per_batch=pairs_per_batch,
        shuffle=True,
        seed=seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
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
    return train_dataset, val_dataset, train_loader, val_loader, batch_sampler


def zero_like_loss(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def masked_node_prob_mse(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum() * a.shape[-1]
    if denom.item() <= 0:
        return zero_like_loss(a)
    diff2 = (a - b) ** 2
    return (diff2 * mask.unsqueeze(-1)).sum() / denom


def masked_edge_prob_mse(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum()
    if denom.item() <= 0:
        return zero_like_loss(a)
    diff2 = (a - b) ** 2
    return (diff2 * mask).sum() / denom


def compute_second_step_consistency_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    by_pair: Dict[str, Dict[str, int]] = {}
    batch_size = len(batch["step3_pair_id"])
    for idx in range(batch_size):
        pair_id = str(batch["step3_pair_id"][idx])
        role = str(batch["step3_transition_role"][idx])
        by_pair.setdefault(pair_id, {})
        by_pair[pair_id][role] = idx

    type_losses: List[torch.Tensor] = []
    state_losses: List[torch.Tensor] = []
    edge_losses: List[torch.Tensor] = []

    for role_map in by_pair.values():
        if "A_to_AB" not in role_map or "B_to_AB" not in role_map:
            continue
        idx_a = role_map["A_to_AB"]
        idx_b = role_map["B_to_AB"]

        node_mask = (batch["node_mask"][idx_a] > 0.5) & (batch["node_mask"][idx_b] > 0.5)
        valid_edge_mask = build_valid_edge_mask(batch["node_mask"][idx_a : idx_a + 1])[0].bool()
        valid_edge_mask = valid_edge_mask & build_valid_edge_mask(batch["node_mask"][idx_b : idx_b + 1])[0].bool()

        type_prob_a = torch.softmax(outputs["type_logits_full"][idx_a], dim=-1)
        type_prob_b = torch.softmax(outputs["type_logits_full"][idx_b], dim=-1)
        type_losses.append(masked_node_prob_mse(type_prob_a, type_prob_b, node_mask.float()))

        state_a = outputs["state_pred_full"][idx_a]
        state_b = outputs["state_pred_full"][idx_b]
        state_losses.append(masked_node_prob_mse(state_a, state_b, node_mask.float()))

        edge_prob_a = torch.sigmoid(outputs["edge_logits_full"][idx_a])
        edge_prob_b = torch.sigmoid(outputs["edge_logits_full"][idx_b])
        edge_losses.append(masked_edge_prob_mse(edge_prob_a, edge_prob_b, valid_edge_mask.float()))

    if not type_losses:
        zero = zero_like_loss(outputs["edge_logits_full"])
        return {
            "type_consistency_loss": zero,
            "state_consistency_loss": zero,
            "edge_consistency_loss": zero,
            "consistency_loss": zero,
            "num_consistency_pairs": 0.0,
        }

    type_loss = torch.stack(type_losses).mean()
    state_loss = torch.stack(state_losses).mean()
    edge_loss = torch.stack(edge_losses).mean()
    total = (type_loss + state_loss + edge_loss) / 3.0
    return {
        "type_consistency_loss": type_loss,
        "state_consistency_loss": state_loss,
        "edge_consistency_loss": edge_loss,
        "consistency_loss": total,
        "num_consistency_pairs": float(len(type_losses)),
    }


def run_train_epoch(
    model: OracleLocalDeltaRewriteModel,
    proposal_model,
    loader: DataLoader,
    batch_sampler: SequentialPairBatchSampler,
    device: torch.device,
    edge_loss_weight: float,
    type_loss_weight: float,
    state_loss_weight: float,
    type_flip_weight: float,
    delta_keep_weight: float,
    delta_add_weight: float,
    delta_delete_weight: float,
    fp_edge_keep_loss_weight: float,
    consistency_loss_weight: float,
    node_threshold: float,
    edge_threshold: float,
    optimizer: torch.optim.Optimizer,
    grad_clip: Optional[float],
    epoch: int,
) -> Dict[str, float]:
    model.train()
    proposal_model.eval()
    batch_sampler.set_epoch(epoch)

    total_supervised = 0.0
    total_fp_keep = 0.0
    total_consistency = 0.0
    total_loss = 0.0
    total_consistency_pairs = 0.0
    num_batches = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        proposal_node_probs, proposal_edge_probs, pred_scope_nodes, pred_scope_edges = get_proposal_predictions(
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

        supervised_loss_dict = oracle_local_delta_rewrite_loss(
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

        valid_edge_mask = build_valid_edge_mask(batch["node_mask"])
        oracle_edge_scope_mask = batch["event_scope_union_edges"] * valid_edge_mask
        predicted_only_edge_mask = pred_scope_edges * valid_edge_mask * (1.0 - oracle_edge_scope_mask)
        fp_edge_keep_loss = masked_edge_keep_regularization_loss_from_pair_mask(
            outputs["edge_delta_logits_local"],
            predicted_only_edge_mask,
        )

        consistency_dict = compute_second_step_consistency_loss(outputs, batch)
        combined_loss = (
            supervised_loss_dict["total_loss"]
            + fp_edge_keep_loss_weight * fp_edge_keep_loss
            + consistency_loss_weight * consistency_dict["consistency_loss"]
        )

        optimizer.zero_grad()
        combined_loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_supervised += supervised_loss_dict["total_loss"].item()
        total_fp_keep += fp_edge_keep_loss.item()
        total_consistency += consistency_dict["consistency_loss"].item()
        total_consistency_pairs += consistency_dict["num_consistency_pairs"]
        total_loss += combined_loss.item()
        num_batches += 1

    denom = max(1, num_batches)
    return {
        "train_supervised_loss": total_supervised / denom,
        "train_fp_keep_loss": total_fp_keep / denom,
        "train_consistency_loss": total_consistency / denom,
        "train_total_loss": total_loss / denom,
        "train_avg_consistency_pairs_per_batch": total_consistency_pairs / denom,
    }


def compute_selection_score(summary: Dict[str, Any]) -> float:
    second = summary["second_step_average"]
    quality_metrics = [
        second["delta_all"],
        second["keep"],
        second["delete"],
        second["changed"],
        second["context"],
    ]
    path_metrics = [
        summary["path_mean_abs_gap_delta_all"],
        summary["path_mean_abs_gap_keep"],
        summary["path_mean_abs_gap_delete"],
        summary["path_mean_abs_gap_changed"],
        summary["path_mean_abs_gap_context"],
    ]
    return float(sum(quality_metrics) / len(quality_metrics) - sum(path_metrics) / len(path_metrics))


def build_model_config_from_args_or_checkpoint(
    args: argparse.Namespace,
    init_rewrite_checkpoint_path: Optional[Path],
    node_feat_dim: int,
    state_dim: int,
) -> OracleLocalDeltaRewriteConfig:
    if init_rewrite_checkpoint_path is not None:
        checkpoint = torch.load(init_rewrite_checkpoint_path, map_location="cpu")
        config_dict = dict(checkpoint["model_config"])
        config_dict["use_proposal_conditioning"] = True
        return OracleLocalDeltaRewriteConfig(**config_dict)
    return OracleLocalDeltaRewriteConfig(
        node_feat_dim=node_feat_dim,
        state_dim=state_dim,
        hidden_dim=args.hidden_dim,
        msg_pass_layers=args.msg_pass_layers,
        node_mlp_layers=args.node_mlp_layers,
        edge_mlp_layers=args.edge_mlp_layers,
        dropout=args.dropout,
        edge_dropout=args.edge_dropout,
        copy_logit_value=args.copy_logit_value,
        use_proposal_conditioning=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--proposal_checkpoint", type=str, required=True)
    parser.add_argument("--init_rewrite_checkpoint", type=str, default=None)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--node_threshold", type=float, default=0.20)
    parser.add_argument("--edge_threshold", type=float, default=0.15)
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
    parser.add_argument("--copy_logit_value", type=float, default=10.0)
    parser.add_argument("--edge_loss_weight", type=float, default=1.0)
    parser.add_argument("--type_loss_weight", type=float, default=1.0)
    parser.add_argument("--state_loss_weight", type=float, default=1.0)
    parser.add_argument("--type_flip_weight", type=float, default=1.0)
    parser.add_argument("--delta_keep_weight", type=float, default=1.10)
    parser.add_argument("--delta_add_weight", type=float, default=1.0)
    parser.add_argument("--delta_delete_weight", type=float, default=3.0)
    parser.add_argument("--fp_edge_keep_loss_weight", type=float, default=0.12)
    parser.add_argument("--consistency_loss_weight", type=float, default=0.10)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    pin_memory = device.type == "cuda"

    train_path = resolve_path(args.train_path)
    val_path = resolve_path(args.val_path)
    proposal_checkpoint_path = resolve_path(args.proposal_checkpoint)
    init_rewrite_checkpoint_path = (
        resolve_path(args.init_rewrite_checkpoint) if args.init_rewrite_checkpoint else None
    )
    save_dir = resolve_path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset, train_loader, val_loader, train_batch_sampler = build_dataloaders(
        train_path=str(train_path),
        val_path=str(val_path),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        seed=args.seed,
    )
    train_support = inspect_step3b_dataset(train_dataset)
    val_support = inspect_step3b_dataset(val_dataset)

    sample_item = train_dataset[0]
    node_feat_dim = int(sample_item["node_feats"].shape[-1])
    state_dim = int(node_feat_dim - 1)

    model_config = build_model_config_from_args_or_checkpoint(
        args=args,
        init_rewrite_checkpoint_path=init_rewrite_checkpoint_path,
        node_feat_dim=node_feat_dim,
        state_dim=state_dim,
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
    print(f"train transitions: {len(train_dataset)}")
    print(f"val transitions: {len(val_dataset)}")
    print(f"train complete pairs: {train_support['complete_pair_count']}")
    print(f"val complete pairs: {val_support['complete_pair_count']}")
    print(f"proposal checkpoint: {proposal_checkpoint_path}")
    print(f"init rewrite checkpoint: {init_rewrite_checkpoint_path}")
    print(f"proposal conditioning enabled: {model_config.use_proposal_conditioning}")
    print(f"consistency loss weight: {args.consistency_loss_weight}")
    print(
        "selection metric: "
        "mean(second_step delta_all/keep/delete/changed/context) - "
        "mean(path gap delta_all/keep/delete/changed/context)"
    )

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_train_epoch(
            model=model,
            proposal_model=proposal_model,
            loader=train_loader,
            batch_sampler=train_batch_sampler,
            device=device,
            edge_loss_weight=args.edge_loss_weight,
            type_loss_weight=args.type_loss_weight,
            state_loss_weight=args.state_loss_weight,
            type_flip_weight=args.type_flip_weight,
            delta_keep_weight=args.delta_keep_weight,
            delta_add_weight=args.delta_add_weight,
            delta_delete_weight=args.delta_delete_weight,
            fp_edge_keep_loss_weight=args.fp_edge_keep_loss_weight,
            consistency_loss_weight=args.consistency_loss_weight,
            node_threshold=args.node_threshold,
            edge_threshold=args.edge_threshold,
            optimizer=optimizer,
            grad_clip=args.grad_clip,
            epoch=epoch,
        )

        val_eval = evaluate_step3b(
            proposal_model=proposal_model,
            rewrite_model=model,
            loader=val_loader,
            device=device,
            node_threshold=args.node_threshold,
            edge_threshold=args.edge_threshold,
            use_proposal_conditioning=model_config.use_proposal_conditioning,
        )
        val_summary = build_pair_summary(val_eval["transition_records"])["overall"]
        selection_score = compute_selection_score(val_summary)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_total={train_metrics['train_total_loss']:.6f} "
            f"train_supervised={train_metrics['train_supervised_loss']:.6f} "
            f"train_fp_keep={train_metrics['train_fp_keep_loss']:.6f} "
            f"train_consistency={train_metrics['train_consistency_loss']:.6f} "
            f"val_second_delta={val_summary['second_step_average']['delta_all']:.6f} "
            f"val_second_keep={val_summary['second_step_average']['keep']:.6f} "
            f"val_second_delete={val_summary['second_step_average']['delete']:.6f} "
            f"val_second_changed={val_summary['second_step_average']['changed']:.6f} "
            f"val_second_context={val_summary['second_step_average']['context']:.6f} "
            f"val_path_gap_delta={val_summary['path_mean_abs_gap_delta_all']:.6f} "
            f"val_path_gap_keep={val_summary['path_mean_abs_gap_keep']:.6f} "
            f"val_path_gap_delete={val_summary['path_mean_abs_gap_delete']:.6f} "
            f"val_path_gap_changed={val_summary['path_mean_abs_gap_changed']:.6f} "
            f"val_path_gap_context={val_summary['path_mean_abs_gap_context']:.6f} "
            f"selection_score={selection_score:.6f}"
        )

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": vars(model_config),
            "args": vars(args),
            "train_metrics": train_metrics,
            "val_step3b_summary": val_summary,
            "selection_metric": "step3b_second_step_quality_minus_path_gaps",
            "selection_score": selection_score,
            "selection_formula": {
                "quality_metrics": ["second_step.delta_all", "second_step.keep", "second_step.delete", "second_step.changed", "second_step.context"],
                "gap_metrics": ["path_gap.delta_all", "path_gap.keep", "path_gap.delete", "path_gap.changed", "path_gap.context"],
            },
            "proposal_checkpoint": str(proposal_checkpoint_path),
            "init_rewrite_checkpoint": str(init_rewrite_checkpoint_path) if init_rewrite_checkpoint_path else None,
            "proposal_conditioning_enabled": model_config.use_proposal_conditioning,
            "fp_edge_keep_loss_weight": args.fp_edge_keep_loss_weight,
            "consistency_loss_weight": args.consistency_loss_weight,
            "step3_dataset_type": "sequential_composition",
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
                    "selection_metric": "step3b_second_step_quality_minus_path_gaps",
                    "train_metrics": train_metrics,
                    "val_step3b_summary": val_summary,
                    "model_config": vars(model_config),
                    "args": vars(args),
                    "proposal_checkpoint": str(proposal_checkpoint_path),
                    "init_rewrite_checkpoint": str(init_rewrite_checkpoint_path) if init_rewrite_checkpoint_path else None,
                    "proposal_conditioning_enabled": model_config.use_proposal_conditioning,
                    "fp_edge_keep_loss_weight": args.fp_edge_keep_loss_weight,
                    "consistency_loss_weight": args.consistency_loss_weight,
                    "step3_dataset_type": "sequential_composition",
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
