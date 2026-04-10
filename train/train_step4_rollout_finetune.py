from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import edge_pairs_to_dense_mask
from models.oracle_local_delta import (
    OracleLocalDeltaRewriteConfig,
    OracleLocalDeltaRewriteModel,
    build_valid_edge_mask,
    masked_edge_keep_regularization_loss_from_pair_mask,
    oracle_local_delta_rewrite_loss,
)
from models.proposal import ScopeProposalConfig, ScopeProposalModel
from train.eval_rollout_stability import (
    evaluate_rollouts,
    load_rollout_dataset,
)


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


def safe_div(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return num / den


class RolloutDataset(Dataset):
    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        self.samples = [
            sample
            for sample in load_rollout_dataset(self.file_path)
            if len(sample.get("graph_steps", [])) >= 2 and len(sample.get("transition_samples", [])) >= 2
        ]
        if not self.samples:
            raise ValueError(f"No usable 2-step rollout samples found in {self.file_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


def rollout_collate_fn(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return batch


def build_rollout_dataloaders(
    train_path: str,
    val_path: str,
    batch_size: int,
    num_workers: int,
) -> tuple[RolloutDataset, RolloutDataset, DataLoader, DataLoader]:
    train_dataset = RolloutDataset(train_path)
    val_dataset = RolloutDataset(val_path)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=rollout_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=rollout_collate_fn,
    )
    return train_dataset, val_dataset, train_loader, val_loader


def load_frozen_proposal(proposal_checkpoint_path: Path, device: torch.device) -> ScopeProposalModel:
    checkpoint = torch.load(proposal_checkpoint_path, map_location="cpu")
    model = ScopeProposalModel(ScopeProposalConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def load_rewrite_from_init(init_checkpoint_path: Path, device: torch.device) -> tuple[OracleLocalDeltaRewriteModel, Dict[str, Any]]:
    checkpoint = torch.load(init_checkpoint_path, map_location="cpu")
    model = OracleLocalDeltaRewriteModel(
        OracleLocalDeltaRewriteConfig(**checkpoint["model_config"])
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model, checkpoint


def dense_edge_mask_from_pairs(edge_pairs: List[List[int]] | List[tuple[int, int]], num_nodes: int, device: torch.device) -> torch.Tensor:
    return edge_pairs_to_dense_mask(edge_pairs, num_nodes=num_nodes, undirected=True).to(device)


def graph_to_tensor_batch(graph: Dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    node_feats = torch.tensor(graph["node_features"], dtype=torch.float32, device=device).unsqueeze(0)
    adj = torch.tensor(graph["adj"], dtype=torch.float32, device=device).unsqueeze(0)
    node_mask = torch.ones((1, node_feats.shape[1]), dtype=torch.float32, device=device)
    return node_feats, adj, node_mask


def transition_targets(
    transition_sample: Dict[str, Any],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    graph_t = transition_sample["graph_t"]
    graph_t1 = transition_sample["graph_t1"]
    current_node_feats, current_adj, node_mask = graph_to_tensor_batch(graph_t, device)
    next_node_feats = torch.tensor(graph_t1["node_features"], dtype=torch.float32, device=device).unsqueeze(0)
    next_adj = torch.tensor(graph_t1["adj"], dtype=torch.float32, device=device).unsqueeze(0)
    num_nodes = current_node_feats.shape[1]

    changed_nodes = torch.tensor(
        np.asarray(transition_sample["changed_nodes"], dtype=np.float32),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    changed_edges = dense_edge_mask_from_pairs(transition_sample["changed_edges"], num_nodes, device).unsqueeze(0)

    scope_nodes = torch.zeros((1, num_nodes), dtype=torch.float32, device=device)
    for node_idx in transition_sample["event_scope_union_nodes"]:
        scope_nodes[0, int(node_idx)] = 1.0
    scope_edges = dense_edge_mask_from_pairs(
        transition_sample["event_scope_union_edges"],
        num_nodes,
        device,
    ).unsqueeze(0)

    return {
        "current_node_feats": current_node_feats,
        "current_adj": current_adj,
        "node_mask": node_mask,
        "next_node_feats": next_node_feats,
        "next_adj": next_adj,
        "changed_nodes": changed_nodes,
        "changed_edges": changed_edges,
        "scope_node_mask": scope_nodes,
        "scope_edge_mask": scope_edges,
    }


@torch.no_grad()
def get_proposal_predictions(
    proposal_model: ScopeProposalModel,
    node_feats: torch.Tensor,
    adj: torch.Tensor,
    node_mask: torch.Tensor,
    node_threshold: float,
    edge_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    proposal_outputs = proposal_model(node_feats=node_feats, adj=adj)
    valid_edge_mask = build_valid_edge_mask(node_mask).bool()
    node_scope_logits = proposal_outputs.get("node_scope_logits", proposal_outputs["scope_logits"])
    proposal_node_probs = torch.sigmoid(node_scope_logits) * node_mask

    if "edge_scope_logits" in proposal_outputs:
        proposal_edge_probs = torch.sigmoid(proposal_outputs["edge_scope_logits"]) * valid_edge_mask.float()
    else:
        proposal_edge_probs = (
            proposal_node_probs.unsqueeze(2) * proposal_node_probs.unsqueeze(1) * valid_edge_mask.float()
        )

    pred_scope_nodes = ((proposal_node_probs >= node_threshold) & node_mask.bool()).float()
    pred_scope_edges = ((proposal_edge_probs >= edge_threshold) & valid_edge_mask).float()
    return proposal_node_probs, proposal_edge_probs, pred_scope_nodes, pred_scope_edges


def decode_predicted_next_graph(outputs: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    pred_type = outputs["type_logits_full"].argmax(dim=-1)
    pred_state = outputs["state_pred_full"]
    pred_adj = (torch.sigmoid(outputs["edge_logits_full"]) >= 0.5).float()
    pred_adj = ((pred_adj + pred_adj.transpose(1, 2)) > 0.5).float()
    diag_mask = torch.eye(pred_adj.shape[1], device=pred_adj.device, dtype=torch.bool).unsqueeze(0)
    pred_adj = pred_adj.masked_fill(diag_mask, 0.0)
    pred_node_feats = torch.cat([pred_type.float().unsqueeze(-1), pred_state], dim=-1)
    return pred_node_feats, pred_adj


def compute_rewrite_loss_for_transition(
    model: OracleLocalDeltaRewriteModel,
    proposal_model: ScopeProposalModel,
    current_node_feats: torch.Tensor,
    current_adj: torch.Tensor,
    next_node_feats: torch.Tensor,
    next_adj: torch.Tensor,
    node_mask: torch.Tensor,
    scope_node_mask: torch.Tensor,
    scope_edge_mask: torch.Tensor,
    node_threshold: float,
    edge_threshold: float,
    edge_loss_weight: float,
    type_loss_weight: float,
    state_loss_weight: float,
    type_flip_weight: float,
    delta_keep_weight: float,
    delta_add_weight: float,
    delta_delete_weight: float,
    fp_edge_keep_loss_weight: float,
) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    proposal_node_probs, proposal_edge_probs, pred_scope_nodes, pred_scope_edges = get_proposal_predictions(
        proposal_model=proposal_model,
        node_feats=current_node_feats,
        adj=current_adj,
        node_mask=node_mask,
        node_threshold=node_threshold,
        edge_threshold=edge_threshold,
    )

    outputs = model(
        node_feats=current_node_feats,
        adj=current_adj,
        scope_node_mask=scope_node_mask,
        scope_edge_mask=scope_edge_mask,
        proposal_node_probs=proposal_node_probs,
        proposal_edge_probs=proposal_edge_probs,
    )

    local_loss_dict = oracle_local_delta_rewrite_loss(
        outputs=outputs,
        current_node_feats=current_node_feats,
        current_adj=current_adj,
        target_node_feats=next_node_feats,
        target_adj=next_adj,
        node_mask=node_mask,
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

    valid_edge_mask = build_valid_edge_mask(node_mask)
    oracle_edge_scope_mask = scope_edge_mask * valid_edge_mask
    predicted_only_edge_mask = pred_scope_edges * valid_edge_mask * (1.0 - oracle_edge_scope_mask)
    fp_edge_keep_loss = masked_edge_keep_regularization_loss_from_pair_mask(
        outputs["edge_delta_logits_local"],
        predicted_only_edge_mask,
    )

    total_loss = local_loss_dict["total_loss"] + fp_edge_keep_loss_weight * fp_edge_keep_loss
    loss_dict = {
        "main_total_loss": local_loss_dict["total_loss"],
        "fp_edge_keep_loss": fp_edge_keep_loss,
        "total_loss": total_loss,
    }
    return outputs, loss_dict


def init_train_accumulator() -> Dict[str, float]:
    return {
        "step1_total_loss_sum": 0.0,
        "step1_main_loss_sum": 0.0,
        "step1_fp_keep_loss_sum": 0.0,
        "step2_total_loss_sum": 0.0,
        "step2_main_loss_sum": 0.0,
        "step2_fp_keep_loss_sum": 0.0,
        "total_loss_sum": 0.0,
        "num_batches": 0.0,
        "num_samples": 0.0,
    }


def finalize_train_accumulator(acc: Dict[str, float]) -> Dict[str, float]:
    num_batches = max(acc["num_batches"], 1.0)
    return {
        "step1_total_loss": acc["step1_total_loss_sum"] / num_batches,
        "step1_main_loss": acc["step1_main_loss_sum"] / num_batches,
        "step1_fp_keep_loss": acc["step1_fp_keep_loss_sum"] / num_batches,
        "step2_total_loss": acc["step2_total_loss_sum"] / num_batches,
        "step2_main_loss": acc["step2_main_loss_sum"] / num_batches,
        "step2_fp_keep_loss": acc["step2_fp_keep_loss_sum"] / num_batches,
        "total_loss": acc["total_loss_sum"] / num_batches,
        "num_samples": acc["num_samples"],
    }


def run_rollout_epoch(
    model: OracleLocalDeltaRewriteModel,
    proposal_model: ScopeProposalModel,
    loader: DataLoader,
    device: torch.device,
    node_threshold: float,
    edge_threshold: float,
    edge_loss_weight: float,
    type_loss_weight: float,
    state_loss_weight: float,
    type_flip_weight: float,
    delta_keep_weight: float,
    delta_add_weight: float,
    delta_delete_weight: float,
    fp_edge_keep_loss_weight: float,
    rollout_loss_weight: float,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_clip: Optional[float] = None,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    proposal_model.eval()

    acc = init_train_accumulator()
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch_samples in loader:
            batch_total_loss = None
            batch_size = len(batch_samples)

            for raw_sample in batch_samples:
                transition_samples = raw_sample["transition_samples"]
                step1 = transition_targets(transition_samples[0], device)
                step2 = transition_targets(transition_samples[1], device)

                step1_outputs, step1_loss = compute_rewrite_loss_for_transition(
                    model=model,
                    proposal_model=proposal_model,
                    current_node_feats=step1["current_node_feats"],
                    current_adj=step1["current_adj"],
                    next_node_feats=step1["next_node_feats"],
                    next_adj=step1["next_adj"],
                    node_mask=step1["node_mask"],
                    scope_node_mask=step1["scope_node_mask"],
                    scope_edge_mask=step1["scope_edge_mask"],
                    node_threshold=node_threshold,
                    edge_threshold=edge_threshold,
                    edge_loss_weight=edge_loss_weight,
                    type_loss_weight=type_loss_weight,
                    state_loss_weight=state_loss_weight,
                    type_flip_weight=type_flip_weight,
                    delta_keep_weight=delta_keep_weight,
                    delta_add_weight=delta_add_weight,
                    delta_delete_weight=delta_delete_weight,
                    fp_edge_keep_loss_weight=fp_edge_keep_loss_weight,
                )

                # Autoregressive conversion matches the rollout evaluator exactly:
                # argmax over full type logits, direct full state prediction, and
                # thresholded full edge logits with undirected symmetrization.
                pred_step1_node_feats, pred_step1_adj = decode_predicted_next_graph(step1_outputs)
                pred_step1_node_feats = pred_step1_node_feats.detach() if not is_train else pred_step1_node_feats
                pred_step1_adj = pred_step1_adj.detach() if not is_train else pred_step1_adj

                _, step2_loss = compute_rewrite_loss_for_transition(
                    model=model,
                    proposal_model=proposal_model,
                    current_node_feats=pred_step1_node_feats,
                    current_adj=pred_step1_adj,
                    next_node_feats=step2["next_node_feats"],
                    next_adj=step2["next_adj"],
                    node_mask=step2["node_mask"],
                    scope_node_mask=step2["scope_node_mask"],
                    scope_edge_mask=step2["scope_edge_mask"],
                    node_threshold=node_threshold,
                    edge_threshold=edge_threshold,
                    edge_loss_weight=edge_loss_weight,
                    type_loss_weight=type_loss_weight,
                    state_loss_weight=state_loss_weight,
                    type_flip_weight=type_flip_weight,
                    delta_keep_weight=delta_keep_weight,
                    delta_add_weight=delta_add_weight,
                    delta_delete_weight=delta_delete_weight,
                    fp_edge_keep_loss_weight=fp_edge_keep_loss_weight,
                )

                sample_total_loss = step1_loss["total_loss"] + rollout_loss_weight * step2_loss["total_loss"]
                batch_total_loss = sample_total_loss if batch_total_loss is None else (batch_total_loss + sample_total_loss)

                acc["step1_total_loss_sum"] += step1_loss["total_loss"].item()
                acc["step1_main_loss_sum"] += step1_loss["main_total_loss"].item()
                acc["step1_fp_keep_loss_sum"] += step1_loss["fp_edge_keep_loss"].item()
                acc["step2_total_loss_sum"] += step2_loss["total_loss"].item()
                acc["step2_main_loss_sum"] += step2_loss["main_total_loss"].item()
                acc["step2_fp_keep_loss_sum"] += step2_loss["fp_edge_keep_loss"].item()
                acc["total_loss_sum"] += sample_total_loss.item()
                acc["num_samples"] += 1.0

            batch_total_loss = batch_total_loss / max(batch_size, 1)
            if is_train:
                optimizer.zero_grad()
                batch_total_loss.backward()
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            acc["num_batches"] += 1.0

    return finalize_train_accumulator(acc)


def compute_selection_score(val_rollout_results: Dict[str, Any]) -> float:
    overall = val_rollout_results["overall_final"]
    metrics = [
        overall.get("full_edge_acc"),
        overall.get("changed_edge_acc"),
        overall.get("context_edge_acc"),
        overall.get("add"),
        overall.get("delete"),
    ]
    vals = [float(v) for v in metrics if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--proposal_checkpoint", type=str, required=True)
    parser.add_argument("--init_rewrite_checkpoint", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--node_threshold", type=float, default=0.20)
    parser.add_argument("--edge_threshold", type=float, default=0.15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--edge_loss_weight", type=float, default=1.0)
    parser.add_argument("--type_loss_weight", type=float, default=1.0)
    parser.add_argument("--state_loss_weight", type=float, default=1.0)
    parser.add_argument("--type_flip_weight", type=float, default=1.0)
    parser.add_argument("--delta_keep_weight", type=float, default=1.10)
    parser.add_argument("--delta_add_weight", type=float, default=1.0)
    parser.add_argument("--delta_delete_weight", type=float, default=3.0)
    parser.add_argument("--fp_edge_keep_loss_weight", type=float, default=0.12)
    parser.add_argument("--rollout_loss_weight", type=float, required=True)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    save_dir = resolve_path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = save_dir / "best.pt"
    last_ckpt_path = save_dir / "last.pt"
    best_metrics_path = save_dir / "best_metrics.json"

    train_path = resolve_path(args.train_path)
    val_path = resolve_path(args.val_path)
    proposal_checkpoint_path = resolve_path(args.proposal_checkpoint)
    init_rewrite_checkpoint_path = resolve_path(args.init_rewrite_checkpoint)

    train_dataset, val_dataset, train_loader, _ = build_rollout_dataloaders(
        train_path=str(train_path),
        val_path=str(val_path),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    val_samples = val_dataset.samples

    proposal_model = load_frozen_proposal(proposal_checkpoint_path, device)
    model, init_checkpoint = load_rewrite_from_init(init_rewrite_checkpoint_path, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_score = float("-inf")
    best_epoch = -1
    epochs_without_improve = 0
    proposal_conditioning_enabled = bool(
        init_checkpoint.get("model_config", {}).get("use_proposal_conditioning", False)
    )

    print(f"device: {device}")
    print(f"train samples: {len(train_dataset)}")
    print(f"val samples: {len(val_dataset)}")
    print(f"proposal checkpoint: {proposal_checkpoint_path}")
    print(f"init rewrite checkpoint: {init_rewrite_checkpoint_path}")
    print(f"save dir: {save_dir}")
    print(f"proposal conditioning enabled: {proposal_conditioning_enabled}")
    print(
        "autoregressive graph-state conversion: argmax(type_logits_full) + "
        "state_pred_full + thresholded/symmetrized edge_logits_full"
    )

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_rollout_epoch(
            model=model,
            proposal_model=proposal_model,
            loader=train_loader,
            device=device,
            node_threshold=args.node_threshold,
            edge_threshold=args.edge_threshold,
            edge_loss_weight=args.edge_loss_weight,
            type_loss_weight=args.type_loss_weight,
            state_loss_weight=args.state_loss_weight,
            type_flip_weight=args.type_flip_weight,
            delta_keep_weight=args.delta_keep_weight,
            delta_add_weight=args.delta_add_weight,
            delta_delete_weight=args.delta_delete_weight,
            fp_edge_keep_loss_weight=args.fp_edge_keep_loss_weight,
            rollout_loss_weight=args.rollout_loss_weight,
            optimizer=optimizer,
            grad_clip=args.grad_clip,
        )

        val_results = evaluate_rollouts(
            proposal_model=proposal_model,
            rewrite_model=model,
            samples=val_samples,
            device=device,
            node_threshold=args.node_threshold,
            edge_threshold=args.edge_threshold,
            use_proposal_conditioning=proposal_conditioning_enabled,
        )
        val_score = compute_selection_score(val_results)

        print(
            f"[epoch {epoch:02d}] "
            f"train_total_loss={train_metrics['total_loss']:.6f} "
            f"train_step1={train_metrics['step1_total_loss']:.6f} "
            f"train_step2={train_metrics['step2_total_loss']:.6f} "
            f"val_full_edge_acc={val_results['overall_final']['full_edge_acc']:.6f} "
            f"val_changed_edge_acc={val_results['overall_final']['changed_edge_acc']:.6f} "
            f"val_context_edge_acc={val_results['overall_final']['context_edge_acc']:.6f} "
            f"val_add={val_results['overall_final']['add']:.6f} "
            f"val_delete={val_results['overall_final']['delete']:.6f} "
            f"val_selection_score={val_score:.6f}"
        )

        checkpoint_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": init_checkpoint["model_config"],
            "train_metrics": train_metrics,
            "val_rollout_metrics": val_results,
            "val_selection_score": val_score,
            "proposal_checkpoint": str(proposal_checkpoint_path),
            "init_rewrite_checkpoint": str(init_rewrite_checkpoint_path),
            "rollout_loss_weight": args.rollout_loss_weight,
            "fp_edge_keep_loss_weight": args.fp_edge_keep_loss_weight,
            "selection_metric": "val_rollout_balanced_full_changed_context_add_delete",
            "state_conversion": {
                "node_type": "argmax(type_logits_full)",
                "node_state": "state_pred_full",
                "edge_adj": "sigmoid(edge_logits_full)>=0.5 with undirected symmetrization",
            },
        }
        torch.save(checkpoint_payload, last_ckpt_path)

        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch
            epochs_without_improve = 0
            torch.save(checkpoint_payload, best_ckpt_path)
            save_json(
                best_metrics_path,
                {
                    "best_epoch": best_epoch,
                    "best_validation_selection_score": best_score,
                    "selection_metric": "val_rollout_balanced_full_changed_context_add_delete",
                    "rollout_loss_weight": args.rollout_loss_weight,
                    "proposal_conditioning_enabled": proposal_conditioning_enabled,
                    "warm_start_used": True,
                    "train_metrics": train_metrics,
                    "val_rollout_metrics": val_results,
                },
            )
            print(f"  new best checkpoint saved at epoch {epoch}")
        else:
            epochs_without_improve += 1
            print(
                f"  no improvement. patience {epochs_without_improve}/{args.patience} "
                f"(best epoch {best_epoch}, best score {best_score:.6f})"
            )

        if epochs_without_improve >= args.patience:
            print("early stopping triggered")
            break

    print(f"training finished. best epoch={best_epoch} best validation selection score={best_score:.6f}")
    print(f"best checkpoint: {best_ckpt_path}")
    print(f"last checkpoint: {last_ckpt_path}")
    print(f"best metrics json: {best_metrics_path}")


if __name__ == "__main__":
    main()
