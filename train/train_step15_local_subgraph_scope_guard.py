from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.collate import graph_event_collate_fn
from data.dataset import GraphEventDataset
from models.baselines import build_mlp
from train.eval_noisy_structured_observation import move_batch_to_device, require_keys, resolve_path
from train.eval_step9_gated_edge_completion import (
    get_base_proposal_outputs,
    load_completion_model,
    load_proposal_model,
)
from train.eval_step9_rescue_frontier import average_precision, auroc


FIXED_RESCUE_BUDGET_FRACTION = 0.10


@dataclass
class LocalSubgraphScopeGuardConfig:
    proposal_hidden_dim: int
    hidden_dim: int = 128
    head_layers: int = 2
    dropout: float = 0.0


class LocalSubgraphScopeGuard(nn.Module):
    """
    Candidate-centric local-subgraph event-scope guard.

    This is the smallest Step 15 representation jump: the proposal backbone and
    Step 9 completion head stay frozen, while this guard pools local structural
    context around each internal candidate edge. 中文说明：这里仍然只判断
    internal rescue candidate 是否属于 event scope，不改节点 proposal、不改 rewrite。
    """

    def __init__(self, config: LocalSubgraphScopeGuardConfig):
        super().__init__()
        self.config = config
        h = config.proposal_hidden_dim
        # endpoint latents, endpoint-neighborhood latents, common-neighbor
        # pooled latents, and a compact scalar local-structure bundle.
        input_dim = (9 * h) + 18
        self.head = build_mlp(
            in_dim=input_dim,
            hidden_dim=config.hidden_dim,
            out_dim=1,
            num_layers=config.head_layers,
            dropout=config.dropout,
        )

    @staticmethod
    def _norm_logits(logits: torch.Tensor) -> torch.Tensor:
        return logits.clamp(min=-10.0, max=10.0) / 10.0

    @staticmethod
    def _pair_expand(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, num_nodes = values.shape[:2]
        trailing = values.shape[2:]
        left = values.unsqueeze(2).expand(bsz, num_nodes, num_nodes, *trailing)
        right = values.unsqueeze(1).expand(bsz, num_nodes, num_nodes, *trailing)
        return left, right

    def forward(
        self,
        node_latents: torch.Tensor,
        input_adj: torch.Tensor,
        pred_scope_nodes: torch.Tensor,
        valid_edge_mask: torch.Tensor,
        node_scope_logits: torch.Tensor,
        proposal_node_probs: torch.Tensor,
        base_edge_logits: torch.Tensor,
        proposal_edge_probs: torch.Tensor,
        completion_logits: torch.Tensor,
    ) -> torch.Tensor:
        bsz, num_nodes, hidden_dim = node_latents.shape
        h_i, h_j = self._pair_expand(node_latents)

        scoped_nodes = pred_scope_nodes.bool()
        scoped_float = scoped_nodes.float()
        scope_count = scoped_float.sum(dim=1).clamp_min(1.0)
        adj_bool = (input_adj > 0.5) & valid_edge_mask
        neighbor_mask = (adj_bool & scoped_nodes.unsqueeze(1)).float()
        neighbor_count = neighbor_mask.sum(dim=-1).clamp_min(1.0)
        neighbor_mean = torch.bmm(neighbor_mask, node_latents) / neighbor_count.unsqueeze(-1)
        neigh_i, neigh_j = self._pair_expand(neighbor_mean)

        # Common-neighbor pooling is the small local-subgraph piece that the
        # scalar Step 14 probe could not represent directly.
        common_mask = neighbor_mask.unsqueeze(2) * neighbor_mask.unsqueeze(1)
        common_count_raw = common_mask.sum(dim=-1)
        common_count = common_count_raw.clamp_min(1.0)
        common_mean = torch.einsum("bijn,bnh->bijh", common_mask, node_latents) / common_count.unsqueeze(-1)

        deg = neighbor_mask.sum(dim=-1) / scope_count.unsqueeze(-1)
        deg_i, deg_j = self._pair_expand(deg.unsqueeze(-1))
        node_log_i, node_log_j = self._pair_expand(self._norm_logits(node_scope_logits).unsqueeze(-1))
        node_prob_i, node_prob_j = self._pair_expand(proposal_node_probs.unsqueeze(-1))
        node_prob_min = torch.minimum(node_prob_i, node_prob_j)
        node_prob_max = torch.maximum(node_prob_i, node_prob_j)
        node_prob_prod = node_prob_i * node_prob_j

        induced_mask = scoped_nodes.unsqueeze(2) & scoped_nodes.unsqueeze(1) & valid_edge_mask
        induced_edges = (adj_bool.float() * induced_mask.float()).sum(dim=(1, 2))
        induced_possible = induced_mask.float().sum(dim=(1, 2)).clamp_min(1.0)
        scope_density = (induced_edges / induced_possible).view(bsz, 1, 1, 1).expand(bsz, num_nodes, num_nodes, 1)

        common_norm = (common_count_raw / scope_count.view(bsz, 1, 1)).unsqueeze(-1)
        scalar_features = [
            input_adj.unsqueeze(-1),
            self._norm_logits(base_edge_logits).unsqueeze(-1),
            proposal_edge_probs.unsqueeze(-1),
            self._norm_logits(completion_logits).unsqueeze(-1),
            torch.sigmoid(completion_logits).unsqueeze(-1),
            node_log_i,
            node_log_j,
            node_prob_i,
            node_prob_j,
            node_prob_min,
            node_prob_max,
            node_prob_prod,
            deg_i,
            deg_j,
            torch.minimum(deg_i, deg_j),
            torch.maximum(deg_i, deg_j),
            common_norm,
            scope_density,
        ]
        pair_feat = torch.cat(
            [
                h_i,
                h_j,
                torch.abs(h_i - h_j),
                h_i * h_j,
                neigh_i,
                neigh_j,
                torch.abs(neigh_i - neigh_j),
                neigh_i * neigh_j,
                common_mean,
                *scalar_features,
            ],
            dim=-1,
        )
        logits = self.head(pair_feat).squeeze(-1)
        logits = 0.5 * (logits + logits.transpose(1, 2))
        diag_mask = torch.eye(num_nodes, device=node_latents.device, dtype=torch.bool).unsqueeze(0)
        return logits.masked_fill(diag_mask, -1e9)


def save_local_guard_checkpoint(
    path: Path,
    model: LocalSubgraphScopeGuard,
    args: argparse.Namespace,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "model_config": asdict(model.config),
        "model_state_dict": model.state_dict(),
        "args": vars(args),
    }
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_local_guard_model(checkpoint_path: Path, device: torch.device) -> LocalSubgraphScopeGuard:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = LocalSubgraphScopeGuard(LocalSubgraphScopeGuardConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


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


def build_dataloaders(
    train_path: str,
    val_path: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple[GraphEventDataset, GraphEventDataset, DataLoader, DataLoader]:
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


def init_sums() -> Dict[str, float]:
    return {
        "loss_sum": 0.0,
        "loss_batches": 0.0,
        "candidate_total": 0.0,
        "event_scope_positive_total": 0.0,
        "changed_positive_total": 0.0,
        "budget_selected_total": 0.0,
        "budget_event_scope_tp": 0.0,
        "budget_changed_tp": 0.0,
        "budget_event_scope_positive_total": 0.0,
        "budget_changed_positive_total": 0.0,
    }


def update_budget_sums(
    sums: Dict[str, float],
    candidates: torch.Tensor,
    ranking_scores: torch.Tensor,
    target_scope: torch.Tensor,
    changed_edges: torch.Tensor,
) -> None:
    candidates = candidates.bool()
    for batch_idx in range(candidates.shape[0]):
        candidate_indices = candidates[batch_idx].nonzero(as_tuple=False)
        num_candidates = candidate_indices.shape[0]
        candidate_target = (target_scope[batch_idx] > 0.5) & candidates[batch_idx]
        candidate_changed = (changed_edges[batch_idx] > 0.5) & candidates[batch_idx]
        sums["budget_event_scope_positive_total"] += candidate_target.float().sum().item()
        sums["budget_changed_positive_total"] += candidate_changed.float().sum().item()
        if num_candidates <= 0:
            continue
        budget = int(num_candidates * FIXED_RESCUE_BUDGET_FRACTION)
        if budget <= 0:
            continue
        scores = ranking_scores[batch_idx][candidates[batch_idx]]
        topk = torch.topk(scores, k=min(budget, num_candidates), largest=True).indices
        chosen = candidate_indices[topk]
        selected = torch.zeros_like(candidates[batch_idx])
        selected[chosen[:, 0], chosen[:, 1]] = True
        sums["budget_selected_total"] += selected.float().sum().item()
        sums["budget_event_scope_tp"] += (selected & candidate_target).float().sum().item()
        sums["budget_changed_tp"] += (selected & candidate_changed).float().sum().item()


def finalize_metrics(
    sums: Dict[str, float],
    scores: list[torch.Tensor],
    event_labels: list[torch.Tensor],
    changed_labels: list[torch.Tensor],
) -> Dict[str, Any]:
    event_precision_at_budget = None
    changed_precision_at_budget = None
    if sums["budget_selected_total"] > 0:
        event_precision_at_budget = sums["budget_event_scope_tp"] / sums["budget_selected_total"]
        changed_precision_at_budget = sums["budget_changed_tp"] / sums["budget_selected_total"]
    event_recall_at_budget = None
    changed_recall_at_budget = None
    if sums["budget_event_scope_positive_total"] > 0:
        event_recall_at_budget = sums["budget_event_scope_tp"] / sums["budget_event_scope_positive_total"]
    if sums["budget_changed_positive_total"] > 0:
        changed_recall_at_budget = sums["budget_changed_tp"] / sums["budget_changed_positive_total"]

    event_ap = None
    event_auc = None
    changed_ap = None
    changed_auc = None
    if scores:
        score_tensor = torch.cat(scores).float()
        event_tensor = torch.cat(event_labels).bool()
        changed_tensor = torch.cat(changed_labels).bool()
        event_ap = average_precision(score_tensor, event_tensor)
        event_auc = auroc(score_tensor, event_tensor)
        changed_ap = average_precision(score_tensor, changed_tensor)
        changed_auc = auroc(score_tensor, changed_tensor)

    return {
        "loss": sums["loss_sum"] / max(sums["loss_batches"], 1.0),
        "candidate_total": int(sums["candidate_total"]),
        "event_scope_positive_total": int(sums["event_scope_positive_total"]),
        "changed_positive_total": int(sums["changed_positive_total"]),
        "event_scope_precision_at_budget": event_precision_at_budget,
        "changed_edge_precision_at_budget": changed_precision_at_budget,
        "event_scope_recall_at_budget": event_recall_at_budget,
        "changed_edge_recall_at_budget": changed_recall_at_budget,
        "event_scope_ap": event_ap,
        "event_scope_auroc": event_auc,
        "changed_edge_ap": changed_ap,
        "changed_edge_auroc": changed_auc,
        "selection_score": event_precision_at_budget,
    }


def guard_bce_loss(
    guard_logits: torch.Tensor,
    target_event_scope: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    if candidate_mask.float().sum().item() <= 0:
        return guard_logits.sum() * 0.0
    bce = F.binary_cross_entropy_with_logits(
        guard_logits,
        target_event_scope.float(),
        reduction="none",
    )
    bce = bce * candidate_mask.float()
    return bce.sum() / candidate_mask.float().sum().clamp_min(1.0)


def run_epoch(
    proposal_model: torch.nn.Module,
    completion_model: torch.nn.Module,
    guard_model: LocalSubgraphScopeGuard,
    loader: DataLoader,
    device: torch.device,
    node_threshold: float,
    edge_threshold: float,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_clip: Optional[float] = None,
) -> Dict[str, Any]:
    is_train = optimizer is not None
    proposal_model.eval()
    completion_model.eval()
    guard_model.train() if is_train else guard_model.eval()
    sums = init_sums()
    all_scores: list[torch.Tensor] = []
    all_event_labels: list[torch.Tensor] = []
    all_changed_labels: list[torch.Tensor] = []

    for batch in loader:
        require_keys(batch, ["node_feats", "adj", "node_mask", "changed_edges", "event_scope_union_edges"])
        batch = move_batch_to_device(batch, device)
        with torch.no_grad():
            base_outputs = get_base_proposal_outputs(
                proposal_model=proposal_model,
                batch=batch,
                node_threshold=node_threshold,
                edge_threshold=edge_threshold,
            )
            completion_logits = completion_model(
                node_latents=base_outputs["node_latents"],
                base_edge_logits=base_outputs["edge_scope_logits"],
            )

        guard_logits = guard_model(
            node_latents=base_outputs["node_latents"].detach(),
            input_adj=base_outputs["input_adj"],
            pred_scope_nodes=base_outputs["pred_scope_nodes"],
            valid_edge_mask=base_outputs["valid_edge_mask"],
            node_scope_logits=base_outputs["node_scope_logits"].detach(),
            proposal_node_probs=base_outputs["proposal_node_probs"].detach(),
            base_edge_logits=base_outputs["edge_scope_logits"].detach(),
            proposal_edge_probs=base_outputs["proposal_edge_probs"].detach(),
            completion_logits=completion_logits.detach(),
        )
        candidates = base_outputs["rescue_candidates"].bool()
        target = batch["event_scope_union_edges"]
        loss = guard_bce_loss(guard_logits, target, candidates)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(guard_model.parameters(), grad_clip)
            optimizer.step()

        with torch.no_grad():
            probs = torch.sigmoid(guard_logits) * base_outputs["valid_edge_mask"].float()
            target_bool = (target > 0.5) & candidates
            changed_bool = (batch["changed_edges"] > 0.5) & candidates
            sums["loss_sum"] += loss.item()
            sums["loss_batches"] += 1.0
            sums["candidate_total"] += candidates.float().sum().item()
            sums["event_scope_positive_total"] += target_bool.float().sum().item()
            sums["changed_positive_total"] += changed_bool.float().sum().item()
            update_budget_sums(
                sums=sums,
                candidates=candidates,
                ranking_scores=probs,
                target_scope=target,
                changed_edges=batch["changed_edges"],
            )
            if candidates.float().sum().item() > 0:
                all_scores.append(probs[candidates].detach().float().cpu())
                all_event_labels.append(target_bool[candidates].detach().bool().cpu())
                all_changed_labels.append(changed_bool[candidates].detach().bool().cpu())

    return finalize_metrics(sums, all_scores, all_event_labels, all_changed_labels)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--proposal_checkpoint_path", type=str, required=True)
    parser.add_argument("--completion_checkpoint_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--node_threshold", type=float, default=0.20)
    parser.add_argument("--edge_threshold", type=float, default=0.15)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--head_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    proposal_checkpoint_path = resolve_path(args.proposal_checkpoint_path)
    completion_checkpoint_path = resolve_path(args.completion_checkpoint_path)
    save_dir = resolve_path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset, train_loader, val_loader = build_dataloaders(
        train_path=str(resolve_path(args.train_path)),
        val_path=str(resolve_path(args.val_path)),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    proposal_model = load_proposal_model(proposal_checkpoint_path, device)
    completion_model = load_completion_model(completion_checkpoint_path, device)
    proposal_model.requires_grad_(False)
    completion_model.requires_grad_(False)

    guard_config = LocalSubgraphScopeGuardConfig(
        proposal_hidden_dim=proposal_model.config.hidden_dim,
        hidden_dim=args.hidden_dim,
        head_layers=args.head_layers,
        dropout=args.dropout,
    )
    guard_model = LocalSubgraphScopeGuard(guard_config).to(device)
    optimizer = torch.optim.Adam(guard_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_score = float("-inf")
    best_epoch = -1
    epochs_without_improve = 0
    history: list[Dict[str, Any]] = []
    best_path = save_dir / "best.pt"
    last_path = save_dir / "last.pt"
    summary_path = save_dir / "training_summary.json"

    print(f"device: {device}")
    print(f"train size: {len(train_dataset)}")
    print(f"val size: {len(val_dataset)}")
    print("guard_representation: learned_local_subgraph_guard")
    print("frozen proposal + frozen completion; training only local-subgraph guard")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            proposal_model=proposal_model,
            completion_model=completion_model,
            guard_model=guard_model,
            loader=train_loader,
            device=device,
            node_threshold=args.node_threshold,
            edge_threshold=args.edge_threshold,
            optimizer=optimizer,
            grad_clip=args.grad_clip,
        )
        val_metrics = run_epoch(
            proposal_model=proposal_model,
            completion_model=completion_model,
            guard_model=guard_model,
            loader=val_loader,
            device=device,
            node_threshold=args.node_threshold,
            edge_threshold=args.edge_threshold,
            optimizer=None,
            grad_clip=None,
        )
        val_score = val_metrics.get("selection_score")
        if val_score is None:
            val_score = -float(val_metrics.get("loss") or 0.0)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics, "selection_score": val_score})
        print(
            f"epoch {epoch:03d} | "
            f"train_loss={train_metrics.get('loss'):.6f} | "
            f"val_loss={val_metrics.get('loss'):.6f} | "
            f"val_precision@B={val_metrics.get('event_scope_precision_at_budget')} | "
            f"val_recall@B={val_metrics.get('event_scope_recall_at_budget')} | "
            f"val_ap={val_metrics.get('event_scope_ap')} | "
            f"score={val_score}"
        )

        if val_score > best_score:
            best_score = float(val_score)
            best_epoch = epoch
            epochs_without_improve = 0
            save_local_guard_checkpoint(
                best_path,
                guard_model,
                args,
                extra={
                    "best_epoch": best_epoch,
                    "best_validation_selection_score": best_score,
                    "best_val_metrics": val_metrics,
                    "proposal_checkpoint_path": str(proposal_checkpoint_path),
                    "completion_checkpoint_path": str(completion_checkpoint_path),
                    "rescue_budget_fraction": FIXED_RESCUE_BUDGET_FRACTION,
                    "guard_representation": "learned_local_subgraph_guard",
                },
            )
        else:
            epochs_without_improve += 1

        save_local_guard_checkpoint(
            last_path,
            guard_model,
            args,
            extra={
                "epoch": epoch,
                "validation_selection_score": val_score,
                "val_metrics": val_metrics,
                "proposal_checkpoint_path": str(proposal_checkpoint_path),
                "completion_checkpoint_path": str(completion_checkpoint_path),
                "rescue_budget_fraction": FIXED_RESCUE_BUDGET_FRACTION,
                "guard_representation": "learned_local_subgraph_guard",
            },
        )

        if epochs_without_improve >= args.patience:
            print(f"early stopping after {epoch} epochs")
            break

    summary = {
        "best_epoch": best_epoch,
        "best_validation_selection_score": best_score,
        "history": history,
        "args": vars(args),
        "model_config": asdict(guard_config),
        "proposal_checkpoint_path": str(proposal_checkpoint_path),
        "completion_checkpoint_path": str(completion_checkpoint_path),
        "rescue_budget_fraction": FIXED_RESCUE_BUDGET_FRACTION,
        "guard_representation": "learned_local_subgraph_guard",
    }
    save_json(summary_path, summary)
    print(f"best epoch: {best_epoch}")
    print(f"best validation selection score: {best_score}")
    print(f"saved best checkpoint: {best_path}")
    print(f"saved summary: {summary_path}")


if __name__ == "__main__":
    main()

