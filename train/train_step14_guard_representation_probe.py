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
from train.eval_step11_guarded_internal_completion import load_guard_model
from train.eval_step9_gated_edge_completion import (
    get_base_proposal_outputs,
    load_completion_model,
    load_proposal_model,
)
from train.eval_step9_rescue_frontier import average_precision, auroc


SCORES_ONLY = "scores_only"
ENRICHED_LOCAL_CONTEXT = "enriched_local_context"
FEATURE_BUNDLES = [SCORES_ONLY, ENRICHED_LOCAL_CONTEXT]
FIXED_RESCUE_BUDGET_FRACTION = 0.10


@dataclass
class CandidateScopeProbeConfig:
    feature_bundle: str
    input_dim: int
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.0


class CandidateScopeProbe(nn.Module):
    """
    Candidate-level event-scope probe.

    The class is intentionally fixed across feature bundles; only the input
    features differ. 中文说明：这里是 probe，不是新的主模型线。
    """

    def __init__(self, config: CandidateScopeProbeConfig):
        super().__init__()
        self.config = config
        self.net = build_mlp(
            in_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            out_dim=1,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def save_probe_checkpoint(
    path: Path,
    model: CandidateScopeProbe,
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


def load_probe_model(checkpoint_path: Path, device: torch.device) -> CandidateScopeProbe:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = CandidateScopeProbe(CandidateScopeProbeConfig(**checkpoint["model_config"])).to(device)
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


def pair_expand_node_values(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, num_nodes = values.shape[:2]
    trailing = values.shape[2:]
    left = values.unsqueeze(2).expand(bsz, num_nodes, num_nodes, *trailing)
    right = values.unsqueeze(1).expand(bsz, num_nodes, num_nodes, *trailing)
    return left, right


def normalize_logits(logits: torch.Tensor) -> torch.Tensor:
    return logits.clamp(min=-10.0, max=10.0) / 10.0


def build_candidate_feature_tensor(
    batch: Dict[str, Any],
    base_outputs: Dict[str, torch.Tensor],
    completion_logits: torch.Tensor,
    reference_guard_logits: torch.Tensor,
    feature_bundle: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if feature_bundle not in FEATURE_BUNDLES:
        raise ValueError(f"Unknown feature bundle: {feature_bundle}")
    candidates = base_outputs["rescue_candidates"].bool()
    valid_edge_mask = base_outputs["valid_edge_mask"].bool()
    input_node_feats = base_outputs["input_node_feats"]
    input_adj = base_outputs["input_adj"]
    pred_scope_nodes = base_outputs["pred_scope_nodes"].bool()

    node_logits_i, node_logits_j = pair_expand_node_values(normalize_logits(base_outputs["node_scope_logits"]).unsqueeze(-1))
    node_probs_i, node_probs_j = pair_expand_node_values(base_outputs["proposal_node_probs"].unsqueeze(-1))
    node_pair_min = torch.minimum(node_probs_i, node_probs_j)
    node_pair_max = torch.maximum(node_probs_i, node_probs_j)
    node_pair_prod = node_probs_i * node_probs_j

    scalar_features = [
        normalize_logits(completion_logits).unsqueeze(-1),
        torch.sigmoid(completion_logits).unsqueeze(-1),
        normalize_logits(base_outputs["edge_scope_logits"]).unsqueeze(-1),
        base_outputs["proposal_edge_probs"].unsqueeze(-1),
        node_logits_i,
        node_logits_j,
        node_probs_i,
        node_probs_j,
        node_pair_min,
        node_pair_max,
        node_pair_prod,
        normalize_logits(reference_guard_logits).unsqueeze(-1),
        torch.sigmoid(reference_guard_logits).unsqueeze(-1),
    ]
    features = scalar_features

    if feature_bundle == ENRICHED_LOCAL_CONTEXT:
        node_feat_i, node_feat_j = pair_expand_node_values(input_node_feats)
        scope_float = pred_scope_nodes.float()
        scope_count = scope_float.sum(dim=1).clamp_min(1.0)
        scoped_adj = input_adj * scope_float.unsqueeze(1)
        deg = scoped_adj.sum(dim=-1) / scope_count.unsqueeze(-1)
        deg_i, deg_j = pair_expand_node_values(deg.unsqueeze(-1))
        neighbor_mask = ((input_adj > 0.5) & pred_scope_nodes.unsqueeze(1)).float()
        common = torch.bmm(neighbor_mask, neighbor_mask.transpose(1, 2)) / scope_count.view(-1, 1, 1)
        induced_mask = pred_scope_nodes.unsqueeze(2) & pred_scope_nodes.unsqueeze(1) & valid_edge_mask
        induced_edge_count = (input_adj > 0.5).float().masked_fill(~induced_mask, 0.0).sum(dim=(1, 2))
        induced_possible = induced_mask.float().sum(dim=(1, 2)).clamp_min(1.0)
        scope_density = (induced_edge_count / induced_possible).view(-1, 1, 1, 1).expand_as(common.unsqueeze(-1))
        features.extend(
            [
                input_adj.unsqueeze(-1),
                node_feat_i,
                node_feat_j,
                torch.abs(node_feat_i - node_feat_j),
                node_feat_i * node_feat_j,
                deg_i,
                deg_j,
                torch.minimum(deg_i, deg_j),
                torch.maximum(deg_i, deg_j),
                common.unsqueeze(-1),
                scope_density,
            ]
        )

    feature_tensor = torch.cat(features, dim=-1)
    candidate_features = feature_tensor[candidates].float()
    event_labels = ((batch["event_scope_union_edges"] > 0.5) & candidates)[candidates].float()
    changed_labels = ((batch["changed_edges"] > 0.5) & candidates)[candidates].float()
    return candidate_features, event_labels, changed_labels, candidates


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
    batch_size = candidates.shape[0]
    for batch_idx in range(batch_size):
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


def run_epoch(
    proposal_model: torch.nn.Module,
    completion_model: torch.nn.Module,
    reference_guard_model: torch.nn.Module,
    probe_model: CandidateScopeProbe,
    loader: DataLoader,
    device: torch.device,
    node_threshold: float,
    edge_threshold: float,
    feature_bundle: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_clip: Optional[float] = None,
) -> Dict[str, Any]:
    is_train = optimizer is not None
    proposal_model.eval()
    completion_model.eval()
    reference_guard_model.eval()
    probe_model.train() if is_train else probe_model.eval()
    sums = init_sums()
    all_scores: list[torch.Tensor] = []
    all_event_labels: list[torch.Tensor] = []
    all_changed_labels: list[torch.Tensor] = []

    for batch in loader:
        require_keys(
            batch,
            ["node_feats", "adj", "node_mask", "changed_edges", "event_scope_union_edges"],
        )
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
            reference_guard_logits = reference_guard_model(
                node_latents=base_outputs["node_latents"],
                base_edge_logits=base_outputs["edge_scope_logits"],
                completion_logits=completion_logits,
            )
            features, event_labels, changed_labels, candidates = build_candidate_feature_tensor(
                batch=batch,
                base_outputs=base_outputs,
                completion_logits=completion_logits,
                reference_guard_logits=reference_guard_logits,
                feature_bundle=feature_bundle,
            )
        if features.numel() <= 0:
            continue
        logits = probe_model(features)
        loss = F.binary_cross_entropy_with_logits(logits, event_labels)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(probe_model.parameters(), grad_clip)
            optimizer.step()

        with torch.no_grad():
            probs_flat = torch.sigmoid(logits)
            ranking_scores = torch.zeros_like(candidates, dtype=probs_flat.dtype)
            ranking_scores[candidates] = probs_flat
            sums["loss_sum"] += loss.item()
            sums["loss_batches"] += 1.0
            sums["candidate_total"] += float(event_labels.numel())
            sums["event_scope_positive_total"] += event_labels.sum().item()
            sums["changed_positive_total"] += changed_labels.sum().item()
            update_budget_sums(
                sums=sums,
                candidates=candidates,
                ranking_scores=ranking_scores,
                target_scope=batch["event_scope_union_edges"],
                changed_edges=batch["changed_edges"],
            )
            all_scores.append(probs_flat.detach().float().cpu())
            all_event_labels.append(event_labels.detach().bool().cpu())
            all_changed_labels.append(changed_labels.detach().bool().cpu())

    return finalize_metrics(sums, all_scores, all_event_labels, all_changed_labels)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


@torch.no_grad()
def infer_feature_dim(
    proposal_model: torch.nn.Module,
    completion_model: torch.nn.Module,
    reference_guard_model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    node_threshold: float,
    edge_threshold: float,
    feature_bundle: str,
) -> int:
    for batch in loader:
        batch = move_batch_to_device(batch, device)
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
        reference_guard_logits = reference_guard_model(
            node_latents=base_outputs["node_latents"],
            base_edge_logits=base_outputs["edge_scope_logits"],
            completion_logits=completion_logits,
        )
        features, _, _, _ = build_candidate_feature_tensor(
            batch=batch,
            base_outputs=base_outputs,
            completion_logits=completion_logits,
            reference_guard_logits=reference_guard_logits,
            feature_bundle=feature_bundle,
        )
        if features.numel() > 0:
            return int(features.shape[-1])
    raise RuntimeError("Could not infer feature dimension because no rescue candidates were found.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--proposal_checkpoint_path", type=str, required=True)
    parser.add_argument("--completion_checkpoint_path", type=str, required=True)
    parser.add_argument("--reference_guard_checkpoint_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--feature_bundle", type=str, choices=FEATURE_BUNDLES, required=True)
    parser.add_argument("--node_threshold", type=float, default=0.20)
    parser.add_argument("--edge_threshold", type=float, default=0.15)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
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
    reference_guard_checkpoint_path = resolve_path(args.reference_guard_checkpoint_path)
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
    reference_guard_model = load_guard_model(reference_guard_checkpoint_path, device)
    proposal_model.requires_grad_(False)
    completion_model.requires_grad_(False)
    reference_guard_model.requires_grad_(False)

    input_dim = infer_feature_dim(
        proposal_model=proposal_model,
        completion_model=completion_model,
        reference_guard_model=reference_guard_model,
        loader=train_loader,
        device=device,
        node_threshold=args.node_threshold,
        edge_threshold=args.edge_threshold,
        feature_bundle=args.feature_bundle,
    )
    probe_config = CandidateScopeProbeConfig(
        feature_bundle=args.feature_bundle,
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    probe_model = CandidateScopeProbe(probe_config).to(device)
    optimizer = torch.optim.Adam(probe_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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
    print(f"feature bundle: {args.feature_bundle}")
    print(f"feature dim: {input_dim}")
    print("frozen proposal + frozen completion + frozen reference guard; training candidate probe")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            proposal_model=proposal_model,
            completion_model=completion_model,
            reference_guard_model=reference_guard_model,
            probe_model=probe_model,
            loader=train_loader,
            device=device,
            node_threshold=args.node_threshold,
            edge_threshold=args.edge_threshold,
            feature_bundle=args.feature_bundle,
            optimizer=optimizer,
            grad_clip=args.grad_clip,
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                proposal_model=proposal_model,
                completion_model=completion_model,
                reference_guard_model=reference_guard_model,
                probe_model=probe_model,
                loader=val_loader,
                device=device,
                node_threshold=args.node_threshold,
                edge_threshold=args.edge_threshold,
                feature_bundle=args.feature_bundle,
                optimizer=None,
                grad_clip=None,
            )
        val_score = val_metrics.get("selection_score")
        if val_score is None:
            val_score = -float(val_metrics.get("loss") or 0.0)
        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "selection_score": val_score,
            }
        )
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
            save_probe_checkpoint(
                best_path,
                probe_model,
                args,
                extra={
                    "best_epoch": best_epoch,
                    "best_validation_selection_score": best_score,
                    "best_val_metrics": val_metrics,
                    "proposal_checkpoint_path": str(proposal_checkpoint_path),
                    "completion_checkpoint_path": str(completion_checkpoint_path),
                    "reference_guard_checkpoint_path": str(reference_guard_checkpoint_path),
                    "rescue_budget_fraction": FIXED_RESCUE_BUDGET_FRACTION,
                },
            )
        else:
            epochs_without_improve += 1

        save_probe_checkpoint(
            last_path,
            probe_model,
            args,
            extra={
                "epoch": epoch,
                "validation_selection_score": val_score,
                "val_metrics": val_metrics,
                "proposal_checkpoint_path": str(proposal_checkpoint_path),
                "completion_checkpoint_path": str(completion_checkpoint_path),
                "reference_guard_checkpoint_path": str(reference_guard_checkpoint_path),
                "rescue_budget_fraction": FIXED_RESCUE_BUDGET_FRACTION,
            },
        )
        save_json(
            summary_path,
            {
                "best_epoch": best_epoch,
                "best_validation_selection_score": best_score,
                "proposal_checkpoint_path": str(proposal_checkpoint_path),
                "completion_checkpoint_path": str(completion_checkpoint_path),
                "reference_guard_checkpoint_path": str(reference_guard_checkpoint_path),
                "rescue_budget_fraction": FIXED_RESCUE_BUDGET_FRACTION,
                "args": vars(args),
                "history": history,
            },
        )

        if args.patience > 0 and epochs_without_improve >= args.patience:
            print(f"early stopping after {epoch} epochs")
            break

    print(f"best epoch: {best_epoch}")
    print(f"best validation selection score: {best_score}")
    print(f"saved best checkpoint: {best_path}")
    print(f"saved summary: {summary_path}")


if __name__ == "__main__":
    main()

