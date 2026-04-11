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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.baselines import build_mlp
from train.eval_noisy_structured_observation import move_batch_to_device, require_keys, resolve_path
from train.eval_step12_guard_ranking_interaction import FIXED_RESCUE_BUDGET_FRACTION, build_ranked_outputs
from train.eval_step9_gated_edge_completion import (
    EDGE_COMPLETION_OFF,
    apply_edge_completion_mode,
    get_base_proposal_outputs,
    load_completion_model,
    load_proposal_model,
    load_rewrite_model,
)
from train.eval_step9_rescue_frontier import average_precision, auroc
from train.train_step17_rescue_fallback_gate import (
    build_dataloaders,
    build_gate_feature_tensor,
    build_gate_targets,
    get_device,
    norm_logits,
    pair_expand,
    save_json,
)


LEARNED_TWO_PATH_LOCAL_SUBGRAPH = "learned_two_path_local_subgraph"


@dataclass
class Step21TwoPathLocalSubgraphConfig:
    chooser_representation: str
    pair_feature_dim: int
    node_feature_dim: int
    node_embed_dim: int = 48
    hidden_dim: int = 96
    node_layers: int = 2
    head_layers: int = 3
    dropout: float = 0.0


class Step21TwoPathLocalSubgraphChooser(nn.Module):
    """
    Small rescued-edge chooser with a local pooling encoder.

    The proposal, Step 9 completion head, and rewrite model stay frozen. This
    chooser only scores rescued edges for "keep Step9c output" vs "fallback to
    base output". 中文说明：这是局部 interface chooser，不是新的 proposal/rewrite
    backbone.
    """

    def __init__(self, config: Step21TwoPathLocalSubgraphConfig):
        super().__init__()
        self.config = config
        self.node_encoder = build_mlp(
            in_dim=config.node_feature_dim,
            hidden_dim=config.hidden_dim,
            out_dim=config.node_embed_dim,
            num_layers=config.node_layers,
            dropout=config.dropout,
        )
        pair_in_dim = config.pair_feature_dim + config.node_embed_dim * 8 + 4
        self.head = build_mlp(
            in_dim=pair_in_dim,
            hidden_dim=config.hidden_dim,
            out_dim=1,
            num_layers=config.head_layers,
            dropout=config.dropout,
        )

    @staticmethod
    def _masked_mean(mask: torch.Tensor, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = mask.float()
        counts = weights.sum(dim=-1).clamp_min(1.0)
        pooled = torch.einsum("bijn,bnd->bijd", weights, values) / counts.unsqueeze(-1)
        return pooled, counts

    @staticmethod
    def _masked_max(mask: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        expanded = values[:, None, None, :, :]
        masked = expanded.masked_fill(~mask[..., None], -1.0e4)
        pooled = masked.max(dim=3).values
        empty = mask.float().sum(dim=-1, keepdim=True) <= 0
        return torch.where(empty, torch.zeros_like(pooled), pooled)

    def forward(
        self,
        pair_features: torch.Tensor,
        node_features: torch.Tensor,
        input_adj: torch.Tensor,
        pred_scope_nodes: torch.Tensor,
        valid_edge_mask: torch.Tensor,
    ) -> torch.Tensor:
        node_emb = self.node_encoder(node_features.float())
        bsz, num_nodes, _ = node_emb.shape

        h_i, h_j = pair_expand(node_emb)
        adj_bool = input_adj > 0.5
        scoped_nodes = pred_scope_nodes.bool()
        scope_count = scoped_nodes.float().sum(dim=1).clamp_min(1.0)

        eye = torch.eye(num_nodes, device=node_emb.device, dtype=torch.bool)
        endpoint_mask = eye.view(1, num_nodes, 1, num_nodes) | eye.view(1, 1, num_nodes, num_nodes)
        adj_i = adj_bool.unsqueeze(2).expand(bsz, num_nodes, num_nodes, num_nodes)
        adj_j = adj_bool.unsqueeze(1).expand(bsz, num_nodes, num_nodes, num_nodes)
        scoped_k = scoped_nodes.view(bsz, 1, 1, num_nodes)

        # Candidate-local region: endpoints plus one-hop neighbors inside the predicted node scope.
        local_mask = scoped_k & (endpoint_mask | adj_i | adj_j)
        common_mask = scoped_k & adj_i & adj_j

        local_mean, local_count = self._masked_mean(local_mask, node_emb)
        local_max = self._masked_max(local_mask, node_emb)
        common_mean, common_count = self._masked_mean(common_mask, node_emb)

        scoped_adj = input_adj * scoped_nodes.float().unsqueeze(1)
        deg = scoped_adj.sum(dim=-1) / scope_count.unsqueeze(-1)
        deg_i, deg_j = pair_expand(deg.unsqueeze(-1))

        local_frac = (local_count / scope_count.view(bsz, 1, 1)).unsqueeze(-1)
        common_frac = (common_count / scope_count.view(bsz, 1, 1)).unsqueeze(-1)

        head_input = torch.cat(
            [
                pair_features.float(),
                h_i,
                h_j,
                torch.abs(h_i - h_j),
                h_i * h_j,
                local_mean,
                local_max,
                common_mean,
                torch.abs(local_mean - common_mean),
                deg_i,
                deg_j,
                local_frac,
                common_frac,
            ],
            dim=-1,
        )
        logits = self.head(head_input).squeeze(-1)
        logits = 0.5 * (logits + logits.transpose(1, 2))
        return logits.masked_fill(~valid_edge_mask.bool(), -1.0e9)


def save_step21_checkpoint(
    path: Path,
    model: Step21TwoPathLocalSubgraphChooser,
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


def load_step21_chooser_model(checkpoint_path: Path, device: torch.device) -> Step21TwoPathLocalSubgraphChooser:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = Step21TwoPathLocalSubgraphChooser(
        Step21TwoPathLocalSubgraphConfig(**checkpoint["model_config"])
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_step21_feature_tensors(
    batch: Dict[str, Any],
    base_outputs: Dict[str, torch.Tensor],
    completion_logits: torch.Tensor,
    rewrite_base: Dict[str, torch.Tensor],
    rewrite_step9c: Dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    pair_features = build_gate_feature_tensor(
        base_outputs=base_outputs,
        completion_logits=completion_logits,
        rewrite_base=rewrite_base,
        rewrite_step9c=rewrite_step9c,
    )

    input_node_feats = base_outputs["input_node_feats"]
    proposal_latents = base_outputs["node_latents"]
    proposal_node_probs = base_outputs["proposal_node_probs"].unsqueeze(-1)
    proposal_node_logits = norm_logits(base_outputs["node_scope_logits"]).unsqueeze(-1)
    pred_scope_nodes = base_outputs["pred_scope_nodes"].float().unsqueeze(-1)

    base_type_probs = torch.softmax(rewrite_base["type_logits_full"], dim=-1)
    step_type_probs = torch.softmax(rewrite_step9c["type_logits_full"], dim=-1)
    base_state = rewrite_base["state_pred_full"]
    step_state = rewrite_step9c["state_pred_full"]
    node_features = torch.cat(
        [
            input_node_feats,
            proposal_latents,
            proposal_node_probs,
            proposal_node_logits,
            pred_scope_nodes,
            base_type_probs,
            step_type_probs,
            torch.abs(step_type_probs - base_type_probs),
            base_state,
            step_state,
            torch.abs(step_state - base_state),
        ],
        dim=-1,
    ).float()
    return pair_features, node_features


def init_sums() -> Dict[str, float]:
    return {
        "loss_sum": 0.0,
        "loss_batches": 0.0,
        "rescued_total": 0.0,
        "target_choose_step_total": 0.0,
    }


def finalize_metrics(
    sums: Dict[str, float],
    scores: list[torch.Tensor],
    labels: list[torch.Tensor],
) -> Dict[str, Any]:
    ap = None
    auc = None
    if scores:
        score_tensor = torch.cat(scores).float()
        label_tensor = torch.cat(labels).bool()
        ap = average_precision(score_tensor, label_tensor)
        auc = auroc(score_tensor, label_tensor)
    return {
        "loss": sums["loss_sum"] / max(sums["loss_batches"], 1.0),
        "rescued_total": int(sums["rescued_total"]),
        "target_choose_step_fraction": sums["target_choose_step_total"] / max(sums["rescued_total"], 1.0),
        "choose_step_ap": ap,
        "choose_step_auroc": auc,
        "selection_score": ap if ap is not None else -sums["loss_sum"] / max(sums["loss_batches"], 1.0),
    }


@torch.no_grad()
def build_frozen_paths(
    proposal_model: torch.nn.Module,
    completion_model: torch.nn.Module,
    rewrite_model: torch.nn.Module,
    batch: Dict[str, Any],
    node_threshold: float,
    edge_threshold: float,
    use_proposal_conditioning: bool,
) -> Dict[str, Any]:
    base_outputs = get_base_proposal_outputs(proposal_model, batch, node_threshold, edge_threshold)
    valid_edge_mask = base_outputs["valid_edge_mask"].bool()
    completion_logits = completion_model(base_outputs["node_latents"], base_outputs["edge_scope_logits"])
    completion_probs = torch.sigmoid(completion_logits) * valid_edge_mask.float()
    mode_base = apply_edge_completion_mode(base_outputs, EDGE_COMPLETION_OFF, None, 0.5)
    mode_step9c = build_ranked_outputs(base_outputs, "step9c_completion_only", completion_probs, completion_probs)
    rewrite_base = rewrite_model(
        node_feats=mode_base["input_node_feats"],
        adj=mode_base["input_adj"],
        scope_node_mask=mode_base["pred_scope_nodes"].float(),
        scope_edge_mask=mode_base["final_pred_scope_edges"].float(),
        proposal_node_probs=mode_base["proposal_node_probs"] if use_proposal_conditioning else None,
        proposal_edge_probs=mode_base["final_proposal_edge_probs"] if use_proposal_conditioning else None,
    )
    rewrite_step9c = rewrite_model(
        node_feats=mode_step9c["input_node_feats"],
        adj=mode_step9c["input_adj"],
        scope_node_mask=mode_step9c["pred_scope_nodes"].float(),
        scope_edge_mask=mode_step9c["final_pred_scope_edges"].float(),
        proposal_node_probs=mode_step9c["proposal_node_probs"] if use_proposal_conditioning else None,
        proposal_edge_probs=mode_step9c["final_proposal_edge_probs"] if use_proposal_conditioning else None,
    )
    pair_features, node_features = build_step21_feature_tensors(
        batch=batch,
        base_outputs=base_outputs,
        completion_logits=completion_logits,
        rewrite_base=rewrite_base,
        rewrite_step9c=rewrite_step9c,
    )
    target, rescued, _, _ = build_gate_targets(
        batch=batch,
        valid_edge_mask=valid_edge_mask,
        rescued_edges=mode_step9c["rescued_edges"],
        rewrite_base=rewrite_base,
        rewrite_step9c=rewrite_step9c,
    )
    return {
        "base_outputs": base_outputs,
        "pair_features": pair_features,
        "node_features": node_features,
        "valid_edge_mask": valid_edge_mask,
        "target": target,
        "rescued": rescued,
    }


def run_epoch(
    proposal_model: torch.nn.Module,
    completion_model: torch.nn.Module,
    rewrite_model: torch.nn.Module,
    chooser_model: Step21TwoPathLocalSubgraphChooser,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    node_threshold: float,
    edge_threshold: float,
    use_proposal_conditioning: bool,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_clip: Optional[float] = None,
) -> Dict[str, Any]:
    is_train = optimizer is not None
    proposal_model.eval()
    completion_model.eval()
    rewrite_model.eval()
    chooser_model.train() if is_train else chooser_model.eval()
    sums = init_sums()
    all_scores: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for batch in loader:
        require_keys(batch, ["node_feats", "adj", "next_adj", "node_mask", "changed_edges", "event_scope_union_edges"])
        batch = move_batch_to_device(batch, device)
        frozen = build_frozen_paths(
            proposal_model=proposal_model,
            completion_model=completion_model,
            rewrite_model=rewrite_model,
            batch=batch,
            node_threshold=node_threshold,
            edge_threshold=edge_threshold,
            use_proposal_conditioning=use_proposal_conditioning,
        )
        rescued = frozen["rescued"]
        if rescued.float().sum().item() <= 0:
            continue

        logits = chooser_model(
            pair_features=frozen["pair_features"],
            node_features=frozen["node_features"],
            input_adj=frozen["base_outputs"]["input_adj"],
            pred_scope_nodes=frozen["base_outputs"]["pred_scope_nodes"],
            valid_edge_mask=frozen["valid_edge_mask"],
        )
        loss = F.binary_cross_entropy_with_logits(logits[rescued], frozen["target"][rescued])
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(chooser_model.parameters(), grad_clip)
            optimizer.step()

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            sums["loss_sum"] += loss.item()
            sums["loss_batches"] += 1.0
            sums["rescued_total"] += rescued.float().sum().item()
            sums["target_choose_step_total"] += frozen["target"][rescued].sum().item()
            all_scores.append(probs[rescued].detach().float().cpu())
            all_labels.append(frozen["target"][rescued].detach().bool().cpu())

    return finalize_metrics(sums, all_scores, all_labels)


@torch.no_grad()
def infer_feature_dims(
    proposal_model: torch.nn.Module,
    completion_model: torch.nn.Module,
    rewrite_model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    node_threshold: float,
    edge_threshold: float,
    use_proposal_conditioning: bool,
) -> tuple[int, int]:
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        frozen = build_frozen_paths(
            proposal_model=proposal_model,
            completion_model=completion_model,
            rewrite_model=rewrite_model,
            batch=batch,
            node_threshold=node_threshold,
            edge_threshold=edge_threshold,
            use_proposal_conditioning=use_proposal_conditioning,
        )
        return int(frozen["pair_features"].shape[-1]), int(frozen["node_features"].shape[-1])
    raise RuntimeError("Could not infer Step21 chooser feature dimensions.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--proposal_checkpoint_path", type=str, required=True)
    parser.add_argument("--completion_checkpoint_path", type=str, required=True)
    parser.add_argument("--rewrite_checkpoint_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--node_threshold", type=float, default=0.20)
    parser.add_argument("--edge_threshold", type=float, default=0.15)
    parser.add_argument("--node_embed_dim", type=int, default=48)
    parser.add_argument("--hidden_dim", type=int, default=96)
    parser.add_argument("--node_layers", type=int, default=2)
    parser.add_argument("--head_layers", type=int, default=3)
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
    rewrite_checkpoint_path = resolve_path(args.rewrite_checkpoint_path)
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
    rewrite_model, use_proposal_conditioning = load_rewrite_model(rewrite_checkpoint_path, device)
    proposal_model.requires_grad_(False)
    completion_model.requires_grad_(False)
    rewrite_model.requires_grad_(False)

    pair_dim, node_dim = infer_feature_dims(
        proposal_model=proposal_model,
        completion_model=completion_model,
        rewrite_model=rewrite_model,
        loader=train_loader,
        device=device,
        node_threshold=args.node_threshold,
        edge_threshold=args.edge_threshold,
        use_proposal_conditioning=use_proposal_conditioning,
    )
    config = Step21TwoPathLocalSubgraphConfig(
        chooser_representation=LEARNED_TWO_PATH_LOCAL_SUBGRAPH,
        pair_feature_dim=pair_dim,
        node_feature_dim=node_dim,
        node_embed_dim=args.node_embed_dim,
        hidden_dim=args.hidden_dim,
        node_layers=args.node_layers,
        head_layers=args.head_layers,
        dropout=args.dropout,
    )
    chooser_model = Step21TwoPathLocalSubgraphChooser(config).to(device)
    optimizer = torch.optim.Adam(chooser_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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
    print(f"pair feature dim: {pair_dim}")
    print(f"node feature dim: {node_dim}")
    print("frozen proposal + frozen completion + frozen rewrite; Step21 two-path local-subgraph chooser")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            proposal_model,
            completion_model,
            rewrite_model,
            chooser_model,
            train_loader,
            device,
            args.node_threshold,
            args.edge_threshold,
            use_proposal_conditioning,
            optimizer=optimizer,
            grad_clip=args.grad_clip,
        )
        val_metrics = run_epoch(
            proposal_model,
            completion_model,
            rewrite_model,
            chooser_model,
            val_loader,
            device,
            args.node_threshold,
            args.edge_threshold,
            use_proposal_conditioning,
            optimizer=None,
            grad_clip=None,
        )
        val_score = val_metrics.get("selection_score")
        if val_score is None:
            val_score = -float(val_metrics.get("loss") or 0.0)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics, "selection_score": val_score})
        print(
            f"epoch {epoch:03d} | train_loss={train_metrics.get('loss'):.6f} | "
            f"val_loss={val_metrics.get('loss'):.6f} | val_ap={val_metrics.get('choose_step_ap')} | "
            f"val_auroc={val_metrics.get('choose_step_auroc')} | score={val_score}"
        )
        if val_score > best_score:
            best_score = float(val_score)
            best_epoch = epoch
            epochs_without_improve = 0
            save_step21_checkpoint(
                best_path,
                chooser_model,
                args,
                extra={
                    "best_epoch": best_epoch,
                    "best_validation_selection_score": best_score,
                    "best_val_metrics": val_metrics,
                    "proposal_checkpoint_path": str(proposal_checkpoint_path),
                    "completion_checkpoint_path": str(completion_checkpoint_path),
                    "rewrite_checkpoint_path": str(rewrite_checkpoint_path),
                    "rescue_budget_fraction": FIXED_RESCUE_BUDGET_FRACTION,
                    "chooser_target": "step9c_path_strictly_better_than_base_path",
                    "tie_rule": "fallback_to_base_when_base_and_step9c_have_equal_correctness",
                    "objective": "plain_bce",
                },
            )
        else:
            epochs_without_improve += 1

        save_step21_checkpoint(
            last_path,
            chooser_model,
            args,
            extra={
                "epoch": epoch,
                "validation_selection_score": val_score,
                "val_metrics": val_metrics,
                "proposal_checkpoint_path": str(proposal_checkpoint_path),
                "completion_checkpoint_path": str(completion_checkpoint_path),
                "rewrite_checkpoint_path": str(rewrite_checkpoint_path),
                "rescue_budget_fraction": FIXED_RESCUE_BUDGET_FRACTION,
                "chooser_target": "step9c_path_strictly_better_than_base_path",
                "tie_rule": "fallback_to_base_when_base_and_step9c_have_equal_correctness",
                "objective": "plain_bce",
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
        "model_config": asdict(config),
        "proposal_checkpoint_path": str(proposal_checkpoint_path),
        "completion_checkpoint_path": str(completion_checkpoint_path),
        "rewrite_checkpoint_path": str(rewrite_checkpoint_path),
        "rescue_budget_fraction": FIXED_RESCUE_BUDGET_FRACTION,
        "chooser_target": "step9c_path_strictly_better_than_base_path",
        "tie_rule": "fallback_to_base_when_base_and_step9c_have_equal_correctness",
        "objective": "plain_bce",
    }
    save_json(summary_path, summary)
    print(f"best epoch: {best_epoch}")
    print(f"best validation selection score: {best_score}")
    print(f"saved best checkpoint: {best_path}")
    print(f"saved summary: {summary_path}")


if __name__ == "__main__":
    main()
