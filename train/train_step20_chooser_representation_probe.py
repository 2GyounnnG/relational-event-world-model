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
    pair_expand,
    save_json,
)


COMPACT_INTERFACE_SCORES = "compact_interface_scores"
ENRICHED_RESCUE_LOCAL_CONTEXT = "enriched_rescue_local_context"
FEATURE_BUNDLES = [COMPACT_INTERFACE_SCORES, ENRICHED_RESCUE_LOCAL_CONTEXT]


@dataclass
class Step20ChooserProbeConfig:
    feature_bundle: str
    input_dim: int
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.0


class Step20ChooserProbe(nn.Module):
    """Fixed shallow probe class for both Step20 feature bundles."""

    def __init__(self, config: Step20ChooserProbeConfig):
        super().__init__()
        self.config = config
        self.net = build_mlp(
            in_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            out_dim=1,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )

    def forward(self, features_full: torch.Tensor) -> torch.Tensor:
        logits = self.net(features_full).squeeze(-1)
        logits = 0.5 * (logits + logits.transpose(1, 2))
        num_nodes = logits.shape[-1]
        diag = torch.eye(num_nodes, device=logits.device, dtype=torch.bool).unsqueeze(0)
        return logits.masked_fill(diag, -1e9)


def save_probe_checkpoint(
    path: Path,
    model: Step20ChooserProbe,
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


def load_step20_probe_model(checkpoint_path: Path, device: torch.device) -> Step20ChooserProbe:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = Step20ChooserProbe(Step20ChooserProbeConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def norm_logits(logits: torch.Tensor) -> torch.Tensor:
    return logits.clamp(min=-10.0, max=10.0) / 10.0


def build_step20_feature_tensor(
    batch: Dict[str, Any],
    base_outputs: Dict[str, torch.Tensor],
    completion_logits: torch.Tensor,
    rewrite_base: Dict[str, torch.Tensor],
    rewrite_step9c: Dict[str, torch.Tensor],
    feature_bundle: str,
) -> torch.Tensor:
    if feature_bundle not in FEATURE_BUNDLES:
        raise ValueError(f"Unknown feature bundle: {feature_bundle}")

    compact = build_gate_feature_tensor(
        base_outputs=base_outputs,
        completion_logits=completion_logits,
        rewrite_base=rewrite_base,
        rewrite_step9c=rewrite_step9c,
    )
    if feature_bundle == COMPACT_INTERFACE_SCORES:
        return compact

    input_node_feats = base_outputs["input_node_feats"]
    input_adj = base_outputs["input_adj"]
    valid_edge_mask = base_outputs["valid_edge_mask"].bool()
    pred_scope_nodes = base_outputs["pred_scope_nodes"].bool()
    scope_float = pred_scope_nodes.float()
    scope_count = scope_float.sum(dim=1).clamp_min(1.0)

    node_feat_i, node_feat_j = pair_expand(input_node_feats)
    scoped_adj = input_adj * scope_float.unsqueeze(1)
    deg = scoped_adj.sum(dim=-1) / scope_count.unsqueeze(-1)
    deg_i, deg_j = pair_expand(deg.unsqueeze(-1))
    neighbor_mask = ((input_adj > 0.5) & pred_scope_nodes.unsqueeze(1)).float()
    common = torch.bmm(neighbor_mask, neighbor_mask.transpose(1, 2)) / scope_count.view(-1, 1, 1)
    induced_mask = pred_scope_nodes.unsqueeze(2) & pred_scope_nodes.unsqueeze(1) & valid_edge_mask
    induced_edge_count = (input_adj > 0.5).float().masked_fill(~induced_mask, 0.0).sum(dim=(1, 2))
    induced_possible = induced_mask.float().sum(dim=(1, 2)).clamp_min(1.0)
    scope_density = (induced_edge_count / induced_possible).view(-1, 1, 1, 1).expand_as(common.unsqueeze(-1))

    base_probs = torch.sigmoid(rewrite_base["edge_logits_full"])
    step_probs = torch.sigmoid(rewrite_step9c["edge_logits_full"])
    base_pred = base_probs >= 0.5
    step_pred = step_probs >= 0.5
    disagreement = (base_pred != step_pred).float() * valid_edge_mask.float()
    prob_diff = torch.abs(step_probs - base_probs) * valid_edge_mask.float()
    scoped_valid = pred_scope_nodes.unsqueeze(1) & valid_edge_mask
    denom = scoped_valid.float().sum(dim=-1).clamp_min(1.0)
    node_disagree = (disagreement * scoped_valid.float()).sum(dim=-1) / denom
    node_prob_diff = (prob_diff * scoped_valid.float()).sum(dim=-1) / denom
    disagree_i, disagree_j = pair_expand(node_disagree.unsqueeze(-1))
    prob_diff_i, prob_diff_j = pair_expand(node_prob_diff.unsqueeze(-1))

    local_features = [
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
        disagreement.unsqueeze(-1),
        prob_diff.unsqueeze(-1),
        disagree_i,
        disagree_j,
        prob_diff_i,
        prob_diff_j,
    ]
    return torch.cat([compact, *local_features], dim=-1).float()


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


def run_epoch(
    proposal_model: torch.nn.Module,
    completion_model: torch.nn.Module,
    rewrite_model: torch.nn.Module,
    probe_model: Step20ChooserProbe,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    node_threshold: float,
    edge_threshold: float,
    feature_bundle: str,
    use_proposal_conditioning: bool,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_clip: Optional[float] = None,
) -> Dict[str, Any]:
    is_train = optimizer is not None
    proposal_model.eval()
    completion_model.eval()
    rewrite_model.eval()
    probe_model.train() if is_train else probe_model.eval()
    sums = init_sums()
    all_scores: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for batch in loader:
        require_keys(batch, ["node_feats", "adj", "next_adj", "node_mask", "changed_edges", "event_scope_union_edges"])
        batch = move_batch_to_device(batch, device)
        with torch.no_grad():
            base_outputs = get_base_proposal_outputs(
                proposal_model=proposal_model,
                batch=batch,
                node_threshold=node_threshold,
                edge_threshold=edge_threshold,
            )
            valid_edge_mask = base_outputs["valid_edge_mask"].bool()
            completion_logits = completion_model(
                node_latents=base_outputs["node_latents"],
                base_edge_logits=base_outputs["edge_scope_logits"],
            )
            completion_probs = torch.sigmoid(completion_logits) * valid_edge_mask.float()
            mode_base = apply_edge_completion_mode(base_outputs, EDGE_COMPLETION_OFF, None, 0.5)
            mode_step9c = build_ranked_outputs(
                base_outputs=base_outputs,
                mode_name="step9c_completion_only",
                ranking_scores=completion_probs,
                completion_probs=completion_probs,
            )
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
            features = build_step20_feature_tensor(
                batch=batch,
                base_outputs=base_outputs,
                completion_logits=completion_logits,
                rewrite_base=rewrite_base,
                rewrite_step9c=rewrite_step9c,
                feature_bundle=feature_bundle,
            )
            target, rescued, _, _ = build_gate_targets(
                batch=batch,
                valid_edge_mask=valid_edge_mask,
                rescued_edges=mode_step9c["rescued_edges"],
                rewrite_base=rewrite_base,
                rewrite_step9c=rewrite_step9c,
            )

        if rescued.float().sum().item() <= 0:
            continue
        logits = probe_model(features)
        loss = F.binary_cross_entropy_with_logits(logits[rescued], target[rescued])
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(probe_model.parameters(), grad_clip)
            optimizer.step()

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            sums["loss_sum"] += loss.item()
            sums["loss_batches"] += 1.0
            sums["rescued_total"] += rescued.float().sum().item()
            sums["target_choose_step_total"] += target[rescued].sum().item()
            all_scores.append(probs[rescued].detach().float().cpu())
            all_labels.append(target[rescued].detach().bool().cpu())

    return finalize_metrics(sums, all_scores, all_labels)


@torch.no_grad()
def infer_feature_dim(
    proposal_model: torch.nn.Module,
    completion_model: torch.nn.Module,
    rewrite_model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    node_threshold: float,
    edge_threshold: float,
    feature_bundle: str,
    use_proposal_conditioning: bool,
) -> int:
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        base_outputs = get_base_proposal_outputs(proposal_model, batch, node_threshold, edge_threshold)
        completion_logits = completion_model(base_outputs["node_latents"], base_outputs["edge_scope_logits"])
        completion_probs = torch.sigmoid(completion_logits) * base_outputs["valid_edge_mask"].float()
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
        features = build_step20_feature_tensor(batch, base_outputs, completion_logits, rewrite_base, rewrite_step9c, feature_bundle)
        return int(features.shape[-1])
    raise RuntimeError("Could not infer Step20 chooser feature dimension.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--proposal_checkpoint_path", type=str, required=True)
    parser.add_argument("--completion_checkpoint_path", type=str, required=True)
    parser.add_argument("--rewrite_checkpoint_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--feature_bundle", choices=FEATURE_BUNDLES, required=True)
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

    input_dim = infer_feature_dim(
        proposal_model=proposal_model,
        completion_model=completion_model,
        rewrite_model=rewrite_model,
        loader=train_loader,
        device=device,
        node_threshold=args.node_threshold,
        edge_threshold=args.edge_threshold,
        feature_bundle=args.feature_bundle,
        use_proposal_conditioning=use_proposal_conditioning,
    )
    config = Step20ChooserProbeConfig(
        feature_bundle=args.feature_bundle,
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    probe_model = Step20ChooserProbe(config).to(device)
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
    print(f"feature_bundle: {args.feature_bundle}")
    print(f"feature dim: {input_dim}")
    print("frozen proposal + frozen completion + frozen rewrite; plain BCE chooser representation probe")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            proposal_model,
            completion_model,
            rewrite_model,
            probe_model,
            train_loader,
            device,
            args.node_threshold,
            args.edge_threshold,
            args.feature_bundle,
            use_proposal_conditioning,
            optimizer=optimizer,
            grad_clip=args.grad_clip,
        )
        val_metrics = run_epoch(
            proposal_model,
            completion_model,
            rewrite_model,
            probe_model,
            val_loader,
            device,
            args.node_threshold,
            args.edge_threshold,
            args.feature_bundle,
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
                    "rewrite_checkpoint_path": str(rewrite_checkpoint_path),
                    "rescue_budget_fraction": FIXED_RESCUE_BUDGET_FRACTION,
                    "chooser_target": "step9c_path_strictly_better_than_base_path",
                    "tie_rule": "fallback_to_base_when_base_and_step9c_have_equal_correctness",
                    "objective": "plain_bce",
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
