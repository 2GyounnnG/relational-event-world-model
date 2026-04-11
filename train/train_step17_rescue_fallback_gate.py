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
from train.eval_step10_rescued_scope_rewrite_decomp import bool_pred_adj
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


@dataclass
class RescueFallbackGateConfig:
    input_dim: int
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.0


class RescueFallbackGate(nn.Module):
    """
    Small interface gate over rescued edges.

    The gate predicts whether to keep the Step 9c rewrite output (`1`) or fall
    back to the base-proposal rewrite output (`0`) for each rescued edge.
    中文说明：这是 chooser / interface gate，不是 event-scope classifier。
    """

    def __init__(self, config: RescueFallbackGateConfig):
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


def save_gate_checkpoint(
    path: Path,
    model: RescueFallbackGate,
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


def load_fallback_gate_model(checkpoint_path: Path, device: torch.device) -> RescueFallbackGate:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = RescueFallbackGate(RescueFallbackGateConfig(**checkpoint["model_config"])).to(device)
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


def pair_expand(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, num_nodes = values.shape[:2]
    trailing = values.shape[2:]
    left = values.unsqueeze(2).expand(bsz, num_nodes, num_nodes, *trailing)
    right = values.unsqueeze(1).expand(bsz, num_nodes, num_nodes, *trailing)
    return left, right


def norm_logits(logits: torch.Tensor) -> torch.Tensor:
    return logits.clamp(min=-10.0, max=10.0) / 10.0


def build_gate_feature_tensor(
    base_outputs: Dict[str, torch.Tensor],
    completion_logits: torch.Tensor,
    rewrite_base: Dict[str, torch.Tensor],
    rewrite_step9c: Dict[str, torch.Tensor],
) -> torch.Tensor:
    node_log_i, node_log_j = pair_expand(norm_logits(base_outputs["node_scope_logits"]).unsqueeze(-1))
    node_prob_i, node_prob_j = pair_expand(base_outputs["proposal_node_probs"].unsqueeze(-1))
    node_min = torch.minimum(node_prob_i, node_prob_j)
    node_max = torch.maximum(node_prob_i, node_prob_j)
    node_prod = node_prob_i * node_prob_j

    base_edge_logits = rewrite_base["edge_logits_full"]
    step_edge_logits = rewrite_step9c["edge_logits_full"]
    base_edge_probs = torch.sigmoid(base_edge_logits)
    step_edge_probs = torch.sigmoid(step_edge_logits)
    base_pred = (base_edge_probs >= 0.5).float()
    step_pred = (step_edge_probs >= 0.5).float()
    disagree = (base_pred != step_pred).float()

    features = [
        base_outputs["input_adj"].unsqueeze(-1),
        norm_logits(base_outputs["edge_scope_logits"]).unsqueeze(-1),
        base_outputs["proposal_edge_probs"].unsqueeze(-1),
        norm_logits(completion_logits).unsqueeze(-1),
        torch.sigmoid(completion_logits).unsqueeze(-1),
        node_log_i,
        node_log_j,
        node_prob_i,
        node_prob_j,
        node_min,
        node_max,
        node_prod,
        norm_logits(base_edge_logits).unsqueeze(-1),
        base_edge_probs.unsqueeze(-1),
        norm_logits(step_edge_logits).unsqueeze(-1),
        step_edge_probs.unsqueeze(-1),
        norm_logits(step_edge_logits - base_edge_logits).unsqueeze(-1),
        (step_edge_probs - base_edge_probs).unsqueeze(-1),
        torch.abs(step_edge_probs - base_edge_probs).unsqueeze(-1),
        base_pred.unsqueeze(-1),
        step_pred.unsqueeze(-1),
        disagree.unsqueeze(-1),
    ]
    return torch.cat(features, dim=-1).float()


def build_gate_targets(
    batch: Dict[str, Any],
    valid_edge_mask: torch.Tensor,
    rescued_edges: torch.Tensor,
    rewrite_base: Dict[str, torch.Tensor],
    rewrite_step9c: Dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_base = bool_pred_adj(rewrite_base["edge_logits_full"], valid_edge_mask)
    pred_step = bool_pred_adj(rewrite_step9c["edge_logits_full"], valid_edge_mask)
    target_adj = (batch["next_adj"] > 0.5) & valid_edge_mask.bool()
    base_correct = pred_base == target_adj
    step_correct = pred_step == target_adj
    # Tie rule: fallback to base when both paths have the same correctness.
    # This makes the learned gate conservative without changing edge accuracy
    # on tied rescued edges.
    choose_step_target = step_correct & (~base_correct)
    rescued = rescued_edges.bool() & valid_edge_mask.bool()
    return choose_step_target.float(), rescued, base_correct, step_correct


def init_sums() -> Dict[str, float]:
    return {
        "loss_sum": 0.0,
        "loss_batches": 0.0,
        "rescued_total": 0.0,
        "target_choose_step_total": 0.0,
        "pred_choose_step_total": 0.0,
        "chosen_correct_total": 0.0,
        "base_correct_total": 0.0,
        "step_correct_total": 0.0,
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
        "pred_choose_step_fraction": sums["pred_choose_step_total"] / max(sums["rescued_total"], 1.0),
        "chosen_correct_rate": sums["chosen_correct_total"] / max(sums["rescued_total"], 1.0),
        "base_path_correct_rate_on_rescued": sums["base_correct_total"] / max(sums["rescued_total"], 1.0),
        "step9c_path_correct_rate_on_rescued": sums["step_correct_total"] / max(sums["rescued_total"], 1.0),
        "choose_step_ap": ap,
        "choose_step_auroc": auc,
        "selection_score": sums["chosen_correct_total"] / max(sums["rescued_total"], 1.0),
    }


def run_epoch(
    proposal_model: torch.nn.Module,
    completion_model: torch.nn.Module,
    rewrite_model: torch.nn.Module,
    gate_model: RescueFallbackGate,
    loader: DataLoader,
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
    gate_model.train() if is_train else gate_model.eval()
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
            mode_base = apply_edge_completion_mode(
                base_outputs=base_outputs,
                edge_completion_mode=EDGE_COMPLETION_OFF,
                completion_model=None,
                completion_threshold=0.5,
            )
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
            features = build_gate_feature_tensor(
                base_outputs=base_outputs,
                completion_logits=completion_logits,
                rewrite_base=rewrite_base,
                rewrite_step9c=rewrite_step9c,
            )
            target, rescued, base_correct, step_correct = build_gate_targets(
                batch=batch,
                valid_edge_mask=valid_edge_mask,
                rescued_edges=mode_step9c["rescued_edges"],
                rewrite_base=rewrite_base,
                rewrite_step9c=rewrite_step9c,
            )

        if rescued.float().sum().item() <= 0:
            continue
        gate_logits = gate_model(features)
        loss = F.binary_cross_entropy_with_logits(gate_logits[rescued], target[rescued])

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(gate_model.parameters(), grad_clip)
            optimizer.step()

        with torch.no_grad():
            probs = torch.sigmoid(gate_logits)
            choose_step = (probs >= 0.5) & rescued
            chosen_correct = torch.where(choose_step, step_correct, base_correct)
            sums["loss_sum"] += loss.item()
            sums["loss_batches"] += 1.0
            sums["rescued_total"] += rescued.float().sum().item()
            sums["target_choose_step_total"] += target[rescued].sum().item()
            sums["pred_choose_step_total"] += choose_step.float().sum().item()
            sums["chosen_correct_total"] += (chosen_correct & rescued).float().sum().item()
            sums["base_correct_total"] += (base_correct & rescued).float().sum().item()
            sums["step_correct_total"] += (step_correct & rescued).float().sum().item()
            all_scores.append(probs[rescued].detach().float().cpu())
            all_labels.append(target[rescued].detach().bool().cpu())

    return finalize_metrics(sums, all_scores, all_labels)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


@torch.no_grad()
def infer_feature_dim(
    proposal_model: torch.nn.Module,
    completion_model: torch.nn.Module,
    rewrite_model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    node_threshold: float,
    edge_threshold: float,
    use_proposal_conditioning: bool,
) -> int:
    for batch in loader:
        batch = move_batch_to_device(batch, device)
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
        features = build_gate_feature_tensor(base_outputs, completion_logits, rewrite_base, rewrite_step9c)
        return int(features.shape[-1])
    raise RuntimeError("Could not infer fallback gate feature dimension.")


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
        use_proposal_conditioning=use_proposal_conditioning,
    )
    gate_config = RescueFallbackGateConfig(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    gate_model = RescueFallbackGate(gate_config).to(device)
    optimizer = torch.optim.Adam(gate_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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
    print(f"feature dim: {input_dim}")
    print("frozen proposal + frozen completion + frozen rewrite; training fallback chooser gate only")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            proposal_model,
            completion_model,
            rewrite_model,
            gate_model,
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
            gate_model,
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
            f"epoch {epoch:03d} | "
            f"train_loss={train_metrics.get('loss'):.6f} | "
            f"val_loss={val_metrics.get('loss'):.6f} | "
            f"val_chosen_correct={val_metrics.get('chosen_correct_rate')} | "
            f"val_choose_step_frac={val_metrics.get('pred_choose_step_fraction')} | "
            f"val_ap={val_metrics.get('choose_step_ap')} | "
            f"score={val_score}"
        )

        if val_score > best_score:
            best_score = float(val_score)
            best_epoch = epoch
            epochs_without_improve = 0
            save_gate_checkpoint(
                best_path,
                gate_model,
                args,
                extra={
                    "best_epoch": best_epoch,
                    "best_validation_selection_score": best_score,
                    "best_val_metrics": val_metrics,
                    "proposal_checkpoint_path": str(proposal_checkpoint_path),
                    "completion_checkpoint_path": str(completion_checkpoint_path),
                    "rewrite_checkpoint_path": str(rewrite_checkpoint_path),
                    "rescue_budget_fraction": FIXED_RESCUE_BUDGET_FRACTION,
                    "fallback_gate_mode": "learned_rescue_conditioned_gate",
                    "tie_rule": "fallback_to_base_when_base_and_step9c_have_equal_correctness",
                },
            )
        else:
            epochs_without_improve += 1

        save_gate_checkpoint(
            last_path,
            gate_model,
            args,
            extra={
                "epoch": epoch,
                "validation_selection_score": val_score,
                "val_metrics": val_metrics,
                "proposal_checkpoint_path": str(proposal_checkpoint_path),
                "completion_checkpoint_path": str(completion_checkpoint_path),
                "rewrite_checkpoint_path": str(rewrite_checkpoint_path),
                "rescue_budget_fraction": FIXED_RESCUE_BUDGET_FRACTION,
                "fallback_gate_mode": "learned_rescue_conditioned_gate",
                "tie_rule": "fallback_to_base_when_base_and_step9c_have_equal_correctness",
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
        "model_config": asdict(gate_config),
        "proposal_checkpoint_path": str(proposal_checkpoint_path),
        "completion_checkpoint_path": str(completion_checkpoint_path),
        "rewrite_checkpoint_path": str(rewrite_checkpoint_path),
        "rescue_budget_fraction": FIXED_RESCUE_BUDGET_FRACTION,
        "fallback_gate_mode": "learned_rescue_conditioned_gate",
        "tie_rule": "fallback_to_base_when_base_and_step9c_have_equal_correctness",
    }
    save_json(summary_path, summary)
    print(f"best epoch: {best_epoch}")
    print(f"best validation selection score: {best_score}")
    print(f"saved best checkpoint: {best_path}")
    print(f"saved summary: {summary_path}")


if __name__ == "__main__":
    main()

