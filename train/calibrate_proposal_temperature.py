from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.oracle_local_delta import build_valid_edge_mask
from train.calibrate_noisy_proposal_thresholds import (
    evaluate_grid_for_rewrite,
    load_proposal_model,
    load_rewrite_model,
    parse_threshold_list,
    save_json,
)
from train.eval_noisy_structured_observation import (
    build_loader,
    get_device,
    move_batch_to_device,
    require_keys,
    resolve_path,
)


class TemperatureScaledProposalModel(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        node_temperature: float,
        edge_temperature: Optional[float],
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.node_temperature = float(node_temperature)
        self.edge_temperature = float(edge_temperature) if edge_temperature is not None else None

    def forward(self, *args, **kwargs):
        outputs = self.base_model(*args, **kwargs)
        scaled = dict(outputs)

        if "node_scope_logits" in scaled:
            scaled["node_scope_logits"] = scaled["node_scope_logits"] / self.node_temperature
        if "scope_logits" in scaled:
            scaled["scope_logits"] = scaled["scope_logits"] / self.node_temperature
        if self.edge_temperature is not None and "edge_scope_logits" in scaled:
            scaled["edge_scope_logits"] = scaled["edge_scope_logits"] / self.edge_temperature
        return scaled


def fit_binary_temperature(
    logits: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    max_iter: int = 50,
) -> float:
    logits = logits.to(device=device, dtype=torch.float32)
    targets = targets.to(device=device, dtype=torch.float32)
    log_temperature = nn.Parameter(torch.zeros((), device=device, dtype=torch.float32))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature)
        scaled_logits = logits / temperature
        loss = nn.functional.binary_cross_entropy_with_logits(scaled_logits, targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_temperature.detach()).item())


@torch.no_grad()
def collect_validation_logits(
    proposal_model: nn.Module,
    loader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    proposal_model.eval()
    node_logits_list: List[torch.Tensor] = []
    node_targets_list: List[torch.Tensor] = []
    edge_logits_list: List[torch.Tensor] = []
    edge_targets_list: List[torch.Tensor] = []
    has_edge_logits = False

    for batch in loader:
        require_keys(
            batch,
            [
                "node_feats",
                "adj",
                "node_mask",
                "event_scope_union_nodes",
                "event_scope_union_edges",
            ],
        )
        batch = move_batch_to_device(batch, device)
        input_node_feats = batch.get("obs_node_feats", batch["node_feats"])
        input_adj = batch.get("obs_adj", batch["adj"])
        node_mask = batch["node_mask"].bool()
        valid_edge_mask = build_valid_edge_mask(batch["node_mask"]).bool()

        proposal_outputs = proposal_model(node_feats=input_node_feats, adj=input_adj)
        node_scope_logits = proposal_outputs.get("node_scope_logits", proposal_outputs["scope_logits"])
        node_logits_list.append(node_scope_logits[node_mask].detach().cpu())
        node_targets_list.append(batch["event_scope_union_nodes"][node_mask].detach().cpu())

        if "edge_scope_logits" in proposal_outputs:
            has_edge_logits = True
            edge_scope_logits = proposal_outputs["edge_scope_logits"]
            edge_logits_list.append(edge_scope_logits[valid_edge_mask].detach().cpu())
            edge_targets_list.append(batch["event_scope_union_edges"][valid_edge_mask].detach().cpu())

    node_logits = torch.cat(node_logits_list, dim=0)
    node_targets = torch.cat(node_targets_list, dim=0)
    if has_edge_logits:
        edge_logits = torch.cat(edge_logits_list, dim=0)
        edge_targets = torch.cat(edge_targets_list, dim=0)
    else:
        edge_logits = None
        edge_targets = None
    return node_logits, node_targets, edge_logits, edge_targets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal_checkpoint_path", type=str, required=True)
    parser.add_argument("--w012_checkpoint_path", type=str, required=True)
    parser.add_argument("--i1520_checkpoint_path", type=str, default=None)
    parser.add_argument("--val_data_path", type=str, required=True)
    parser.add_argument("--test_data_path", type=str, required=True)
    parser.add_argument("--node_thresholds", type=str, default="0.15,0.20,0.25")
    parser.add_argument("--edge_thresholds", type=str, default="0.10,0.15,0.20,0.25")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    proposal_checkpoint_path = resolve_path(args.proposal_checkpoint_path)
    w012_checkpoint_path = resolve_path(args.w012_checkpoint_path)
    i1520_checkpoint_path = resolve_path(args.i1520_checkpoint_path) if args.i1520_checkpoint_path else None
    val_data_path = resolve_path(args.val_data_path)
    test_data_path = resolve_path(args.test_data_path)
    node_thresholds = parse_threshold_list(args.node_thresholds)
    edge_thresholds = parse_threshold_list(args.edge_thresholds)
    device = get_device(args.device)

    _, val_loader = build_loader(
        data_path=str(val_data_path),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    _, test_loader = build_loader(
        data_path=str(test_data_path),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    proposal_model = load_proposal_model(proposal_checkpoint_path, device)
    node_logits, node_targets, edge_logits, edge_targets = collect_validation_logits(
        proposal_model=proposal_model,
        loader=val_loader,
        device=device,
    )

    node_temperature = fit_binary_temperature(node_logits, node_targets, device=device)
    edge_temperature = None
    if edge_logits is not None and edge_targets is not None:
        edge_temperature = fit_binary_temperature(edge_logits, edge_targets, device=device)

    scaled_proposal_model = TemperatureScaledProposalModel(
        base_model=proposal_model,
        node_temperature=node_temperature,
        edge_temperature=edge_temperature,
    ).to(device)
    scaled_proposal_model.eval()

    w012_model, w012_use_conditioning = load_rewrite_model(w012_checkpoint_path, device)

    threshold_payload: Dict[str, Any] = {
        "metadata": {
            "proposal_checkpoint_path": str(proposal_checkpoint_path),
            "temperature_calibration_mode": "validation_based_post_hoc_temperature_scaling",
            "val_data_path": str(val_data_path),
            "test_data_path": str(test_data_path),
            "node_temperature": node_temperature,
            "edge_temperature": edge_temperature,
            "node_threshold_grid": node_thresholds,
            "edge_threshold_grid": edge_thresholds,
            "selection_mode": "validation_based_after_temperature_scaling",
            "selection_formula": "0.35*full_edge_acc + 0.35*context_edge_acc + 0.15*changed_edge_acc + 0.15*delete",
        },
        "rewrites": {},
    }

    print(f"device: {device}")
    print(f"proposal checkpoint: {proposal_checkpoint_path}")
    print(f"val data: {val_data_path}")
    print(f"test data: {test_data_path}")
    print(f"learned node temperature: {node_temperature:.6f}")
    if edge_temperature is None:
        print("learned edge temperature: NA")
    else:
        print(f"learned edge temperature: {edge_temperature:.6f}")
    print(f"node thresholds: {node_thresholds}")
    print(f"edge thresholds: {edge_thresholds}")

    threshold_payload["rewrites"]["W012"] = evaluate_grid_for_rewrite(
        proposal_model=scaled_proposal_model,
        rewrite_model=w012_model,
        use_proposal_conditioning=w012_use_conditioning,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        node_thresholds=node_thresholds,
        edge_thresholds=edge_thresholds,
    )
    best_w012 = threshold_payload["rewrites"]["W012"]["best_val"]
    print(
        f"W012 best on val: node_threshold={best_w012['node_threshold']:.2f} "
        f"edge_threshold={best_w012['edge_threshold']:.2f} "
        f"score={best_w012['val_selection_score']:.6f}"
    )

    if i1520_checkpoint_path is not None and i1520_checkpoint_path.exists():
        i1520_model, i1520_use_conditioning = load_rewrite_model(i1520_checkpoint_path, device)
        threshold_payload["rewrites"]["I1520"] = evaluate_grid_for_rewrite(
            proposal_model=scaled_proposal_model,
            rewrite_model=i1520_model,
            use_proposal_conditioning=i1520_use_conditioning,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            node_thresholds=node_thresholds,
            edge_thresholds=edge_thresholds,
        )
        best_i1520 = threshold_payload["rewrites"]["I1520"]["best_val"]
        print(
            f"I1520 best on val: node_threshold={best_i1520['node_threshold']:.2f} "
            f"edge_threshold={best_i1520['edge_threshold']:.2f} "
            f"score={best_i1520['val_selection_score']:.6f}"
        )
    else:
        threshold_payload["rewrites"]["I1520"] = None
        print("I1520 checkpoint not available; skipped")

    temperature_payload = {
        "metadata": {
            "proposal_checkpoint_path": str(proposal_checkpoint_path),
            "val_data_path": str(val_data_path),
            "calibration_method": "scalar_temperature_scaling",
            "uses_noisy_observation_input": True,
            "objective": "masked BCEWithLogits on oracle node/edge scope targets",
        },
        "temperatures": {
            "node_temperature": node_temperature,
            "edge_temperature": edge_temperature,
        },
    }

    temperature_json_path = proposal_checkpoint_path.parent / "proposal_temperature_calibration.json"
    threshold_json_path = proposal_checkpoint_path.parent / "noisy_threshold_calibration_temperature_scaled.json"
    save_json(temperature_json_path, temperature_payload)
    save_json(threshold_json_path, threshold_payload)
    print(f"saved temperature calibration json: {temperature_json_path}")
    print(f"saved threshold calibration json: {threshold_json_path}")


if __name__ == "__main__":
    main()
