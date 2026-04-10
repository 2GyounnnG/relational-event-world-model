from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.oracle_local_delta import OracleLocalDeltaRewriteConfig, OracleLocalDeltaRewriteModel
from models.proposal import ScopeProposalConfig, ScopeProposalModel
from train.eval_noisy_structured_observation import (
    build_loader,
    evaluate,
    get_device,
    resolve_path,
)


def parse_threshold_list(raw: str) -> List[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def compute_selection_score(results: Dict[str, Any]) -> float:
    overall = results["overall"]
    return (
        0.35 * float(overall["full_edge_acc"])
        + 0.35 * float(overall["context_edge_acc"])
        + 0.15 * float(overall["changed_edge_acc"])
        + 0.15 * float(overall["delete"])
    )


def load_proposal_model(checkpoint_path: Path, device: torch.device) -> ScopeProposalModel:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = ScopeProposalModel(ScopeProposalConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def load_rewrite_model(checkpoint_path: Path, device: torch.device) -> tuple[OracleLocalDeltaRewriteModel, bool]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = OracleLocalDeltaRewriteModel(
        OracleLocalDeltaRewriteConfig(**checkpoint["model_config"])
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    use_proposal_conditioning = bool(
        checkpoint.get("model_config", {}).get("use_proposal_conditioning", False)
    )
    return model, use_proposal_conditioning


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def evaluate_grid_for_rewrite(
    proposal_model: ScopeProposalModel,
    rewrite_model: OracleLocalDeltaRewriteModel,
    use_proposal_conditioning: bool,
    val_loader,
    test_loader,
    device: torch.device,
    node_thresholds: List[float],
    edge_thresholds: List[float],
) -> Dict[str, Any]:
    grid_results = []
    best_entry = None

    for node_threshold in node_thresholds:
        for edge_threshold in edge_thresholds:
            val_results = evaluate(
                proposal_model=proposal_model,
                rewrite_model=rewrite_model,
                loader=val_loader,
                device=device,
                node_threshold=node_threshold,
                edge_threshold=edge_threshold,
                use_proposal_conditioning=use_proposal_conditioning,
            )
            val_score = compute_selection_score(val_results)
            entry = {
                "node_threshold": node_threshold,
                "edge_threshold": edge_threshold,
                "val_selection_score": val_score,
                "val_results": val_results,
            }
            grid_results.append(entry)
            if best_entry is None or val_score > best_entry["val_selection_score"]:
                best_entry = entry

    assert best_entry is not None
    best_test_results = evaluate(
        proposal_model=proposal_model,
        rewrite_model=rewrite_model,
        loader=test_loader,
        device=device,
        node_threshold=best_entry["node_threshold"],
        edge_threshold=best_entry["edge_threshold"],
        use_proposal_conditioning=use_proposal_conditioning,
    )

    default_test_results = evaluate(
        proposal_model=proposal_model,
        rewrite_model=rewrite_model,
        loader=test_loader,
        device=device,
        node_threshold=0.20,
        edge_threshold=0.15,
        use_proposal_conditioning=use_proposal_conditioning,
    )

    return {
        "grid_results": grid_results,
        "best_val": best_entry,
        "best_test_results": best_test_results,
        "default_test_results": default_test_results,
        "default_test_selection_score": compute_selection_score(default_test_results),
        "best_test_selection_score": compute_selection_score(best_test_results),
    }


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
    w012_model, w012_use_conditioning = load_rewrite_model(w012_checkpoint_path, device)

    payload: Dict[str, Any] = {
        "metadata": {
            "proposal_checkpoint_path": str(proposal_checkpoint_path),
            "val_data_path": str(val_data_path),
            "test_data_path": str(test_data_path),
            "selection_mode": "validation_based",
            "selection_formula": "0.35*full_edge_acc + 0.35*context_edge_acc + 0.15*changed_edge_acc + 0.15*delete",
            "node_threshold_grid": node_thresholds,
            "edge_threshold_grid": edge_thresholds,
        },
        "rewrites": {},
    }

    print(f"device: {device}")
    print(f"proposal checkpoint: {proposal_checkpoint_path}")
    print(f"val data: {val_data_path}")
    print(f"test data: {test_data_path}")
    print(f"node thresholds: {node_thresholds}")
    print(f"edge thresholds: {edge_thresholds}")

    payload["rewrites"]["W012"] = evaluate_grid_for_rewrite(
        proposal_model=proposal_model,
        rewrite_model=w012_model,
        use_proposal_conditioning=w012_use_conditioning,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        node_thresholds=node_thresholds,
        edge_thresholds=edge_thresholds,
    )
    best_w012 = payload["rewrites"]["W012"]["best_val"]
    print(
        f"W012 best on val: node_threshold={best_w012['node_threshold']:.2f} "
        f"edge_threshold={best_w012['edge_threshold']:.2f} "
        f"score={best_w012['val_selection_score']:.6f}"
    )

    if i1520_checkpoint_path is not None and i1520_checkpoint_path.exists():
        i1520_model, i1520_use_conditioning = load_rewrite_model(i1520_checkpoint_path, device)
        payload["rewrites"]["I1520"] = evaluate_grid_for_rewrite(
            proposal_model=proposal_model,
            rewrite_model=i1520_model,
            use_proposal_conditioning=i1520_use_conditioning,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            node_thresholds=node_thresholds,
            edge_thresholds=edge_thresholds,
        )
        best_i1520 = payload["rewrites"]["I1520"]["best_val"]
        print(
            f"I1520 best on val: node_threshold={best_i1520['node_threshold']:.2f} "
            f"edge_threshold={best_i1520['edge_threshold']:.2f} "
            f"score={best_i1520['val_selection_score']:.6f}"
        )
    else:
        payload["rewrites"]["I1520"] = None
        print("I1520 checkpoint not available; skipped")

    out_path = proposal_checkpoint_path.parent / "noisy_threshold_calibration.json"
    save_json(out_path, payload)
    print(f"saved calibration json: {out_path}")


if __name__ == "__main__":
    main()
