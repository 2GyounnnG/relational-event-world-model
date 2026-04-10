from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.collate import graph_event_collate_fn
from data.dataset import GraphEventDataset
from models.oracle_local_delta import OracleLocalDeltaRewriteConfig, OracleLocalDeltaRewriteModel
from models.proposal import ScopeProposalConfig, ScopeProposalModel
from train.eval_noisy_structured_observation import evaluate, get_device, resolve_path


CORRUPTION_SETTINGS = ("N1", "N2", "N3")


def parse_threshold_list(raw: str) -> List[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def compute_selection_score(overall: Dict[str, Any]) -> float:
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


def build_loader_from_indices(
    dataset: GraphEventDataset,
    indices: List[int],
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=graph_event_collate_fn,
        pin_memory=pin_memory,
    )


def collect_setting_indices(dataset: GraphEventDataset) -> Dict[str, List[int]]:
    setting_to_indices: Dict[str, List[int]] = {setting: [] for setting in CORRUPTION_SETTINGS}
    for idx, sample in enumerate(dataset.samples):
        setting = sample.get("step6a_corruption_setting")
        if setting in setting_to_indices:
            setting_to_indices[str(setting)].append(idx)
    return setting_to_indices


def aggregate_overall(setting_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    metric_names = [
        "full_type_acc",
        "full_state_mae",
        "full_edge_acc",
        "changed_type_acc",
        "flip_acc",
        "changed_edge_acc",
        "context_edge_acc",
        "delta_all",
        "keep",
        "add",
        "delete",
        "proposal_node_precision",
        "proposal_node_recall",
        "proposal_node_f1",
        "proposal_edge_precision",
        "proposal_edge_recall",
        "proposal_edge_f1",
    ]
    total_count = sum(int(result["count"]) for result in setting_results.values())
    overall: Dict[str, Any] = {"count": total_count}
    for metric_name in metric_names:
        weighted_sum = 0.0
        weight_total = 0
        for result in setting_results.values():
            value = result.get(metric_name)
            count = int(result["count"])
            if value is None:
                continue
            weighted_sum += float(value) * count
            weight_total += count
        overall[metric_name] = (weighted_sum / weight_total) if weight_total > 0 else None
    return overall


def evaluate_default_thresholds_by_regime(
    proposal_model: ScopeProposalModel,
    rewrite_model: OracleLocalDeltaRewriteModel,
    use_proposal_conditioning: bool,
    test_loaders: Dict[str, DataLoader],
    device: torch.device,
) -> Dict[str, Any]:
    by_setting = {}
    for setting, loader in test_loaders.items():
        results = evaluate(
            proposal_model=proposal_model,
            rewrite_model=rewrite_model,
            loader=loader,
            device=device,
            node_threshold=0.20,
            edge_threshold=0.15,
            use_proposal_conditioning=use_proposal_conditioning,
        )
        by_setting[setting] = results["overall"]
    return {
        "overall": aggregate_overall(by_setting),
        "by_corruption_setting": by_setting,
    }


def evaluate_by_regime_for_rewrite(
    proposal_model: ScopeProposalModel,
    rewrite_model: OracleLocalDeltaRewriteModel,
    use_proposal_conditioning: bool,
    val_loaders: Dict[str, DataLoader],
    test_loaders: Dict[str, DataLoader],
    device: torch.device,
    node_thresholds: List[float],
    edge_thresholds: List[float],
) -> Dict[str, Any]:
    regime_grid_results: Dict[str, List[Dict[str, Any]]] = {}
    best_by_regime: Dict[str, Dict[str, Any]] = {}
    test_by_regime: Dict[str, Dict[str, Any]] = {}

    for setting in CORRUPTION_SETTINGS:
        loader_val = val_loaders[setting]
        loader_test = test_loaders[setting]
        grid_results = []
        best_entry = None
        for node_threshold in node_thresholds:
            for edge_threshold in edge_thresholds:
                val_results = evaluate(
                    proposal_model=proposal_model,
                    rewrite_model=rewrite_model,
                    loader=loader_val,
                    device=device,
                    node_threshold=node_threshold,
                    edge_threshold=edge_threshold,
                    use_proposal_conditioning=use_proposal_conditioning,
                )
                overall = val_results["overall"]
                val_score = compute_selection_score(overall)
                entry = {
                    "node_threshold": node_threshold,
                    "edge_threshold": edge_threshold,
                    "val_selection_score": val_score,
                    "val_results": overall,
                }
                grid_results.append(entry)
                if best_entry is None or val_score > best_entry["val_selection_score"]:
                    best_entry = entry
        assert best_entry is not None
        regime_grid_results[setting] = grid_results
        best_by_regime[setting] = best_entry
        best_test_results = evaluate(
            proposal_model=proposal_model,
            rewrite_model=rewrite_model,
            loader=loader_test,
            device=device,
            node_threshold=float(best_entry["node_threshold"]),
            edge_threshold=float(best_entry["edge_threshold"]),
            use_proposal_conditioning=use_proposal_conditioning,
        )
        test_by_regime[setting] = best_test_results["overall"]

    aggregated_overall = aggregate_overall(test_by_regime)
    default_test_results = evaluate_default_thresholds_by_regime(
        proposal_model=proposal_model,
        rewrite_model=rewrite_model,
        use_proposal_conditioning=use_proposal_conditioning,
        test_loaders=test_loaders,
        device=device,
    )
    return {
        "grid_results_by_regime": regime_grid_results,
        "best_val_by_regime": best_by_regime,
        "best_test_results": {
            "overall": aggregated_overall,
            "by_corruption_setting": test_by_regime,
        },
        "default_test_results": default_test_results,
        "best_test_selection_score": compute_selection_score(aggregated_overall),
        "default_test_selection_score": compute_selection_score(default_test_results["overall"]),
    }


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def build_setting_loaders(
    data_path: Path,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple[GraphEventDataset, Dict[str, DataLoader], Dict[str, int]]:
    dataset = GraphEventDataset(str(data_path))
    setting_indices = collect_setting_indices(dataset)
    loaders = {
        setting: build_loader_from_indices(
            dataset=dataset,
            indices=indices,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        for setting, indices in setting_indices.items()
    }
    counts = {setting: len(indices) for setting, indices in setting_indices.items()}
    return dataset, loaders, counts


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

    _, val_loaders, val_counts = build_setting_loaders(
        data_path=val_data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    _, test_loaders, test_counts = build_setting_loaders(
        data_path=test_data_path,
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
            "selection_mode": "validation_based_by_corruption_regime",
            "selection_formula": "0.35*full_edge_acc + 0.35*context_edge_acc + 0.15*changed_edge_acc + 0.15*delete",
            "node_threshold_grid": node_thresholds,
            "edge_threshold_grid": edge_thresholds,
            "val_counts_by_regime": val_counts,
            "test_counts_by_regime": test_counts,
        },
        "rewrites": {},
    }

    print(f"device: {device}")
    print(f"proposal checkpoint: {proposal_checkpoint_path}")
    print(f"val data: {val_data_path}")
    print(f"test data: {test_data_path}")
    print(f"node thresholds: {node_thresholds}")
    print(f"edge thresholds: {edge_thresholds}")
    print(f"val counts by regime: {val_counts}")
    print(f"test counts by regime: {test_counts}")

    payload["rewrites"]["W012"] = evaluate_by_regime_for_rewrite(
        proposal_model=proposal_model,
        rewrite_model=w012_model,
        use_proposal_conditioning=w012_use_conditioning,
        val_loaders=val_loaders,
        test_loaders=test_loaders,
        device=device,
        node_thresholds=node_thresholds,
        edge_thresholds=edge_thresholds,
    )
    print("W012 best by regime:")
    for setting in CORRUPTION_SETTINGS:
        best = payload["rewrites"]["W012"]["best_val_by_regime"][setting]
        print(
            f"  {setting}: node_threshold={best['node_threshold']:.2f} "
            f"edge_threshold={best['edge_threshold']:.2f} "
            f"score={best['val_selection_score']:.6f}"
        )

    if i1520_checkpoint_path is not None and i1520_checkpoint_path.exists():
        i1520_model, i1520_use_conditioning = load_rewrite_model(i1520_checkpoint_path, device)
        payload["rewrites"]["I1520"] = evaluate_by_regime_for_rewrite(
            proposal_model=proposal_model,
            rewrite_model=i1520_model,
            use_proposal_conditioning=i1520_use_conditioning,
            val_loaders=val_loaders,
            test_loaders=test_loaders,
            device=device,
            node_thresholds=node_thresholds,
            edge_thresholds=edge_thresholds,
        )
        print("I1520 best by regime:")
        for setting in CORRUPTION_SETTINGS:
            best = payload["rewrites"]["I1520"]["best_val_by_regime"][setting]
            print(
                f"  {setting}: node_threshold={best['node_threshold']:.2f} "
                f"edge_threshold={best['edge_threshold']:.2f} "
                f"score={best['val_selection_score']:.6f}"
            )
    else:
        payload["rewrites"]["I1520"] = None
        print("I1520 checkpoint not available; skipped")

    out_path = proposal_checkpoint_path.parent / "noisy_threshold_calibration_by_regime.json"
    save_json(out_path, payload)
    print(f"saved calibration json: {out_path}")


if __name__ == "__main__":
    main()
