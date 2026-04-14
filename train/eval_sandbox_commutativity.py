from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.sandbox_local_event_operator import SandboxLocalEventOperator  # noqa: E402
from models.sandbox_monolithic_baseline import SandboxMonolithicBaseline  # noqa: E402


MODEL_TYPES = ("local_operator", "monolithic_baseline")
PAIR_TYPES = ("impulse+impulse", "impulse+break", "break+break")


def resolve_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate sandbox independent-pair commutativity.")
    parser.add_argument("--model_type", choices=MODEL_TYPES, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pairs_path", default="artifacts/sandbox_local_event_pairs/pairs.pkl")
    parser.add_argument("--output_dir", default="artifacts/sandbox_commutativity_eval")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_pairs(path: Path) -> list[Dict[str, Any]]:
    with open(path, "rb") as f:
        pairs = pickle.load(f)
    if len(pairs) == 0:
        raise ValueError(f"no pairs found in {path}")
    return pairs


def build_model(model_type: str, checkpoint_path: Path, sample: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_model_type = checkpoint.get("model_type")
    if checkpoint_model_type != model_type:
        raise ValueError(
            f"checkpoint model_type {checkpoint_model_type!r} does not match requested {model_type!r}"
        )
    hidden_dim = int(checkpoint.get("args", {}).get("hidden_dim", 64))
    node_dim = int(sample["node_features_t"].shape[-1])
    edge_dim = int(sample["edge_features_t"].shape[-1])
    event_param_dim = int(sample["event_a"]["event_params"].shape[-1])
    if model_type == "local_operator":
        model = SandboxLocalEventOperator(
            node_feature_dim=node_dim,
            edge_feature_dim=edge_dim,
            event_param_dim=event_param_dim,
            hidden_dim=hidden_dim,
        )
    else:
        model = SandboxMonolithicBaseline(
            node_feature_dim=node_dim,
            edge_feature_dim=edge_dim,
            event_param_dim=event_param_dim,
            hidden_dim=hidden_dim,
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def tensor(x: Any, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(x, dtype=dtype, device=device)


def run_event(
    model: torch.nn.Module,
    model_type: str,
    node_features: torch.Tensor,
    edge_features: torch.Tensor,
    edge_index: torch.Tensor,
    event: Dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    device = node_features.device
    num_nodes = int(node_features.shape[0])
    num_edges = int(edge_features.shape[0])
    common = {
        "node_features_t": node_features,
        "edge_index": edge_index,
        "edge_features_t": edge_features,
        "event_type_id": tensor([event["event_type_id"]], device, torch.long),
        "event_params": tensor(event["event_params"], device, torch.float32).view(1, -1),
        "node_batch_index": torch.zeros(num_nodes, dtype=torch.long, device=device),
        "edge_batch_index": torch.zeros(num_edges, dtype=torch.long, device=device),
        "num_nodes_per_graph": torch.tensor([num_nodes], dtype=torch.long, device=device),
        "num_edges_per_graph": torch.tensor([num_edges], dtype=torch.long, device=device),
    }
    if model_type == "local_operator":
        outputs = model(
            **common,
            event_node_mask=tensor(event["event_node_mask"], device, torch.float32),
            event_edge_mask=tensor(event["event_edge_mask"], device, torch.float32),
            event_scope_node_mask=tensor(event["event_scope_node_mask"], device, torch.float32),
            event_scope_edge_mask=tensor(event["event_scope_edge_mask"], device, torch.float32),
        )
    else:
        outputs = model(**common)
    return outputs["node_features_pred"], outputs["edge_features_pred"]


def state_rmse(
    node_a: torch.Tensor,
    edge_a: torch.Tensor,
    node_b: torch.Tensor,
    edge_b: torch.Tensor,
) -> float:
    node_sse = (node_a - node_b).pow(2).sum()
    edge_sse = (edge_a - edge_b).pow(2).sum()
    count = node_a.numel() + edge_a.numel()
    return float(torch.sqrt((node_sse + edge_sse) / max(count, 1)).detach().cpu())


def make_bucket() -> Dict[str, float]:
    return defaultdict(float)


def update_bucket(bucket: Dict[str, float], mismatch: float, error_ab: float, error_ba: float, oracle_disc: float) -> None:
    bucket["num_pairs"] += 1.0
    bucket["prediction_mismatch_rmse_sum"] += mismatch
    bucket["target_error_ab_rmse_sum"] += error_ab
    bucket["target_error_ba_rmse_sum"] += error_ba
    bucket["oracle_ab_ba_discrepancy_sum"] += oracle_disc
    bucket["oracle_ab_ba_discrepancy_max"] = max(bucket["oracle_ab_ba_discrepancy_max"], oracle_disc)


def finalize_bucket(bucket: Dict[str, float]) -> Dict[str, float]:
    count = max(bucket["num_pairs"], 1.0)
    return {
        "num_pairs": int(bucket["num_pairs"]),
        "prediction_mismatch_rmse": bucket["prediction_mismatch_rmse_sum"] / count,
        "target_error_ab_rmse": bucket["target_error_ab_rmse_sum"] / count,
        "target_error_ba_rmse": bucket["target_error_ba_rmse_sum"] / count,
        "oracle_ab_ba_discrepancy_mean": bucket["oracle_ab_ba_discrepancy_sum"] / count,
        "oracle_ab_ba_discrepancy_max": bucket["oracle_ab_ba_discrepancy_max"],
    }


def evaluate(model: torch.nn.Module, model_type: str, pairs: list[Dict[str, Any]], device: torch.device) -> Dict[str, Any]:
    overall = make_bucket()
    by_type = {pair_type: make_bucket() for pair_type in PAIR_TYPES}
    with torch.no_grad():
        for sample in pairs:
            node_t = tensor(sample["node_features_t"], device, torch.float32)
            edge_t = tensor(sample["edge_features_t"], device, torch.float32)
            edge_index = tensor(sample["edge_index"], device, torch.long)

            node_a, edge_a = run_event(model, model_type, node_t, edge_t, edge_index, sample["event_a"])
            pred_ab_node, pred_ab_edge = run_event(model, model_type, node_a, edge_a, edge_index, sample["event_b"])
            node_b, edge_b = run_event(model, model_type, node_t, edge_t, edge_index, sample["event_b"])
            pred_ba_node, pred_ba_edge = run_event(model, model_type, node_b, edge_b, edge_index, sample["event_a"])

            target_ab_node = tensor(sample["ab_node_features_next"], device, torch.float32)
            target_ab_edge = tensor(sample["ab_edge_features_next"], device, torch.float32)
            target_ba_node = tensor(sample["ba_node_features_next"], device, torch.float32)
            target_ba_edge = tensor(sample["ba_edge_features_next"], device, torch.float32)

            mismatch = state_rmse(pred_ab_node, pred_ab_edge, pred_ba_node, pred_ba_edge)
            error_ab = state_rmse(pred_ab_node, pred_ab_edge, target_ab_node, target_ab_edge)
            error_ba = state_rmse(pred_ba_node, pred_ba_edge, target_ba_node, target_ba_edge)
            oracle_disc = float(sample["oracle_ab_ba_discrepancy"])
            update_bucket(overall, mismatch, error_ab, error_ba, oracle_disc)
            update_bucket(by_type[sample["pair_type"]], mismatch, error_ab, error_ba, oracle_disc)

    return {
        "overall": finalize_bucket(overall),
        "pair_type_breakdown": {pair_type: finalize_bucket(bucket) for pair_type, bucket in by_type.items()},
    }


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def save_csv(path: Path, breakdown: Dict[str, Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pair_type",
        "num_pairs",
        "prediction_mismatch_rmse",
        "target_error_ab_rmse",
        "target_error_ba_rmse",
        "oracle_ab_ba_discrepancy_mean",
        "oracle_ab_ba_discrepancy_max",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pair_type in PAIR_TYPES:
            row = {"pair_type": pair_type}
            row.update(breakdown[pair_type])
            writer.writerow(row)


def print_summary(results: Dict[str, Any]) -> None:
    overall = results["overall"]
    print("commutativity evaluation complete")
    print(
        f"overall: pairs={overall['num_pairs']} "
        f"pred_mismatch={overall['prediction_mismatch_rmse']:.6f} "
        f"target_ab={overall['target_error_ab_rmse']:.6f} "
        f"target_ba={overall['target_error_ba_rmse']:.6f}"
    )
    for pair_type, row in results["pair_type_breakdown"].items():
        print(
            f"{pair_type}: pairs={row['num_pairs']} "
            f"pred_mismatch={row['prediction_mismatch_rmse']:.6f} "
            f"target_ab={row['target_error_ab_rmse']:.6f} "
            f"target_ba={row['target_error_ba_rmse']:.6f}"
        )


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    pairs_path = resolve_path(args.pairs_path)
    checkpoint_path = resolve_path(args.checkpoint)
    output_dir = resolve_path(args.output_dir)

    pairs = load_pairs(pairs_path)
    model = build_model(args.model_type, checkpoint_path, pairs[0], device)
    results = evaluate(model, args.model_type, pairs, device)
    summary = {
        "model_type": args.model_type,
        "checkpoint": str(checkpoint_path),
        "pairs_path": str(pairs_path),
        **results,
    }

    summary_path = output_dir / f"{args.model_type}_commutativity_summary.json"
    breakdown_path = output_dir / f"{args.model_type}_pair_type_breakdown.csv"
    save_json(summary_path, summary)
    save_csv(breakdown_path, results["pair_type_breakdown"])
    print_summary(results)
    print(f"summary_json={summary_path}")
    print(f"pair_type_breakdown_csv={breakdown_path}")


if __name__ == "__main__":
    main()
