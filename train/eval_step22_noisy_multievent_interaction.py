from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import edge_pairs_to_dense_mask
from models.oracle_local_delta import (
    EDGE_DELTA_ADD,
    EDGE_DELTA_DELETE,
    EDGE_DELTA_KEEP,
    OracleLocalDeltaRewriteConfig,
    OracleLocalDeltaRewriteModel,
    build_edge_delta_targets,
    build_valid_edge_mask,
)
from models.proposal import ScopeProposalConfig, ScopeProposalModel
from train.eval_rollout_stability import accuracy_from_mask, graph_to_tensors, mae_from_mask


CORE_METRICS = [
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
]


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def safe_div(num: float, den: float) -> Optional[float]:
    if den <= 0:
        return None
    return num / den


def load_samples(path: Path) -> list[Dict[str, Any]]:
    with open(path, "rb") as f:
        samples = pickle.load(f)
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"Step 22 dataset is empty or malformed: {path}")
    return samples


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def get_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if has_mps:
        return torch.device("mps")
    return torch.device("cpu")


def inspect_dataset(samples: list[Dict[str, Any]], observation_regime: str) -> Dict[str, Any]:
    bucket_counts = Counter(str(sample.get("step5_dependency_bucket", "unknown")) for sample in samples)
    setting_counts = Counter(str(sample.get("step22_corruption_setting", "clean")) for sample in samples)
    horizon_counts = Counter(int(sample.get("horizon", len(sample.get("graph_steps", [])))) for sample in samples)
    signature_counts = Counter(str(sample.get("step5_ordered_signature", "unknown")) for sample in samples)
    return {
        "sample_count": len(samples),
        "observation_regime": observation_regime,
        "horizon_distribution": {str(k): int(v) for k, v in sorted(horizon_counts.items())},
        "dependency_bucket_counts": dict(sorted(bucket_counts.items())),
        "corruption_setting_counts": dict(sorted(setting_counts.items())),
        "top_ordered_signatures": [
            {"ordered_signature": signature, "count": int(count)}
            for signature, count in signature_counts.most_common(12)
        ],
        "dataset_keys": sorted(samples[0].keys()) if samples else [],
    }


def current_clean_graph(sample: Dict[str, Any], step_idx: int) -> Dict[str, Any]:
    if step_idx == 0:
        return sample["graph_0"]
    return sample["graph_steps"][step_idx - 1]


def current_observed_graph(sample: Dict[str, Any], step_idx: int, observation_regime: str) -> Dict[str, Any]:
    if observation_regime == "clean":
        return current_clean_graph(sample, step_idx)
    obs_inputs = sample.get("obs_graph_inputs")
    if obs_inputs is None:
        raise KeyError("Noisy observation_regime requires `obs_graph_inputs` in Step 22 samples.")
    return obs_inputs[step_idx]


def event_scope_masks(sample: Dict[str, Any], step_idx: int, num_nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
    transition_samples = sample.get("transition_samples")
    if transition_samples:
        transition = transition_samples[step_idx]
        node_scope = torch.zeros(num_nodes, dtype=torch.bool)
        for node_idx in transition.get("event_scope_union_nodes", []):
            node_scope[int(node_idx)] = True
        edge_scope = edge_pairs_to_dense_mask(
            transition.get("event_scope_union_edges", []),
            num_nodes=num_nodes,
            undirected=True,
            dtype=torch.float32,
        ).bool()
        return node_scope, edge_scope

    event = sample["events"][step_idx]
    node_scope = torch.zeros(num_nodes, dtype=torch.bool)
    for node_idx in event.get("node_scope", []):
        node_scope[int(node_idx)] = True
    edge_scope = edge_pairs_to_dense_mask(
        event.get("edge_scope", []),
        num_nodes=num_nodes,
        undirected=True,
        dtype=torch.float32,
    ).bool()
    return node_scope, edge_scope


def binary_scope_counts(
    pred_mask: torch.Tensor,
    true_mask: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Dict[str, float]:
    pred = pred_mask.bool() & valid_mask.bool()
    true = true_mask.bool() & valid_mask.bool()
    tp = (pred & true).float().sum().item()
    pred_pos = pred.float().sum().item()
    true_pos = true.float().sum().item()
    excess = (pred & (~true)).float().sum().item()
    return {"tp": tp, "pred_pos": pred_pos, "true_pos": true_pos, "excess": excess}


def summarize_records(records: list[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"count": len(records)}
    for metric in CORE_METRICS:
        vals = [record[metric] for record in records if record.get(metric) is not None]
        summary[metric] = sum(vals) / len(vals) if vals else None

    node_tp = sum(float(record.get("proposal_node_scope_tp", 0.0)) for record in records)
    node_true = sum(float(record.get("proposal_node_scope_true_pos", 0.0)) for record in records)
    node_pred = sum(float(record.get("proposal_node_scope_pred_pos", 0.0)) for record in records)
    node_excess = sum(float(record.get("proposal_node_scope_excess", 0.0)) for record in records)
    edge_tp = sum(float(record.get("proposal_edge_scope_tp", 0.0)) for record in records)
    edge_true = sum(float(record.get("proposal_edge_scope_true_pos", 0.0)) for record in records)
    edge_pred = sum(float(record.get("proposal_edge_scope_pred_pos", 0.0)) for record in records)
    edge_excess = sum(float(record.get("proposal_edge_scope_excess", 0.0)) for record in records)
    changed_edge_tp = sum(float(record.get("proposal_changed_edge_tp", 0.0)) for record in records)
    changed_edge_true = sum(float(record.get("proposal_changed_edge_true_pos", 0.0)) for record in records)

    summary["proposal_node_scope_recall"] = safe_div(node_tp, node_true)
    summary["proposal_node_scope_precision"] = safe_div(node_tp, node_pred)
    summary["proposal_node_scope_excess_ratio"] = safe_div(node_excess, node_pred)
    summary["proposal_edge_scope_recall"] = safe_div(edge_tp, edge_true)
    summary["proposal_edge_scope_precision"] = safe_div(edge_tp, edge_pred)
    summary["proposal_edge_scope_excess_ratio"] = safe_div(edge_excess, edge_pred)
    summary["proposal_changed_edge_recall"] = safe_div(changed_edge_tp, changed_edge_true)
    changed_recall = summary["proposal_changed_edge_recall"]
    summary["proposal_out_of_scope_miss_edge"] = None if changed_recall is None else 1.0 - changed_recall
    summary["proposal_counts"] = {
        "node_tp": node_tp,
        "node_true_pos": node_true,
        "node_pred_pos": node_pred,
        "edge_tp": edge_tp,
        "edge_true_pos": edge_true,
        "edge_pred_pos": edge_pred,
        "changed_edge_tp": changed_edge_tp,
        "changed_edge_true_pos": changed_edge_true,
    }
    return summary


def format_grouped(records_by_group: Dict[str, list[Dict[str, Any]]]) -> Dict[str, Any]:
    return {group: summarize_records(records) for group, records in sorted(records_by_group.items())}


def evaluate_transition_with_observation(
    proposal_model: ScopeProposalModel,
    rewrite_model: OracleLocalDeltaRewriteModel,
    clean_current_graph: Dict[str, Any],
    observed_current_graph: Dict[str, Any],
    target_graph: Dict[str, Any],
    oracle_node_scope: torch.Tensor,
    oracle_edge_scope: torch.Tensor,
    device: torch.device,
    node_threshold: float,
    edge_threshold: float,
    use_proposal_conditioning: bool,
) -> Dict[str, Any]:
    input_node_feats, input_adj, node_mask = graph_to_tensors(observed_current_graph, device)
    clean_node_feats, clean_adj, _ = graph_to_tensors(clean_current_graph, device)
    target_node_feats, target_adj, _ = graph_to_tensors(target_graph, device)
    valid_edge_mask = build_valid_edge_mask(node_mask).bool()

    proposal_outputs = proposal_model(node_feats=input_node_feats, adj=input_adj)
    node_scope_logits = proposal_outputs.get("node_scope_logits", proposal_outputs["scope_logits"])
    proposal_node_probs = torch.sigmoid(node_scope_logits) * node_mask
    pred_scope_nodes = (proposal_node_probs >= node_threshold) & node_mask.bool()
    if "edge_scope_logits" in proposal_outputs:
        proposal_edge_probs = torch.sigmoid(proposal_outputs["edge_scope_logits"]) * valid_edge_mask.float()
        pred_scope_edges = (proposal_edge_probs >= edge_threshold) & valid_edge_mask
    else:
        proposal_edge_probs = proposal_node_probs.unsqueeze(2) * proposal_node_probs.unsqueeze(1) * valid_edge_mask.float()
        pred_scope_edges = pred_scope_nodes.unsqueeze(2) & pred_scope_nodes.unsqueeze(1) & valid_edge_mask

    rewrite_outputs = rewrite_model(
        node_feats=input_node_feats,
        adj=input_adj,
        scope_node_mask=pred_scope_nodes.float(),
        scope_edge_mask=pred_scope_edges.float(),
        proposal_node_probs=proposal_node_probs if use_proposal_conditioning else None,
        proposal_edge_probs=proposal_edge_probs if use_proposal_conditioning else None,
    )

    pred_type = rewrite_outputs["type_logits_full"].argmax(dim=-1)[0].detach().cpu()
    pred_state = rewrite_outputs["state_pred_full"][0].detach().cpu()
    pred_adj = (torch.sigmoid(rewrite_outputs["edge_logits_full"]) >= 0.5).float()
    pred_adj = ((pred_adj + pred_adj.transpose(1, 2)) > 0.5).float()
    diag_mask = torch.eye(pred_adj.shape[1], device=pred_adj.device, dtype=torch.bool).unsqueeze(0)
    pred_adj = pred_adj.masked_fill(diag_mask, 0.0)[0].detach().cpu()

    clean_node_feats_cpu = clean_node_feats[0].detach().cpu()
    clean_adj_cpu = clean_adj[0].detach().cpu()
    target_node_feats_cpu = target_node_feats[0].detach().cpu()
    target_adj_cpu = target_adj[0].detach().cpu()
    node_mask_cpu = node_mask[0].detach().cpu()
    valid_edge_cpu = valid_edge_mask[0].detach().cpu().bool()

    target_type = target_node_feats_cpu[:, 0].long()
    current_type = clean_node_feats_cpu[:, 0].long()
    target_state = target_node_feats_cpu[:, 1:]
    changed_nodes = torch.any((clean_node_feats_cpu - target_node_feats_cpu).abs() > 1e-6, dim=-1)
    flip_mask = current_type != target_type
    changed_edges = (clean_adj_cpu != target_adj_cpu) & valid_edge_cpu
    context_edges = (~changed_edges) & valid_edge_cpu
    target_adj_bool = target_adj_cpu > 0.5
    pred_adj_bool = pred_adj > 0.5
    target_delta = build_edge_delta_targets(clean_adj_cpu, target_adj_cpu)
    pred_delta = build_edge_delta_targets(clean_adj_cpu, pred_adj)

    pred_nodes_cpu = pred_scope_nodes[0].detach().cpu().bool()
    pred_edges_cpu = pred_scope_edges[0].detach().cpu().bool()
    oracle_node_scope = oracle_node_scope.bool() & node_mask_cpu.bool()
    oracle_edge_scope = oracle_edge_scope.bool() & valid_edge_cpu

    node_counts = binary_scope_counts(pred_nodes_cpu, oracle_node_scope, node_mask_cpu.bool())
    edge_counts = binary_scope_counts(pred_edges_cpu, oracle_edge_scope, valid_edge_cpu)
    changed_counts = binary_scope_counts(pred_edges_cpu, changed_edges, valid_edge_cpu)

    return {
        "full_type_acc": accuracy_from_mask(pred_type, target_type, node_mask_cpu),
        "full_state_mae": mae_from_mask(pred_state, target_state, node_mask_cpu),
        "full_edge_acc": accuracy_from_mask(pred_adj_bool, target_adj_bool, valid_edge_cpu),
        "changed_type_acc": accuracy_from_mask(pred_type, target_type, changed_nodes),
        "flip_acc": accuracy_from_mask(pred_type, target_type, flip_mask),
        "changed_edge_acc": accuracy_from_mask(pred_adj_bool, target_adj_bool, changed_edges),
        "context_edge_acc": accuracy_from_mask(pred_adj_bool, target_adj_bool, context_edges),
        "delta_all": accuracy_from_mask(pred_delta, target_delta, valid_edge_cpu),
        "keep": accuracy_from_mask(pred_delta, target_delta, valid_edge_cpu & (target_delta == EDGE_DELTA_KEEP)),
        "add": accuracy_from_mask(pred_delta, target_delta, valid_edge_cpu & (target_delta == EDGE_DELTA_ADD)),
        "delete": accuracy_from_mask(pred_delta, target_delta, valid_edge_cpu & (target_delta == EDGE_DELTA_DELETE)),
        "proposal_node_scope_tp": node_counts["tp"],
        "proposal_node_scope_pred_pos": node_counts["pred_pos"],
        "proposal_node_scope_true_pos": node_counts["true_pos"],
        "proposal_node_scope_excess": node_counts["excess"],
        "proposal_edge_scope_tp": edge_counts["tp"],
        "proposal_edge_scope_pred_pos": edge_counts["pred_pos"],
        "proposal_edge_scope_true_pos": edge_counts["true_pos"],
        "proposal_edge_scope_excess": edge_counts["excess"],
        "proposal_changed_edge_tp": changed_counts["tp"],
        "proposal_changed_edge_true_pos": changed_counts["true_pos"],
    }


@torch.no_grad()
def evaluate_step22(
    proposal_model: ScopeProposalModel,
    rewrite_model: OracleLocalDeltaRewriteModel,
    samples: list[Dict[str, Any]],
    device: torch.device,
    observation_regime: str,
    node_threshold: float,
    edge_threshold: float,
    use_proposal_conditioning: bool,
) -> Dict[str, Any]:
    all_step_records: list[Dict[str, Any]] = []
    final_records: list[Dict[str, Any]] = []
    by_step: dict[int, list[Dict[str, Any]]] = defaultdict(list)
    by_bucket_final: dict[str, list[Dict[str, Any]]] = defaultdict(list)
    by_bucket_all: dict[str, list[Dict[str, Any]]] = defaultdict(list)
    by_event_type: dict[str, list[Dict[str, Any]]] = defaultdict(list)
    by_corruption_final: dict[str, list[Dict[str, Any]]] = defaultdict(list)
    by_corruption_all: dict[str, list[Dict[str, Any]]] = defaultdict(list)

    for sample in samples:
        graph_steps = sample["graph_steps"]
        dependency_bucket = str(sample.get("step5_dependency_bucket", "unknown"))
        corruption_setting = str(sample.get("step22_corruption_setting", "clean"))
        for step_idx, target_graph in enumerate(graph_steps):
            clean_current = current_clean_graph(sample, step_idx)
            observed_current = current_observed_graph(sample, step_idx, observation_regime)
            num_nodes = len(clean_current["node_features"])
            oracle_node_scope, oracle_edge_scope = event_scope_masks(sample, step_idx, num_nodes)
            event = sample["events"][step_idx]
            event_type = str(event.get("event_type", "unknown"))
            record = evaluate_transition_with_observation(
                proposal_model=proposal_model,
                rewrite_model=rewrite_model,
                clean_current_graph=clean_current,
                observed_current_graph=observed_current,
                target_graph=target_graph,
                oracle_node_scope=oracle_node_scope,
                oracle_edge_scope=oracle_edge_scope,
                device=device,
                node_threshold=node_threshold,
                edge_threshold=edge_threshold,
                use_proposal_conditioning=use_proposal_conditioning,
            )
            record.update(
                {
                    "step_index": step_idx + 1,
                    "dependency_bucket": dependency_bucket,
                    "event_type": event_type,
                    "corruption_setting": corruption_setting,
                }
            )
            all_step_records.append(record)
            by_step[step_idx + 1].append(record)
            by_bucket_all[dependency_bucket].append(record)
            by_event_type[event_type].append(record)
            by_corruption_all[corruption_setting].append(record)
            if step_idx == len(graph_steps) - 1:
                final_records.append(record)
                by_bucket_final[dependency_bucket].append(record)
                by_corruption_final[corruption_setting].append(record)

    return {
        "overall_final": summarize_records(final_records),
        "overall_all_steps": summarize_records(all_step_records),
        "per_step_summary": [
            {"step_index": step_idx, **summarize_records(records)}
            for step_idx, records in sorted(by_step.items())
        ],
        "by_dependency_bucket_final": format_grouped(by_bucket_final),
        "by_dependency_bucket_all_steps": format_grouped(by_bucket_all),
        "by_event_type_all_steps": format_grouped(by_event_type),
        "by_corruption_setting_final": format_grouped(by_corruption_final),
        "by_corruption_setting_all_steps": format_grouped(by_corruption_all),
    }


def load_proposal_model(path: Path, device: torch.device) -> ScopeProposalModel:
    checkpoint = torch.load(path, map_location="cpu")
    model = ScopeProposalModel(ScopeProposalConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def load_rewrite_model(path: Path, device: torch.device) -> tuple[OracleLocalDeltaRewriteModel, bool]:
    checkpoint = torch.load(path, map_location="cpu")
    model = OracleLocalDeltaRewriteModel(
        OracleLocalDeltaRewriteConfig(**checkpoint["model_config"])
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    use_proposal_conditioning = bool(checkpoint.get("model_config", {}).get("use_proposal_conditioning", False))
    return model, use_proposal_conditioning


def write_summary_csv(path: Path, payload: Dict[str, Any]) -> None:
    rows = []
    results = payload["results"]
    sections = [
        ("overall_final", {"overall": results["overall_final"]}),
        ("overall_all_steps", {"overall": results["overall_all_steps"]}),
        ("by_dependency_bucket_final", results["by_dependency_bucket_final"]),
        ("by_dependency_bucket_all_steps", results["by_dependency_bucket_all_steps"]),
        ("by_event_type_all_steps", results["by_event_type_all_steps"]),
        ("by_corruption_setting_final", results["by_corruption_setting_final"]),
        ("by_corruption_setting_all_steps", results["by_corruption_setting_all_steps"]),
    ]
    for section, groups in sections:
        for group, metrics in groups.items():
            rows.append(
                {
                    "section": section,
                    "group": group,
                    "count": metrics.get("count"),
                    "full_edge": metrics.get("full_edge_acc"),
                    "context_edge": metrics.get("context_edge_acc"),
                    "changed_edge": metrics.get("changed_edge_acc"),
                    "add": metrics.get("add"),
                    "delete": metrics.get("delete"),
                    "full_type": metrics.get("full_type_acc"),
                    "state_mae": metrics.get("full_state_mae"),
                    "proposal_node_scope_recall": metrics.get("proposal_node_scope_recall"),
                    "proposal_edge_scope_recall": metrics.get("proposal_edge_scope_recall"),
                    "proposal_edge_scope_excess_ratio": metrics.get("proposal_edge_scope_excess_ratio"),
                    "proposal_out_of_scope_miss_edge": metrics.get("proposal_out_of_scope_miss_edge"),
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal_checkpoint_path", type=str, required=True)
    parser.add_argument("--rewrite_checkpoint_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--observation_regime", choices=["clean", "noisy"], required=True)
    parser.add_argument("--split_name", type=str, default="test")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--artifact_dir", type=str, default="artifacts/step22_noisy_multievent_interaction")
    parser.add_argument("--node_threshold", type=float, default=0.20)
    parser.add_argument("--edge_threshold", type=float, default=0.15)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    proposal_checkpoint_path = resolve_path(args.proposal_checkpoint_path)
    rewrite_checkpoint_path = resolve_path(args.rewrite_checkpoint_path)
    data_path = resolve_path(args.data_path)
    artifact_dir = resolve_path(args.artifact_dir)
    device = get_device(args.device)

    samples = load_samples(data_path)
    dataset_support = inspect_dataset(samples, args.observation_regime)
    proposal_model = load_proposal_model(proposal_checkpoint_path, device)
    rewrite_model, use_proposal_conditioning = load_rewrite_model(rewrite_checkpoint_path, device)
    results = evaluate_step22(
        proposal_model=proposal_model,
        rewrite_model=rewrite_model,
        samples=samples,
        device=device,
        observation_regime=args.observation_regime,
        node_threshold=args.node_threshold,
        edge_threshold=args.edge_threshold,
        use_proposal_conditioning=use_proposal_conditioning,
    )

    run_name = args.run_name or f"{args.split_name}_{args.observation_regime}_{rewrite_checkpoint_path.parent.name}"
    out_path = artifact_dir / f"{run_name}.json"
    csv_path = artifact_dir / f"{run_name}.csv"
    payload = {
        "metadata": {
            "proposal_checkpoint_path": str(proposal_checkpoint_path),
            "rewrite_checkpoint_path": str(rewrite_checkpoint_path),
            "data_path": str(data_path),
            "split_name": args.split_name,
            "observation_regime": args.observation_regime,
            "node_threshold": args.node_threshold,
            "edge_threshold": args.edge_threshold,
            "rewrite_uses_proposal_conditioning": use_proposal_conditioning,
            "evaluation_mode": "step22_observed_current_state_per_transition",
            "model_input": "clean current graph for observation_regime=clean; obs_graph_inputs[k] for observation_regime=noisy",
            "targets": "clean Step 5 graph_steps[k]",
            "note": "This isolates observation_regime on the Step 5 multievent substrate; it is not an autoregressive feedback benchmark.",
        },
        "dataset_support": dataset_support,
        "results": results,
    }
    save_json(out_path, payload)
    write_summary_csv(csv_path, payload)
    compact = {
        "overall_final": results["overall_final"],
        "overall_all_steps": results["overall_all_steps"],
        "by_dependency_bucket_final": results["by_dependency_bucket_final"],
        "by_event_type_all_steps": results["by_event_type_all_steps"],
    }
    print(f"device: {device}")
    print(f"saved json: {out_path}")
    print(f"saved csv: {csv_path}")
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
