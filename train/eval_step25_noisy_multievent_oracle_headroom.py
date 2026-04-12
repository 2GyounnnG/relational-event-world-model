from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.oracle_local_delta import (
    EDGE_DELTA_ADD,
    EDGE_DELTA_DELETE,
    EDGE_DELTA_KEEP,
    OracleLocalDeltaRewriteModel,
    build_edge_delta_targets,
    build_valid_edge_mask,
)
from models.proposal import ScopeProposalModel
from train.eval_rollout_stability import accuracy_from_mask, graph_to_tensors, mae_from_mask
from train.eval_step22_noisy_multievent_interaction import (
    CORE_METRICS,
    binary_scope_counts,
    current_clean_graph,
    current_observed_graph,
    event_scope_masks,
    format_grouped,
    inspect_dataset,
    load_proposal_model,
    load_rewrite_model,
    load_samples,
    resolve_path,
    save_json,
    summarize_records,
    write_summary_csv as write_eval_csv,
    get_device,
)


SYSTEM_METRICS = ["full_edge_acc", "context_edge_acc", "changed_edge_acc", "add", "delete"]
PROPOSAL_METRICS = [
    "proposal_node_scope_recall",
    "proposal_edge_scope_recall",
    "proposal_edge_scope_excess_ratio",
    "proposal_out_of_scope_miss_edge",
]


def evaluate_transition_with_scope_source(
    proposal_model: Optional[ScopeProposalModel],
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
    scope_source: str,
) -> Dict[str, Any]:
    input_node_feats, input_adj, node_mask = graph_to_tensors(observed_current_graph, device)
    clean_node_feats, clean_adj, _ = graph_to_tensors(clean_current_graph, device)
    target_node_feats, target_adj, _ = graph_to_tensors(target_graph, device)
    valid_edge_mask = build_valid_edge_mask(node_mask).bool()

    oracle_node_scope_device = oracle_node_scope.to(device).bool().unsqueeze(0) & node_mask.bool()
    oracle_edge_scope_device = oracle_edge_scope.to(device).bool().unsqueeze(0) & valid_edge_mask

    if scope_source == "learned":
        if proposal_model is None:
            raise ValueError("scope_source=learned requires a proposal checkpoint/model.")
        proposal_outputs = proposal_model(node_feats=input_node_feats, adj=input_adj)
        node_scope_logits = proposal_outputs.get("node_scope_logits", proposal_outputs["scope_logits"])
        proposal_node_probs = torch.sigmoid(node_scope_logits) * node_mask
        pred_scope_nodes = (proposal_node_probs >= node_threshold) & node_mask.bool()
        if "edge_scope_logits" in proposal_outputs:
            proposal_edge_probs = torch.sigmoid(proposal_outputs["edge_scope_logits"]) * valid_edge_mask.float()
            pred_scope_edges = (proposal_edge_probs >= edge_threshold) & valid_edge_mask
        else:
            proposal_edge_probs = (
                proposal_node_probs.unsqueeze(2) * proposal_node_probs.unsqueeze(1) * valid_edge_mask.float()
            )
            pred_scope_edges = pred_scope_nodes.unsqueeze(2) & pred_scope_nodes.unsqueeze(1) & valid_edge_mask
    elif scope_source == "oracle":
        # Oracle mode probes event-scope headroom. When rewrite checkpoints use proposal conditioning,
        # we feed oracle scope probabilities too, so the whole proposal/rewrite interface sees oracle scope.
        pred_scope_nodes = oracle_node_scope_device
        pred_scope_edges = oracle_edge_scope_device
        proposal_node_probs = pred_scope_nodes.float() * node_mask.float()
        proposal_edge_probs = pred_scope_edges.float() * valid_edge_mask.float()
    else:
        raise ValueError(f"Unsupported scope_source: {scope_source}")

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
    oracle_node_scope_cpu = oracle_node_scope.bool() & node_mask_cpu.bool()
    oracle_edge_scope_cpu = oracle_edge_scope.bool() & valid_edge_cpu

    node_counts = binary_scope_counts(pred_nodes_cpu, oracle_node_scope_cpu, node_mask_cpu.bool())
    edge_counts = binary_scope_counts(pred_edges_cpu, oracle_edge_scope_cpu, valid_edge_cpu)
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
def evaluate_step25(
    proposal_model: Optional[ScopeProposalModel],
    rewrite_model: OracleLocalDeltaRewriteModel,
    samples: list[Dict[str, Any]],
    device: torch.device,
    observation_regime: str,
    node_threshold: float,
    edge_threshold: float,
    use_proposal_conditioning: bool,
    scope_source: str,
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
            event_type = str(sample["events"][step_idx].get("event_type", "unknown"))
            record = evaluate_transition_with_scope_source(
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
                scope_source=scope_source,
            )
            record.update(
                {
                    "step_index": step_idx + 1,
                    "dependency_bucket": dependency_bucket,
                    "event_type": event_type,
                    "corruption_setting": corruption_setting,
                    "scope_source": scope_source,
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


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def metric_delta(candidate: Optional[float], baseline: Optional[float]) -> Optional[float]:
    if candidate is None or baseline is None:
        return None
    return candidate - baseline


def compare_groups(candidate: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Optional[float]]:
    return {
        metric: metric_delta(candidate.get(metric), baseline.get(metric))
        for metric in SYSTEM_METRICS + PROPOSAL_METRICS
    }


def build_summary(artifact_dir: Path) -> Dict[str, Any]:
    run_paths = {
        "learned_noisy_p2_rft1": PROJECT_ROOT / "artifacts/step22_noisy_multievent_interaction/noisy_p2_rft1.json",
        "learned_step24_joint": PROJECT_ROOT / "artifacts/step24_noisy_interaction_joint/noisy_step24_joint.json",
        "oracle_rft1": artifact_dir / "oracle_scope_rft1.json",
        "oracle_step24_rewrite": artifact_dir / "oracle_scope_step24_rewrite.json",
        "oracle_w012_optional": artifact_dir / "oracle_scope_w012.json",
        "noisy_p2_i1520_optional": PROJECT_ROOT / "artifacts/step22_noisy_multievent_interaction/noisy_p2_i1520.json",
    }
    runs: Dict[str, Dict[str, Any]] = {}
    missing: Dict[str, str] = {}
    for name, path in run_paths.items():
        payload = load_json(path)
        if payload is None:
            missing[name] = str(path)
        else:
            runs[name] = payload

    comparisons: Dict[str, Any] = {}
    for oracle_name, learned_name in (
        ("oracle_rft1", "learned_noisy_p2_rft1"),
        ("oracle_step24_rewrite", "learned_step24_joint"),
    ):
        if oracle_name in runs and learned_name in runs:
            comparisons[f"{oracle_name}_minus_{learned_name}"] = {
                "overall_final": compare_groups(
                    runs[oracle_name]["results"]["overall_final"],
                    runs[learned_name]["results"]["overall_final"],
                ),
                "strongly_interacting_final": compare_groups(
                    runs[oracle_name]["results"]["by_dependency_bucket_final"].get("strongly_interacting", {}),
                    runs[learned_name]["results"]["by_dependency_bucket_final"].get("strongly_interacting", {}),
                ),
                "fully_independent_final": compare_groups(
                    runs[oracle_name]["results"]["by_dependency_bucket_final"].get("fully_independent", {}),
                    runs[learned_name]["results"]["by_dependency_bucket_final"].get("fully_independent", {}),
                ),
                "partially_dependent_final": compare_groups(
                    runs[oracle_name]["results"]["by_dependency_bucket_final"].get("partially_dependent", {}),
                    runs[learned_name]["results"]["by_dependency_bucket_final"].get("partially_dependent", {}),
                ),
            }
    if "oracle_step24_rewrite" in runs and "oracle_rft1" in runs:
        comparisons["oracle_step24_rewrite_minus_oracle_rft1"] = {
            "overall_final": compare_groups(
                runs["oracle_step24_rewrite"]["results"]["overall_final"],
                runs["oracle_rft1"]["results"]["overall_final"],
            ),
            "strongly_interacting_final": compare_groups(
                runs["oracle_step24_rewrite"]["results"]["by_dependency_bucket_final"].get("strongly_interacting", {}),
                runs["oracle_rft1"]["results"]["by_dependency_bucket_final"].get("strongly_interacting", {}),
            ),
        }

    return {
        "metadata": {
            "evaluation": "Step 25 noisy multievent oracle-scope headroom",
            "single_variable": "scope_source = learned | oracle",
            "fixed_thresholds": {"node_threshold": 0.15, "edge_threshold": 0.10},
            "note": "Oracle rows use GT event scope as hard masks and as proposal-conditioning probabilities when rewrite uses proposal conditioning.",
        },
        "run_paths": {name: str(path) for name, path in run_paths.items()},
        "missing_runs": missing,
        "runs": {
            name: {
                "metadata": payload.get("metadata", {}),
                "overall_final": payload["results"]["overall_final"],
                "by_dependency_bucket_final": payload["results"].get("by_dependency_bucket_final", {}),
                "by_corruption_setting_final": payload["results"].get("by_corruption_setting_final", {}),
                "by_event_type_all_steps": payload["results"].get("by_event_type_all_steps", {}),
            }
            for name, payload in sorted(runs.items())
        },
        "comparisons": comparisons,
    }


def write_consolidated_summary_csv(path: Path, summary: Dict[str, Any]) -> None:
    rows = []
    for run_name, run in summary["runs"].items():
        sections = [
            ("overall_final", {"overall": run["overall_final"]}),
            ("by_dependency_bucket_final", run["by_dependency_bucket_final"]),
            ("by_corruption_setting_final", run["by_corruption_setting_final"]),
            ("by_event_type_all_steps", run["by_event_type_all_steps"]),
        ]
        for section, groups in sections:
            for group, metrics in groups.items():
                row = {"run": run_name, "section": section, "group": group, "count": metrics.get("count")}
                for metric in SYSTEM_METRICS + PROPOSAL_METRICS:
                    row[metric] = metrics.get(metric)
                rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["run", "section", "group", "count"] + SYSTEM_METRICS + PROPOSAL_METRICS
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rewrite_checkpoint_path", type=str, help="Rewrite checkpoint to evaluate.")
    parser.add_argument("--proposal_checkpoint_path", type=str, default="", help="Required only for scope_source=learned.")
    parser.add_argument("--data_path", type=str, default="data/graph_event_step22_noisy_step5_test.pkl")
    parser.add_argument("--observation_regime", choices=["clean", "noisy"], default="noisy")
    parser.add_argument("--scope_source", choices=["learned", "oracle"], default="oracle")
    parser.add_argument("--split_name", type=str, default="test")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--artifact_dir", type=str, default="artifacts/step25_noisy_multievent_oracle_headroom")
    parser.add_argument("--node_threshold", type=float, default=0.15)
    parser.add_argument("--edge_threshold", type=float, default=0.10)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--write_consolidated_summary", action="store_true")
    args = parser.parse_args()

    artifact_dir = resolve_path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if args.write_consolidated_summary:
        summary = build_summary(artifact_dir)
        out_json = artifact_dir / "summary.json"
        out_csv = artifact_dir / "summary.csv"
        save_json(out_json, summary)
        write_consolidated_summary_csv(out_csv, summary)
        print(f"saved summary json: {out_json}")
        print(f"saved summary csv: {out_csv}")
        if summary["missing_runs"]:
            print(f"missing optional/expected runs: {summary['missing_runs']}")
        print(json.dumps(summary["comparisons"], indent=2, ensure_ascii=False))
        return

    if not args.rewrite_checkpoint_path:
        raise ValueError("--rewrite_checkpoint_path is required unless --write_consolidated_summary is set.")

    device = get_device(args.device)
    data_path = resolve_path(args.data_path)
    rewrite_checkpoint_path = resolve_path(args.rewrite_checkpoint_path)
    proposal_checkpoint_path = resolve_path(args.proposal_checkpoint_path) if args.proposal_checkpoint_path else None

    samples = load_samples(data_path)
    dataset_support = inspect_dataset(samples, args.observation_regime)
    rewrite_model, use_proposal_conditioning = load_rewrite_model(rewrite_checkpoint_path, device)
    proposal_model = None
    if args.scope_source == "learned":
        if proposal_checkpoint_path is None:
            raise ValueError("scope_source=learned requires --proposal_checkpoint_path.")
        proposal_model = load_proposal_model(proposal_checkpoint_path, device)

    results = evaluate_step25(
        proposal_model=proposal_model,
        rewrite_model=rewrite_model,
        samples=samples,
        device=device,
        observation_regime=args.observation_regime,
        node_threshold=args.node_threshold,
        edge_threshold=args.edge_threshold,
        use_proposal_conditioning=use_proposal_conditioning,
        scope_source=args.scope_source,
    )

    default_run = f"{args.scope_source}_scope_{rewrite_checkpoint_path.parent.name}"
    run_name = args.run_name or default_run
    out_path = artifact_dir / f"{run_name}.json"
    csv_path = artifact_dir / f"{run_name}.csv"
    payload = {
        "metadata": {
            "proposal_checkpoint_path": str(proposal_checkpoint_path) if proposal_checkpoint_path else None,
            "rewrite_checkpoint_path": str(rewrite_checkpoint_path),
            "data_path": str(data_path),
            "split_name": args.split_name,
            "observation_regime": args.observation_regime,
            "scope_source": args.scope_source,
            "node_threshold": args.node_threshold,
            "edge_threshold": args.edge_threshold,
            "rewrite_uses_proposal_conditioning": use_proposal_conditioning,
            "evaluation_mode": "step25_noisy_multievent_oracle_headroom",
            "model_input": "obs_graph_inputs[k] for noisy observation; clean graph is used only for target/delta analysis.",
            "targets": "clean Step 5 graph_steps[k]",
            "oracle_scope_conditioning_note": (
                "For scope_source=oracle, GT event-scope masks are also passed as proposal probabilities "
                "when the rewrite checkpoint uses proposal conditioning."
            ),
        },
        "dataset_support": dataset_support,
        "results": results,
    }
    save_json(out_path, payload)
    write_eval_csv(csv_path, payload)
    compact = {
        "overall_final": results["overall_final"],
        "by_dependency_bucket_final": results["by_dependency_bucket_final"],
        "by_event_type_all_steps": results["by_event_type_all_steps"],
    }
    print(f"device: {device}")
    print(f"saved json: {out_path}")
    print(f"saved csv: {csv_path}")
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
