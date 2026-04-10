from __future__ import annotations

import argparse
from collections import defaultdict
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.oracle_local_delta import OracleLocalDeltaRewriteConfig, OracleLocalDeltaRewriteModel
from models.proposal import ScopeProposalConfig, ScopeProposalModel
from train.eval_rollout_stability import (
    evaluate_transition,
    get_device,
    graph_to_tensors,
    inspect_rollout_dataset,
    load_rollout_dataset,
    predict_next_graph,
    resolve_path,
    save_json,
    summarize_metric_records,
)


def inspect_step5_dataset(samples: list[Dict[str, Any]]) -> Dict[str, Any]:
    support = inspect_rollout_dataset(samples)
    bucket_counts: dict[str, int] = defaultdict(int)
    ordered_signature_counts: dict[str, int] = defaultdict(int)
    for sample in samples:
        bucket_counts[str(sample.get("step5_dependency_bucket", "unknown"))] += 1
        ordered_signature_counts[str(sample.get("step5_ordered_signature", "unknown"))] += 1
    support["dependency_bucket_counts"] = dict(sorted(bucket_counts.items()))
    support["ordered_signature_counts"] = dict(
        sorted(ordered_signature_counts.items(), key=lambda item: (-item[1], item[0]))
    )
    return support


@torch.no_grad()
def evaluate_step5_regime(
    proposal_model: ScopeProposalModel,
    rewrite_model: OracleLocalDeltaRewriteModel,
    samples: list[Dict[str, Any]],
    device: torch.device,
    node_threshold: float,
    edge_threshold: float,
    use_proposal_conditioning: bool,
) -> Dict[str, Any]:
    final_records_overall: list[Dict[str, Any]] = []
    per_step_records: dict[int, list[Dict[str, Any]]] = defaultdict(list)
    bucket_records: dict[str, list[Dict[str, Any]]] = defaultdict(list)
    ordered_signature_records: dict[str, list[Dict[str, Any]]] = defaultdict(list)

    for sample in samples:
        graph_0 = sample["graph_0"]
        graph_steps = sample["graph_steps"]
        dependency_bucket = str(sample.get("step5_dependency_bucket", "unknown"))
        ordered_signature = str(sample.get("step5_ordered_signature", "unknown"))

        current_pred_node_feats, current_pred_adj, _ = graph_to_tensors(graph_0, device)
        current_gt_graph = graph_0
        final_metrics: Dict[str, Any] | None = None

        for step_idx, target_gt_graph in enumerate(graph_steps, start=1):
            pred_next_node_feats, pred_next_adj = predict_next_graph(
                proposal_model=proposal_model,
                rewrite_model=rewrite_model,
                current_node_feats=current_pred_node_feats,
                current_adj=current_pred_adj,
                node_threshold=node_threshold,
                edge_threshold=edge_threshold,
                use_proposal_conditioning=use_proposal_conditioning,
            )

            metrics = evaluate_transition(
                current_gt_graph=current_gt_graph,
                target_gt_graph=target_gt_graph,
                pred_next_node_feats=pred_next_node_feats[0].detach().cpu(),
                pred_next_adj=pred_next_adj[0].detach().cpu(),
            )
            per_step_records[step_idx].append(metrics)
            current_pred_node_feats = pred_next_node_feats
            current_pred_adj = pred_next_adj
            current_gt_graph = target_gt_graph
            final_metrics = metrics

        if final_metrics is None:
            continue
        enriched = dict(final_metrics)
        enriched["dependency_bucket"] = dependency_bucket
        enriched["ordered_signature"] = ordered_signature
        final_records_overall.append(enriched)
        bucket_records[dependency_bucket].append(enriched)
        ordered_signature_records[ordered_signature].append(enriched)

    overall_summary = summarize_metric_records(final_records_overall)
    per_step_summary = [
        {"step_index": step_idx, **summarize_metric_records(records)}
        for step_idx, records in sorted(per_step_records.items())
    ]
    dependency_bucket_summary = {
        bucket: summarize_metric_records(records)
        for bucket, records in sorted(bucket_records.items())
    }
    ordered_signature_summary = []
    for signature, records in ordered_signature_records.items():
        summary = summarize_metric_records(records)
        ordered_signature_summary.append(
            {
                "ordered_signature": signature,
                "count": len(records),
                "delta_all": summary.get("delta_all"),
                "changed": summary.get("changed_edge_acc"),
                "context": summary.get("context_edge_acc"),
                "flip": summary.get("flip_acc"),
            }
        )
    ordered_signature_summary.sort(key=lambda row: (-row["count"], row["ordered_signature"]))

    return {
        "overall_final": overall_summary,
        "per_step_summary": per_step_summary,
        "dependency_bucket_summary": dependency_bucket_summary,
        "ordered_signature_summary": ordered_signature_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal_checkpoint_path", type=str, required=True)
    parser.add_argument("--rewrite_checkpoint_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--split_name", type=str, default="eval")
    parser.add_argument("--node_threshold", type=float, default=0.20)
    parser.add_argument("--edge_threshold", type=float, default=0.15)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    proposal_checkpoint_path = resolve_path(args.proposal_checkpoint_path)
    rewrite_checkpoint_path = resolve_path(args.rewrite_checkpoint_path)
    data_path = resolve_path(args.data_path)
    device = get_device(args.device)

    samples = load_rollout_dataset(data_path)
    dataset_support = inspect_step5_dataset(samples)

    proposal_checkpoint = torch.load(proposal_checkpoint_path, map_location="cpu")
    proposal_model = ScopeProposalModel(ScopeProposalConfig(**proposal_checkpoint["model_config"])).to(device)
    proposal_model.load_state_dict(proposal_checkpoint["model_state_dict"])
    proposal_model.eval()

    rewrite_checkpoint = torch.load(rewrite_checkpoint_path, map_location="cpu")
    rewrite_model = OracleLocalDeltaRewriteModel(
        OracleLocalDeltaRewriteConfig(**rewrite_checkpoint["model_config"])
    ).to(device)
    rewrite_model.load_state_dict(rewrite_checkpoint["model_state_dict"])
    rewrite_model.eval()
    use_proposal_conditioning = bool(
        rewrite_checkpoint.get("model_config", {}).get("use_proposal_conditioning", False)
    )

    results = evaluate_step5_regime(
        proposal_model=proposal_model,
        rewrite_model=rewrite_model,
        samples=samples,
        device=device,
        node_threshold=args.node_threshold,
        edge_threshold=args.edge_threshold,
        use_proposal_conditioning=use_proposal_conditioning,
    )

    payload = {
        "metadata": {
            "proposal_checkpoint_path": str(proposal_checkpoint_path),
            "rewrite_checkpoint_path": str(rewrite_checkpoint_path),
            "data_path": str(data_path),
            "split_name": args.split_name,
            "node_threshold": args.node_threshold,
            "edge_threshold": args.edge_threshold,
            "rewrite_uses_proposal_conditioning": use_proposal_conditioning,
            "evaluation_mode": "autoregressive_3_event_structural_regime",
        },
        "dataset_support": dataset_support,
        "results": results,
    }

    out_path = rewrite_checkpoint_path.parent / f"{args.split_name}_step5_multievent_regime.json"
    save_json(out_path, payload)

    print(f"device: {device}")
    print(f"proposal checkpoint: {proposal_checkpoint_path}")
    print(f"rewrite checkpoint: {rewrite_checkpoint_path}")
    print(f"data: {data_path}")
    print(f"total Step 5 samples: {dataset_support['total_rollout_sample_count']}")
    print(f"dependency buckets: {dataset_support['dependency_bucket_counts']}")
    print(f"saved json: {out_path}")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
