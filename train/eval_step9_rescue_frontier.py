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

from train.eval_edge_scope_miss_anatomy import make_group_names
from train.eval_noisy_structured_observation import (
    build_loader,
    get_device,
    move_batch_to_device,
    require_keys,
    resolve_path,
    save_json,
)
from train.eval_scope_edit_localization_gap import safe_div, slugify
from train.eval_step9_gated_edge_completion import (
    EDGE_COMPLETION_LEARNED,
    EDGE_COMPLETION_NAIVE,
    EDGE_COMPLETION_OFF,
    apply_edge_completion_mode,
    build_downstream_sample_stats,
    build_proposal_sample_stats,
    finalize_downstream_bucket,
    finalize_proposal_bucket,
    get_base_proposal_outputs,
    init_downstream_bucket,
    init_proposal_bucket,
    load_completion_model,
    load_proposal_model,
    load_rewrite_model,
    update_bucket,
)


BUDGET_FRACTIONS = (0.02, 0.05, 0.10, 0.20)
DEFAULT_MODE = "learned_default_threshold_0.5"
NAIVE_MODE = EDGE_COMPLETION_NAIVE
BASE_MODE = EDGE_COMPLETION_OFF


def budget_mode_name(fraction: float) -> str:
    return f"learned_budget_{fraction:.2f}"


def init_score_bucket() -> Dict[str, list[torch.Tensor]]:
    return {
        "scores": [],
        "oracle_scope_labels": [],
        "changed_edge_labels": [],
    }


def append_score_bucket(
    bucket: Dict[str, list[torch.Tensor]],
    scores: torch.Tensor,
    oracle_labels: torch.Tensor,
    changed_labels: torch.Tensor,
) -> None:
    if scores.numel() <= 0:
        return
    bucket["scores"].append(scores.detach().float().cpu())
    bucket["oracle_scope_labels"].append(oracle_labels.detach().bool().cpu())
    bucket["changed_edge_labels"].append(changed_labels.detach().bool().cpu())


def tensor_quantiles(values: torch.Tensor) -> Dict[str, Optional[float]]:
    if values.numel() <= 0:
        return {q: None for q in ["p01", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "p99"]}
    probs = torch.tensor([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    qs = torch.quantile(values.float(), probs)
    return {
        "p01": qs[0].item(),
        "p05": qs[1].item(),
        "p10": qs[2].item(),
        "p25": qs[3].item(),
        "p50": qs[4].item(),
        "p75": qs[5].item(),
        "p90": qs[6].item(),
        "p95": qs[7].item(),
        "p99": qs[8].item(),
    }


def average_precision(scores: torch.Tensor, labels: torch.Tensor) -> Optional[float]:
    labels = labels.bool()
    pos_total = labels.float().sum().item()
    if pos_total <= 0:
        return None
    order = torch.argsort(scores, descending=True)
    sorted_labels = labels[order].float()
    cum_tp = torch.cumsum(sorted_labels, dim=0)
    ranks = torch.arange(1, sorted_labels.numel() + 1, device=sorted_labels.device, dtype=torch.float32)
    precision_at_k = cum_tp / ranks
    ap = (precision_at_k * sorted_labels).sum().item() / pos_total
    return ap


def auroc(scores: torch.Tensor, labels: torch.Tensor) -> Optional[float]:
    labels = labels.bool()
    pos = labels.float().sum().item()
    neg = (~labels).float().sum().item()
    if pos <= 0 or neg <= 0:
        return None
    order = torch.argsort(scores)
    sorted_labels = labels[order]
    ranks = torch.arange(1, scores.numel() + 1, dtype=torch.float64)
    pos_rank_sum = ranks[sorted_labels].sum().item()
    return (pos_rank_sum - pos * (pos + 1.0) / 2.0) / (pos * neg)


def finalize_score_bucket(bucket: Dict[str, list[torch.Tensor]]) -> Dict[str, Any]:
    if not bucket["scores"]:
        return {
            "candidate_count": 0,
            "oracle_positive_count": 0,
            "changed_positive_count": 0,
            "oracle_ap": None,
            "oracle_auroc": None,
            "changed_ap": None,
            "changed_auroc": None,
            "oracle_positive_scores": {},
            "oracle_negative_scores": {},
            "changed_positive_scores": {},
            "changed_negative_scores": {},
        }
    scores = torch.cat(bucket["scores"]).float()
    oracle_labels = torch.cat(bucket["oracle_scope_labels"]).bool()
    changed_labels = torch.cat(bucket["changed_edge_labels"]).bool()

    def score_summary(labels: torch.Tensor) -> Dict[str, Any]:
        pos_scores = scores[labels]
        neg_scores = scores[~labels]
        return {
            "positive_count": int(pos_scores.numel()),
            "negative_count": int(neg_scores.numel()),
            "positive_mean": pos_scores.mean().item() if pos_scores.numel() > 0 else None,
            "negative_mean": neg_scores.mean().item() if neg_scores.numel() > 0 else None,
            "positive_std": pos_scores.std(unbiased=False).item() if pos_scores.numel() > 0 else None,
            "negative_std": neg_scores.std(unbiased=False).item() if neg_scores.numel() > 0 else None,
            "positive_quantiles": tensor_quantiles(pos_scores),
            "negative_quantiles": tensor_quantiles(neg_scores),
        }

    oracle_summary = score_summary(oracle_labels)
    changed_summary = score_summary(changed_labels)
    return {
        "candidate_count": int(scores.numel()),
        "oracle_positive_count": int(oracle_labels.float().sum().item()),
        "changed_positive_count": int(changed_labels.float().sum().item()),
        "oracle_ap": average_precision(scores, oracle_labels),
        "oracle_auroc": auroc(scores, oracle_labels),
        "changed_ap": average_precision(scores, changed_labels),
        "changed_auroc": auroc(scores, changed_labels),
        "oracle_positive_scores": {
            "mean": oracle_summary["positive_mean"],
            "std": oracle_summary["positive_std"],
            **oracle_summary["positive_quantiles"],
        },
        "oracle_negative_scores": {
            "mean": oracle_summary["negative_mean"],
            "std": oracle_summary["negative_std"],
            **oracle_summary["negative_quantiles"],
        },
        "changed_positive_scores": {
            "mean": changed_summary["positive_mean"],
            "std": changed_summary["positive_std"],
            **changed_summary["positive_quantiles"],
        },
        "changed_negative_scores": {
            "mean": changed_summary["negative_mean"],
            "std": changed_summary["negative_std"],
            **changed_summary["negative_quantiles"],
        },
    }


def build_budgeted_outputs(
    base_outputs: Dict[str, torch.Tensor],
    completion_probs: torch.Tensor,
    rescue_budget_fraction: float,
) -> Dict[str, torch.Tensor]:
    candidates = base_outputs["rescue_candidates"].bool()
    selected = torch.zeros_like(candidates)
    batch_size = candidates.shape[0]
    for batch_idx in range(batch_size):
        candidate_indices = candidates[batch_idx].nonzero(as_tuple=False)
        num_candidates = candidate_indices.shape[0]
        if num_candidates <= 0:
            continue
        budget = int(num_candidates * rescue_budget_fraction)
        if budget <= 0:
            continue
        scores = completion_probs[batch_idx][candidates[batch_idx]]
        topk = torch.topk(scores, k=min(budget, num_candidates), largest=True).indices
        chosen_indices = candidate_indices[topk]
        selected[batch_idx, chosen_indices[:, 0], chosen_indices[:, 1]] = True

    valid_edge_mask = base_outputs["valid_edge_mask"]
    final_pred_edges = (base_outputs["pred_scope_edges"] | selected) & valid_edge_mask
    final_edge_probs = torch.where(
        selected,
        torch.maximum(base_outputs["proposal_edge_probs"], completion_probs),
        base_outputs["proposal_edge_probs"],
    )
    return {
        **base_outputs,
        "edge_completion_mode": budget_mode_name(rescue_budget_fraction),
        "completion_probs": completion_probs,
        "rescued_edges": selected & valid_edge_mask,
        "final_pred_scope_edges": final_pred_edges,
        "final_proposal_edge_probs": final_edge_probs * valid_edge_mask.float(),
    }


def add_efficiency_against_naive(results: Dict[str, Any], mode_names: list[str]) -> None:
    base = results[BASE_MODE]["overall"]
    naive = results[NAIVE_MODE]["overall"]
    for mode in mode_names:
        if mode in {BASE_MODE, NAIVE_MODE}:
            continue
        metrics = results[mode]["overall"]
        metrics["recall_recovery_fraction_vs_naive"] = safe_div(
            metrics["proposal_changed_region_recall_edge"] - base["proposal_changed_region_recall_edge"],
            naive["proposal_changed_region_recall_edge"] - base["proposal_changed_region_recall_edge"],
        )
        metrics["cost_fraction_vs_naive"] = safe_div(
            metrics["edge_pred_total"] - base["edge_pred_total"],
            naive["edge_pred_total"] - base["edge_pred_total"],
        )

    for event_type in ["edge_add", "edge_delete"]:
        base_et = results[BASE_MODE]["by_event_type"].get(event_type)
        naive_et = results[NAIVE_MODE]["by_event_type"].get(event_type)
        if not base_et or not naive_et:
            continue
        for mode in mode_names:
            if mode in {BASE_MODE, NAIVE_MODE}:
                continue
            et_metrics = results[mode]["by_event_type"].get(event_type)
            if not et_metrics:
                continue
            et_metrics["recall_recovery_fraction_vs_naive"] = safe_div(
                et_metrics["proposal_changed_region_recall_edge"] - base_et["proposal_changed_region_recall_edge"],
                naive_et["proposal_changed_region_recall_edge"] - base_et["proposal_changed_region_recall_edge"],
            )
            et_metrics["cost_fraction_vs_naive"] = safe_div(
                et_metrics["edge_pred_total"] - base_et["edge_pred_total"],
                naive_et["edge_pred_total"] - base_et["edge_pred_total"],
            )


@torch.no_grad()
def evaluate_rescue_frontier(
    proposal_model: torch.nn.Module,
    rewrite_model: torch.nn.Module,
    completion_model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    node_threshold: float,
    edge_threshold: float,
    completion_threshold: float,
    rescue_budget_fractions: tuple[float, ...],
    use_proposal_conditioning: bool,
) -> Dict[str, Any]:
    proposal_model.eval()
    rewrite_model.eval()
    completion_model.eval()

    mode_names = [BASE_MODE, DEFAULT_MODE]
    mode_names.extend(budget_mode_name(frac) for frac in rescue_budget_fractions)
    mode_names.append(NAIVE_MODE)

    proposal_buckets: dict[str, dict[str, Dict[str, float]]] = {
        mode: defaultdict(init_proposal_bucket) for mode in mode_names
    }
    downstream_buckets: dict[str, dict[str, Dict[str, float]]] = {
        mode: defaultdict(init_downstream_bucket) for mode in mode_names
    }
    score_buckets: dict[str, Dict[str, list[torch.Tensor]]] = defaultdict(init_score_bucket)

    for batch in loader:
        require_keys(
            batch,
            [
                "node_feats",
                "adj",
                "next_adj",
                "node_mask",
                "changed_edges",
                "event_scope_union_edges",
                "events",
            ],
        )
        batch = move_batch_to_device(batch, device)
        base_outputs = get_base_proposal_outputs(
            proposal_model=proposal_model,
            batch=batch,
            node_threshold=node_threshold,
            edge_threshold=edge_threshold,
        )
        completion_logits = completion_model(
            node_latents=base_outputs["node_latents"],
            base_edge_logits=base_outputs["edge_scope_logits"],
        )
        completion_probs = torch.sigmoid(completion_logits) * base_outputs["valid_edge_mask"].float()
        valid_edge_mask = base_outputs["valid_edge_mask"].bool()
        candidates = base_outputs["rescue_candidates"].bool() & valid_edge_mask

        mode_outputs: dict[str, Dict[str, torch.Tensor]] = {
            BASE_MODE: apply_edge_completion_mode(
                base_outputs=base_outputs,
                edge_completion_mode=EDGE_COMPLETION_OFF,
                completion_model=None,
                completion_threshold=completion_threshold,
            ),
            DEFAULT_MODE: apply_edge_completion_mode(
                base_outputs=base_outputs,
                edge_completion_mode=EDGE_COMPLETION_LEARNED,
                completion_model=completion_model,
                completion_threshold=completion_threshold,
            ),
            NAIVE_MODE: apply_edge_completion_mode(
                base_outputs=base_outputs,
                edge_completion_mode=EDGE_COMPLETION_NAIVE,
                completion_model=None,
                completion_threshold=completion_threshold,
            ),
        }
        for fraction in rescue_budget_fractions:
            mode_outputs[budget_mode_name(fraction)] = build_budgeted_outputs(
                base_outputs=base_outputs,
                completion_probs=completion_probs,
                rescue_budget_fraction=fraction,
            )

        rewrite_outputs_by_mode = {}
        for mode, outputs in mode_outputs.items():
            rewrite_outputs_by_mode[mode] = rewrite_model(
                node_feats=outputs["input_node_feats"],
                adj=outputs["input_adj"],
                scope_node_mask=outputs["pred_scope_nodes"].float(),
                scope_edge_mask=outputs["final_pred_scope_edges"].float(),
                proposal_node_probs=outputs["proposal_node_probs"] if use_proposal_conditioning else None,
                proposal_edge_probs=outputs["final_proposal_edge_probs"] if use_proposal_conditioning else None,
            )

        batch_size = batch["node_feats"].shape[0]
        for sample_idx in range(batch_size):
            group_names = make_group_names(batch, sample_idx)
            event_type_groups = [name for name in group_names if name.startswith("event_type::")]
            groups = ["overall"] + event_type_groups
            if "step6a_corruption_setting" in batch:
                setting = str(batch["step6a_corruption_setting"][sample_idx])
                groups.append(f"corruption::{setting}")

            candidate_mask = candidates[sample_idx]
            candidate_scores = completion_probs[sample_idx][candidate_mask]
            oracle_labels = (batch["event_scope_union_edges"][sample_idx] > 0.5)[candidate_mask]
            changed_labels = (batch["changed_edges"][sample_idx] > 0.5)[candidate_mask]
            for group_name in groups:
                append_score_bucket(
                    score_buckets[group_name],
                    candidate_scores,
                    oracle_labels,
                    changed_labels,
                )

            for mode, outputs in mode_outputs.items():
                proposal_stats = build_proposal_sample_stats(
                    batch=batch,
                    final_pred_edges=outputs["final_pred_scope_edges"],
                    rescued_edges=outputs["rescued_edges"],
                    valid_edge_mask=valid_edge_mask,
                    sample_idx=sample_idx,
                )
                downstream_stats = build_downstream_sample_stats(
                    batch=batch,
                    rewrite_outputs=rewrite_outputs_by_mode[mode],
                    valid_edge_mask=valid_edge_mask,
                    sample_idx=sample_idx,
                )
                for group_name in groups:
                    update_bucket(proposal_buckets[mode][group_name], proposal_stats)
                    update_bucket(downstream_buckets[mode][group_name], downstream_stats)

    proposal_results: dict[str, Dict[str, Any]] = {}
    downstream_results: dict[str, Dict[str, Any]] = {}
    for mode in mode_names:
        finalized_proposal = {
            name: finalize_proposal_bucket(bucket)
            for name, bucket in proposal_buckets[mode].items()
        }
        finalized_downstream = {
            name: finalize_downstream_bucket(bucket)
            for name, bucket in downstream_buckets[mode].items()
        }
        proposal_results[mode] = {
            "overall": finalized_proposal.get("overall", finalize_proposal_bucket(init_proposal_bucket())),
            "by_event_type": {
                name.split("event_type::", 1)[1]: value
                for name, value in finalized_proposal.items()
                if name.startswith("event_type::")
            },
            "by_corruption_setting": {
                name.split("corruption::", 1)[1]: value
                for name, value in finalized_proposal.items()
                if name.startswith("corruption::")
            },
        }
        downstream_results[mode] = {
            "overall": finalized_downstream.get("overall", finalize_downstream_bucket(init_downstream_bucket())),
            "by_event_type": {
                name.split("event_type::", 1)[1]: value
                for name, value in finalized_downstream.items()
                if name.startswith("event_type::")
            },
            "by_corruption_setting": {
                name.split("corruption::", 1)[1]: value
                for name, value in finalized_downstream.items()
                if name.startswith("corruption::")
            },
        }

    add_efficiency_against_naive(proposal_results, mode_names)
    score_results = {
        "overall": finalize_score_bucket(score_buckets["overall"]),
        "by_event_type": {
            name.split("event_type::", 1)[1]: finalize_score_bucket(bucket)
            for name, bucket in score_buckets.items()
            if name.startswith("event_type::")
        },
        "by_corruption_setting": {
            name.split("corruption::", 1)[1]: finalize_score_bucket(bucket)
            for name, bucket in score_buckets.items()
            if name.startswith("corruption::")
        },
    }
    return {
        "score_separability": score_results,
        "proposal_side": proposal_results,
        "downstream": downstream_results,
    }


def write_summary_csv(path: Path, payload: Dict[str, Any]) -> None:
    rows = []
    for mode, metrics in payload["results"]["proposal_side"].items():
        proposal_overall = metrics["overall"]
        downstream_overall = payload["results"]["downstream"][mode]["overall"]
        rows.append(
            {
                "section": "frontier_overall",
                "mode": mode,
                "group": "overall",
                "edge_recall": proposal_overall.get("proposal_changed_region_recall_edge"),
                "out_of_scope_miss_edge": proposal_overall.get("out_of_scope_miss_edge"),
                "edge_pred_total": proposal_overall.get("edge_pred_total"),
                "recall_recovery_fraction_vs_naive": proposal_overall.get(
                    "recall_recovery_fraction_vs_naive"
                ),
                "cost_fraction_vs_naive": proposal_overall.get("cost_fraction_vs_naive"),
                "changed_edge": downstream_overall.get("changed_edge"),
                "context_edge": downstream_overall.get("context_edge"),
                "add": downstream_overall.get("add"),
                "delete": downstream_overall.get("delete"),
            }
        )
        for event_type in ["edge_add", "edge_delete"]:
            proposal_event = metrics.get("by_event_type", {}).get(event_type)
            downstream_event = payload["results"]["downstream"][mode].get("by_event_type", {}).get(event_type)
            if not proposal_event or not downstream_event:
                continue
            rows.append(
                {
                    "section": "frontier_event_type",
                    "mode": mode,
                    "group": event_type,
                    "edge_recall": proposal_event.get("proposal_changed_region_recall_edge"),
                    "out_of_scope_miss_edge": proposal_event.get("out_of_scope_miss_edge"),
                    "edge_pred_total": proposal_event.get("edge_pred_total"),
                    "recall_recovery_fraction_vs_naive": proposal_event.get(
                        "recall_recovery_fraction_vs_naive"
                    ),
                    "cost_fraction_vs_naive": proposal_event.get("cost_fraction_vs_naive"),
                    "changed_edge": downstream_event.get("changed_edge"),
                    "context_edge": downstream_event.get("context_edge"),
                    "add": downstream_event.get("add"),
                    "delete": downstream_event.get("delete"),
                }
            )

    for group, metrics in payload["results"]["score_separability"].get("by_event_type", {}).items():
        if group not in {"edge_add", "edge_delete"}:
            continue
        rows.append(
            {
                "section": "separability_event_type",
                "mode": "scores",
                "group": group,
                "edge_recall": None,
                "out_of_scope_miss_edge": None,
                "edge_pred_total": metrics.get("candidate_count"),
                "recall_recovery_fraction_vs_naive": metrics.get("oracle_ap"),
                "cost_fraction_vs_naive": metrics.get("oracle_auroc"),
                "changed_edge": metrics.get("changed_ap"),
                "context_edge": metrics.get("changed_auroc"),
                "add": None,
                "delete": None,
            }
        )

    fieldnames = [
        "section",
        "mode",
        "group",
        "edge_recall",
        "out_of_scope_miss_edge",
        "edge_pred_total",
        "recall_recovery_fraction_vs_naive",
        "cost_fraction_vs_naive",
        "changed_edge",
        "context_edge",
        "add",
        "delete",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal_checkpoint_path", type=str, required=True)
    parser.add_argument("--rewrite_checkpoint_path", type=str, required=True)
    parser.add_argument("--completion_checkpoint_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--split_name", type=str, default="test")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--artifact_dir", type=str, default="artifacts/step9b_rescue_frontier")
    parser.add_argument("--node_threshold", type=float, default=0.20)
    parser.add_argument("--edge_threshold", type=float, default=0.15)
    parser.add_argument("--completion_threshold", type=float, default=0.50)
    parser.add_argument("--rescue_budget_fractions", type=float, nargs="*", default=list(BUDGET_FRACTIONS))
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    proposal_checkpoint_path = resolve_path(args.proposal_checkpoint_path)
    rewrite_checkpoint_path = resolve_path(args.rewrite_checkpoint_path)
    completion_checkpoint_path = resolve_path(args.completion_checkpoint_path)
    data_path = resolve_path(args.data_path)
    artifact_dir = resolve_path(args.artifact_dir)
    device = get_device(args.device)

    if tuple(args.rescue_budget_fractions) != BUDGET_FRACTIONS:
        raise ValueError(f"Use the fixed Step 9b budget grid: {BUDGET_FRACTIONS}")

    _, loader = build_loader(
        data_path=str(data_path),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    proposal_model = load_proposal_model(proposal_checkpoint_path, device)
    rewrite_model, use_proposal_conditioning = load_rewrite_model(rewrite_checkpoint_path, device)
    completion_model = load_completion_model(completion_checkpoint_path, device)

    results = evaluate_rescue_frontier(
        proposal_model=proposal_model,
        rewrite_model=rewrite_model,
        completion_model=completion_model,
        loader=loader,
        device=device,
        node_threshold=args.node_threshold,
        edge_threshold=args.edge_threshold,
        completion_threshold=args.completion_threshold,
        rescue_budget_fractions=tuple(args.rescue_budget_fractions),
        use_proposal_conditioning=use_proposal_conditioning,
    )

    run_name = args.run_name or slugify(f"{args.split_name}_{rewrite_checkpoint_path.parent.name}")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifact_dir / f"{run_name}.json"
    csv_path = artifact_dir / f"{run_name}.csv"
    payload = {
        "metadata": {
            "proposal_checkpoint_path": str(proposal_checkpoint_path),
            "rewrite_checkpoint_path": str(rewrite_checkpoint_path),
            "completion_checkpoint_path": str(completion_checkpoint_path),
            "data_path": str(data_path),
            "split_name": args.split_name,
            "node_threshold": args.node_threshold,
            "edge_threshold": args.edge_threshold,
            "completion_threshold": args.completion_threshold,
            "rescue_budget_fractions": list(args.rescue_budget_fractions),
            "budget_definition": "per-sample top-k over internal candidates; k=floor(fraction * candidate_count)",
            "candidate_definition": "both endpoints inside predicted node scope and base edge proposal did not select the edge",
            "rewrite_uses_proposal_conditioning": use_proposal_conditioning,
        },
        "results": results,
    }
    save_json(out_path, payload)
    write_summary_csv(csv_path, payload)

    compact = {
        "score_separability_overall": results["score_separability"]["overall"],
        "proposal_overall": {
            mode: metrics["overall"]
            for mode, metrics in results["proposal_side"].items()
        },
        "downstream_overall": {
            mode: metrics["overall"]
            for mode, metrics in results["downstream"].items()
        },
    }
    print(f"device: {device}")
    print(f"saved json: {out_path}")
    print(f"saved csv: {csv_path}")
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
