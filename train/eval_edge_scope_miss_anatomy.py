from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.oracle_local_delta import build_valid_edge_mask
from models.proposal import ScopeProposalConfig, ScopeProposalModel
from train.eval_noisy_structured_observation import (
    build_loader,
    get_device,
    move_batch_to_device,
    require_keys,
    resolve_path,
    save_json,
)
from train.eval_scope_edit_localization_gap import (
    EVENT_TYPE_ORDER,
    extract_event_type_list,
    safe_div,
    slugify,
)


CLOSURE_NONE = "none"
CLOSURE_INDUCED = "induced_edge_closure_over_predicted_nodes"
CLOSURE_MODES = (CLOSURE_NONE, CLOSURE_INDUCED)


def init_closure_bucket() -> Dict[str, float]:
    return {
        "num_samples": 0.0,
        "edge_changed_total": 0.0,
        "edge_changed_covered": 0.0,
        "edge_pred_total": 0.0,
        "edge_pred_context_total": 0.0,
        "edge_out_scope_changed_total": 0.0,
    }


def init_anatomy_bucket() -> Dict[str, float]:
    return {
        "num_samples": 0.0,
        "edge_changed_total": 0.0,
        "edge_out_scope_changed_total": 0.0,
        "miss_both_endpoints_in_predicted_node_scope": 0.0,
        "miss_one_endpoint_in_predicted_node_scope": 0.0,
        "miss_neither_endpoint_in_predicted_node_scope": 0.0,
    }


def update_bucket(bucket: Dict[str, float], sample_stats: Dict[str, float]) -> None:
    for key, value in sample_stats.items():
        bucket[key] += value


def finalize_closure_bucket(bucket: Dict[str, float]) -> Dict[str, Any]:
    return {
        "num_samples": int(bucket["num_samples"]),
        "proposal_changed_region_recall_edge": safe_div(
            bucket["edge_changed_covered"], bucket["edge_changed_total"]
        ),
        "out_of_scope_miss_edge": safe_div(
            bucket["edge_out_scope_changed_total"], bucket["edge_changed_total"]
        ),
        "proposal_scope_excess_ratio_edge": safe_div(
            bucket["edge_pred_context_total"], bucket["edge_pred_total"]
        ),
        "edge_changed_total": int(bucket["edge_changed_total"]),
        "edge_pred_total": int(bucket["edge_pred_total"]),
        "edge_out_scope_changed_total": int(bucket["edge_out_scope_changed_total"]),
    }


def finalize_anatomy_bucket(bucket: Dict[str, float]) -> Dict[str, Any]:
    missed_total = bucket["edge_out_scope_changed_total"]
    changed_total = bucket["edge_changed_total"]
    both = bucket["miss_both_endpoints_in_predicted_node_scope"]
    one = bucket["miss_one_endpoint_in_predicted_node_scope"]
    neither = bucket["miss_neither_endpoint_in_predicted_node_scope"]
    return {
        "num_samples": int(bucket["num_samples"]),
        "edge_changed_total": int(changed_total),
        "missed_changed_edge_total": int(missed_total),
        "miss_both_endpoints_in_predicted_node_scope": int(both),
        "miss_one_endpoint_in_predicted_node_scope": int(one),
        "miss_neither_endpoint_in_predicted_node_scope": int(neither),
        "miss_both_share_of_missed": safe_div(both, missed_total),
        "miss_one_share_of_missed": safe_div(one, missed_total),
        "miss_neither_share_of_missed": safe_div(neither, missed_total),
        "miss_both_share_of_total_changed": safe_div(both, changed_total),
        "miss_one_share_of_total_changed": safe_div(one, changed_total),
        "miss_neither_share_of_total_changed": safe_div(neither, changed_total),
    }


def make_group_names(batch: Dict[str, Any], sample_idx: int) -> list[str]:
    names = ["overall"]

    events_item = batch.get("events", [None] * (sample_idx + 1))[sample_idx]
    event_types = sorted(set(extract_event_type_list(events_item)))
    for event_type in event_types:
        if event_type in EVENT_TYPE_ORDER:
            names.append(f"event_type::{event_type}")

    if "step6a_corruption_setting" in batch:
        setting = str(batch["step6a_corruption_setting"][sample_idx])
        names.append(f"corruption::{setting}")
        for event_type in event_types:
            if event_type in EVENT_TYPE_ORDER:
                names.append(f"corruption::{setting}::event_type::{event_type}")

    return names


def build_predicted_scopes(
    proposal_model: ScopeProposalModel,
    batch: Dict[str, Any],
    node_threshold: float,
    edge_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    input_node_feats = batch.get("obs_node_feats", batch["node_feats"])
    input_adj = batch.get("obs_adj", batch["adj"])
    node_mask = batch["node_mask"].bool()
    valid_edge_mask = build_valid_edge_mask(batch["node_mask"]).bool()

    proposal_outputs = proposal_model(node_feats=input_node_feats, adj=input_adj)
    node_scope_logits = proposal_outputs.get("node_scope_logits", proposal_outputs["scope_logits"])
    proposal_node_probs = torch.sigmoid(node_scope_logits) * batch["node_mask"]
    pred_scope_nodes = (proposal_node_probs >= node_threshold) & node_mask

    if "edge_scope_logits" in proposal_outputs:
        proposal_edge_probs = torch.sigmoid(proposal_outputs["edge_scope_logits"]) * valid_edge_mask.float()
        pred_scope_edges = (proposal_edge_probs >= edge_threshold) & valid_edge_mask
    else:
        proposal_edge_probs = (
            proposal_node_probs.unsqueeze(2) * proposal_node_probs.unsqueeze(1) * valid_edge_mask.float()
        )
        pred_scope_edges = (
            pred_scope_nodes.unsqueeze(2) & pred_scope_nodes.unsqueeze(1) & valid_edge_mask
        )

    return input_node_feats, input_adj, pred_scope_nodes, pred_scope_edges, valid_edge_mask


def apply_closure_mode(
    pred_scope_nodes: torch.Tensor,
    pred_scope_edges: torch.Tensor,
    valid_edge_mask: torch.Tensor,
    closure_mode: str,
) -> torch.Tensor:
    if closure_mode == CLOSURE_NONE:
        return pred_scope_edges
    if closure_mode == CLOSURE_INDUCED:
        node_induced_edges = (
            pred_scope_nodes.unsqueeze(2) & pred_scope_nodes.unsqueeze(1) & valid_edge_mask
        )
        return pred_scope_edges | node_induced_edges
    raise ValueError(f"Unsupported closure mode: {closure_mode}")


def build_closure_sample_stats(
    batch: Dict[str, Any],
    pred_scope_edges: torch.Tensor,
    valid_edge_mask: torch.Tensor,
    sample_idx: int,
) -> Dict[str, float]:
    changed_edges = (batch["changed_edges"][sample_idx] > 0.5) & valid_edge_mask[sample_idx]
    context_edges = (~changed_edges) & valid_edge_mask[sample_idx]
    pred_edge_mask = pred_scope_edges[sample_idx] & valid_edge_mask[sample_idx]
    changed_covered = changed_edges & pred_edge_mask
    changed_out = changed_edges & (~pred_edge_mask)
    context_in_pred = context_edges & pred_edge_mask

    return {
        "num_samples": 1.0,
        "edge_changed_total": changed_edges.float().sum().item(),
        "edge_changed_covered": changed_covered.float().sum().item(),
        "edge_pred_total": pred_edge_mask.float().sum().item(),
        "edge_pred_context_total": context_in_pred.float().sum().item(),
        "edge_out_scope_changed_total": changed_out.float().sum().item(),
    }


def build_miss_anatomy_sample_stats(
    batch: Dict[str, Any],
    pred_scope_nodes: torch.Tensor,
    pred_scope_edges: torch.Tensor,
    valid_edge_mask: torch.Tensor,
    sample_idx: int,
) -> Dict[str, float]:
    changed_edges = (batch["changed_edges"][sample_idx] > 0.5) & valid_edge_mask[sample_idx]
    missed_edges = changed_edges & (~pred_scope_edges[sample_idx]) & valid_edge_mask[sample_idx]
    pred_nodes = pred_scope_nodes[sample_idx]

    endpoint_u_in = pred_nodes.unsqueeze(1).expand_as(missed_edges)
    endpoint_v_in = pred_nodes.unsqueeze(0).expand_as(missed_edges)
    both = missed_edges & endpoint_u_in & endpoint_v_in
    one = missed_edges & (endpoint_u_in ^ endpoint_v_in)
    neither = missed_edges & (~endpoint_u_in) & (~endpoint_v_in)

    return {
        "num_samples": 1.0,
        "edge_changed_total": changed_edges.float().sum().item(),
        "edge_out_scope_changed_total": missed_edges.float().sum().item(),
        "miss_both_endpoints_in_predicted_node_scope": both.float().sum().item(),
        "miss_one_endpoint_in_predicted_node_scope": one.float().sum().item(),
        "miss_neither_endpoint_in_predicted_node_scope": neither.float().sum().item(),
    }


@torch.no_grad()
def evaluate_edge_scope_miss_anatomy(
    proposal_model: ScopeProposalModel,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    node_threshold: float,
    edge_threshold: float,
) -> Dict[str, Any]:
    proposal_model.eval()

    anatomy_buckets: dict[str, Dict[str, float]] = defaultdict(init_anatomy_bucket)
    closure_buckets: dict[str, dict[str, Dict[str, float]]] = {
        closure_mode: defaultdict(init_closure_bucket) for closure_mode in CLOSURE_MODES
    }

    for batch in loader:
        require_keys(
            batch,
            [
                "node_feats",
                "adj",
                "next_adj",
                "node_mask",
                "changed_edges",
                "events",
            ],
        )
        batch = move_batch_to_device(batch, device)
        _, _, pred_scope_nodes, pred_scope_edges, valid_edge_mask = build_predicted_scopes(
            proposal_model=proposal_model,
            batch=batch,
            node_threshold=node_threshold,
            edge_threshold=edge_threshold,
        )

        batch_size = batch["node_feats"].shape[0]
        for sample_idx in range(batch_size):
            group_names = make_group_names(batch, sample_idx)
            anatomy_stats = build_miss_anatomy_sample_stats(
                batch=batch,
                pred_scope_nodes=pred_scope_nodes,
                pred_scope_edges=pred_scope_edges,
                valid_edge_mask=valid_edge_mask,
                sample_idx=sample_idx,
            )
            for group_name in group_names:
                update_bucket(anatomy_buckets[group_name], anatomy_stats)

            for closure_mode in CLOSURE_MODES:
                closure_edge_mask = apply_closure_mode(
                    pred_scope_nodes=pred_scope_nodes,
                    pred_scope_edges=pred_scope_edges,
                    valid_edge_mask=valid_edge_mask,
                    closure_mode=closure_mode,
                )
                closure_stats = build_closure_sample_stats(
                    batch=batch,
                    pred_scope_edges=closure_edge_mask,
                    valid_edge_mask=valid_edge_mask,
                    sample_idx=sample_idx,
                )
                for group_name in group_names:
                    update_bucket(closure_buckets[closure_mode][group_name], closure_stats)

    anatomy = {
        "overall": finalize_anatomy_bucket(anatomy_buckets["overall"]),
        "by_event_type": {
            event_type: finalize_anatomy_bucket(anatomy_buckets[f"event_type::{event_type}"])
            for event_type in EVENT_TYPE_ORDER
            if f"event_type::{event_type}" in anatomy_buckets
        },
        "by_corruption_setting": {},
    }

    for group_name in sorted(anatomy_buckets.keys()):
        if not group_name.startswith("corruption::"):
            continue
        parts = group_name.split("::")
        if len(parts) == 2:
            setting = parts[1]
            anatomy["by_corruption_setting"].setdefault(setting, {"overall": None, "by_event_type": {}})
            anatomy["by_corruption_setting"][setting]["overall"] = finalize_anatomy_bucket(anatomy_buckets[group_name])
        elif len(parts) == 4 and parts[2] == "event_type":
            setting = parts[1]
            event_type = parts[3]
            anatomy["by_corruption_setting"].setdefault(setting, {"overall": None, "by_event_type": {}})
            anatomy["by_corruption_setting"][setting]["by_event_type"][event_type] = finalize_anatomy_bucket(
                anatomy_buckets[group_name]
            )

    closure_results: Dict[str, Any] = {}
    for closure_mode in CLOSURE_MODES:
        closure_results[closure_mode] = {
            "overall": finalize_closure_bucket(closure_buckets[closure_mode]["overall"]),
            "by_event_type": {
                event_type: finalize_closure_bucket(closure_buckets[closure_mode][f"event_type::{event_type}"])
                for event_type in EVENT_TYPE_ORDER
                if f"event_type::{event_type}" in closure_buckets[closure_mode]
            },
            "by_corruption_setting": {},
        }

        for group_name in sorted(closure_buckets[closure_mode].keys()):
            if not group_name.startswith("corruption::"):
                continue
            parts = group_name.split("::")
            if len(parts) == 2:
                setting = parts[1]
                closure_results[closure_mode]["by_corruption_setting"].setdefault(
                    setting, {"overall": None, "by_event_type": {}}
                )
                closure_results[closure_mode]["by_corruption_setting"][setting]["overall"] = finalize_closure_bucket(
                    closure_buckets[closure_mode][group_name]
                )
            elif len(parts) == 4 and parts[2] == "event_type":
                setting = parts[1]
                event_type = parts[3]
                closure_results[closure_mode]["by_corruption_setting"].setdefault(
                    setting, {"overall": None, "by_event_type": {}}
                )
                closure_results[closure_mode]["by_corruption_setting"][setting]["by_event_type"][event_type] = (
                    finalize_closure_bucket(closure_buckets[closure_mode][group_name])
                )

    def delta_entry(base: Dict[str, Any], closure: Dict[str, Any]) -> Dict[str, Optional[float]]:
        keys = [
            "proposal_changed_region_recall_edge",
            "out_of_scope_miss_edge",
            "proposal_scope_excess_ratio_edge",
        ]
        out: Dict[str, Optional[float]] = {}
        for key in keys:
            b = base.get(key)
            c = closure.get(key)
            out[key] = None if b is None or c is None else (c - b)
        return out

    closure_delta = {
        "overall": delta_entry(
            closure_results[CLOSURE_NONE]["overall"],
            closure_results[CLOSURE_INDUCED]["overall"],
        ),
        "by_event_type": {},
        "by_corruption_setting": {},
    }
    for event_type in EVENT_TYPE_ORDER:
        if event_type in closure_results[CLOSURE_NONE]["by_event_type"] and event_type in closure_results[CLOSURE_INDUCED]["by_event_type"]:
            closure_delta["by_event_type"][event_type] = delta_entry(
                closure_results[CLOSURE_NONE]["by_event_type"][event_type],
                closure_results[CLOSURE_INDUCED]["by_event_type"][event_type],
            )
    for setting in sorted(closure_results[CLOSURE_NONE]["by_corruption_setting"].keys()):
        if setting not in closure_results[CLOSURE_INDUCED]["by_corruption_setting"]:
            continue
        closure_delta["by_corruption_setting"][setting] = {
            "overall": delta_entry(
                closure_results[CLOSURE_NONE]["by_corruption_setting"][setting]["overall"],
                closure_results[CLOSURE_INDUCED]["by_corruption_setting"][setting]["overall"],
            )
        }

    return {
        "miss_anatomy": anatomy,
        "closure_probe": closure_results,
        "closure_delta_induced_minus_none": closure_delta,
    }


def build_payload(
    args: argparse.Namespace,
    proposal_checkpoint_path: Path,
    results: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "metadata": {
            "proposal_checkpoint_path": str(proposal_checkpoint_path),
            "rewrite_checkpoint_path": str(resolve_path(args.rewrite_checkpoint_path)) if args.rewrite_checkpoint_path else None,
            "data_path": str(resolve_path(args.data_path)),
            "split_name": args.split_name,
            "run_name": args.run_name,
            "node_threshold": args.node_threshold,
            "edge_threshold": args.edge_threshold,
            "evaluation_mode": "edge_scope_miss_anatomy_and_induced_closure_probe",
            "model_input": "obs_graph_t when present; clean graph_t otherwise",
            "note_cn": "本评估主要是 proposal 侧，因此同一 proposal 前端下，换 rewrite 不会改变 anatomy / closure 统计。",
        },
        "definitions": {
            "miss_both_endpoints_in_predicted_node_scope": "missed GT changed edges whose two endpoints are both already inside predicted node scope",
            "miss_one_endpoint_in_predicted_node_scope": "missed GT changed edges with exactly one endpoint in predicted node scope",
            "miss_neither_endpoint_in_predicted_node_scope": "missed GT changed edges with neither endpoint in predicted node scope",
            "closure_mode::none": "use original predicted edge scope",
            "closure_mode::induced_edge_closure_over_predicted_nodes": "include every valid edge slot whose two endpoints are both in predicted node scope",
        },
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal_checkpoint_path", type=str, required=True)
    parser.add_argument("--rewrite_checkpoint_path", type=str, default=None)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--split_name", type=str, default="eval")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--artifact_dir", type=str, default="artifacts/edge_scope_closure_probe")
    parser.add_argument("--node_threshold", type=float, default=0.20)
    parser.add_argument("--edge_threshold", type=float, default=0.15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    proposal_checkpoint_path = resolve_path(args.proposal_checkpoint_path)
    data_path = resolve_path(args.data_path)
    artifact_dir = resolve_path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)

    _, loader = build_loader(
        data_path=str(data_path),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    proposal_checkpoint = torch.load(proposal_checkpoint_path, map_location="cpu")
    proposal_model = ScopeProposalModel(ScopeProposalConfig(**proposal_checkpoint["model_config"])).to(device)
    proposal_model.load_state_dict(proposal_checkpoint["model_state_dict"])
    proposal_model.eval()

    results = evaluate_edge_scope_miss_anatomy(
        proposal_model=proposal_model,
        loader=loader,
        device=device,
        node_threshold=args.node_threshold,
        edge_threshold=args.edge_threshold,
    )
    payload = build_payload(args=args, proposal_checkpoint_path=proposal_checkpoint_path, results=results)

    if args.run_name is None:
        run_name = "__".join(
            [
                slugify(proposal_checkpoint_path.parent.name),
                slugify(Path(args.rewrite_checkpoint_path).parent.name if args.rewrite_checkpoint_path else "proposal_only"),
                slugify(data_path.stem),
            ]
        )
    else:
        run_name = slugify(args.run_name)

    out_path = artifact_dir / f"{run_name}.json"
    save_json(out_path, payload)

    print(f"device: {device}")
    print(f"proposal checkpoint: {proposal_checkpoint_path}")
    if args.rewrite_checkpoint_path:
        print(f"rewrite checkpoint (metadata only): {resolve_path(args.rewrite_checkpoint_path)}")
    print(f"data: {data_path}")
    print(f"saved json: {out_path}")
    print(json.dumps(results["miss_anatomy"]["overall"], indent=2, ensure_ascii=False))
    print(json.dumps(results["closure_delta_induced_minus_none"]["overall"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
