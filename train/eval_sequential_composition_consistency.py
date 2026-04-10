from __future__ import annotations

import argparse
from collections import defaultdict
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.collate import graph_event_collate_fn
from data.dataset import GraphEventDataset
from models.oracle_local import build_valid_edge_mask
from models.oracle_local_delta import (
    EDGE_DELTA_ADD,
    EDGE_DELTA_DELETE,
    EDGE_DELTA_KEEP,
    OracleLocalDeltaRewriteConfig,
    OracleLocalDeltaRewriteModel,
    build_edge_delta_targets,
)
from models.proposal import ScopeProposalConfig, ScopeProposalModel


REQUIRED_ROLES = ("base_to_A", "base_to_B", "A_to_AB", "B_to_AB")


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def get_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if has_mps:
        return torch.device("mps")
    return torch.device("cpu")


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in batch.items():
        out[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return out


def require_keys(batch: Dict[str, Any], required_keys: Iterable[str]) -> None:
    missing = [key for key in required_keys if key not in batch]
    if missing:
        raise KeyError(f"Missing required batch keys: {missing}")


def safe_div(num: float, den: float) -> Optional[float]:
    if den <= 0:
        return None
    return num / den


def edge_correct_and_total(
    edge_logits: torch.Tensor,
    target_adj: torch.Tensor,
    pair_mask: torch.Tensor,
) -> tuple[float, float]:
    pred_adj = (torch.sigmoid(edge_logits) >= 0.5).float()
    pair_mask_f = pair_mask.float()
    correct = ((pred_adj == target_adj.float()).float() * pair_mask_f).sum().item()
    total = pair_mask_f.sum().item()
    return correct, total


def edge_delta_correct_and_total(
    edge_delta_logits: torch.Tensor,
    current_adj: torch.Tensor,
    target_adj: torch.Tensor,
    pair_mask: torch.Tensor,
    label_id: int | None = None,
) -> tuple[float, float]:
    pred_delta = edge_delta_logits.argmax(dim=-1)
    target_delta = build_edge_delta_targets(current_adj, target_adj)
    mask = pair_mask.bool()
    if label_id is not None:
        mask = mask & (target_delta == label_id)
    correct = ((pred_delta == target_delta) & mask).float().sum().item()
    total = mask.float().sum().item()
    return correct, total


def type_correct_and_total(
    pred_type: torch.Tensor,
    target_type: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[float, float]:
    mask = mask.bool()
    correct = ((pred_type == target_type) & mask).float().sum().item()
    total = mask.float().sum().item()
    return correct, total


def summarize_records(records: list[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    metric_names = [
        "delta_all",
        "keep",
        "add",
        "delete",
        "changed",
        "context",
        "full",
        "changed_type",
        "flip",
        "nonflip",
        "scope",
        "scope_edge",
    ]
    out: Dict[str, Optional[float]] = {}
    for metric_name in metric_names:
        vals = [record[metric_name] for record in records if record.get(metric_name) is not None]
        out[metric_name] = (sum(vals) / len(vals)) if vals else None
    out["count"] = len(records)
    return out


def compute_sample_metrics(
    rewrite_outputs: Dict[str, torch.Tensor],
    batch: Dict[str, Any],
    sample_idx_in_batch: int,
    pred_scope_nodes: torch.Tensor,
    pred_scope_edges: torch.Tensor,
) -> Dict[str, Optional[float]]:
    node_mask = batch["node_mask"][sample_idx_in_batch].bool()
    valid_edge_mask = build_valid_edge_mask(batch["node_mask"][sample_idx_in_batch : sample_idx_in_batch + 1])[0].bool()
    changed_nodes = (batch["changed_nodes"][sample_idx_in_batch] > 0.5) & node_mask
    changed_edges = (batch["changed_edges"][sample_idx_in_batch] > 0.5) & valid_edge_mask
    current_type = batch["node_feats"][sample_idx_in_batch, :, 0].long()
    target_type = batch["next_node_feats"][sample_idx_in_batch, :, 0].long()
    flip_target_mask = (current_type != target_type) & node_mask
    nonflip_changed_mask = changed_nodes & (~flip_target_mask)
    context_edges = pred_scope_edges & (~changed_edges)

    pred_full_type = rewrite_outputs["type_logits_full"][sample_idx_in_batch].argmax(dim=-1)
    pred_scope_type = rewrite_outputs["type_logits_local"][sample_idx_in_batch].argmax(dim=-1)
    current_adj = batch["adj"][sample_idx_in_batch]
    target_adj = batch["next_adj"][sample_idx_in_batch]

    c, t = edge_correct_and_total(rewrite_outputs["edge_logits_local"][sample_idx_in_batch], target_adj, pred_scope_edges.float())
    scope_edge = safe_div(c, t)
    c, t = edge_delta_correct_and_total(
        rewrite_outputs["edge_delta_logits_local"][sample_idx_in_batch],
        current_adj,
        target_adj,
        pred_scope_edges.float(),
    )
    delta_all = safe_div(c, t)
    c, t = edge_delta_correct_and_total(
        rewrite_outputs["edge_delta_logits_local"][sample_idx_in_batch],
        current_adj,
        target_adj,
        pred_scope_edges.float(),
        label_id=EDGE_DELTA_KEEP,
    )
    keep = safe_div(c, t)
    c, t = edge_delta_correct_and_total(
        rewrite_outputs["edge_delta_logits_local"][sample_idx_in_batch],
        current_adj,
        target_adj,
        pred_scope_edges.float(),
        label_id=EDGE_DELTA_ADD,
    )
    add = safe_div(c, t)
    c, t = edge_delta_correct_and_total(
        rewrite_outputs["edge_delta_logits_local"][sample_idx_in_batch],
        current_adj,
        target_adj,
        pred_scope_edges.float(),
        label_id=EDGE_DELTA_DELETE,
    )
    delete = safe_div(c, t)
    c, t = edge_correct_and_total(rewrite_outputs["edge_logits_local"][sample_idx_in_batch], target_adj, changed_edges.float())
    changed = safe_div(c, t)
    c, t = edge_correct_and_total(rewrite_outputs["edge_logits_local"][sample_idx_in_batch], target_adj, context_edges.float())
    context = safe_div(c, t)

    c, t = type_correct_and_total(pred_full_type, target_type, node_mask)
    full_type = safe_div(c, t)
    c, t = type_correct_and_total(pred_full_type, target_type, changed_nodes)
    changed_type = safe_div(c, t)
    c, t = type_correct_and_total(pred_full_type, target_type, flip_target_mask)
    flip = safe_div(c, t)
    c, t = type_correct_and_total(pred_full_type, target_type, nonflip_changed_mask)
    nonflip = safe_div(c, t)
    c, t = type_correct_and_total(pred_scope_type, target_type, pred_scope_nodes.float())
    scope = safe_div(c, t)

    return {
        "delta_all": delta_all,
        "keep": keep,
        "add": add,
        "delete": delete,
        "changed": changed,
        "context": context,
        "full": full_type,
        "changed_type": changed_type,
        "flip": flip,
        "nonflip": nonflip,
        "scope": scope,
        "scope_edge": scope_edge,
    }


def build_loader(
    data_path: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple[GraphEventDataset, DataLoader]:
    dataset = GraphEventDataset(data_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=graph_event_collate_fn,
        pin_memory=pin_memory,
    )
    return dataset, loader


def inspect_step3b_dataset(dataset: GraphEventDataset) -> Dict[str, Any]:
    raw_samples = dataset.samples
    support_keys = set(raw_samples[0].keys()) if raw_samples else set()
    pair_to_indices: dict[str, list[int]] = defaultdict(list)
    pair_to_roles: dict[str, set[str]] = defaultdict(set)

    for idx, raw in enumerate(raw_samples):
        pair_id = raw.get("step3_pair_id")
        role = raw.get("step3_transition_role")
        if pair_id is None or role is None:
            continue
        pair_id = str(pair_id)
        pair_to_indices[pair_id].append(idx)
        pair_to_roles[pair_id].add(str(role))

    valid_pair_ids = sorted(pair_to_indices.keys())
    complete_pair_ids = [
        pair_id
        for pair_id in valid_pair_ids
        if pair_to_roles[pair_id] == set(REQUIRED_ROLES) and len(pair_to_indices[pair_id]) == 4
    ]

    return {
        "total_transition_count": len(raw_samples),
        "valid_pair_id_count": len(valid_pair_ids),
        "complete_pair_count": len(complete_pair_ids),
        "dataset_keys": sorted(support_keys),
        "complete_pair_ids": complete_pair_ids,
    }


@torch.no_grad()
def evaluate_step3b(
    proposal_model: ScopeProposalModel,
    rewrite_model: OracleLocalDeltaRewriteModel,
    loader: DataLoader,
    device: torch.device,
    node_threshold: float,
    edge_threshold: float,
    use_proposal_conditioning: bool,
) -> Dict[str, Any]:
    proposal_model.eval()
    rewrite_model.eval()

    records: list[Dict[str, Any]] = []
    total_seen = 0

    for batch in loader:
        require_keys(
            batch,
            [
                "node_feats",
                "adj",
                "next_adj",
                "next_node_feats",
                "node_mask",
                "changed_nodes",
                "changed_edges",
                "events",
                "step3_pair_id",
                "step3_transition_role",
                "step3_unordered_signature",
            ],
        )
        batch = move_batch_to_device(batch, device)

        proposal_outputs = proposal_model(node_feats=batch["node_feats"], adj=batch["adj"])
        node_mask = batch["node_mask"].bool()
        valid_edge_mask = build_valid_edge_mask(batch["node_mask"]).bool()
        node_scope_logits = proposal_outputs.get("node_scope_logits", proposal_outputs["scope_logits"])
        proposal_node_probs = torch.sigmoid(node_scope_logits) * node_mask.float()
        pred_scope_nodes = (proposal_node_probs >= node_threshold) & node_mask

        if "edge_scope_logits" in proposal_outputs:
            proposal_edge_probs = torch.sigmoid(proposal_outputs["edge_scope_logits"]) * valid_edge_mask.float()
            pred_scope_edges = (proposal_edge_probs >= edge_threshold) & valid_edge_mask
        else:
            proposal_edge_probs = (
                proposal_node_probs.unsqueeze(2) * proposal_node_probs.unsqueeze(1) * valid_edge_mask.float()
            )
            pred_scope_edges = pred_scope_nodes.unsqueeze(2) & pred_scope_nodes.unsqueeze(1) & valid_edge_mask

        rewrite_outputs = rewrite_model(
            node_feats=batch["node_feats"],
            adj=batch["adj"],
            scope_node_mask=pred_scope_nodes.float(),
            scope_edge_mask=pred_scope_edges.float(),
            proposal_node_probs=proposal_node_probs if use_proposal_conditioning else None,
            proposal_edge_probs=proposal_edge_probs if use_proposal_conditioning else None,
        )

        batch_size = batch["node_feats"].shape[0]
        for i in range(batch_size):
            record = compute_sample_metrics(
                rewrite_outputs,
                batch,
                i,
                pred_scope_nodes[i],
                pred_scope_edges[i],
            )
            record["dataset_index"] = total_seen + i
            record["step3_pair_id"] = batch["step3_pair_id"][i]
            record["step3_transition_role"] = batch["step3_transition_role"][i]
            record["unordered_signature"] = batch["step3_unordered_signature"][i]
            records.append(record)

        total_seen += batch_size

    return {"transition_records": records}


def build_pair_summary(records: list[Dict[str, Any]]) -> Dict[str, Any]:
    by_pair_id: dict[str, dict[str, Dict[str, Any]]] = defaultdict(dict)
    for record in records:
        by_pair_id[str(record["step3_pair_id"])][str(record["step3_transition_role"])] = record

    valid_pairs: list[Dict[str, Any]] = []
    path_metric_names = ["delta_all", "keep", "add", "delete", "changed", "context", "full", "changed_type", "flip"]

    for pair_id, role_map in by_pair_id.items():
        if set(role_map.keys()) != set(REQUIRED_ROLES):
            continue
        rec_a = role_map["A_to_AB"]
        rec_b = role_map["B_to_AB"]
        pair_summary: Dict[str, Any] = {
            "step3_pair_id": pair_id,
            "unordered_signature": rec_a["unordered_signature"],
        }
        for metric_name in path_metric_names:
            a_val = rec_a.get(metric_name)
            b_val = rec_b.get(metric_name)
            pair_summary[f"path_gap_{metric_name}"] = abs(a_val - b_val) if a_val is not None and b_val is not None else None
        valid_pairs.append(pair_summary)

    first_step_records = [
        record for record in records if record["step3_transition_role"] in {"base_to_A", "base_to_B"}
    ]
    second_step_records = [
        record for record in records if record["step3_transition_role"] in {"A_to_AB", "B_to_AB"}
    ]

    overall: Dict[str, Any] = {
        "valid_sequential_pair_count": len(valid_pairs),
        "first_step_average": summarize_records(first_step_records),
        "second_step_average": summarize_records(second_step_records),
    }
    for metric_name in path_metric_names:
        vals = [pair[f"path_gap_{metric_name}"] for pair in valid_pairs if pair.get(f"path_gap_{metric_name}") is not None]
        overall[f"path_mean_abs_gap_{metric_name}"] = (sum(vals) / len(vals)) if vals else None

    by_signature: dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for pair in valid_pairs:
        by_signature[str(pair["unordered_signature"])].append(pair)

    signature_summary: list[Dict[str, Any]] = []
    for unordered_sig, pairs in by_signature.items():
        row: Dict[str, Any] = {
            "unordered_signature": unordered_sig,
            "count": len(pairs),
        }
        for metric_name in path_metric_names:
            vals = [pair[f"path_gap_{metric_name}"] for pair in pairs if pair.get(f"path_gap_{metric_name}") is not None]
            row[f"path_mean_abs_gap_{metric_name}"] = (sum(vals) / len(vals)) if vals else None
        signature_summary.append(row)
    signature_summary.sort(key=lambda row: (-int(row["count"]), row["unordered_signature"]))

    return {
        "overall": overall,
        "signature_summary": signature_summary,
    }


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal_checkpoint_path", type=str, required=True)
    parser.add_argument("--rewrite_checkpoint_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--split_name", type=str, default="eval")
    parser.add_argument("--node_threshold", type=float, default=0.20)
    parser.add_argument("--edge_threshold", type=float, default=0.15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = get_device(args.device)
    pin_memory = device.type == "cuda"

    proposal_checkpoint_path = resolve_path(args.proposal_checkpoint_path)
    rewrite_checkpoint_path = resolve_path(args.rewrite_checkpoint_path)
    data_path = resolve_path(args.data_path)

    dataset, loader = build_loader(
        data_path=str(data_path),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    dataset_support = inspect_step3b_dataset(dataset)

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

    evaluation = evaluate_step3b(
        proposal_model=proposal_model,
        rewrite_model=rewrite_model,
        loader=loader,
        device=device,
        node_threshold=args.node_threshold,
        edge_threshold=args.edge_threshold,
        use_proposal_conditioning=use_proposal_conditioning,
    )
    pair_summary = build_pair_summary(evaluation["transition_records"])

    output_payload = {
        "metadata": {
            "proposal_checkpoint_path": str(proposal_checkpoint_path),
            "rewrite_checkpoint_path": str(rewrite_checkpoint_path),
            "data_path": str(data_path),
            "split_name": args.split_name,
            "node_threshold": args.node_threshold,
            "edge_threshold": args.edge_threshold,
            "rewrite_uses_proposal_conditioning": use_proposal_conditioning,
            "path_construction_mode": "ground_truth_intermediate_states_with_exact_second_step_final_predictions",
        },
        "dataset_support": {
            "total_transition_count": dataset_support["total_transition_count"],
            "valid_pair_id_count": dataset_support["valid_pair_id_count"],
            "complete_pair_count": dataset_support["complete_pair_count"],
            "dataset_keys": dataset_support["dataset_keys"],
        },
        "results": {
            "overall": pair_summary["overall"],
            "signature_summary": pair_summary["signature_summary"],
        },
    }

    out_path = rewrite_checkpoint_path.parent / f"{args.split_name}_sequential_composition_consistency.json"
    save_json(out_path, output_payload)

    print(f"device: {device}")
    print(f"proposal checkpoint: {proposal_checkpoint_path}")
    print(f"rewrite checkpoint: {rewrite_checkpoint_path}")
    print(f"data: {data_path}")
    print(f"total transitions: {dataset_support['total_transition_count']}")
    print(f"valid pair ids: {dataset_support['valid_pair_id_count']}")
    print(f"complete four-transition structures: {dataset_support['complete_pair_count']}")
    print(f"saved json: {out_path}")
    print(json.dumps(output_payload["results"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
