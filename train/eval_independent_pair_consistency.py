from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
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
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def require_keys(batch: Dict[str, Any], required_keys: Iterable[str]) -> None:
    missing = [k for k in required_keys if k not in batch]
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


def safe_event_type_name(x: Any) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        for key in ["event_type", "type", "name", "kind", "event_name"]:
            if key in x:
                return str(x[key])
        if len(x) == 1:
            return str(next(iter(x.keys())))
    return str(x)


def extract_event_type_list(events_item: Any) -> list[str]:
    if events_item is None:
        return []
    if not isinstance(events_item, list):
        return [safe_event_type_name(events_item)]
    return [safe_event_type_name(e) for e in events_item]


def infer_num_events(num_events_item: Any, events_item: Any) -> int:
    if num_events_item is not None:
        try:
            return int(num_events_item)
        except (TypeError, ValueError):
            pass
    if isinstance(events_item, list):
        return len(events_item)
    return 0


def canonical_json(x: Any) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def graph_state_hash(sample: Dict[str, Any]) -> str:
    node_feats = torch.as_tensor(sample["node_feats"]).cpu().numpy().tobytes()
    adj = torch.as_tensor(sample["adj"]).cpu().numpy().tobytes()
    h = hashlib.sha1()
    h.update(node_feats)
    h.update(adj)
    return h.hexdigest()


def full_event_signature(events_item: Any) -> str:
    return canonical_json(events_item)


def ordered_signature(event_types: list[str]) -> str:
    return "->".join(event_types) if event_types else "NA"


def unordered_signature(event_types: list[str]) -> str:
    return "+".join(sorted(event_types)) if event_types else "NA"


def independent_sample_info(dataset: GraphEventDataset) -> Dict[str, Any]:
    total_size = len(dataset)
    raw_samples = dataset.samples
    two_event_count = 0
    independent_two_event_count = 0
    filtered_raw_indices: list[int] = []
    support_keys = set(raw_samples[0].keys()) if raw_samples else set()
    has_exact_metadata = {
        "step3_pair_id",
        "step3_ordered_variant",
        "step3_ordered_signature",
        "step3_unordered_signature",
    }.issubset(support_keys)

    reverse_lookup: dict[tuple[str, str], list[int]] = defaultdict(list)
    sample_infos: dict[int, Dict[str, Any]] = {}
    pair_id_to_indices: dict[str, list[int]] = defaultdict(list)

    for idx, raw in enumerate(raw_samples):
        events_item = raw.get("events")
        num_events = len(events_item) if isinstance(events_item, list) else 0
        if num_events == 2:
            two_event_count += 1
        independent_pairs = raw.get("independent_pairs")
        is_independent = bool(independent_pairs)
        if num_events == 2 and is_independent:
            independent_two_event_count += 1
            filtered_raw_indices.append(idx)
            graph_hash = graph_state_hash(
                {
                    "node_feats": raw["graph_t"]["node_features"],
                    "adj": raw["graph_t"]["adj"],
                }
            )
            events_sig = full_event_signature(events_item)
            reverse_events_sig = canonical_json(list(reversed(events_item)))
            reverse_lookup[(graph_hash, events_sig)].append(idx)
            sample_infos[idx] = {
                "graph_hash": graph_hash,
                "events_sig": events_sig,
                "reverse_events_sig": reverse_events_sig,
                "ordered_signature": ordered_signature(extract_event_type_list(events_item)),
                "unordered_signature": unordered_signature(extract_event_type_list(events_item)),
                "step3_pair_id": raw.get("step3_pair_id"),
                "step3_ordered_variant": raw.get("step3_ordered_variant"),
            }
            if raw.get("step3_pair_id") is not None:
                pair_id_to_indices[str(raw["step3_pair_id"])].append(idx)

    exact_reverse_pair_count = 0
    exact_matching_source = "heuristic_reverse_lookup"
    seen_pairs: set[tuple[int, int]] = set()
    if has_exact_metadata:
        for _, indices in pair_id_to_indices.items():
            if len(indices) != 2:
                continue
            variants = {raw_samples[i].get("step3_ordered_variant") for i in indices}
            if len(variants) == 2:
                exact_reverse_pair_count += 1
        exact_matching_source = "metadata_pair_id"
    else:
        for idx in filtered_raw_indices:
            info = sample_infos[idx]
            candidates = reverse_lookup.get((info["graph_hash"], info["reverse_events_sig"]), [])
            for other_idx in candidates:
                if other_idx == idx:
                    continue
                pair = tuple(sorted((idx, other_idx)))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    exact_reverse_pair_count += 1

    return {
        "total_dataset_size": total_size,
        "two_event_sample_count": two_event_count,
        "independent_two_event_sample_count": independent_two_event_count,
        "supports_exact_reverse_matching": exact_reverse_pair_count > 0,
        "exact_reverse_pair_count": exact_reverse_pair_count,
        "exact_matching_source": exact_matching_source,
        "dataset_keys": sorted(support_keys),
        "filtered_raw_indices": filtered_raw_indices,
        "sample_infos": sample_infos,
        "has_exact_pair_metadata": has_exact_metadata,
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
    for name in metric_names:
        vals = [r[name] for r in records if r.get(name) is not None]
        out[name] = (sum(vals) / len(vals)) if vals else None
    out["count"] = len(records)
    return out


def top_bucket_summary(
    records: list[Dict[str, Any]],
    key_name: str,
    top_k: int = 12,
) -> list[Dict[str, Any]]:
    buckets: dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        buckets[record[key_name]].append(record)
    items = []
    for key, bucket_records in buckets.items():
        summary = summarize_records(bucket_records)
        summary[key_name] = key
        items.append(summary)
    items.sort(key=lambda x: (-int(x["count"]), x[key_name]))
    return items[:top_k]


def reverse_gap_summary(
    records: list[Dict[str, Any]],
    sample_infos: dict[int, Dict[str, Any]],
    supports_exact_reverse_matching: bool,
) -> list[Dict[str, Any]]:
    by_unordered_then_ordered: dict[str, dict[str, list[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    by_idx = {record["dataset_index"]: record for record in records}
    for record in records:
        by_unordered_then_ordered[record["unordered_signature"]][record["ordered_signature"]].append(record)

    summaries: list[Dict[str, Any]] = []
    paired_seen: set[tuple[int, int]] = set()

    for unordered_sig, ordered_groups in by_unordered_then_ordered.items():
        if len(ordered_groups) < 2:
            continue
        ordered_names = sorted(ordered_groups.keys())
        if len(ordered_names) != 2:
            continue
        order_a, order_b = ordered_names
        records_a = ordered_groups[order_a]
        records_b = ordered_groups[order_b]
        summary: Dict[str, Any] = {
            "unordered_signature": unordered_sig,
            "order_a": order_a,
            "order_b": order_b,
            "count_a": len(records_a),
            "count_b": len(records_b),
            "comparison_mode": "bucket_level",
        }
        metrics = ["delta_all", "delete", "changed", "context", "flip"]

        if supports_exact_reverse_matching:
            pair_gaps: dict[str, list[float]] = defaultdict(list)
            for rec in records_a:
                info = sample_infos[rec["dataset_index"]]
                for other in records_b:
                    other_info = sample_infos[other["dataset_index"]]
                    if (
                        info["graph_hash"] == other_info["graph_hash"]
                        and info["reverse_events_sig"] == other_info["events_sig"]
                    ):
                        pair = tuple(sorted((rec["dataset_index"], other["dataset_index"])))
                        if pair in paired_seen:
                            continue
                        paired_seen.add(pair)
                        for metric_name in metrics:
                            if rec.get(metric_name) is not None and other.get(metric_name) is not None:
                                pair_gaps[metric_name].append(abs(rec[metric_name] - other[metric_name]))
                        break
            if pair_gaps:
                summary["comparison_mode"] = "exact_pair"
                summary["exact_pair_count"] = max(len(v) for v in pair_gaps.values())
                for metric_name in metrics:
                    vals = pair_gaps.get(metric_name, [])
                    summary[f"gap_{metric_name}"] = (sum(vals) / len(vals)) if vals else None
                summaries.append(summary)
                continue

        mean_a = summarize_records(records_a)
        mean_b = summarize_records(records_b)
        for metric_name in metrics:
            a_val = mean_a.get(metric_name)
            b_val = mean_b.get(metric_name)
            summary[f"gap_{metric_name}"] = abs(a_val - b_val) if a_val is not None and b_val is not None else None
        summaries.append(summary)

    summaries.sort(key=lambda x: (-max(x["count_a"], x["count_b"]), x["unordered_signature"]))
    return summaries


@torch.no_grad()
def evaluate_independent_pairs(
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

    per_sample_records: list[Dict[str, Any]] = []
    total_seen = 0

    for batch in loader:
        require_keys(
            batch,
            [
                "node_feats",
                "adj",
                "next_adj",
                "node_mask",
                "changed_nodes",
                "changed_edges",
                "event_scope_union_nodes",
                "event_scope_union_edges",
                "next_node_feats",
                "events",
                "num_events",
                "independent_pairs",
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
            events_item = batch["events"][i]
            num_events_item = batch["num_events"][i]
            independent_pairs = batch["independent_pairs"][i]
            num_events = infer_num_events(num_events_item, events_item)
            is_independent = bool(independent_pairs)
            dataset_index = total_seen + i
            if num_events != 2 or not is_independent:
                continue

            event_types = extract_event_type_list(events_item)
            record = compute_sample_metrics(
                rewrite_outputs,
                batch,
                i,
                pred_scope_nodes[i],
                pred_scope_edges[i],
            )
            record["dataset_index"] = dataset_index
            record["ordered_signature"] = (
                batch["step3_ordered_signature"][i]
                if "step3_ordered_signature" in batch
                else ordered_signature(event_types)
            )
            record["unordered_signature"] = (
                batch["step3_unordered_signature"][i]
                if "step3_unordered_signature" in batch
                else unordered_signature(event_types)
            )
            record["events"] = events_item
            record["step3_pair_id"] = batch["step3_pair_id"][i] if "step3_pair_id" in batch else None
            record["step3_ordered_variant"] = (
                batch["step3_ordered_variant"][i] if "step3_ordered_variant" in batch else None
            )
            record["step3_base_graph_id"] = (
                batch["step3_base_graph_id"][i] if "step3_base_graph_id" in batch else None
            )
            per_sample_records.append(record)

        total_seen += batch_size

    return {
        "num_step3_samples": len(per_sample_records),
        "overall": summarize_records(per_sample_records),
        "ordered_signature_summary": top_bucket_summary(per_sample_records, "ordered_signature"),
        "unordered_signature_summary": top_bucket_summary(per_sample_records, "unordered_signature"),
        "per_sample_records": per_sample_records,
    }


def exact_pair_gap_summary(records: list[Dict[str, Any]]) -> Dict[str, Any]:
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
    ]
    by_pair_id: dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        pair_id = record.get("step3_pair_id")
        if pair_id is not None:
            by_pair_id[str(pair_id)].append(record)

    pair_summaries: list[Dict[str, Any]] = []
    metric_gaps: dict[str, list[float]] = defaultdict(list)
    metric_wins: dict[str, Counter[str]] = defaultdict(Counter)

    for pair_id, pair_records in by_pair_id.items():
        if len(pair_records) != 2:
            continue
        variants = {r.get("step3_ordered_variant") for r in pair_records}
        if len(variants) != 2:
            continue
        pair_records = sorted(pair_records, key=lambda r: str(r.get("step3_ordered_variant")))
        rec_a, rec_b = pair_records
        pair_summary: Dict[str, Any] = {
            "step3_pair_id": pair_id,
            "unordered_signature": rec_a.get("unordered_signature"),
            "ordered_variant_a": rec_a.get("step3_ordered_variant"),
            "ordered_variant_b": rec_b.get("step3_ordered_variant"),
        }
        for metric_name in metric_names:
            a_val = rec_a.get(metric_name)
            b_val = rec_b.get(metric_name)
            if a_val is None or b_val is None:
                pair_summary[f"abs_gap_{metric_name}"] = None
                continue
            abs_gap = abs(a_val - b_val)
            pair_summary[f"abs_gap_{metric_name}"] = abs_gap
            metric_gaps[metric_name].append(abs_gap)
            if a_val > b_val:
                metric_wins[metric_name]["variant_a_better"] += 1
            elif b_val > a_val:
                metric_wins[metric_name]["variant_b_better"] += 1
            else:
                metric_wins[metric_name]["tie"] += 1
        pair_summaries.append(pair_summary)

    overall: Dict[str, Any] = {"exact_pair_count": len(pair_summaries)}
    for metric_name in metric_names:
        vals = metric_gaps.get(metric_name, [])
        overall[f"mean_abs_gap_{metric_name}"] = (sum(vals) / len(vals)) if vals else None
        overall[f"wins_{metric_name}"] = dict(metric_wins.get(metric_name, {}))

    by_signature: dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for pair_summary in pair_summaries:
        by_signature[str(pair_summary["unordered_signature"])].append(pair_summary)

    signature_summary: list[Dict[str, Any]] = []
    for unordered_sig, rows in by_signature.items():
        row_summary: Dict[str, Any] = {
            "unordered_signature": unordered_sig,
            "count": len(rows),
        }
        for metric_name in metric_names:
            vals = [r[f"abs_gap_{metric_name}"] for r in rows if r.get(f"abs_gap_{metric_name}") is not None]
            row_summary[f"mean_abs_gap_{metric_name}"] = (sum(vals) / len(vals)) if vals else None
        signature_summary.append(row_summary)
    signature_summary.sort(key=lambda row: (-int(row["count"]), row["unordered_signature"]))

    return {
        "overall": overall,
        "signature_summary": signature_summary,
        "pair_summaries": pair_summaries,
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

    dataset_support = independent_sample_info(dataset)

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

    evaluation = evaluate_independent_pairs(
        proposal_model=proposal_model,
        rewrite_model=rewrite_model,
        loader=loader,
        device=device,
        node_threshold=args.node_threshold,
        edge_threshold=args.edge_threshold,
        use_proposal_conditioning=use_proposal_conditioning,
    )

    reverse_summary = reverse_gap_summary(
        evaluation["per_sample_records"],
        dataset_support["sample_infos"],
        dataset_support["supports_exact_reverse_matching"],
    )
    exact_summary = exact_pair_gap_summary(evaluation["per_sample_records"])

    output_payload = {
        "metadata": {
            "proposal_checkpoint_path": str(proposal_checkpoint_path),
            "rewrite_checkpoint_path": str(rewrite_checkpoint_path),
            "data_path": str(data_path),
            "split_name": args.split_name,
            "node_threshold": args.node_threshold,
            "edge_threshold": args.edge_threshold,
            "rewrite_uses_proposal_conditioning": use_proposal_conditioning,
        },
        "dataset_support": {
            "total_dataset_size": dataset_support["total_dataset_size"],
            "two_event_sample_count": dataset_support["two_event_sample_count"],
            "independent_two_event_sample_count": dataset_support["independent_two_event_sample_count"],
            "supports_exact_reverse_matching": dataset_support["supports_exact_reverse_matching"],
            "exact_reverse_pair_count": dataset_support["exact_reverse_pair_count"],
            "exact_matching_source": dataset_support["exact_matching_source"],
            "dataset_keys": dataset_support["dataset_keys"],
            "evaluation_mode": (
                "exact_pair_and_bucket_level"
                if dataset_support["supports_exact_reverse_matching"]
                else "distribution_level_bucket_only"
            ),
        },
        "results": {
            "num_step3_samples": evaluation["num_step3_samples"],
            "overall": evaluation["overall"],
            "ordered_signature_summary": evaluation["ordered_signature_summary"],
            "unordered_signature_summary": evaluation["unordered_signature_summary"],
            "exact_matched_summary": exact_summary["overall"],
            "exact_matched_signature_summary": exact_summary["signature_summary"],
            "reverse_order_gap_summary": reverse_summary,
        },
    }

    out_path = rewrite_checkpoint_path.parent / f"{args.split_name}_independent_pair_consistency.json"
    save_json(out_path, output_payload)

    print(f"device: {device}")
    print(f"proposal checkpoint: {proposal_checkpoint_path}")
    print(f"rewrite checkpoint: {rewrite_checkpoint_path}")
    print(f"data: {data_path}")
    print(f"dataset size: {len(dataset)}")
    print(f"two-event samples: {dataset_support['two_event_sample_count']}")
    print(f"independent two-event samples: {dataset_support['independent_two_event_sample_count']}")
    print(f"supports exact reverse matching: {dataset_support['supports_exact_reverse_matching']}")
    print(f"exact reverse pair count: {dataset_support['exact_reverse_pair_count']}")
    print(f"exact matching source: {dataset_support['exact_matching_source']}")
    print(f"saved json: {out_path}")
    print(json.dumps(output_payload["results"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
