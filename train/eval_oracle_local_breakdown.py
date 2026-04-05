from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------
# Make project root importable when running:
#   python train/eval_oracle_local_breakdown.py
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import GraphEventDataset
from data.collate import graph_event_collate_fn
from models.oracle_local import (
    OracleLocalRewriteConfig,
    OracleLocalRewriteModel,
    build_valid_edge_mask,
)


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


def extract_event_type_list(events_item: Any) -> List[str]:
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


def infer_has_independent_pair(independent_pairs_item: Any) -> bool:
    if independent_pairs_item is None:
        return False
    if isinstance(independent_pairs_item, bool):
        return independent_pairs_item
    if isinstance(independent_pairs_item, (list, tuple, set, dict)):
        return len(independent_pairs_item) > 0
    return bool(independent_pairs_item)


# ---------------------------------------------------------------------
# Metric numerators / denominators
# ---------------------------------------------------------------------
def type_correct_and_total(
    logits: torch.Tensor,
    target_node_feats: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[float, float]:
    pred = logits.argmax(dim=-1)
    target = target_node_feats[:, 0].long()
    mask_f = mask.float()
    correct = ((pred == target).float() * mask_f).sum().item()
    total = mask_f.sum().item()
    return correct, total


def state_abs_sum_and_total_dims(
    pred_state: torch.Tensor,
    target_node_feats: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[float, float]:
    target_state = target_node_feats[:, 1:]
    mask_f = mask.float().unsqueeze(-1)
    abs_sum = (torch.abs(pred_state - target_state) * mask_f).sum().item()
    total_dims = (mask.float().sum() * pred_state.shape[-1]).item()
    return abs_sum, total_dims


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


# ---------------------------------------------------------------------
# Bucket accumulation
# ---------------------------------------------------------------------
def init_bucket() -> Dict[str, float]:
    return {
        "num_samples": 0.0,
        "scope_node_fraction_sum": 0.0,
        "scope_edge_fraction_sum": 0.0,
        "full_type_correct": 0.0,
        "full_type_total": 0.0,
        "full_state_abs_sum": 0.0,
        "full_state_total_dims": 0.0,
        "full_edge_correct": 0.0,
        "full_edge_total": 0.0,
        "scope_type_correct": 0.0,
        "scope_type_total": 0.0,
        "scope_state_abs_sum": 0.0,
        "scope_state_total_dims": 0.0,
        "scope_edge_correct": 0.0,
        "scope_edge_total": 0.0,
        "changed_node_type_correct": 0.0,
        "changed_node_type_total": 0.0,
        "changed_node_state_abs_sum": 0.0,
        "changed_node_state_total_dims": 0.0,
        "context_node_type_correct": 0.0,
        "context_node_type_total": 0.0,
        "context_node_state_abs_sum": 0.0,
        "context_node_state_total_dims": 0.0,
        "changed_edge_correct": 0.0,
        "changed_edge_total": 0.0,
        "context_edge_correct": 0.0,
        "context_edge_total": 0.0,
    }


def bucket_for(stats: Dict[str, Dict[str, float]], name: str) -> Dict[str, float]:
    if name not in stats:
        stats[name] = init_bucket()
    return stats[name]


def update_bucket(bucket: Dict[str, float], sample_stats: Dict[str, float]) -> None:
    for k, v in sample_stats.items():
        bucket[k] += v


def safe_div(num: float, den: float) -> Optional[float]:
    if den <= 0:
        return None
    return num / den


def finalize_bucket(bucket: Dict[str, float]) -> Dict[str, Any]:
    return {
        "num_samples": int(bucket["num_samples"]),
        "avg_scope_node_fraction": safe_div(bucket["scope_node_fraction_sum"], bucket["num_samples"]),
        "avg_scope_edge_fraction": safe_div(bucket["scope_edge_fraction_sum"], bucket["num_samples"]),
        "full_type_acc": safe_div(bucket["full_type_correct"], bucket["full_type_total"]),
        "full_state_mae": safe_div(bucket["full_state_abs_sum"], bucket["full_state_total_dims"]),
        "full_edge_acc": safe_div(bucket["full_edge_correct"], bucket["full_edge_total"]),
        "scope_type_acc": safe_div(bucket["scope_type_correct"], bucket["scope_type_total"]),
        "scope_state_mae": safe_div(bucket["scope_state_abs_sum"], bucket["scope_state_total_dims"]),
        "scope_edge_acc": safe_div(bucket["scope_edge_correct"], bucket["scope_edge_total"]),
        "changed_node_type_acc": safe_div(bucket["changed_node_type_correct"], bucket["changed_node_type_total"]),
        "changed_node_state_mae": safe_div(bucket["changed_node_state_abs_sum"], bucket["changed_node_state_total_dims"]),
        "context_node_type_acc": safe_div(bucket["context_node_type_correct"], bucket["context_node_type_total"]),
        "context_node_state_mae": safe_div(bucket["context_node_state_abs_sum"], bucket["context_node_state_total_dims"]),
        "changed_edge_acc": safe_div(bucket["changed_edge_correct"], bucket["changed_edge_total"]),
        "context_edge_acc": safe_div(bucket["context_edge_correct"], bucket["context_edge_total"]),
        "full_type_total": int(bucket["full_type_total"]),
        "scope_type_total": int(bucket["scope_type_total"]),
        "scope_edge_total": int(bucket["scope_edge_total"]),
        "changed_edge_total": int(bucket["changed_edge_total"]),
        "context_edge_total": int(bucket["context_edge_total"]),
    }


def build_sample_stats(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, Any],
    sample_idx: int,
) -> Dict[str, float]:
    node_mask = batch["node_mask"][sample_idx].bool()
    valid_edge_mask = build_valid_edge_mask(batch["node_mask"][sample_idx : sample_idx + 1]).squeeze(0).bool()

    scope_node_mask = (batch["event_scope_union_nodes"][sample_idx] > 0.5) & node_mask
    scope_edge_mask = (batch["event_scope_union_edges"][sample_idx] > 0.5) & valid_edge_mask

    if "changed_nodes" in batch:
        changed_node_mask = (batch["changed_nodes"][sample_idx] > 0.5) & node_mask
    else:
        changed_node_mask = torch.zeros_like(node_mask)
    if "changed_edges" in batch:
        changed_edge_mask = (batch["changed_edges"][sample_idx] > 0.5) & valid_edge_mask
    else:
        changed_edge_mask = torch.zeros_like(valid_edge_mask)

    context_node_mask = scope_node_mask & (~changed_node_mask)
    context_edge_mask = scope_edge_mask & (~changed_edge_mask)

    target_node_feats = batch["next_node_feats"][sample_idx]
    target_adj = batch["next_adj"][sample_idx]

    sample_stats: Dict[str, float] = {
        "num_samples": 1.0,
        "scope_node_fraction_sum": safe_div(scope_node_mask.float().sum().item(), node_mask.float().sum().item()) or 0.0,
        "scope_edge_fraction_sum": safe_div(scope_edge_mask.float().sum().item(), valid_edge_mask.float().sum().item()) or 0.0,
    }

    full_type_correct, full_type_total = type_correct_and_total(
        outputs["type_logits_full"][sample_idx],
        target_node_feats,
        node_mask,
    )
    full_state_abs_sum, full_state_total_dims = state_abs_sum_and_total_dims(
        outputs["state_pred_full"][sample_idx],
        target_node_feats,
        node_mask,
    )
    full_edge_correct, full_edge_total = edge_correct_and_total(
        outputs["edge_logits_full"][sample_idx],
        target_adj,
        valid_edge_mask,
    )

    scope_type_correct, scope_type_total = type_correct_and_total(
        outputs["type_logits_local"][sample_idx],
        target_node_feats,
        scope_node_mask,
    )
    scope_state_abs_sum, scope_state_total_dims = state_abs_sum_and_total_dims(
        outputs["state_pred_local"][sample_idx],
        target_node_feats,
        scope_node_mask,
    )
    scope_edge_correct, scope_edge_total = edge_correct_and_total(
        outputs["edge_logits_local"][sample_idx],
        target_adj,
        scope_edge_mask,
    )

    changed_node_type_correct, changed_node_type_total = type_correct_and_total(
        outputs["type_logits_local"][sample_idx],
        target_node_feats,
        changed_node_mask,
    )
    changed_node_state_abs_sum, changed_node_state_total_dims = state_abs_sum_and_total_dims(
        outputs["state_pred_local"][sample_idx],
        target_node_feats,
        changed_node_mask,
    )

    context_node_type_correct, context_node_type_total = type_correct_and_total(
        outputs["type_logits_local"][sample_idx],
        target_node_feats,
        context_node_mask,
    )
    context_node_state_abs_sum, context_node_state_total_dims = state_abs_sum_and_total_dims(
        outputs["state_pred_local"][sample_idx],
        target_node_feats,
        context_node_mask,
    )

    changed_edge_correct, changed_edge_total = edge_correct_and_total(
        outputs["edge_logits_local"][sample_idx],
        target_adj,
        changed_edge_mask,
    )
    context_edge_correct, context_edge_total = edge_correct_and_total(
        outputs["edge_logits_local"][sample_idx],
        target_adj,
        context_edge_mask,
    )

    sample_stats.update(
        {
            "full_type_correct": full_type_correct,
            "full_type_total": full_type_total,
            "full_state_abs_sum": full_state_abs_sum,
            "full_state_total_dims": full_state_total_dims,
            "full_edge_correct": full_edge_correct,
            "full_edge_total": full_edge_total,
            "scope_type_correct": scope_type_correct,
            "scope_type_total": scope_type_total,
            "scope_state_abs_sum": scope_state_abs_sum,
            "scope_state_total_dims": scope_state_total_dims,
            "scope_edge_correct": scope_edge_correct,
            "scope_edge_total": scope_edge_total,
            "changed_node_type_correct": changed_node_type_correct,
            "changed_node_type_total": changed_node_type_total,
            "changed_node_state_abs_sum": changed_node_state_abs_sum,
            "changed_node_state_total_dims": changed_node_state_total_dims,
            "context_node_type_correct": context_node_type_correct,
            "context_node_type_total": context_node_type_total,
            "context_node_state_abs_sum": context_node_state_abs_sum,
            "context_node_state_total_dims": context_node_state_total_dims,
            "changed_edge_correct": changed_edge_correct,
            "changed_edge_total": changed_edge_total,
            "context_edge_correct": context_edge_correct,
            "context_edge_total": context_edge_total,
        }
    )
    return sample_stats


def sort_bucket_names(names: Iterable[str]) -> List[str]:
    def _key(name: str):
        if name == "all":
            return (0, name)
        if name.startswith("num_events="):
            return (1, name)
        if name.startswith("single_event_type::"):
            return (2, name)
        if name.startswith("contains_event_type::"):
            return (3, name)
        if name.startswith("event_signature::"):
            return (4, name)
        if name.startswith("two_event_independent::"):
            return (5, name)
        return (99, name)

    return sorted(names, key=_key)


def fmt(v: Optional[float]) -> str:
    if v is None:
        return "NA"
    return f"{v:.6f}"


def print_section(title: str, section: Dict[str, Dict[str, Any]]) -> None:
    if not section:
        return

    print(f"\n[{title}]")
    header = (
        f"{'bucket':<36} {'n':>6} {'scope_type':>12} {'scope_state':>12} {'scope_edge':>12} "
        f"{'changed_edge':>13} {'context_edge':>13} {'scope_n_frac':>12} {'scope_e_frac':>12}"
    )
    print(header)
    print("-" * len(header))

    for bucket_name in sort_bucket_names(section.keys()):
        m = section[bucket_name]
        print(
            f"{bucket_name:<36} "
            f"{m['num_samples']:>6d} "
            f"{fmt(m['scope_type_acc']):>12} "
            f"{fmt(m['scope_state_mae']):>12} "
            f"{fmt(m['scope_edge_acc']):>12} "
            f"{fmt(m['changed_edge_acc']):>13} "
            f"{fmt(m['context_edge_acc']):>13} "
            f"{fmt(m['avg_scope_node_fraction']):>12} "
            f"{fmt(m['avg_scope_edge_fraction']):>12}"
        )


@torch.no_grad()
def evaluate_breakdown(
    model: OracleLocalRewriteModel,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    model.eval()

    sections_raw: Dict[str, Dict[str, Dict[str, float]]] = {
        "overall": {},
        "by_num_events": {},
        "by_single_event_type": {},
        "by_contains_event_type": {},
        "by_event_signature": {},
        "by_two_event_independence": {},
    }

    for batch in loader:
        require_keys(
            batch,
            [
                "node_feats",
                "adj",
                "next_node_feats",
                "next_adj",
                "node_mask",
                "event_scope_union_nodes",
                "event_scope_union_edges",
            ],
        )
        batch = move_batch_to_device(batch, device)

        outputs = model(
            node_feats=batch["node_feats"],
            adj=batch["adj"],
            scope_node_mask=batch["event_scope_union_nodes"],
            scope_edge_mask=batch["event_scope_union_edges"],
        )

        batch_size = batch["node_feats"].shape[0]
        events_meta = batch.get("events", [None] * batch_size)
        num_events_meta = batch.get("num_events", [None] * batch_size)
        independent_pairs_meta = batch.get("independent_pairs", [None] * batch_size)

        for i in range(batch_size):
            sample_stats = build_sample_stats(outputs, batch, i)
            update_bucket(bucket_for(sections_raw["overall"], "all"), sample_stats)

            events_item = events_meta[i] if i < len(events_meta) else None
            num_events_item = num_events_meta[i] if i < len(num_events_meta) else None
            independent_pairs_item = independent_pairs_meta[i] if i < len(independent_pairs_meta) else None

            event_types = extract_event_type_list(events_item)
            num_events = infer_num_events(num_events_item, events_item)
            has_independent_pair = infer_has_independent_pair(independent_pairs_item)

            update_bucket(
                bucket_for(sections_raw["by_num_events"], f"num_events={num_events}"),
                sample_stats,
            )

            if num_events == 1 and len(event_types) == 1:
                update_bucket(
                    bucket_for(sections_raw["by_single_event_type"], f"single_event_type::{event_types[0]}"),
                    sample_stats,
                )

            if event_types:
                unique_types = sorted(set(event_types))
                for event_type in unique_types:
                    update_bucket(
                        bucket_for(sections_raw["by_contains_event_type"], f"contains_event_type::{event_type}"),
                        sample_stats,
                    )
                signature = "+".join(unique_types)
                update_bucket(
                    bucket_for(sections_raw["by_event_signature"], f"event_signature::{signature}"),
                    sample_stats,
                )
            else:
                update_bucket(
                    bucket_for(sections_raw["by_event_signature"], "event_signature::<missing>"),
                    sample_stats,
                )

            if num_events == 2:
                key = "two_event_independent::yes" if has_independent_pair else "two_event_independent::no"
                update_bucket(bucket_for(sections_raw["by_two_event_independence"], key), sample_stats)

    sections_final: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for section_name, section_buckets in sections_raw.items():
        sections_final[section_name] = {
            bucket_name: finalize_bucket(bucket_stats)
            for bucket_name, bucket_stats in section_buckets.items()
        }
    return sections_final


def build_loader(data_path: str, batch_size: int, num_workers: int, pin_memory: bool) -> tuple[GraphEventDataset, DataLoader]:
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


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--split_name", type=str, default="eval")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save_json_path", type=str, default=None)
    args = parser.parse_args()

    device = get_device(args.device)
    pin_memory = device.type == "cuda"

    checkpoint_path = resolve_path(args.checkpoint_path)
    data_path = resolve_path(args.data_path)

    ckpt = torch.load(checkpoint_path, map_location='cpu')
    if "model_config" not in ckpt:
        raise KeyError(f"Checkpoint does not contain model_config: {checkpoint_path}")

    model_config = OracleLocalRewriteConfig(**ckpt["model_config"])
    model = OracleLocalRewriteModel(model_config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dataset, loader = build_loader(
        data_path=str(data_path),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    sections = evaluate_breakdown(model=model, loader=loader, device=device)

    payload = {
        "checkpoint_path": str(checkpoint_path),
        "data_path": str(data_path),
        "split_name": args.split_name,
        "device": str(device),
        "dataset_size": len(dataset),
        "checkpoint_epoch": ckpt.get("epoch"),
        "model_config": ckpt.get("model_config"),
        "sections": sections,
    }

    if args.save_json_path is not None:
        save_json_path = resolve_path(args.save_json_path)
    else:
        save_json_path = checkpoint_path.parent / f"{args.split_name}_breakdown.json"
    save_json(save_json_path, payload)

    print(f"device: {device}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"data: {data_path}")
    print(f"dataset size: {len(dataset)}")
    print(f"checkpoint epoch: {ckpt.get('epoch')}")
    print(f"saved json: {save_json_path}")

    print_section("overall", sections["overall"])
    print_section("by_num_events", sections["by_num_events"])
    print_section("by_single_event_type", sections["by_single_event_type"])
    print_section("by_contains_event_type", sections["by_contains_event_type"])
    print_section("by_event_signature", sections["by_event_signature"])
    print_section("by_two_event_independence", sections["by_two_event_independence"])


if __name__ == "__main__":
    main()
