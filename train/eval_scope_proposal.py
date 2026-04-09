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
#   python train/eval_scope_proposal.py
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.collate import graph_event_collate_fn
from data.dataset import GraphEventDataset
from models.oracle_local import build_valid_edge_mask
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


def safe_div(num: float, den: float) -> Optional[float]:
    if den <= 0:
        return None
    return num / den


def init_bucket() -> Dict[str, float]:
    return {
        "num_samples": 0.0,
        "node_tp": 0.0,
        "node_pred_pos": 0.0,
        "node_true_pos": 0.0,
        "edge_tp": 0.0,
        "edge_pred_pos": 0.0,
        "edge_true_pos": 0.0,
        "changed_node_covered": 0.0,
        "changed_node_total": 0.0,
        "changed_edge_covered": 0.0,
        "changed_edge_total": 0.0,
        "pred_node_scope_fraction_sum": 0.0,
        "oracle_node_scope_fraction_sum": 0.0,
        "pred_edge_scope_fraction_sum": 0.0,
        "oracle_edge_scope_fraction_sum": 0.0,
    }


def bucket_for(stats: Dict[str, Dict[str, float]], name: str) -> Dict[str, float]:
    if name not in stats:
        stats[name] = init_bucket()
    return stats[name]


def update_bucket(bucket: Dict[str, float], sample_stats: Dict[str, float]) -> None:
    for k, v in sample_stats.items():
        bucket[k] += v


def finalize_bucket(bucket: Dict[str, float]) -> Dict[str, Any]:
    node_precision = safe_div(bucket["node_tp"], bucket["node_pred_pos"])
    node_recall = safe_div(bucket["node_tp"], bucket["node_true_pos"])
    node_f1 = None
    if node_precision is not None and node_recall is not None and node_precision + node_recall > 0:
        node_f1 = 2.0 * node_precision * node_recall / (node_precision + node_recall)

    edge_precision = safe_div(bucket["edge_tp"], bucket["edge_pred_pos"])
    edge_recall = safe_div(bucket["edge_tp"], bucket["edge_true_pos"])
    edge_f1 = None
    if edge_precision is not None and edge_recall is not None and edge_precision + edge_recall > 0:
        edge_f1 = 2.0 * edge_precision * edge_recall / (edge_precision + edge_recall)

    return {
        "num_samples": int(bucket["num_samples"]),
        "node_scope_precision": node_precision,
        "node_scope_recall": node_recall,
        "node_scope_f1": node_f1,
        "edge_scope_precision": edge_precision,
        "edge_scope_recall": edge_recall,
        "edge_scope_f1": edge_f1,
        "changed_node_coverage": safe_div(bucket["changed_node_covered"], bucket["changed_node_total"]),
        "changed_edge_coverage": safe_div(bucket["changed_edge_covered"], bucket["changed_edge_total"]),
        "avg_predicted_node_scope_fraction": safe_div(bucket["pred_node_scope_fraction_sum"], bucket["num_samples"]),
        "avg_oracle_node_scope_fraction": safe_div(bucket["oracle_node_scope_fraction_sum"], bucket["num_samples"]),
        "avg_predicted_edge_scope_fraction": safe_div(bucket["pred_edge_scope_fraction_sum"], bucket["num_samples"]),
        "avg_oracle_edge_scope_fraction": safe_div(bucket["oracle_edge_scope_fraction_sum"], bucket["num_samples"]),
        "changed_node_total": int(bucket["changed_node_total"]),
        "changed_edge_total": int(bucket["changed_edge_total"]),
    }


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


def build_sample_stats(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, Any],
    sample_idx: int,
    node_threshold: float,
    edge_threshold: float,
) -> Dict[str, float]:
    node_mask = batch["node_mask"][sample_idx].bool()
    valid_edge_mask = build_valid_edge_mask(batch["node_mask"][sample_idx : sample_idx + 1]).squeeze(0).bool()

    pred_node_scope_mask = (torch.sigmoid(outputs["node_scope_logits"][sample_idx]) >= node_threshold) & node_mask
    oracle_node_scope_mask = (batch["event_scope_union_nodes"][sample_idx] > 0.5) & node_mask
    pred_edge_scope_mask = (torch.sigmoid(outputs["edge_scope_logits"][sample_idx]) >= edge_threshold) & valid_edge_mask
    oracle_edge_scope_mask = (batch["event_scope_union_edges"][sample_idx] > 0.5) & valid_edge_mask
    changed_node_mask = (batch["changed_nodes"][sample_idx] > 0.5) & node_mask
    changed_edge_mask = (batch["changed_edges"][sample_idx] > 0.5) & valid_edge_mask

    node_tp = (pred_node_scope_mask & oracle_node_scope_mask).float().sum().item()
    node_pred_pos = pred_node_scope_mask.float().sum().item()
    node_true_pos = oracle_node_scope_mask.float().sum().item()
    edge_tp = (pred_edge_scope_mask & oracle_edge_scope_mask).float().sum().item()
    edge_pred_pos = pred_edge_scope_mask.float().sum().item()
    edge_true_pos = oracle_edge_scope_mask.float().sum().item()
    changed_node_covered = (pred_node_scope_mask & changed_node_mask).float().sum().item()
    changed_node_total = changed_node_mask.float().sum().item()
    changed_edge_covered = (pred_edge_scope_mask & changed_edge_mask).float().sum().item()
    changed_edge_total = changed_edge_mask.float().sum().item()

    sample_stats: Dict[str, float] = {
        "num_samples": 1.0,
        "node_tp": node_tp,
        "node_pred_pos": node_pred_pos,
        "node_true_pos": node_true_pos,
        "edge_tp": edge_tp,
        "edge_pred_pos": edge_pred_pos,
        "edge_true_pos": edge_true_pos,
        "changed_node_covered": changed_node_covered,
        "changed_node_total": changed_node_total,
        "changed_edge_covered": changed_edge_covered,
        "changed_edge_total": changed_edge_total,
        "pred_node_scope_fraction_sum": safe_div(
            pred_node_scope_mask.float().sum().item(),
            node_mask.float().sum().item(),
        ) or 0.0,
        "oracle_node_scope_fraction_sum": safe_div(
            oracle_node_scope_mask.float().sum().item(),
            node_mask.float().sum().item(),
        ) or 0.0,
        "pred_edge_scope_fraction_sum": safe_div(
            pred_edge_scope_mask.float().sum().item(),
            valid_edge_mask.float().sum().item(),
        ) or 0.0,
        "oracle_edge_scope_fraction_sum": safe_div(
            oracle_edge_scope_mask.float().sum().item(),
            valid_edge_mask.float().sum().item(),
        ) or 0.0,
    }
    return sample_stats


def print_section(title: str, section: Dict[str, Dict[str, Any]]) -> None:
    if not section:
        return

    print(f"\n[{title}]")
    header = (
        f"{'bucket':<36} {'n':>6} {'n_f1':>10} {'e_f1':>10} {'chg_node':>10} {'chg_edge':>10} "
        f"{'n_pred':>10} {'n_oracle':>10} {'e_pred':>10} {'e_oracle':>10}"
    )
    print(header)
    print("-" * len(header))

    for bucket_name in sort_bucket_names(section.keys()):
        m = section[bucket_name]
        print(
            f"{bucket_name:<36} "
            f"{m['num_samples']:>6d} "
            f"{fmt(m['node_scope_f1']):>10} "
            f"{fmt(m['edge_scope_f1']):>10} "
            f"{fmt(m['changed_node_coverage']):>10} "
            f"{fmt(m['changed_edge_coverage']):>10} "
            f"{fmt(m['avg_predicted_node_scope_fraction']):>10} "
            f"{fmt(m['avg_oracle_node_scope_fraction']):>10} "
            f"{fmt(m['avg_predicted_edge_scope_fraction']):>10} "
            f"{fmt(m['avg_oracle_edge_scope_fraction']):>10}"
        )


@torch.no_grad()
def evaluate_breakdown(
    model: ScopeProposalModel,
    loader: DataLoader,
    device: torch.device,
    node_threshold: float = 0.5,
    edge_threshold: float = 0.5,
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
                "node_mask",
                "changed_nodes",
                "changed_edges",
                "event_scope_union_nodes",
                "event_scope_union_edges",
                "events",
                "num_events",
                "independent_pairs",
            ],
        )
        batch = move_batch_to_device(batch, device)

        outputs = model(
            node_feats=batch["node_feats"],
            adj=batch["adj"],
        )

        batch_size = batch["node_feats"].shape[0]
        events_meta = batch.get("events", [None] * batch_size)
        num_events_meta = batch.get("num_events", [None] * batch_size)
        independent_pairs_meta = batch.get("independent_pairs", [None] * batch_size)

        for i in range(batch_size):
            sample_stats = build_sample_stats(
                outputs,
                batch,
                i,
                node_threshold=node_threshold,
                edge_threshold=edge_threshold,
            )
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


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--split_name", type=str, default="eval")
    parser.add_argument("--node_threshold", type=float, default=0.5)
    parser.add_argument("--edge_threshold", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    checkpoint_path = resolve_path(args.checkpoint_path)
    data_path = resolve_path(args.data_path)
    device = get_device(args.device)
    pin_memory = device.type == "cuda"

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_config = ScopeProposalConfig(**checkpoint["model_config"])
    model = ScopeProposalModel(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset, loader = build_loader(
        data_path=str(data_path),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    sections = evaluate_breakdown(
        model=model,
        loader=loader,
        device=device,
        node_threshold=args.node_threshold,
        edge_threshold=args.edge_threshold,
    )

    out_path = checkpoint_path.parent / f"{args.split_name}_scope_proposal.json"
    save_json(
        out_path,
        {
            "checkpoint_path": str(checkpoint_path),
            "data_path": str(data_path),
            "split_name": args.split_name,
            "node_threshold": args.node_threshold,
            "edge_threshold": args.edge_threshold,
            "checkpoint_epoch": checkpoint.get("epoch", None),
            "sections": sections,
        },
    )

    print(f"device: {device}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"data: {data_path}")
    print(f"dataset size: {len(dataset)}")
    print(f"checkpoint epoch: {checkpoint.get('epoch', 'NA')}")
    print(f"node threshold: {args.node_threshold}")
    print(f"edge threshold: {args.edge_threshold}")
    print(f"saved json: {out_path}")

    for section_name in [
        "overall",
        "by_num_events",
        "by_single_event_type",
        "by_contains_event_type",
        "by_event_signature",
        "by_two_event_independence",
    ]:
        print_section(section_name, sections.get(section_name, {}))


if __name__ == "__main__":
    main()
