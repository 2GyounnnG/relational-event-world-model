from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------
# Make project root importable when running:
#   python train/eval_type_breakdown.py
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.collate import graph_event_collate_fn
from data.dataset import GraphEventDataset
from models.baselines import GlobalBaselineConfig, GlobalTransitionBaseline
from models.oracle_local import OracleLocalRewriteConfig, OracleLocalRewriteModel
from models.oracle_local_delta import OracleLocalDeltaRewriteConfig, OracleLocalDeltaRewriteModel


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


def init_bucket(include_scope: bool) -> Dict[str, float]:
    bucket = {
        "num_samples": 0.0,
        "full_type_correct": 0.0,
        "full_type_total": 0.0,
        "changed_type_correct": 0.0,
        "changed_type_total": 0.0,
        "flip_target_type_correct": 0.0,
        "flip_target_type_total": 0.0,
        "nonflip_changed_type_correct": 0.0,
        "nonflip_changed_type_total": 0.0,
    }
    if include_scope:
        bucket["scope_type_correct"] = 0.0
        bucket["scope_type_total"] = 0.0
    return bucket


def bucket_for(stats: Dict[str, Dict[str, float]], name: str, include_scope: bool) -> Dict[str, float]:
    if name not in stats:
        stats[name] = init_bucket(include_scope)
    return stats[name]


def update_bucket(bucket: Dict[str, float], sample_stats: Dict[str, float]) -> None:
    for k, v in sample_stats.items():
        if k in bucket:
            bucket[k] += v


def finalize_bucket(bucket: Dict[str, float], include_scope: bool) -> Dict[str, Any]:
    out = {
        "num_samples": int(bucket["num_samples"]),
        "full_type_acc": safe_div(bucket["full_type_correct"], bucket["full_type_total"]),
        "changed_type_acc": safe_div(bucket["changed_type_correct"], bucket["changed_type_total"]),
        "flip_target_type_acc": safe_div(bucket["flip_target_type_correct"], bucket["flip_target_type_total"]),
        "nonflip_changed_type_acc": safe_div(
            bucket["nonflip_changed_type_correct"],
            bucket["nonflip_changed_type_total"],
        ),
        "full_type_total": int(bucket["full_type_total"]),
        "changed_type_total": int(bucket["changed_type_total"]),
        "flip_target_type_total": int(bucket["flip_target_type_total"]),
        "nonflip_changed_type_total": int(bucket["nonflip_changed_type_total"]),
    }
    if include_scope:
        out["scope_type_acc"] = safe_div(bucket["scope_type_correct"], bucket["scope_type_total"])
        out["scope_type_total"] = int(bucket["scope_type_total"])
    return out


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


def mask_correct_and_total(
    pred_type: torch.Tensor,
    target_type: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[float, float]:
    mask_f = mask.float()
    correct = ((pred_type == target_type).float() * mask_f).sum().item()
    total = mask_f.sum().item()
    return correct, total


def get_type_outputs(
    outputs: Dict[str, torch.Tensor],
    model_kind: str,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    if model_kind == "global_typed":
        return outputs["type_logits"], None
    if model_kind == "oracle_local_typed":
        return outputs["type_logits_full"], outputs["type_logits_local"]
    if model_kind == "oracle_local_delta":
        return outputs["type_logits_full"], outputs["type_logits_local"]
    raise ValueError(f"Unsupported model_kind: {model_kind}")


def run_model(
    model: torch.nn.Module,
    batch: Dict[str, Any],
    model_kind: str,
) -> Dict[str, torch.Tensor]:
    if model_kind == "global_typed":
        return model(batch["node_feats"], batch["adj"])
    if model_kind in {"oracle_local_typed", "oracle_local_delta"}:
        return model(
            node_feats=batch["node_feats"],
            adj=batch["adj"],
            scope_node_mask=batch["event_scope_union_nodes"],
            scope_edge_mask=batch["event_scope_union_edges"],
        )
    raise ValueError(f"Unsupported model_kind: {model_kind}")


def build_sample_stats(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, Any],
    sample_idx: int,
    model_kind: str,
) -> Dict[str, float]:
    full_logits, scope_logits = get_type_outputs(outputs, model_kind)

    node_feats = batch["node_feats"][sample_idx]
    target_node_feats = batch["next_node_feats"][sample_idx]
    node_mask = torch.ones_like(node_feats[:, 0], dtype=torch.bool)
    if "node_mask" in batch:
        node_mask = batch["node_mask"][sample_idx].bool()

    current_type = node_feats[:, 0].long()
    target_type = target_node_feats[:, 0].long()
    pred_full_type = full_logits[sample_idx].argmax(dim=-1)

    changed_node_mask = (batch["changed_nodes"][sample_idx] > 0.5) & node_mask
    flip_target_mask = (current_type != target_type) & node_mask
    nonflip_changed_mask = changed_node_mask & (~flip_target_mask)

    sample_stats: Dict[str, float] = {"num_samples": 1.0}

    full_type_correct, full_type_total = mask_correct_and_total(pred_full_type, target_type, node_mask)
    changed_type_correct, changed_type_total = mask_correct_and_total(pred_full_type, target_type, changed_node_mask)
    flip_target_type_correct, flip_target_type_total = mask_correct_and_total(
        pred_full_type, target_type, flip_target_mask
    )
    nonflip_changed_type_correct, nonflip_changed_type_total = mask_correct_and_total(
        pred_full_type,
        target_type,
        nonflip_changed_mask,
    )

    sample_stats.update(
        {
            "full_type_correct": full_type_correct,
            "full_type_total": full_type_total,
            "changed_type_correct": changed_type_correct,
            "changed_type_total": changed_type_total,
            "flip_target_type_correct": flip_target_type_correct,
            "flip_target_type_total": flip_target_type_total,
            "nonflip_changed_type_correct": nonflip_changed_type_correct,
            "nonflip_changed_type_total": nonflip_changed_type_total,
        }
    )

    if model_kind in {"oracle_local_typed", "oracle_local_delta"}:
        scope_node_mask = (batch["event_scope_union_nodes"][sample_idx] > 0.5) & node_mask
        pred_scope_type = scope_logits[sample_idx].argmax(dim=-1)
        scope_type_correct, scope_type_total = mask_correct_and_total(pred_scope_type, target_type, scope_node_mask)
        sample_stats["scope_type_correct"] = scope_type_correct
        sample_stats["scope_type_total"] = scope_type_total

    return sample_stats


def update_flip_confusions(
    old_to_new: Counter[str],
    old_to_pred: Counter[str],
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, Any],
    model_kind: str,
) -> None:
    full_logits, _ = get_type_outputs(outputs, model_kind)
    pred_type = full_logits.argmax(dim=-1)
    current_type = batch["node_feats"][:, :, 0].long()
    target_type = batch["next_node_feats"][:, :, 0].long()
    node_mask = batch["node_mask"].bool()
    flip_target_mask = (current_type != target_type) & node_mask

    batch_size = batch["node_feats"].shape[0]
    for i in range(batch_size):
        mask = flip_target_mask[i]
        if not mask.any():
            continue
        old_types = current_type[i][mask].tolist()
        new_types = target_type[i][mask].tolist()
        pred_types = pred_type[i][mask].tolist()
        for old_type, new_type, pred_t in zip(old_types, new_types, pred_types):
            old_to_new[f"{old_type}->{new_type}"] += 1
            old_to_pred[f"{old_type}->{pred_t}"] += 1


def print_section(title: str, section: Dict[str, Dict[str, Any]], include_scope: bool) -> None:
    if not section:
        return

    print(f"\n[{title}]")
    header = (
        f"{'bucket':<36} {'n':>6} {'full':>10} {'changed':>10} {'flip':>10} {'nonflip':>10}"
    )
    if include_scope:
        header += f" {'scope':>10}"
    print(header)
    print("-" * len(header))

    for bucket_name in sort_bucket_names(section.keys()):
        m = section[bucket_name]
        row = (
            f"{bucket_name:<36} "
            f"{m['num_samples']:>6d} "
            f"{fmt(m['full_type_acc']):>10} "
            f"{fmt(m['changed_type_acc']):>10} "
            f"{fmt(m['flip_target_type_acc']):>10} "
            f"{fmt(m['nonflip_changed_type_acc']):>10}"
        )
        if include_scope:
            row += f" {fmt(m.get('scope_type_acc')):>10}"
        print(row)


def print_flip_confusion_summary(confusions: Dict[str, Dict[str, int]], top_k: int = 12) -> None:
    print("\n[flip_confusion_target]")
    if not confusions["old_to_new"]:
        print("no flip-target nodes found")
    else:
        for key, count in sorted(confusions["old_to_new"].items(), key=lambda x: (-x[1], x[0]))[:top_k]:
            print(f"{key:<12} {count}")

    print("\n[flip_confusion_predicted]")
    if not confusions["old_to_pred"]:
        print("no flip-target nodes found")
    else:
        for key, count in sorted(confusions["old_to_pred"].items(), key=lambda x: (-x[1], x[0]))[:top_k]:
            print(f"{key:<12} {count}")


@torch.no_grad()
def evaluate_breakdown(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    model_kind: str,
) -> tuple[Dict[str, Dict[str, Dict[str, Any]]], Dict[str, Dict[str, int]]]:
    model.eval()
    include_scope = model_kind in {"oracle_local_typed", "oracle_local_delta"}

    sections_raw: Dict[str, Dict[str, Dict[str, float]]] = {
        "overall": {},
        "by_num_events": {},
        "by_single_event_type": {},
        "by_contains_event_type": {},
        "by_event_signature": {},
        "by_two_event_independence": {},
    }
    flip_old_to_new: Counter[str] = Counter()
    flip_old_to_pred: Counter[str] = Counter()

    for batch in loader:
        require_keys(
            batch,
            [
                "node_feats",
                "adj",
                "next_node_feats",
                "changed_nodes",
                "node_mask",
                "events",
                "num_events",
                "independent_pairs",
            ],
        )
        if include_scope:
            require_keys(batch, ["event_scope_union_nodes", "event_scope_union_edges"])

        batch = move_batch_to_device(batch, device)
        outputs = run_model(model, batch, model_kind)
        update_flip_confusions(flip_old_to_new, flip_old_to_pred, outputs, batch, model_kind)

        batch_size = batch["node_feats"].shape[0]
        events_meta = batch.get("events", [None] * batch_size)
        num_events_meta = batch.get("num_events", [None] * batch_size)
        independent_pairs_meta = batch.get("independent_pairs", [None] * batch_size)

        for i in range(batch_size):
            sample_stats = build_sample_stats(outputs, batch, i, model_kind)
            update_bucket(bucket_for(sections_raw["overall"], "all", include_scope), sample_stats)

            events_item = events_meta[i] if i < len(events_meta) else None
            num_events_item = num_events_meta[i] if i < len(num_events_meta) else None
            independent_pairs_item = independent_pairs_meta[i] if i < len(independent_pairs_meta) else None

            event_types = extract_event_type_list(events_item)
            num_events = infer_num_events(num_events_item, events_item)
            has_independent_pair = infer_has_independent_pair(independent_pairs_item)

            update_bucket(
                bucket_for(sections_raw["by_num_events"], f"num_events={num_events}", include_scope),
                sample_stats,
            )

            if num_events == 1 and len(event_types) == 1:
                update_bucket(
                    bucket_for(
                        sections_raw["by_single_event_type"],
                        f"single_event_type::{event_types[0]}",
                        include_scope,
                    ),
                    sample_stats,
                )

            if event_types:
                unique_types = sorted(set(event_types))
                for event_type in unique_types:
                    update_bucket(
                        bucket_for(
                            sections_raw["by_contains_event_type"],
                            f"contains_event_type::{event_type}",
                            include_scope,
                        ),
                        sample_stats,
                    )
                signature = "+".join(unique_types)
                update_bucket(
                    bucket_for(
                        sections_raw["by_event_signature"],
                        f"event_signature::{signature}",
                        include_scope,
                    ),
                    sample_stats,
                )
            else:
                update_bucket(
                    bucket_for(
                        sections_raw["by_event_signature"],
                        "event_signature::<missing>",
                        include_scope,
                    ),
                    sample_stats,
                )

            if num_events == 2:
                key = "two_event_independent::yes" if has_independent_pair else "two_event_independent::no"
                update_bucket(
                    bucket_for(sections_raw["by_two_event_independence"], key, include_scope),
                    sample_stats,
                )

    sections_final: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for section_name, section_buckets in sections_raw.items():
        sections_final[section_name] = {
            bucket_name: finalize_bucket(bucket_stats, include_scope)
            for bucket_name, bucket_stats in section_buckets.items()
        }

    flip_confusions = {
        "old_to_new": dict(sorted(flip_old_to_new.items(), key=lambda x: (-x[1], x[0]))),
        "old_to_pred": dict(sorted(flip_old_to_pred.items(), key=lambda x: (-x[1], x[0]))),
    }
    return sections_final, flip_confusions


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


def load_model(checkpoint: Dict[str, Any], model_kind: str, device: torch.device) -> torch.nn.Module:
    if "model_config" not in checkpoint:
        raise KeyError("Checkpoint does not contain model_config")

    if model_kind == "global_typed":
        model_config = GlobalBaselineConfig(**checkpoint["model_config"])
        model = GlobalTransitionBaseline(model_config).to(device)
    elif model_kind == "oracle_local_typed":
        model_config = OracleLocalRewriteConfig(**checkpoint["model_config"])
        model = OracleLocalRewriteModel(model_config).to(device)
    elif model_kind == "oracle_local_delta":
        model_config = OracleLocalDeltaRewriteConfig(**checkpoint["model_config"])
        model = OracleLocalDeltaRewriteModel(model_config).to(device)
    else:
        raise ValueError(f"Unsupported model_kind: {model_kind}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_kind",
        type=str,
        required=True,
        choices=["global_typed", "oracle_local_typed", "oracle_local_delta"],
    )
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--split_name", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save_json_path", type=str, default=None)
    args = parser.parse_args()

    checkpoint_path = resolve_path(args.checkpoint_path)
    data_path = resolve_path(args.data_path)
    device = get_device(args.device)
    pin_memory = device.type == "cuda"

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = load_model(checkpoint, args.model_kind, device)

    dataset, loader = build_loader(
        data_path=str(data_path),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    sections, flip_confusions = evaluate_breakdown(
        model=model,
        loader=loader,
        device=device,
        model_kind=args.model_kind,
    )

    if args.save_json_path is not None:
        save_json_path = resolve_path(args.save_json_path)
    else:
        save_json_path = checkpoint_path.parent / f"{args.split_name}_type_breakdown_{args.model_kind}.json"

    payload = {
        "model_kind": args.model_kind,
        "checkpoint_path": str(checkpoint_path),
        "data_path": str(data_path),
        "split_name": args.split_name,
        "device": str(device),
        "dataset_size": len(dataset),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "model_config": checkpoint.get("model_config"),
        "sections": sections,
        "flip_confusions": flip_confusions,
    }
    save_json(save_json_path, payload)

    print(f"device: {device}")
    print(f"model_kind: {args.model_kind}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"data: {data_path}")
    print(f"dataset size: {len(dataset)}")
    print(f"checkpoint epoch: {checkpoint.get('epoch')}")
    print(f"saved json: {save_json_path}")

    include_scope = args.model_kind in {"oracle_local_typed", "oracle_local_delta"}
    print_section("overall", sections["overall"], include_scope=include_scope)
    print_section("by_num_events", sections["by_num_events"], include_scope=include_scope)
    print_section(
        "by_single_event_type",
        sections["by_single_event_type"],
        include_scope=include_scope,
    )
    print_section(
        "by_contains_event_type",
        sections["by_contains_event_type"],
        include_scope=include_scope,
    )
    print_section(
        "by_event_signature",
        sections["by_event_signature"],
        include_scope=include_scope,
    )
    print_section(
        "by_two_event_independence",
        sections["by_two_event_independence"],
        include_scope=include_scope,
    )
    print_flip_confusion_summary(flip_confusions)


if __name__ == "__main__":
    main()
