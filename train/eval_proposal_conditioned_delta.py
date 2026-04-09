from __future__ import annotations

import argparse
from collections import Counter
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------
# Make project root importable when running:
#   python train/eval_proposal_conditioned_delta.py
# ---------------------------------------------------------------------
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


@torch.no_grad()
def evaluate(
    proposal_model: ScopeProposalModel,
    rewrite_model: OracleLocalDeltaRewriteModel,
    loader: DataLoader,
    device: torch.device,
    node_threshold: float,
    edge_threshold: float,
) -> Dict[str, Any]:
    proposal_model.eval()
    rewrite_model.eval()

    acc = {
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
        "num_samples": 0.0,
        "scope_edge_correct": 0.0,
        "scope_edge_total": 0.0,
        "scope_edge_delta_correct": 0.0,
        "scope_edge_delta_total": 0.0,
        "scope_keep_correct": 0.0,
        "scope_keep_total": 0.0,
        "scope_add_correct": 0.0,
        "scope_add_total": 0.0,
        "scope_delete_correct": 0.0,
        "scope_delete_total": 0.0,
        "changed_edge_correct": 0.0,
        "changed_edge_total_rewrite": 0.0,
        "context_edge_correct": 0.0,
        "context_edge_total": 0.0,
        "full_type_correct": 0.0,
        "full_type_total": 0.0,
        "changed_type_correct": 0.0,
        "changed_type_total": 0.0,
        "flip_target_type_correct": 0.0,
        "flip_target_type_total": 0.0,
        "nonflip_changed_type_correct": 0.0,
        "nonflip_changed_type_total": 0.0,
        "scope_type_correct": 0.0,
        "scope_type_total": 0.0,
        "motif_full_type_correct": 0.0,
        "motif_full_type_total": 0.0,
        "motif_changed_type_correct": 0.0,
        "motif_changed_type_total": 0.0,
        "motif_flip_target_type_correct": 0.0,
        "motif_flip_target_type_total": 0.0,
        "motif_nonflip_changed_type_correct": 0.0,
        "motif_nonflip_changed_type_total": 0.0,
        "motif_scope_type_correct": 0.0,
        "motif_scope_type_total": 0.0,
        "motif_num_samples": 0.0,
    }
    flip_old_to_pred: Counter[str] = Counter()

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
            ],
        )
        batch = move_batch_to_device(batch, device)

        proposal_outputs = proposal_model(
            node_feats=batch["node_feats"],
            adj=batch["adj"],
        )

        node_mask = batch["node_mask"].bool()
        valid_edge_mask = build_valid_edge_mask(batch["node_mask"]).bool()
        pred_scope_mask = (torch.sigmoid(proposal_outputs["scope_logits"]) >= node_threshold) & node_mask
        oracle_scope_mask = (batch["event_scope_union_nodes"] > 0.5) & node_mask
        oracle_edge_scope_mask = (batch["event_scope_union_edges"] > 0.5) & valid_edge_mask
        changed_node_mask = (batch["changed_nodes"] > 0.5) & node_mask
        changed_edge_mask = (batch["changed_edges"] > 0.5) & valid_edge_mask
        current_type = batch["node_feats"][:, :, 0].long()
        target_type = batch["next_node_feats"][:, :, 0].long()
        flip_target_mask = (current_type != target_type) & node_mask
        nonflip_changed_mask = changed_node_mask & (~flip_target_mask)
        if "edge_scope_logits" in proposal_outputs:
            pred_edge_scope_mask = (
                (torch.sigmoid(proposal_outputs["edge_scope_logits"]) >= edge_threshold) & valid_edge_mask
            )
        else:
            pred_edge_scope_mask = pred_scope_mask.unsqueeze(2) & pred_scope_mask.unsqueeze(1) & valid_edge_mask
        context_edge_mask = pred_edge_scope_mask & (~changed_edge_mask)

        rewrite_outputs = rewrite_model(
            node_feats=batch["node_feats"],
            adj=batch["adj"],
            scope_node_mask=pred_scope_mask.float(),
            scope_edge_mask=pred_edge_scope_mask.float(),
        )
        pred_full_type = rewrite_outputs["type_logits_full"].argmax(dim=-1)
        pred_scope_type = rewrite_outputs["type_logits_local"].argmax(dim=-1)

        batch_size = batch["node_feats"].shape[0]
        acc["num_samples"] += batch_size

        acc["node_tp"] += (pred_scope_mask & oracle_scope_mask).float().sum().item()
        acc["node_pred_pos"] += pred_scope_mask.float().sum().item()
        acc["node_true_pos"] += oracle_scope_mask.float().sum().item()
        acc["edge_tp"] += (pred_edge_scope_mask & oracle_edge_scope_mask).float().sum().item()
        acc["edge_pred_pos"] += pred_edge_scope_mask.float().sum().item()
        acc["edge_true_pos"] += oracle_edge_scope_mask.float().sum().item()
        acc["changed_node_covered"] += (pred_scope_mask & changed_node_mask).float().sum().item()
        acc["changed_node_total"] += changed_node_mask.float().sum().item()
        acc["changed_edge_covered"] += (pred_edge_scope_mask & changed_edge_mask).float().sum().item()
        acc["changed_edge_total"] += changed_edge_mask.float().sum().item()
        c, t = type_correct_and_total(pred_full_type, target_type, node_mask)
        acc["full_type_correct"] += c
        acc["full_type_total"] += t
        c, t = type_correct_and_total(pred_full_type, target_type, changed_node_mask)
        acc["changed_type_correct"] += c
        acc["changed_type_total"] += t
        c, t = type_correct_and_total(pred_full_type, target_type, flip_target_mask)
        acc["flip_target_type_correct"] += c
        acc["flip_target_type_total"] += t
        c, t = type_correct_and_total(pred_full_type, target_type, nonflip_changed_mask)
        acc["nonflip_changed_type_correct"] += c
        acc["nonflip_changed_type_total"] += t
        c, t = type_correct_and_total(pred_scope_type, target_type, pred_scope_mask)
        acc["scope_type_correct"] += c
        acc["scope_type_total"] += t

        for i in range(batch_size):
            sample_node_mask = node_mask[i]
            sample_valid_edge_mask = valid_edge_mask[i]
            sample_pred_scope_mask = pred_scope_mask[i]
            sample_oracle_scope_mask = oracle_scope_mask[i]
            sample_pred_edge_scope_mask = pred_edge_scope_mask[i]
            sample_oracle_edge_scope_mask = oracle_edge_scope_mask[i]
            sample_changed_edge_mask = changed_edge_mask[i]
            sample_context_edge_mask = context_edge_mask[i]
            sample_current_type = current_type[i]
            sample_target_type = target_type[i]
            sample_pred_full_type = pred_full_type[i]
            sample_pred_scope_type = pred_scope_type[i]
            sample_changed_node_mask = changed_node_mask[i]
            sample_flip_target_mask = flip_target_mask[i]
            sample_nonflip_changed_mask = nonflip_changed_mask[i]

            acc["pred_node_scope_fraction_sum"] += (
                sample_pred_scope_mask.float().sum().item() / sample_node_mask.float().sum().item()
            )
            acc["oracle_node_scope_fraction_sum"] += (
                sample_oracle_scope_mask.float().sum().item() / sample_node_mask.float().sum().item()
            )
            acc["pred_edge_scope_fraction_sum"] += (
                sample_pred_edge_scope_mask.float().sum().item() / sample_valid_edge_mask.float().sum().item()
            )
            acc["oracle_edge_scope_fraction_sum"] += (
                sample_oracle_edge_scope_mask.float().sum().item() / sample_valid_edge_mask.float().sum().item()
            )

            for old_type, pred_t in zip(
                sample_current_type[sample_flip_target_mask].tolist(),
                sample_pred_full_type[sample_flip_target_mask].tolist(),
            ):
                flip_old_to_pred[f"{old_type}->{pred_t}"] += 1

            events_item = batch["events"][i] if i < len(batch["events"]) else None
            num_events_item = batch["num_events"][i] if i < len(batch["num_events"]) else None
            event_types = extract_event_type_list(events_item)
            num_events = infer_num_events(num_events_item, events_item)
            if num_events == 1 and len(event_types) == 1 and event_types[0] == "motif_type_flip":
                acc["motif_num_samples"] += 1
                c, t = type_correct_and_total(sample_pred_full_type, sample_target_type, sample_node_mask)
                acc["motif_full_type_correct"] += c
                acc["motif_full_type_total"] += t
                c, t = type_correct_and_total(sample_pred_full_type, sample_target_type, sample_changed_node_mask)
                acc["motif_changed_type_correct"] += c
                acc["motif_changed_type_total"] += t
                c, t = type_correct_and_total(sample_pred_full_type, sample_target_type, sample_flip_target_mask)
                acc["motif_flip_target_type_correct"] += c
                acc["motif_flip_target_type_total"] += t
                c, t = type_correct_and_total(sample_pred_full_type, sample_target_type, sample_nonflip_changed_mask)
                acc["motif_nonflip_changed_type_correct"] += c
                acc["motif_nonflip_changed_type_total"] += t
                c, t = type_correct_and_total(sample_pred_scope_type, sample_target_type, sample_pred_scope_mask)
                acc["motif_scope_type_correct"] += c
                acc["motif_scope_type_total"] += t

            target_adj = batch["next_adj"][i]
            current_adj = batch["adj"][i]

            c, t = edge_correct_and_total(
                rewrite_outputs["edge_logits_local"][i],
                target_adj,
                sample_pred_edge_scope_mask.float(),
            )
            acc["scope_edge_correct"] += c
            acc["scope_edge_total"] += t

            c, t = edge_delta_correct_and_total(
                rewrite_outputs["edge_delta_logits_local"][i],
                current_adj,
                target_adj,
                sample_pred_edge_scope_mask.float(),
            )
            acc["scope_edge_delta_correct"] += c
            acc["scope_edge_delta_total"] += t

            c, t = edge_delta_correct_and_total(
                rewrite_outputs["edge_delta_logits_local"][i],
                current_adj,
                target_adj,
                sample_pred_edge_scope_mask.float(),
                label_id=EDGE_DELTA_KEEP,
            )
            acc["scope_keep_correct"] += c
            acc["scope_keep_total"] += t

            c, t = edge_delta_correct_and_total(
                rewrite_outputs["edge_delta_logits_local"][i],
                current_adj,
                target_adj,
                sample_pred_edge_scope_mask.float(),
                label_id=EDGE_DELTA_ADD,
            )
            acc["scope_add_correct"] += c
            acc["scope_add_total"] += t

            c, t = edge_delta_correct_and_total(
                rewrite_outputs["edge_delta_logits_local"][i],
                current_adj,
                target_adj,
                sample_pred_edge_scope_mask.float(),
                label_id=EDGE_DELTA_DELETE,
            )
            acc["scope_delete_correct"] += c
            acc["scope_delete_total"] += t

            c, t = edge_correct_and_total(
                rewrite_outputs["edge_logits_local"][i],
                target_adj,
                sample_changed_edge_mask.float(),
            )
            acc["changed_edge_correct"] += c
            acc["changed_edge_total_rewrite"] += t

            c, t = edge_correct_and_total(
                rewrite_outputs["edge_logits_local"][i],
                target_adj,
                sample_context_edge_mask.float(),
            )
            acc["context_edge_correct"] += c
            acc["context_edge_total"] += t

    proposal_precision = safe_div(acc["node_tp"], acc["node_pred_pos"])
    proposal_recall = safe_div(acc["node_tp"], acc["node_true_pos"])
    proposal_f1 = None
    if proposal_precision is not None and proposal_recall is not None and proposal_precision + proposal_recall > 0:
        proposal_f1 = 2.0 * proposal_precision * proposal_recall / (proposal_precision + proposal_recall)
    edge_precision = safe_div(acc["edge_tp"], acc["edge_pred_pos"])
    edge_recall = safe_div(acc["edge_tp"], acc["edge_true_pos"])
    edge_f1 = None
    if edge_precision is not None and edge_recall is not None and edge_precision + edge_recall > 0:
        edge_f1 = 2.0 * edge_precision * edge_recall / (edge_precision + edge_recall)

    return {
        "proposal_summary": {
            "node_precision": proposal_precision,
            "node_recall": proposal_recall,
            "node_f1": proposal_f1,
            "edge_precision": edge_precision,
            "edge_recall": edge_recall,
            "edge_f1": edge_f1,
            "changed_node_coverage": safe_div(acc["changed_node_covered"], acc["changed_node_total"]),
            "changed_edge_coverage": safe_div(acc["changed_edge_covered"], acc["changed_edge_total"]),
            "avg_predicted_node_scope_fraction": safe_div(acc["pred_node_scope_fraction_sum"], acc["num_samples"]),
            "avg_oracle_node_scope_fraction": safe_div(acc["oracle_node_scope_fraction_sum"], acc["num_samples"]),
            "avg_predicted_edge_scope_fraction": safe_div(acc["pred_edge_scope_fraction_sum"], acc["num_samples"]),
            "avg_oracle_edge_scope_fraction": safe_div(acc["oracle_edge_scope_fraction_sum"], acc["num_samples"]),
        },
        "rewrite_edge_overall": {
            "scope_edge": safe_div(acc["scope_edge_correct"], acc["scope_edge_total"]),
            "delta_all": safe_div(acc["scope_edge_delta_correct"], acc["scope_edge_delta_total"]),
            "keep": safe_div(acc["scope_keep_correct"], acc["scope_keep_total"]),
            "add": safe_div(acc["scope_add_correct"], acc["scope_add_total"]),
            "delete": safe_div(acc["scope_delete_correct"], acc["scope_delete_total"]),
            "changed": safe_div(acc["changed_edge_correct"], acc["changed_edge_total_rewrite"]),
            "context": safe_div(acc["context_edge_correct"], acc["context_edge_total"]),
        },
        "rewrite_type_overall": {
            "full": safe_div(acc["full_type_correct"], acc["full_type_total"]),
            "changed": safe_div(acc["changed_type_correct"], acc["changed_type_total"]),
            "flip": safe_div(acc["flip_target_type_correct"], acc["flip_target_type_total"]),
            "nonflip": safe_div(acc["nonflip_changed_type_correct"], acc["nonflip_changed_type_total"]),
            "scope": safe_div(acc["scope_type_correct"], acc["scope_type_total"]),
        },
        "rewrite_type_by_single_event_type": {
            "single_event_type::motif_type_flip": {
                "num_samples": int(acc["motif_num_samples"]),
                "full": safe_div(acc["motif_full_type_correct"], acc["motif_full_type_total"]),
                "changed": safe_div(acc["motif_changed_type_correct"], acc["motif_changed_type_total"]),
                "flip": safe_div(acc["motif_flip_target_type_correct"], acc["motif_flip_target_type_total"]),
                "nonflip": safe_div(
                    acc["motif_nonflip_changed_type_correct"],
                    acc["motif_nonflip_changed_type_total"],
                ),
                "scope": safe_div(acc["motif_scope_type_correct"], acc["motif_scope_type_total"]),
            }
        },
        "flip_confusion_predicted": dict(sorted(flip_old_to_pred.items(), key=lambda x: (-x[1], x[0]))),
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
    parser.add_argument("--node_threshold", type=float, default=0.25)
    parser.add_argument("--edge_threshold", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    proposal_checkpoint_path = resolve_path(args.proposal_checkpoint_path)
    rewrite_checkpoint_path = resolve_path(args.rewrite_checkpoint_path)
    data_path = resolve_path(args.data_path)
    device = get_device(args.device)
    pin_memory = device.type == "cuda"

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

    dataset, loader = build_loader(
        data_path=str(data_path),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    results = evaluate(
        proposal_model=proposal_model,
        rewrite_model=rewrite_model,
        loader=loader,
        device=device,
        node_threshold=args.node_threshold,
        edge_threshold=args.edge_threshold,
    )

    out_path = rewrite_checkpoint_path.parent / f"{args.split_name}_proposal_conditioned_delta.json"
    save_json(
        out_path,
        {
            "proposal_checkpoint_path": str(proposal_checkpoint_path),
            "rewrite_checkpoint_path": str(rewrite_checkpoint_path),
            "data_path": str(data_path),
            "split_name": args.split_name,
            "node_threshold": args.node_threshold,
            "edge_threshold": args.edge_threshold,
            "dataset_size": len(dataset),
            "results": results,
        },
    )

    print(f"device: {device}")
    print(f"proposal checkpoint: {proposal_checkpoint_path}")
    print(f"rewrite checkpoint: {rewrite_checkpoint_path}")
    print(f"data: {data_path}")
    print(f"dataset size: {len(dataset)}")
    print(f"node threshold: {args.node_threshold}")
    print(f"edge threshold: {args.edge_threshold}")
    print(f"saved json: {out_path}")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
