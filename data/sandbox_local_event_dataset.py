from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from torch.utils.data import DataLoader, Dataset


REQUIRED_SAMPLE_KEYS = (
    "node_features_t",
    "edge_index",
    "edge_features_t",
    "event_type",
    "event_type_id",
    "event_node_mask",
    "event_edge_mask",
    "event_params",
    "event_scope_node_mask",
    "event_scope_edge_mask",
    "changed_node_mask",
    "changed_edge_mask",
    "node_features_next",
    "edge_features_next",
    "copy_node_features_next",
    "copy_edge_features_next",
)

NODE_LEVEL_KEYS = (
    "node_features_t",
    "node_features_next",
    "copy_node_features_next",
    "event_node_mask",
    "event_scope_node_mask",
    "changed_node_mask",
)

EDGE_LEVEL_KEYS = (
    "edge_features_t",
    "edge_features_next",
    "copy_edge_features_next",
    "event_edge_mask",
    "event_scope_edge_mask",
    "changed_edge_mask",
)


def resolve_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[1] / path


def split_path(output_prefix: str | Path, split: str) -> Path:
    if split not in {"train", "val", "test"}:
        raise ValueError(f"split must be one of train, val, test; got {split!r}")
    prefix = resolve_path(output_prefix)
    return prefix.with_name(f"{prefix.name}_{split}.pkl")


class SandboxLocalEventDataset(Dataset):
    """PyTorch Dataset for generated sandbox local-event MVP pickle splits."""

    def __init__(
        self,
        file_path: str | Path | None = None,
        *,
        split: str | None = None,
        output_prefix: str | Path = "data/sandbox_local_event_mvp",
    ):
        if file_path is None:
            if split is None:
                raise ValueError("provide either file_path or split")
            file_path = split_path(output_prefix, split)
        elif split is not None:
            raise ValueError("provide file_path or split, not both")

        self.file_path = resolve_path(file_path)
        with open(self.file_path, "rb") as f:
            self.samples: List[Dict[str, Any]] = pickle.load(f)

        if len(self.samples) == 0:
            raise ValueError(f"Dataset is empty: {self.file_path}")
        self._validate_sample(self.samples[0], index=0)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        raw = self.samples[index]
        self._validate_sample(raw, index=index)

        item: Dict[str, Any] = {
            "edge_index": torch.as_tensor(raw["edge_index"], dtype=torch.long),
            "event_type": raw["event_type"],
            "event_type_id": torch.tensor(int(raw["event_type_id"]), dtype=torch.long),
            "event_params": torch.as_tensor(raw["event_params"], dtype=torch.float32),
        }

        for key in NODE_LEVEL_KEYS + EDGE_LEVEL_KEYS:
            item[key] = torch.as_tensor(raw[key], dtype=torch.float32)

        return item

    @staticmethod
    def _validate_sample(sample: Dict[str, Any], index: int) -> None:
        missing = [key for key in REQUIRED_SAMPLE_KEYS if key not in sample]
        if missing:
            raise KeyError(f"sample {index} is missing required keys: {missing}")


def sandbox_local_event_collate_fn(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if len(batch) == 0:
        raise ValueError("cannot collate an empty batch")

    out: Dict[str, Any] = {}
    node_counts = torch.tensor([int(sample["node_features_t"].shape[0]) for sample in batch], dtype=torch.long)
    edge_counts = torch.tensor([int(sample["edge_features_t"].shape[0]) for sample in batch], dtype=torch.long)

    out["num_nodes_per_graph"] = node_counts
    out["num_edges_per_graph"] = edge_counts
    out["node_batch_index"] = torch.repeat_interleave(
        torch.arange(len(batch), dtype=torch.long),
        node_counts,
    )
    out["edge_batch_index"] = torch.repeat_interleave(
        torch.arange(len(batch), dtype=torch.long),
        edge_counts,
    )

    for key in NODE_LEVEL_KEYS:
        out[key] = torch.cat([sample[key] for sample in batch], dim=0)
    for key in EDGE_LEVEL_KEYS:
        out[key] = torch.cat([sample[key] for sample in batch], dim=0)

    edge_indices = []
    node_offset = 0
    for sample in batch:
        edge_indices.append(sample["edge_index"] + node_offset)
        node_offset += int(sample["node_features_t"].shape[0])
    out["edge_index"] = torch.cat(edge_indices, dim=1)

    out["event_params"] = torch.stack([sample["event_params"] for sample in batch], dim=0)
    out["event_type_id"] = torch.stack([sample["event_type_id"] for sample in batch], dim=0)
    out["event_type"] = [sample["event_type"] for sample in batch]
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-check the sandbox local-event dataset batching.")
    parser.add_argument("--path", type=str, default="")
    parser.add_argument("--split", type=str, default="train", choices=("train", "val", "test"))
    parser.add_argument("--output_prefix", type=str, default="data/sandbox_local_event_mvp")
    parser.add_argument("--batch_size", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.path:
        dataset = SandboxLocalEventDataset(args.path)
    else:
        dataset = SandboxLocalEventDataset(split=args.split, output_prefix=args.output_prefix)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=sandbox_local_event_collate_fn,
    )
    batch = next(iter(loader))

    print(f"path={dataset.file_path}")
    print(f"num_samples={len(dataset)}")
    for key in (
        "node_features_t",
        "edge_index",
        "edge_features_t",
        "event_params",
        "event_type_id",
        "node_batch_index",
        "edge_batch_index",
        "num_nodes_per_graph",
        "num_edges_per_graph",
        "node_features_next",
        "edge_features_next",
    ):
        value = batch[key]
        print(f"{key}: shape={tuple(value.shape)} dtype={value.dtype}")
    print(f"event_type: {batch['event_type']}")


if __name__ == "__main__":
    main()
