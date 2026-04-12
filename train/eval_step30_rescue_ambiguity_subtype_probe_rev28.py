from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.step30_dataset import Step30WeakObservationDataset, step30_weak_observation_collate_fn
from models.encoder_step30 import build_pair_mask
from train.eval_step30_encoder_recovery import load_model, move_batch_to_device
from train.eval_step30_rescue_candidate_latent_probe_rev23 import (
    CLASS_NAMES,
    binary_metrics,
    clean_json_numbers,
    rescue_class_label,
    sigmoid_np,
)


SUBTYPE_NAMES = [
    "weak_positive_ambiguous",
    "warning_dominated_ambiguous",
    "conflicting_evidence_ambiguous",
    "relation_borderline_ambiguous",
    "low_confidence_ambiguous",
]


@dataclass
class Rev28Rows:
    target: np.ndarray
    variant: np.ndarray
    relation: np.ndarray
    support: np.ndarray
    signed_witness: np.ndarray
    bundle_positive: np.ndarray
    bundle_warning: np.ndarray
    bundle_corroboration: np.ndarray
    bundle_endpoint_compat: np.ndarray
    is_candidate: np.ndarray
    candidate_label: np.ndarray
    is_ambiguous_signal: np.ndarray
    subtype: np.ndarray
    rev6_score: np.ndarray
    rev24_safe_logit: np.ndarray
    rev24_safe_prob: np.ndarray
    rev24_false_prob: np.ndarray
    rev24_ambiguous_prob: np.ndarray
    rev26_binary_prob: np.ndarray
    rev26_ambiguous_prob: np.ndarray


def parse_thresholds(value: str) -> Dict[str, float]:
    out: Dict[str, float] = {"default": 0.5}
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        key, raw_val = part.split(":", 1)
        out[key.strip()] = float(raw_val)
    return out


def threshold_array(variant: np.ndarray, thresholds: Dict[str, float]) -> np.ndarray:
    return np.asarray(
        [thresholds.get(str(v), thresholds.get("default", 0.5)) for v in variant],
        dtype=np.float32,
    )


def write_csv(path: Path, rows: list[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def edge_metrics_from_arrays(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    tp = float(((pred == 1) & (target == 1)).sum())
    fp = float(((pred == 1) & (target == 0)).sum())
    fn = float(((pred == 0) & (target == 1)).sum())
    tn = float(((pred == 0) & (target == 0)).sum())
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        **binary_metrics(tp, fp, fn),
    }


def subtype_for_candidate(
    relation: float,
    positive: float,
    warning: float,
    corroboration: float,
    endpoint_compat: float,
    signed_witness: float,
) -> str:
    margin = positive - warning
    if margin >= 0.12 and (
        corroboration >= 0.45 or endpoint_compat >= 0.50 or signed_witness >= 0.05
    ):
        return "weak_positive_ambiguous"
    if margin <= -0.12:
        return "warning_dominated_ambiguous"
    if abs(margin) < 0.10 and max(positive, warning) >= 0.45:
        return "conflicting_evidence_ambiguous"
    if relation >= 0.45:
        return "relation_borderline_ambiguous"
    return "low_confidence_ambiguous"


def rev25_score(rows: Rev28Rows) -> np.ndarray:
    return rows.rev24_safe_prob - np.maximum(rows.rev24_false_prob, rows.rev24_ambiguous_prob)


def rev26_score(rows: Rev28Rows) -> np.ndarray:
    return rows.rev26_binary_prob - rows.rev26_ambiguous_prob


def base_prediction(rows: Rev28Rows, thresholds: Dict[str, float]) -> np.ndarray:
    return (rows.rev6_score >= threshold_array(rows.variant, thresholds)).astype(np.int64)


def top20_budget(rows: Rev28Rows) -> int:
    return max(1, int(round(0.20 * int(rows.is_candidate.sum()))))


def selected_additions(
    rows: Rev28Rows,
    score: np.ndarray,
    thresholds: Dict[str, float],
    budget: int,
) -> np.ndarray:
    base_pred = base_prediction(rows, thresholds)
    candidate_idx = np.flatnonzero(rows.is_candidate & (base_pred == 0))
    chosen = np.zeros_like(rows.target, dtype=bool)
    if len(candidate_idx) == 0:
        return chosen
    budget = min(int(budget), len(candidate_idx))
    order = np.argsort(-score[candidate_idx])
    chosen[candidate_idx[order[:budget]]] = True
    return chosen


def prediction_with_additions(
    rows: Rev28Rows,
    additions: np.ndarray,
    thresholds: Dict[str, float],
) -> np.ndarray:
    return np.maximum(base_prediction(rows, thresholds), additions.astype(np.int64))


@torch.no_grad()
def collect_rows(
    data_path: str,
    models: Dict[str, Any],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    rescue_relation_max: float,
    rescue_support_min: float,
) -> Rev28Rows:
    dataset = Step30WeakObservationDataset(data_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=step30_weak_observation_collate_fn,
        pin_memory=device.type == "cuda",
    )

    out: Dict[str, list[Any]] = {
        "target": [],
        "variant": [],
        "relation": [],
        "support": [],
        "signed_witness": [],
        "bundle_positive": [],
        "bundle_warning": [],
        "bundle_corroboration": [],
        "bundle_endpoint_compat": [],
        "is_candidate": [],
        "candidate_label": [],
        "is_ambiguous_signal": [],
        "subtype": [],
        "rev6_score": [],
        "rev24_safe_logit": [],
        "rev24_safe_prob": [],
        "rev24_false_prob": [],
        "rev24_ambiguous_prob": [],
        "rev26_binary_prob": [],
        "rev26_ambiguous_prob": [],
    }

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        variants = [str(v) for v in batch.get("step30_observation_variant", ["unknown"] * len(batch["target_adj"]))]
        target_adj = batch["target_adj"].float()
        node_mask = batch["node_mask"].float()
        relation_hints = batch["weak_relation_hints"].float()
        pair_support_hints = batch["weak_pair_support_hints"].float()
        signed_witness = batch["weak_signed_pair_witness"].float()
        pair_bundle = batch["weak_pair_evidence_bundle"].float()
        pair_mask = build_pair_mask(node_mask).bool()

        outputs = {
            name: model(
                weak_slot_features=batch["weak_slot_features"],
                weak_relation_hints=relation_hints,
                weak_pair_support_hints=pair_support_hints,
                weak_signed_pair_witness=signed_witness,
                weak_pair_evidence_bundle=pair_bundle,
            )
            for name, model in models.items()
        }
        rev6_prob = torch.sigmoid(outputs["rev6"]["edge_logits"]).detach().cpu().numpy()

        rev24_logits = outputs["rev24"]["rescue_candidate_logits"]
        rev24_safe_logit = outputs["rev24"]["rescue_candidate_safe_logits"].detach().cpu().numpy()
        rev24_probs = torch.softmax(rev24_logits, dim=-1).detach().cpu().numpy()

        rev26_logits = outputs["rev26"]["rescue_candidate_logits"]
        rev26_probs = torch.softmax(rev26_logits, dim=-1).detach().cpu().numpy()
        rev26_binary_logits_tensor = outputs["rev26"].get("rescue_candidate_binary_logits")
        if rev26_binary_logits_tensor is None:
            rev26_binary_prob = rev26_probs[..., 0]
        else:
            rev26_binary_prob = sigmoid_np(rev26_binary_logits_tensor.detach().cpu().numpy())

        for batch_idx in range(int(target_adj.shape[0])):
            n = int(node_mask[batch_idx].sum().item())
            target_np = target_adj[batch_idx, :n, :n].detach().cpu().numpy()
            relation_np = relation_hints[batch_idx, :n, :n].detach().cpu().numpy()
            support_np = pair_support_hints[batch_idx, :n, :n].detach().cpu().numpy()
            witness_np = signed_witness[batch_idx, :n, :n].detach().cpu().numpy()
            bundle_np = pair_bundle[batch_idx, :n, :n, :].detach().cpu().numpy()
            valid_np = pair_mask[batch_idx, :n, :n].detach().cpu().numpy().astype(bool)

            for i in range(n):
                for j in range(i + 1, n):
                    if not bool(valid_np[i, j]):
                        continue
                    relation = float(relation_np[i, j])
                    support = float(support_np[i, j])
                    target = int(target_np[i, j] >= 0.5)
                    positive = float(bundle_np[i, j, 0])
                    warning = float(bundle_np[i, j, 1])
                    corroboration = float(bundle_np[i, j, 2])
                    endpoint_compat = float(bundle_np[i, j, 3])
                    witness = float(witness_np[i, j])
                    is_candidate = (
                        variants[batch_idx] == "noisy"
                        and relation < rescue_relation_max
                        and support >= rescue_support_min
                    )
                    ambiguous_signal = (
                        is_candidate
                        and (
                            relation >= 0.45
                            or support < 0.65
                            or abs(positive - warning) < 0.10
                        )
                    )
                    label = -1
                    subtype = "not_ambiguous_signal"
                    if is_candidate:
                        label = rescue_class_label(
                            target,
                            relation,
                            support,
                            bundle_np[i, j, :],
                        )
                    if ambiguous_signal:
                        subtype = subtype_for_candidate(
                            relation=relation,
                            positive=positive,
                            warning=warning,
                            corroboration=corroboration,
                            endpoint_compat=endpoint_compat,
                            signed_witness=witness,
                        )

                    out["target"].append(target)
                    out["variant"].append(variants[batch_idx])
                    out["relation"].append(relation)
                    out["support"].append(support)
                    out["signed_witness"].append(witness)
                    out["bundle_positive"].append(positive)
                    out["bundle_warning"].append(warning)
                    out["bundle_corroboration"].append(corroboration)
                    out["bundle_endpoint_compat"].append(endpoint_compat)
                    out["is_candidate"].append(bool(is_candidate))
                    out["candidate_label"].append(label)
                    out["is_ambiguous_signal"].append(bool(ambiguous_signal))
                    out["subtype"].append(subtype)
                    out["rev6_score"].append(float(rev6_prob[batch_idx, i, j]))
                    out["rev24_safe_logit"].append(float(rev24_safe_logit[batch_idx, i, j]))
                    out["rev24_safe_prob"].append(float(rev24_probs[batch_idx, i, j, 0]))
                    out["rev24_false_prob"].append(float(rev24_probs[batch_idx, i, j, 1]))
                    out["rev24_ambiguous_prob"].append(float(rev24_probs[batch_idx, i, j, 2]))
                    out["rev26_binary_prob"].append(float(rev26_binary_prob[batch_idx, i, j]))
                    out["rev26_ambiguous_prob"].append(float(rev26_probs[batch_idx, i, j, 2]))

    return Rev28Rows(
        target=np.asarray(out["target"], dtype=np.int64),
        variant=np.asarray(out["variant"]),
        relation=np.asarray(out["relation"], dtype=np.float32),
        support=np.asarray(out["support"], dtype=np.float32),
        signed_witness=np.asarray(out["signed_witness"], dtype=np.float32),
        bundle_positive=np.asarray(out["bundle_positive"], dtype=np.float32),
        bundle_warning=np.asarray(out["bundle_warning"], dtype=np.float32),
        bundle_corroboration=np.asarray(out["bundle_corroboration"], dtype=np.float32),
        bundle_endpoint_compat=np.asarray(out["bundle_endpoint_compat"], dtype=np.float32),
        is_candidate=np.asarray(out["is_candidate"], dtype=bool),
        candidate_label=np.asarray(out["candidate_label"], dtype=np.int64),
        is_ambiguous_signal=np.asarray(out["is_ambiguous_signal"], dtype=bool),
        subtype=np.asarray(out["subtype"]),
        rev6_score=np.asarray(out["rev6_score"], dtype=np.float32),
        rev24_safe_logit=np.asarray(out["rev24_safe_logit"], dtype=np.float32),
        rev24_safe_prob=np.asarray(out["rev24_safe_prob"], dtype=np.float32),
        rev24_false_prob=np.asarray(out["rev24_false_prob"], dtype=np.float32),
        rev24_ambiguous_prob=np.asarray(out["rev24_ambiguous_prob"], dtype=np.float32),
        rev26_binary_prob=np.asarray(out["rev26_binary_prob"], dtype=np.float32),
        rev26_ambiguous_prob=np.asarray(out["rev26_ambiguous_prob"], dtype=np.float32),
    )


def mean_or_none(values: np.ndarray) -> float | None:
    if len(values) == 0:
        return None
    return float(values.mean())


def subtype_rows(
    rows: Rev28Rows,
    additions_by_name: Dict[str, np.ndarray],
) -> list[Dict[str, Any]]:
    out_rows: list[Dict[str, Any]] = []
    for subtype in SUBTYPE_NAMES:
        mask = rows.is_ambiguous_signal & (rows.subtype == subtype)
        if not mask.any():
            continue
        row: Dict[str, Any] = {
            "subtype": subtype,
            "count": int(mask.sum()),
            "gt_positive_rate": float(rows.target[mask].mean()),
            "safe_label_rate": float((rows.candidate_label[mask] == 0).mean()),
            "current_ambiguous_label_rate": float((rows.candidate_label[mask] == 2).mean()),
            "avg_relation_hint": float(rows.relation[mask].mean()),
            "avg_pair_support_hint": float(rows.support[mask].mean()),
            "avg_signed_witness": float(rows.signed_witness[mask].mean()),
            "avg_bundle_positive": float(rows.bundle_positive[mask].mean()),
            "avg_bundle_warning": float(rows.bundle_warning[mask].mean()),
            "avg_bundle_corroboration": float(rows.bundle_corroboration[mask].mean()),
            "avg_bundle_endpoint_compat": float(rows.bundle_endpoint_compat[mask].mean()),
            "avg_rev6_score": float(rows.rev6_score[mask].mean()),
            "avg_rev24_safe_prob": float(rows.rev24_safe_prob[mask].mean()),
            "avg_rev24_ambiguous_prob": float(rows.rev24_ambiguous_prob[mask].mean()),
            "avg_rev26_binary_prob": float(rows.rev26_binary_prob[mask].mean()),
            "avg_rev26_ambiguous_prob": float(rows.rev26_ambiguous_prob[mask].mean()),
        }
        for name, additions in additions_by_name.items():
            selected = additions[mask].astype(bool)
            row[f"{name}_admitted"] = int(selected.sum())
            row[f"{name}_admission_rate"] = float(selected.mean()) if mask.any() else 0.0
            row[f"{name}_selected_precision"] = (
                float(rows.target[mask][selected].mean()) if selected.any() else None
            )
            positives = int(rows.target[mask].sum())
            row[f"{name}_positive_recall_contribution"] = (
                float((selected & (rows.target[mask] == 1)).sum() / max(positives, 1))
            )
        out_rows.append(row)
    return out_rows


def false_admission_composition_rows(
    rows: Rev28Rows,
    additions_by_name: Dict[str, np.ndarray],
) -> list[Dict[str, Any]]:
    out_rows: list[Dict[str, Any]] = []
    ambiguous_false = rows.is_ambiguous_signal & (rows.candidate_label == 2)
    for policy, additions in additions_by_name.items():
        selected_false = ambiguous_false & additions
        total = int(selected_false.sum())
        for subtype in SUBTYPE_NAMES:
            mask = selected_false & (rows.subtype == subtype)
            out_rows.append(
                {
                    "policy": policy,
                    "subtype": subtype,
                    "ambiguous_fp_count": int(mask.sum()),
                    "fraction_of_policy_ambiguous_fp": float(int(mask.sum()) / max(total, 1)),
                    "policy_ambiguous_fp_total": total,
                }
            )
    return out_rows


def summarize_prediction(
    rows: Rev28Rows,
    additions: np.ndarray,
    thresholds: Dict[str, float],
    row_name: str,
) -> Dict[str, Any]:
    pred = prediction_with_additions(rows, additions, thresholds)
    noisy = rows.variant == "noisy"
    candidate = rows.is_candidate
    selected = additions[candidate].astype(np.int64)
    rescue_metrics = edge_metrics_from_arrays(selected, rows.target[candidate])
    row: Dict[str, Any] = {
        "row": row_name,
        "selected_additions": int(additions.sum()),
        "noisy_precision": edge_metrics_from_arrays(pred[noisy], rows.target[noisy])["precision"],
        "noisy_recall": edge_metrics_from_arrays(pred[noisy], rows.target[noisy])["recall"],
        "noisy_f1": edge_metrics_from_arrays(pred[noisy], rows.target[noisy])["f1"],
        "selected_precision": rescue_metrics["precision"],
        "selected_recall": rescue_metrics["recall"],
        "selected_f1": rescue_metrics["f1"],
    }
    return row


def counterfactual_rows(
    rows: Rev28Rows,
    rev26_additions: np.ndarray,
    thresholds: Dict[str, float],
    budget: int,
) -> list[Dict[str, Any]]:
    out_rows: list[Dict[str, Any]] = []
    base_score = rev26_score(rows)
    out_rows.append(summarize_prediction(rows, rev26_additions, thresholds, "rev26_baseline"))
    for subtype in SUBTYPE_NAMES:
        subtype_mask = rows.is_ambiguous_signal & (rows.subtype == subtype)

        suppressed = rev26_additions.copy()
        suppressed[subtype_mask] = False
        row = summarize_prediction(
            rows,
            suppressed,
            thresholds,
            f"suppress_{subtype}",
        )
        row["policy"] = "suppress_selected_subtype"
        row["subtype"] = subtype
        row["removed_selected"] = int((rev26_additions & subtype_mask).sum())
        out_rows.append(row)

        boosted_score = base_score.copy()
        boosted_score[subtype_mask] += 0.25
        boosted = selected_additions(rows, boosted_score, thresholds, budget)
        row = summarize_prediction(
            rows,
            boosted,
            thresholds,
            f"boost_{subtype}",
        )
        row["policy"] = "boost_subtype_plus_0.25_fixed_budget"
        row["subtype"] = subtype
        row["delta_selected_in_subtype"] = int(
            (boosted & subtype_mask).sum() - (rev26_additions & subtype_mask).sum()
        )
        out_rows.append(row)
    return out_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", default="data/graph_event_step30_weak_obs_rev17_test.pkl")
    parser.add_argument("--rev6_checkpoint", default="checkpoints/step30_encoder_recovery_rev6/best.pt")
    parser.add_argument("--rev24_checkpoint", default="checkpoints/step30_encoder_recovery_rev24/best.pt")
    parser.add_argument("--rev26_checkpoint", default="checkpoints/step30_encoder_recovery_rev26/best.pt")
    parser.add_argument("--output_dir", default="artifacts/step30_rescue_ambiguity_subtype_probe_rev28")
    parser.add_argument("--thresholds", default="clean:0.50,noisy:0.55")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rescue_relation_max", type=float, default=0.50)
    parser.add_argument("--rescue_support_min", type=float, default=0.55)
    args = parser.parse_args()

    if args.device != "auto":
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    models = {
        "rev6": load_model(args.rev6_checkpoint, device),
        "rev24": load_model(args.rev24_checkpoint, device),
        "rev26": load_model(args.rev26_checkpoint, device),
    }
    thresholds = parse_thresholds(args.thresholds)
    rows = collect_rows(
        args.test_path,
        models=models,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        rescue_relation_max=args.rescue_relation_max,
        rescue_support_min=args.rescue_support_min,
    )

    budget = top20_budget(rows)
    additions_by_name = {
        "rev24_safe_only": selected_additions(rows, rows.rev24_safe_logit, thresholds, budget),
        "rev25_class_aware": selected_additions(rows, rev25_score(rows), thresholds, budget),
        "rev26_calibrated": selected_additions(rows, rev26_score(rows), thresholds, budget),
    }
    subtype_summary = subtype_rows(rows, additions_by_name)
    fp_composition = false_admission_composition_rows(rows, additions_by_name)
    counterfactual_summary = counterfactual_rows(
        rows,
        additions_by_name["rev26_calibrated"],
        thresholds,
        budget,
    )

    ambiguous_signal = rows.is_ambiguous_signal
    current_ambiguous = rows.is_ambiguous_signal & (rows.candidate_label == 2)
    diagnosis = {
        "candidate_count": int(rows.is_candidate.sum()),
        "ambiguous_signal_count": int(ambiguous_signal.sum()),
        "ambiguous_signal_gt_positive_rate": float(rows.target[ambiguous_signal].mean()),
        "current_ambiguous_label_count": int(current_ambiguous.sum()),
        "current_ambiguous_label_gt_positive_rate": float(rows.target[current_ambiguous].mean())
        if current_ambiguous.any()
        else None,
        "selected_budget": int(budget),
        "strongest_safe_subtype_by_gt_positive_rate": max(
            subtype_summary,
            key=lambda row: float(row["gt_positive_rate"]),
        )["subtype"],
        "most_harmful_rev26_ambiguous_fp_subtype": max(
            [
                row
                for row in fp_composition
                if row["policy"] == "rev26_calibrated"
            ],
            key=lambda row: int(row["ambiguous_fp_count"]),
        )["subtype"],
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "ambiguity_subtype_summary.csv", subtype_summary)
    write_csv(output_dir / "false_admission_composition.csv", fp_composition)
    write_csv(output_dir / "counterfactual_summary.csv", counterfactual_summary)
    summary = {
        "step": "step30_rev28_rescue_ambiguity_subtype_probe",
        "backend_rerun": False,
        "backend_rerun_reason": "rev28 is diagnostic-only and does not meet a recovery gate by itself.",
        "subtype_rules": {
            "weak_positive_ambiguous": "positive-warning >= 0.12 and corroboration/endpoint/witness support is present",
            "warning_dominated_ambiguous": "warning-positive >= 0.12",
            "conflicting_evidence_ambiguous": "abs(positive-warning) < 0.10 and max(positive, warning) >= 0.45",
            "relation_borderline_ambiguous": "relation_hint >= 0.45 after stronger signal rules",
            "low_confidence_ambiguous": "remaining ambiguous-signal candidates",
        },
        "diagnosis": diagnosis,
        "ambiguity_subtype_summary": subtype_summary,
        "false_admission_composition": fp_composition,
        "counterfactual_summary": counterfactual_summary,
        "args": vars(args),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(clean_json_numbers(summary), f, indent=2)
    print(json.dumps(clean_json_numbers(summary), indent=2))


if __name__ == "__main__":
    main()
