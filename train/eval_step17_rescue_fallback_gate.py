from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

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
from train.eval_step10_rescued_scope_rewrite_decomp import (
    bool_pred_adj,
    finalize_decomp_bucket,
    init_decomp_bucket,
    update_decomp_sample,
)
from train.eval_step12_guard_ranking_interaction import FIXED_RESCUE_BUDGET_FRACTION, build_ranked_outputs
from train.eval_step16_rescue_aware_rewrite_probe import (
    apply_oracle_false_scope_fallback,
    clone_rewrite_with_edge_logits,
)
from train.eval_step9_gated_edge_completion import (
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
from train.train_step17_rescue_fallback_gate import (
    build_gate_feature_tensor,
    build_gate_targets,
    load_fallback_gate_model,
)


BASE_MODE = "base"
STEP9C_MODE = "step9c_completion_only"
ORACLE_FALSE_SCOPE_MODE = "step9c_oracle_false_scope_fallback"
LEARNED_GATE_MODE = "step17_learned_rescue_conditioned_gate"
ORACLE_CHOOSE_MODE = "oracle_choose_better_of_two_paths"
MODE_ORDER = [BASE_MODE, STEP9C_MODE, ORACLE_FALSE_SCOPE_MODE, LEARNED_GATE_MODE, ORACLE_CHOOSE_MODE]
DECOMP_MODES = [STEP9C_MODE, ORACLE_FALSE_SCOPE_MODE, LEARNED_GATE_MODE, ORACLE_CHOOSE_MODE]
GATE_MODES = [LEARNED_GATE_MODE, ORACLE_CHOOSE_MODE]


def apply_choose_step_gate(
    rescued_edges: torch.Tensor,
    choose_step_mask: torch.Tensor,
    base_edge_logits_full: torch.Tensor,
    step_edge_logits_full: torch.Tensor,
) -> torch.Tensor:
    rescued = rescued_edges.bool() | rescued_edges.bool().transpose(1, 2)
    choose_step = choose_step_mask.bool() | choose_step_mask.bool().transpose(1, 2)
    fallback_mask = rescued & (~choose_step)
    return torch.where(fallback_mask, base_edge_logits_full, step_edge_logits_full)


def init_gate_bucket() -> Dict[str, float]:
    return {
        "rescued_total": 0.0,
        "fallback_total": 0.0,
        "step9c_kept_total": 0.0,
        "fallback_false_scope_total": 0.0,
        "fallback_changed_total": 0.0,
        "step9c_kept_false_scope_total": 0.0,
        "step9c_kept_changed_total": 0.0,
        "target_choose_step_total": 0.0,
        "chosen_correct_total": 0.0,
    }


def update_gate_bucket(
    bucket: Dict[str, float],
    batch: Dict[str, Any],
    valid_edge_mask: torch.Tensor,
    rescued_edges: torch.Tensor,
    choose_step_mask: torch.Tensor,
    target_choose_step: torch.Tensor,
    base_correct: torch.Tensor,
    step_correct: torch.Tensor,
    sample_idx: int,
) -> None:
    valid = valid_edge_mask[sample_idx].bool()
    rescued = rescued_edges[sample_idx].bool() & valid
    choose_step = choose_step_mask[sample_idx].bool() & rescued
    fallback = rescued & (~choose_step)
    event_scope = (batch["event_scope_union_edges"][sample_idx] > 0.5) & valid
    changed = (batch["changed_edges"][sample_idx] > 0.5) & valid
    false_scope = (~event_scope) & valid
    chosen_correct = torch.where(choose_step, step_correct[sample_idx], base_correct[sample_idx])

    bucket["rescued_total"] += rescued.float().sum().item()
    bucket["fallback_total"] += fallback.float().sum().item()
    bucket["step9c_kept_total"] += choose_step.float().sum().item()
    bucket["fallback_false_scope_total"] += (fallback & false_scope).float().sum().item()
    bucket["fallback_changed_total"] += (fallback & changed).float().sum().item()
    bucket["step9c_kept_false_scope_total"] += (choose_step & false_scope).float().sum().item()
    bucket["step9c_kept_changed_total"] += (choose_step & changed).float().sum().item()
    bucket["target_choose_step_total"] += ((target_choose_step[sample_idx] > 0.5) & rescued).float().sum().item()
    bucket["chosen_correct_total"] += (chosen_correct & rescued).float().sum().item()


def finalize_gate_bucket(bucket: Dict[str, float]) -> Dict[str, Any]:
    rescued = bucket["rescued_total"]
    fallback = bucket["fallback_total"]
    kept = bucket["step9c_kept_total"]
    return {
        "rescued_total": int(rescued),
        "fallback_total": int(fallback),
        "step9c_kept_total": int(kept),
        "fallback_fraction": safe_div(fallback, rescued),
        "step9c_kept_fraction": safe_div(kept, rescued),
        "fallback_false_scope_fraction": safe_div(bucket["fallback_false_scope_total"], fallback),
        "fallback_changed_fraction": safe_div(bucket["fallback_changed_total"], fallback),
        "step9c_kept_false_scope_fraction": safe_div(bucket["step9c_kept_false_scope_total"], kept),
        "step9c_kept_changed_fraction": safe_div(bucket["step9c_kept_changed_total"], kept),
        "target_choose_step_fraction": safe_div(bucket["target_choose_step_total"], rescued),
        "chosen_correct_rate_on_rescued": safe_div(bucket["chosen_correct_total"], rescued),
    }


@torch.no_grad()
def evaluate_step17(
    proposal_model: torch.nn.Module,
    completion_model: torch.nn.Module,
    rewrite_model: torch.nn.Module,
    gate_model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    node_threshold: float,
    edge_threshold: float,
    completion_threshold: float,
    use_proposal_conditioning: bool,
) -> Dict[str, Any]:
    proposal_model.eval()
    completion_model.eval()
    rewrite_model.eval()
    gate_model.eval()

    proposal_buckets = {mode: defaultdict(init_proposal_bucket) for mode in MODE_ORDER}
    downstream_buckets = {mode: defaultdict(init_downstream_bucket) for mode in MODE_ORDER}
    decomp_buckets = {mode: defaultdict(init_decomp_bucket) for mode in DECOMP_MODES}
    gate_buckets = {mode: defaultdict(init_gate_bucket) for mode in GATE_MODES}

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
        valid_edge_mask = base_outputs["valid_edge_mask"].bool()
        completion_logits = completion_model(
            node_latents=base_outputs["node_latents"],
            base_edge_logits=base_outputs["edge_scope_logits"],
        )
        completion_probs = torch.sigmoid(completion_logits) * valid_edge_mask.float()
        mode_outputs: Dict[str, Dict[str, torch.Tensor]] = {
            BASE_MODE: apply_edge_completion_mode(
                base_outputs=base_outputs,
                edge_completion_mode=EDGE_COMPLETION_OFF,
                completion_model=None,
                completion_threshold=completion_threshold,
            ),
            STEP9C_MODE: build_ranked_outputs(
                base_outputs=base_outputs,
                mode_name=STEP9C_MODE,
                ranking_scores=completion_probs,
                completion_probs=completion_probs,
            ),
        }
        mode_outputs[ORACLE_FALSE_SCOPE_MODE] = {**mode_outputs[STEP9C_MODE], "edge_completion_mode": ORACLE_FALSE_SCOPE_MODE}
        mode_outputs[LEARNED_GATE_MODE] = {**mode_outputs[STEP9C_MODE], "edge_completion_mode": LEARNED_GATE_MODE}
        mode_outputs[ORACLE_CHOOSE_MODE] = {**mode_outputs[STEP9C_MODE], "edge_completion_mode": ORACLE_CHOOSE_MODE}

        rewrite_outputs_by_mode: Dict[str, Dict[str, torch.Tensor]] = {}
        for mode in [BASE_MODE, STEP9C_MODE]:
            outputs = mode_outputs[mode]
            rewrite_outputs_by_mode[mode] = rewrite_model(
                node_feats=outputs["input_node_feats"],
                adj=outputs["input_adj"],
                scope_node_mask=outputs["pred_scope_nodes"].float(),
                scope_edge_mask=outputs["final_pred_scope_edges"].float(),
                proposal_node_probs=outputs["proposal_node_probs"] if use_proposal_conditioning else None,
                proposal_edge_probs=outputs["final_proposal_edge_probs"] if use_proposal_conditioning else None,
            )

        features = build_gate_feature_tensor(
            base_outputs=base_outputs,
            completion_logits=completion_logits,
            rewrite_base=rewrite_outputs_by_mode[BASE_MODE],
            rewrite_step9c=rewrite_outputs_by_mode[STEP9C_MODE],
        )
        gate_probs = torch.sigmoid(gate_model(features)) * valid_edge_mask.float()
        learned_choose_step = (gate_probs >= 0.5) & mode_outputs[STEP9C_MODE]["rescued_edges"].bool()
        target_choose_step, rescued, base_correct, step_correct = build_gate_targets(
            batch=batch,
            valid_edge_mask=valid_edge_mask,
            rescued_edges=mode_outputs[STEP9C_MODE]["rescued_edges"],
            rewrite_base=rewrite_outputs_by_mode[BASE_MODE],
            rewrite_step9c=rewrite_outputs_by_mode[STEP9C_MODE],
        )
        oracle_choose_step = (target_choose_step > 0.5) & rescued

        fallback_logits = apply_oracle_false_scope_fallback(
            batch=batch,
            valid_edge_mask=valid_edge_mask,
            rescued_edges=mode_outputs[STEP9C_MODE]["rescued_edges"],
            base_edge_logits_full=rewrite_outputs_by_mode[BASE_MODE]["edge_logits_full"],
            budget_edge_logits_full=rewrite_outputs_by_mode[STEP9C_MODE]["edge_logits_full"],
        )
        learned_logits = apply_choose_step_gate(
            rescued_edges=mode_outputs[STEP9C_MODE]["rescued_edges"],
            choose_step_mask=learned_choose_step,
            base_edge_logits_full=rewrite_outputs_by_mode[BASE_MODE]["edge_logits_full"],
            step_edge_logits_full=rewrite_outputs_by_mode[STEP9C_MODE]["edge_logits_full"],
        )
        oracle_choose_logits = apply_choose_step_gate(
            rescued_edges=mode_outputs[STEP9C_MODE]["rescued_edges"],
            choose_step_mask=oracle_choose_step,
            base_edge_logits_full=rewrite_outputs_by_mode[BASE_MODE]["edge_logits_full"],
            step_edge_logits_full=rewrite_outputs_by_mode[STEP9C_MODE]["edge_logits_full"],
        )
        rewrite_outputs_by_mode[ORACLE_FALSE_SCOPE_MODE] = clone_rewrite_with_edge_logits(
            rewrite_outputs_by_mode[STEP9C_MODE],
            fallback_logits,
        )
        rewrite_outputs_by_mode[LEARNED_GATE_MODE] = clone_rewrite_with_edge_logits(
            rewrite_outputs_by_mode[STEP9C_MODE],
            learned_logits,
        )
        rewrite_outputs_by_mode[ORACLE_CHOOSE_MODE] = clone_rewrite_with_edge_logits(
            rewrite_outputs_by_mode[STEP9C_MODE],
            oracle_choose_logits,
        )

        pred_adj_base = bool_pred_adj(rewrite_outputs_by_mode[BASE_MODE]["edge_logits_full"], valid_edge_mask)
        pred_adj_by_mode = {
            mode: bool_pred_adj(rewrite_outputs_by_mode[mode]["edge_logits_full"], valid_edge_mask)
            for mode in DECOMP_MODES
        }

        for sample_idx in range(batch["node_feats"].shape[0]):
            group_names = make_group_names(batch, sample_idx)
            groups = ["overall"] + [name for name in group_names if name.startswith("event_type::")]
            if "step6a_corruption_setting" in batch:
                groups.append(f"corruption::{batch['step6a_corruption_setting'][sample_idx]}")

            for mode in MODE_ORDER:
                outputs = mode_outputs[mode]
                prop_stats = build_proposal_sample_stats(
                    batch=batch,
                    final_pred_edges=outputs["final_pred_scope_edges"],
                    rescued_edges=outputs["rescued_edges"],
                    valid_edge_mask=valid_edge_mask,
                    sample_idx=sample_idx,
                )
                down_stats = build_downstream_sample_stats(
                    batch=batch,
                    rewrite_outputs=rewrite_outputs_by_mode[mode],
                    valid_edge_mask=valid_edge_mask,
                    sample_idx=sample_idx,
                )
                for group in groups:
                    update_bucket(proposal_buckets[mode][group], prop_stats)
                    update_bucket(downstream_buckets[mode][group], down_stats)

            for mode in DECOMP_MODES:
                for group in groups:
                    update_decomp_sample(
                        bucket=decomp_buckets[mode][group],
                        batch=batch,
                        base_pred_edges=base_outputs["pred_scope_edges"],
                        rescued_edges=mode_outputs[mode]["rescued_edges"],
                        pred_adj_base=pred_adj_base,
                        pred_adj_budget=pred_adj_by_mode[mode],
                        valid_edge_mask=valid_edge_mask,
                        sample_idx=sample_idx,
                    )
            for mode, choose_step_mask in [
                (LEARNED_GATE_MODE, learned_choose_step),
                (ORACLE_CHOOSE_MODE, oracle_choose_step),
            ]:
                for group in groups:
                    update_gate_bucket(
                        bucket=gate_buckets[mode][group],
                        batch=batch,
                        valid_edge_mask=valid_edge_mask,
                        rescued_edges=mode_outputs[STEP9C_MODE]["rescued_edges"],
                        choose_step_mask=choose_step_mask,
                        target_choose_step=target_choose_step,
                        base_correct=base_correct,
                        step_correct=step_correct,
                        sample_idx=sample_idx,
                    )

    proposal_results: dict[str, Any] = {}
    downstream_results: dict[str, Any] = {}
    decomp_results: dict[str, Any] = {}
    gate_results: dict[str, Any] = {}
    for mode in MODE_ORDER:
        finalized_prop = {name: finalize_proposal_bucket(bucket) for name, bucket in proposal_buckets[mode].items()}
        finalized_down = {name: finalize_downstream_bucket(bucket) for name, bucket in downstream_buckets[mode].items()}
        proposal_results[mode] = format_grouped_results(finalized_prop, finalize_proposal_bucket(init_proposal_bucket()))
        downstream_results[mode] = format_grouped_results(finalized_down, finalize_downstream_bucket(init_downstream_bucket()))
    for mode in DECOMP_MODES:
        finalized_decomp = {name: finalize_decomp_bucket(bucket) for name, bucket in decomp_buckets[mode].items()}
        decomp_results[mode] = format_grouped_results(finalized_decomp, finalize_decomp_bucket(init_decomp_bucket()))
    for mode in GATE_MODES:
        finalized_gate = {name: finalize_gate_bucket(bucket) for name, bucket in gate_buckets[mode].items()}
        gate_results[mode] = format_grouped_results(finalized_gate, finalize_gate_bucket(init_gate_bucket()))

    return {
        "proposal_side": proposal_results,
        "downstream": downstream_results,
        "decomposition": decomp_results,
        "gate_behavior": gate_results,
        "deltas_vs_step9c": compute_deltas_vs_step9c(downstream_results),
    }


def format_grouped_results(finalized: Dict[str, Any], empty_overall: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "overall": finalized.get("overall", empty_overall),
        "by_event_type": {
            name.split("event_type::", 1)[1]: value
            for name, value in finalized.items()
            if name.startswith("event_type::")
        },
        "by_corruption_setting": {
            name.split("corruption::", 1)[1]: value
            for name, value in finalized.items()
            if name.startswith("corruption::")
        },
    }


def compute_deltas_vs_step9c(downstream_results: Dict[str, Any]) -> Dict[str, Any]:
    step9c = downstream_results[STEP9C_MODE]["overall"]
    return {
        mode: {
            key: None if metrics["overall"].get(key) is None else metrics["overall"][key] - step9c[key]
            for key in ["full_edge", "context_edge", "changed_edge", "add", "delete"]
        }
        for mode, metrics in downstream_results.items()
    }


def write_summary_csv(path: Path, payload: Dict[str, Any]) -> None:
    rows = []
    results = payload["results"]
    for mode, metrics_by_group in results["downstream"].items():
        down = metrics_by_group["overall"]
        decomp = results.get("decomposition", {}).get(mode, {}).get("overall", {})
        gate = results.get("gate_behavior", {}).get(mode, {}).get("overall", {})
        delta = results.get("deltas_vs_step9c", {}).get(mode, {})
        rows.append(
            {
                "mode": mode,
                "group": "overall",
                "full_edge": down.get("full_edge"),
                "context_edge": down.get("context_edge"),
                "changed_edge": down.get("changed_edge"),
                "add": down.get("add"),
                "delete": down.get("delete"),
                "delta_full_edge_vs_step9c": delta.get("full_edge"),
                "delta_context_edge_vs_step9c": delta.get("context_edge"),
                "delta_changed_edge_vs_step9c": delta.get("changed_edge"),
                "delta_add_vs_step9c": delta.get("add"),
                "delta_delete_vs_step9c": delta.get("delete"),
                "false_scope_preserve_rate": decomp.get("preserve_rate_rescued_false_scope_budget"),
                "true_changed_correct_rate": decomp.get("correct_edit_rate_rescued_true_changed_budget"),
                "false_scope_extra_context_errors": decomp.get("extra_context_errors_rescued_false_scope"),
                "fallback_fraction": gate.get("fallback_fraction"),
                "step9c_kept_fraction": gate.get("step9c_kept_fraction"),
                "fallback_false_scope_fraction": gate.get("fallback_false_scope_fraction"),
                "fallback_changed_fraction": gate.get("fallback_changed_fraction"),
                "step9c_kept_false_scope_fraction": gate.get("step9c_kept_false_scope_fraction"),
                "step9c_kept_changed_fraction": gate.get("step9c_kept_changed_fraction"),
                "chosen_correct_rate_on_rescued": gate.get("chosen_correct_rate_on_rescued"),
            }
        )
    fieldnames = list(rows[0].keys()) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal_checkpoint_path", type=str, required=True)
    parser.add_argument("--completion_checkpoint_path", type=str, required=True)
    parser.add_argument("--rewrite_checkpoint_path", type=str, required=True)
    parser.add_argument("--fallback_gate_checkpoint_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--split_name", type=str, default="test")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--artifact_dir", type=str, default="artifacts/step17_rescue_fallback_gate")
    parser.add_argument("--node_threshold", type=float, default=0.20)
    parser.add_argument("--edge_threshold", type=float, default=0.15)
    parser.add_argument("--completion_threshold", type=float, default=0.50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = get_device(args.device)
    proposal_checkpoint_path = resolve_path(args.proposal_checkpoint_path)
    completion_checkpoint_path = resolve_path(args.completion_checkpoint_path)
    rewrite_checkpoint_path = resolve_path(args.rewrite_checkpoint_path)
    fallback_gate_checkpoint_path = resolve_path(args.fallback_gate_checkpoint_path)
    data_path = resolve_path(args.data_path)
    artifact_dir = resolve_path(args.artifact_dir)

    _, loader = build_loader(str(data_path), args.batch_size, args.num_workers, pin_memory=(device.type == "cuda"))
    proposal_model = load_proposal_model(proposal_checkpoint_path, device)
    completion_model = load_completion_model(completion_checkpoint_path, device)
    rewrite_model, use_proposal_conditioning = load_rewrite_model(rewrite_checkpoint_path, device)
    gate_model = load_fallback_gate_model(fallback_gate_checkpoint_path, device)
    results = evaluate_step17(
        proposal_model=proposal_model,
        completion_model=completion_model,
        rewrite_model=rewrite_model,
        gate_model=gate_model,
        loader=loader,
        device=device,
        node_threshold=args.node_threshold,
        edge_threshold=args.edge_threshold,
        completion_threshold=args.completion_threshold,
        use_proposal_conditioning=use_proposal_conditioning,
    )

    run_name = args.run_name or slugify(f"{args.split_name}_{rewrite_checkpoint_path.parent.name}")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifact_dir / f"{run_name}.json"
    csv_path = artifact_dir / f"{run_name}.csv"
    payload = {
        "metadata": {
            "proposal_checkpoint_path": str(proposal_checkpoint_path),
            "completion_checkpoint_path": str(completion_checkpoint_path),
            "rewrite_checkpoint_path": str(rewrite_checkpoint_path),
            "fallback_gate_checkpoint_path": str(fallback_gate_checkpoint_path),
            "data_path": str(data_path),
            "split_name": args.split_name,
            "node_threshold": args.node_threshold,
            "edge_threshold": args.edge_threshold,
            "completion_threshold": args.completion_threshold,
            "rescue_budget_fraction": FIXED_RESCUE_BUDGET_FRACTION,
            "fallback_gate_mode": "learned_rescue_conditioned_gate",
            "tie_rule": "fallback_to_base_when_base_and_step9c_have_equal_correctness",
            "rewrite_uses_proposal_conditioning": use_proposal_conditioning,
        },
        "results": results,
    }
    save_json(out_path, payload)
    write_summary_csv(csv_path, payload)
    compact = {
        "downstream_overall": {mode: metrics["overall"] for mode, metrics in results["downstream"].items()},
        "gate_behavior_overall": {mode: metrics["overall"] for mode, metrics in results["gate_behavior"].items()},
        "decomposition_overall": {mode: metrics["overall"] for mode, metrics in results["decomposition"].items()},
        "deltas_vs_step9c": results["deltas_vs_step9c"],
    }
    print(f"device: {device}")
    print(f"saved json: {out_path}")
    print(f"saved csv: {csv_path}")
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

