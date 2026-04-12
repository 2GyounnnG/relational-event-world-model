from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SYSTEM_METRICS = ["full_edge_acc", "context_edge_acc", "changed_edge_acc", "add", "delete"]
PROPOSAL_METRICS = [
    "proposal_node_scope_recall",
    "proposal_edge_scope_recall",
    "proposal_edge_scope_excess_ratio",
    "proposal_out_of_scope_miss_edge",
]
SUMMARY_SECTIONS = [
    "overall_final",
    "by_dependency_bucket_final",
    "by_corruption_setting_final",
    "by_event_type_all_steps",
]


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def metric_subset(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {"count": metrics.get("count"), **{key: metrics.get(key) for key in SYSTEM_METRICS + PROPOSAL_METRICS}}


def metric_delta(candidate: Optional[float], baseline: Optional[float]) -> Optional[float]:
    if candidate is None or baseline is None:
        return None
    return candidate - baseline


def compare_metric_groups(candidate: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: metric_delta(candidate.get(key), baseline.get(key))
        for key in SYSTEM_METRICS + PROPOSAL_METRICS
    }


def extract_sections(payload: Dict[str, Any]) -> Dict[str, Any]:
    results = payload["results"]
    extracted: Dict[str, Any] = {}
    for section in SUMMARY_SECTIONS:
        if section == "overall_final":
            extracted[section] = {"overall": metric_subset(results["overall_final"])}
        else:
            extracted[section] = {
                group: metric_subset(metrics)
                for group, metrics in sorted(results.get(section, {}).items())
            }
    return extracted


def add_comparison(
    comparisons: Dict[str, Any],
    comparison_name: str,
    candidate: Optional[Dict[str, Any]],
    baseline: Optional[Dict[str, Any]],
) -> None:
    if candidate is None or baseline is None:
        return
    comparisons[comparison_name] = {
        "overall_final": compare_metric_groups(
            candidate["results"]["overall_final"],
            baseline["results"]["overall_final"],
        ),
        "strongly_interacting_final": compare_metric_groups(
            candidate["results"]["by_dependency_bucket_final"].get("strongly_interacting", {}),
            baseline["results"]["by_dependency_bucket_final"].get("strongly_interacting", {}),
        ),
        "fully_independent_final": compare_metric_groups(
            candidate["results"]["by_dependency_bucket_final"].get("fully_independent", {}),
            baseline["results"]["by_dependency_bucket_final"].get("fully_independent", {}),
        ),
        "partially_dependent_final": compare_metric_groups(
            candidate["results"]["by_dependency_bucket_final"].get("partially_dependent", {}),
            baseline["results"]["by_dependency_bucket_final"].get("partially_dependent", {}),
        ),
    }


def write_csv(path: Path, runs: Dict[str, Dict[str, Any]]) -> None:
    rows = []
    for run_name, payload in sorted(runs.items()):
        for section, groups in extract_sections(payload).items():
            for group, metrics in groups.items():
                row = {"run": run_name, "section": section, "group": group}
                row.update(metrics)
                rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["run", "section", "group", "count"] + SYSTEM_METRICS + PROPOSAL_METRICS
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step22_artifact_dir", type=str, default="artifacts/step22_noisy_multievent_interaction")
    parser.add_argument("--step23_artifact_dir", type=str, default="artifacts/step23_noisy_interaction_aware_proposal")
    parser.add_argument("--step24_artifact_dir", type=str, default="artifacts/step24_noisy_interaction_joint")
    parser.add_argument("--step26_artifact_dir", type=str, default="artifacts/step26_noisy_interaction_joint_deeper")
    parser.add_argument("--step27_artifact_dir", type=str, default="artifacts/step27_step26_factorization")
    parser.add_argument("--step28_artifact_dir", type=str, default="artifacts/step28_rft1_anchored_joint")
    parser.add_argument("--artifact_dir", type=str, default="artifacts/step29_retained_proposal_branch")
    parser.add_argument("--output_name", type=str, default="summary")
    args = parser.parse_args()

    step22_dir = resolve_path(args.step22_artifact_dir)
    step23_dir = resolve_path(args.step23_artifact_dir)
    step24_dir = resolve_path(args.step24_artifact_dir)
    step26_dir = resolve_path(args.step26_artifact_dir)
    step27_dir = resolve_path(args.step27_artifact_dir)
    step28_dir = resolve_path(args.step28_artifact_dir)
    artifact_dir = resolve_path(args.artifact_dir)

    run_paths = {
        "baseline_noisy_p2_rft1": step22_dir / "noisy_p2_rft1.json",
        "baseline_noisy_p2_i1520": step22_dir / "noisy_p2_i1520.json",
        "step23_proposal_only_rft1_optional": step23_dir / "noisy_step23_p2_rft1.json",
        "step24_light_joint": step24_dir / "noisy_step24_joint.json",
        "step26_full_joint": step26_dir / "noisy_step26_joint_deeper.json",
        "step26_proposal_rft1_retained_candidate": step27_dir / "step26_proposal_baseline_rewrite.json",
        "step28_rft1_anchored_joint": step28_dir / "noisy_step28_rft1_anchored_joint.json",
    }

    runs: Dict[str, Dict[str, Any]] = {}
    missing: Dict[str, str] = {}
    for run_name, path in run_paths.items():
        payload = load_json(path)
        if payload is None:
            missing[run_name] = str(path)
        else:
            runs[run_name] = payload

    candidate = runs.get("step26_proposal_rft1_retained_candidate")
    comparisons: Dict[str, Any] = {}
    for baseline_name in (
        "baseline_noisy_p2_rft1",
        "baseline_noisy_p2_i1520",
        "step23_proposal_only_rft1_optional",
        "step24_light_joint",
        "step26_full_joint",
        "step28_rft1_anchored_joint",
    ):
        add_comparison(
            comparisons,
            f"retained_candidate_minus_{baseline_name}",
            candidate,
            runs.get(baseline_name),
        )

    summary = {
        "metadata": {
            "evaluation": "Step 29 retained noisy interaction-aware proposal branch consolidation",
            "main_question": "Should Step26 proposal + RFT1 be retained as a noisy interaction-aware branch candidate?",
            "single_retained_candidate": "Step26 proposal + RFT1 rewrite",
            "fixed_thresholds": {"node_threshold": 0.15, "edge_threshold": 0.10},
            "note": "No model was trained. This script consolidates existing frozen-evaluation artifacts.",
        },
        "run_paths": {name: str(path) for name, path in run_paths.items()},
        "missing_runs": missing,
        "runs": {
            run_name: {
                "metadata": payload.get("metadata", {}),
                "sections": extract_sections(payload),
            }
            for run_name, payload in sorted(runs.items())
        },
        "comparisons": comparisons,
    }

    artifact_dir.mkdir(parents=True, exist_ok=True)
    out_json = artifact_dir / f"{args.output_name}.json"
    out_csv = artifact_dir / f"{args.output_name}.csv"
    save_json(out_json, summary)
    write_csv(out_csv, runs)
    print(f"saved summary json: {out_json}")
    print(f"saved summary csv: {out_csv}")
    if missing:
        print(f"missing optional/expected runs: {missing}")
    if comparisons:
        print(json.dumps(comparisons, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
