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
    return {key: metrics.get(key) for key in SYSTEM_METRICS + PROPOSAL_METRICS}


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


def write_csv(path: Path, runs: Dict[str, Dict[str, Any]]) -> None:
    rows = []
    for run_name, payload in runs.items():
        for section, groups in extract_sections(payload).items():
            for group, metrics in groups.items():
                row = {"run": run_name, "section": section, "group": group}
                row.update(metrics)
                rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["run", "section", "group"] + SYSTEM_METRICS + PROPOSAL_METRICS
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step22_artifact_dir", type=str, default="artifacts/step22_noisy_multievent_interaction")
    parser.add_argument("--step23_artifact_dir", type=str, default="artifacts/step23_noisy_interaction_aware_proposal")
    parser.add_argument("--step24_artifact_dir", type=str, default="artifacts/step24_noisy_interaction_joint")
    parser.add_argument("--step24_run_name", type=str, default="noisy_step24_joint")
    parser.add_argument("--output_name", type=str, default="summary")
    args = parser.parse_args()

    step22_dir = resolve_path(args.step22_artifact_dir)
    step23_dir = resolve_path(args.step23_artifact_dir)
    step24_dir = resolve_path(args.step24_artifact_dir)
    run_paths = {
        "baseline_noisy_p2_w012": step22_dir / "noisy_p2_w012.json",
        "baseline_noisy_p2_rft1": step22_dir / "noisy_p2_rft1.json",
        "baseline_noisy_p2_i1520_optional": step22_dir / "noisy_p2_i1520.json",
        "step23_proposal_only_rft1": step23_dir / "noisy_step23_p2_rft1.json",
        "step24_joint_candidate": step24_dir / f"{args.step24_run_name}.json",
    }
    runs: Dict[str, Dict[str, Any]] = {}
    missing: Dict[str, str] = {}
    for run_name, path in run_paths.items():
        payload = load_json(path)
        if payload is None:
            missing[run_name] = str(path)
        else:
            runs[run_name] = payload

    comparisons: Dict[str, Any] = {}
    candidate = runs.get("step24_joint_candidate")
    add_comparison(comparisons, "step24_minus_baseline_noisy_p2_rft1", candidate, runs.get("baseline_noisy_p2_rft1"))
    add_comparison(comparisons, "step24_minus_step23_proposal_only", candidate, runs.get("step23_proposal_only_rft1"))
    add_comparison(comparisons, "step24_minus_noisy_p2_i1520_optional", candidate, runs.get("baseline_noisy_p2_i1520_optional"))

    summary = {
        "metadata": {
            "evaluation": "Step 24 light joint noisy interaction-aware proposal+rewrite comparison",
            "single_mechanism_variable": "coupling_regime = frozen_rewrite_baseline | light_joint_noisy_interaction",
            "fixed_thresholds": {"node_threshold": 0.15, "edge_threshold": 0.10},
            "step22_artifact_dir": str(step22_dir),
            "step23_artifact_dir": str(step23_dir),
            "step24_artifact_dir": str(step24_dir),
            "note": "This script consolidates Step 22/23/24 evaluator outputs; it does not run inference itself.",
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
    out_json = step24_dir / f"{args.output_name}.json"
    out_csv = step24_dir / f"{args.output_name}.csv"
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
