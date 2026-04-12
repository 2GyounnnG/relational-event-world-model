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

from train.eval_step22_noisy_multievent_interaction import (
    evaluate_step22,
    get_device,
    inspect_dataset,
    load_proposal_model,
    load_rewrite_model,
    load_samples,
    save_json,
    write_summary_csv,
)
from train.eval_step25_noisy_multievent_oracle_headroom import evaluate_step25


SYSTEM_METRICS = ["full_edge_acc", "context_edge_acc", "changed_edge_acc", "add", "delete"]
PROPOSAL_METRICS = [
    "proposal_node_scope_recall",
    "proposal_edge_scope_recall",
    "proposal_edge_scope_excess_ratio",
    "proposal_out_of_scope_miss_edge",
]
SUMMARY_SECTIONS = [
    "overall_final",
    "overall_all_steps",
    "by_dependency_bucket_final",
    "by_dependency_bucket_all_steps",
    "by_corruption_setting_final",
    "by_corruption_setting_all_steps",
    "by_event_type_all_steps",
]


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


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
        elif section == "overall_all_steps":
            extracted[section] = {"overall": metric_subset(results["overall_all_steps"])}
        else:
            extracted[section] = {
                group: metric_subset(metrics)
                for group, metrics in sorted(results.get(section, {}).items())
            }
    return extracted


def compare_payloads(candidate: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
    comparisons = {
        "overall_final": compare_metric_groups(
            candidate["results"]["overall_final"],
            baseline["results"]["overall_final"],
        ),
        "overall_all_steps": compare_metric_groups(
            candidate["results"]["overall_all_steps"],
            baseline["results"]["overall_all_steps"],
        ),
    }
    for bucket in ("fully_independent", "partially_dependent", "strongly_interacting"):
        comparisons[f"{bucket}_final"] = compare_metric_groups(
            candidate["results"]["by_dependency_bucket_final"].get(bucket, {}),
            baseline["results"]["by_dependency_bucket_final"].get(bucket, {}),
        )
    return comparisons


def write_consolidated_csv(path: Path, runs: Dict[str, Dict[str, Any]]) -> None:
    rows = []
    for run_name, payload in sorted(runs.items()):
        metadata = payload.get("metadata", {})
        for section, groups in extract_sections(payload).items():
            for group, metrics in groups.items():
                row = {
                    "run": run_name,
                    "scope_source": metadata.get("scope_source"),
                    "proposal_component_source": metadata.get("proposal_component_source"),
                    "rewrite_component_source": metadata.get("rewrite_component_source"),
                    "section": section,
                    "group": group,
                }
                row.update(metrics)
                rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run",
        "scope_source",
        "proposal_component_source",
        "rewrite_component_source",
        "section",
        "group",
        "count",
    ] + SYSTEM_METRICS + PROPOSAL_METRICS
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_run_payload(
    *,
    run_name: str,
    results: Dict[str, Any],
    dataset_support: Dict[str, Any],
    data_path: Path,
    proposal_checkpoint_path: Optional[Path],
    rewrite_checkpoint_path: Path,
    proposal_component_source: str,
    rewrite_component_source: str,
    scope_source: str,
    use_proposal_conditioning: bool,
    node_threshold: float,
    edge_threshold: float,
) -> Dict[str, Any]:
    return {
        "metadata": {
            "run_name": run_name,
            "evaluation": "step27_step26_factorization",
            "single_variable": "component_source = baseline | step26 applied independently to proposal and rewrite",
            "scope_source": scope_source,
            "proposal_component_source": proposal_component_source,
            "rewrite_component_source": rewrite_component_source,
            "proposal_checkpoint_path": str(proposal_checkpoint_path) if proposal_checkpoint_path else None,
            "rewrite_checkpoint_path": str(rewrite_checkpoint_path),
            "data_path": str(data_path),
            "observation_regime": "noisy",
            "node_threshold": node_threshold,
            "edge_threshold": edge_threshold,
            "rewrite_uses_proposal_conditioning": use_proposal_conditioning,
            "note": "No training, no threshold changes. This factors Step26 into proposal-side and rewrite-side component swaps.",
        },
        "dataset_support": dataset_support,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/graph_event_step22_noisy_step5_test.pkl")
    parser.add_argument("--baseline_proposal_checkpoint", type=str, default="checkpoints/proposal_noisy_obs_p2/best.pt")
    parser.add_argument("--baseline_rewrite_checkpoint", type=str, default="checkpoints/step6_noisy_rewrite_rft1/best.pt")
    parser.add_argument(
        "--step26_proposal_checkpoint",
        type=str,
        default="checkpoints/step26_noisy_interaction_joint_deeper/proposal_best.pt",
    )
    parser.add_argument(
        "--step26_rewrite_checkpoint",
        type=str,
        default="checkpoints/step26_noisy_interaction_joint_deeper/rewrite_best.pt",
    )
    parser.add_argument("--artifact_dir", type=str, default="artifacts/step27_step26_factorization")
    parser.add_argument("--node_threshold", type=float, default=0.15)
    parser.add_argument("--edge_threshold", type=float, default=0.10)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--skip_oracle", action="store_true")
    args = parser.parse_args()

    data_path = resolve_path(args.data_path)
    artifact_dir = resolve_path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)
    samples = load_samples(data_path)
    dataset_support = inspect_dataset(samples, "noisy")

    proposal_paths = {
        "baseline": resolve_path(args.baseline_proposal_checkpoint),
        "step26": resolve_path(args.step26_proposal_checkpoint),
    }
    rewrite_paths = {
        "baseline": resolve_path(args.baseline_rewrite_checkpoint),
        "step26": resolve_path(args.step26_rewrite_checkpoint),
    }
    proposal_models = {
        name: load_proposal_model(path, device)
        for name, path in proposal_paths.items()
    }
    rewrite_models = {
        name: load_rewrite_model(path, device)
        for name, path in rewrite_paths.items()
    }

    learned_specs = [
        ("baseline_proposal_baseline_rewrite", "baseline", "baseline"),
        ("step26_proposal_baseline_rewrite", "step26", "baseline"),
        ("baseline_proposal_step26_rewrite", "baseline", "step26"),
        ("step26_proposal_step26_rewrite", "step26", "step26"),
    ]
    runs: Dict[str, Dict[str, Any]] = {}
    for run_name, proposal_source, rewrite_source in learned_specs:
        rewrite_model, use_conditioning = rewrite_models[rewrite_source]
        results = evaluate_step22(
            proposal_model=proposal_models[proposal_source],
            rewrite_model=rewrite_model,
            samples=samples,
            device=device,
            observation_regime="noisy",
            node_threshold=args.node_threshold,
            edge_threshold=args.edge_threshold,
            use_proposal_conditioning=use_conditioning,
        )
        payload = build_run_payload(
            run_name=run_name,
            results=results,
            dataset_support=dataset_support,
            data_path=data_path,
            proposal_checkpoint_path=proposal_paths[proposal_source],
            rewrite_checkpoint_path=rewrite_paths[rewrite_source],
            proposal_component_source=proposal_source,
            rewrite_component_source=rewrite_source,
            scope_source="learned",
            use_proposal_conditioning=use_conditioning,
            node_threshold=args.node_threshold,
            edge_threshold=args.edge_threshold,
        )
        runs[run_name] = payload
        save_json(artifact_dir / f"{run_name}.json", payload)
        write_summary_csv(artifact_dir / f"{run_name}.csv", payload)
        print(f"saved learned run: {artifact_dir / f'{run_name}.json'}")

    if not args.skip_oracle:
        for run_name, rewrite_source in (
            ("oracle_scope_baseline_rewrite", "baseline"),
            ("oracle_scope_step26_rewrite", "step26"),
        ):
            rewrite_model, use_conditioning = rewrite_models[rewrite_source]
            results = evaluate_step25(
                proposal_model=None,
                rewrite_model=rewrite_model,
                samples=samples,
                device=device,
                observation_regime="noisy",
                node_threshold=args.node_threshold,
                edge_threshold=args.edge_threshold,
                use_proposal_conditioning=use_conditioning,
                scope_source="oracle",
            )
            payload = build_run_payload(
                run_name=run_name,
                results=results,
                dataset_support=dataset_support,
                data_path=data_path,
                proposal_checkpoint_path=None,
                rewrite_checkpoint_path=rewrite_paths[rewrite_source],
                proposal_component_source="oracle",
                rewrite_component_source=rewrite_source,
                scope_source="oracle",
                use_proposal_conditioning=use_conditioning,
                node_threshold=args.node_threshold,
                edge_threshold=args.edge_threshold,
            )
            runs[run_name] = payload
            save_json(artifact_dir / f"{run_name}.json", payload)
            write_summary_csv(artifact_dir / f"{run_name}.csv", payload)
            print(f"saved oracle run: {artifact_dir / f'{run_name}.json'}")

    comparisons: Dict[str, Any] = {}
    baseline = runs["baseline_proposal_baseline_rewrite"]
    full_step26 = runs["step26_proposal_step26_rewrite"]
    step26_prop_safe_rewrite = runs["step26_proposal_baseline_rewrite"]
    safe_prop_step26_rewrite = runs["baseline_proposal_step26_rewrite"]
    comparisons["full_step26_minus_baseline"] = compare_payloads(full_step26, baseline)
    comparisons["step26_proposal_with_baseline_rewrite_minus_baseline"] = compare_payloads(
        step26_prop_safe_rewrite,
        baseline,
    )
    comparisons["step26_rewrite_with_baseline_proposal_minus_baseline"] = compare_payloads(
        safe_prop_step26_rewrite,
        baseline,
    )
    comparisons["step26_rewrite_effect_under_step26_proposal"] = compare_payloads(
        full_step26,
        step26_prop_safe_rewrite,
    )
    comparisons["step26_proposal_effect_under_step26_rewrite"] = compare_payloads(
        full_step26,
        safe_prop_step26_rewrite,
    )
    if "oracle_scope_baseline_rewrite" in runs and "oracle_scope_step26_rewrite" in runs:
        comparisons["oracle_step26_rewrite_minus_oracle_baseline_rewrite"] = compare_payloads(
            runs["oracle_scope_step26_rewrite"],
            runs["oracle_scope_baseline_rewrite"],
        )

    summary = {
        "metadata": {
            "evaluation": "Step 27 Step26 proposal/rewrite factorization",
            "single_variable": "component_source = baseline | step26 applied independently to proposal and rewrite",
            "fixed_thresholds": {"node_threshold": args.node_threshold, "edge_threshold": args.edge_threshold},
            "data_path": str(data_path),
            "artifact_dir": str(artifact_dir),
            "device": str(device),
            "baseline_proposal_checkpoint": str(proposal_paths["baseline"]),
            "baseline_rewrite_checkpoint": str(rewrite_paths["baseline"]),
            "step26_proposal_checkpoint": str(proposal_paths["step26"]),
            "step26_rewrite_checkpoint": str(rewrite_paths["step26"]),
            "oracle_rows_included": not args.skip_oracle,
        },
        "run_paths": {run_name: str(artifact_dir / f"{run_name}.json") for run_name in sorted(runs)},
        "runs": {
            run_name: {
                "metadata": payload.get("metadata", {}),
                "sections": extract_sections(payload),
            }
            for run_name, payload in sorted(runs.items())
        },
        "comparisons": comparisons,
    }
    save_json(artifact_dir / "summary.json", summary)
    write_consolidated_csv(artifact_dir / "summary.csv", runs)
    print(f"saved summary json: {artifact_dir / 'summary.json'}")
    print(f"saved summary csv: {artifact_dir / 'summary.csv'}")
    print(json.dumps(comparisons, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
