from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train.eval_noisy_structured_observation import build_loader, get_device, resolve_path, save_json
from train.eval_scope_edit_localization_gap import slugify
from train.eval_step9_gated_edge_completion import (
    load_completion_model,
    load_proposal_model,
    load_rewrite_model,
)
from train.eval_step9_rescue_frontier import (
    budget_mode_name,
    evaluate_rescue_frontier,
    write_summary_csv,
)


FIXED_RESCUE_BUDGET_FRACTION = 0.10


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal_checkpoint_path", type=str, required=True)
    parser.add_argument("--rewrite_checkpoint_path", type=str, required=True)
    parser.add_argument("--completion_checkpoint_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--split_name", type=str, default="test")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--artifact_dir", type=str, default="artifacts/step9c_budgeted_internal_completion")
    parser.add_argument("--node_threshold", type=float, default=0.20)
    parser.add_argument("--edge_threshold", type=float, default=0.15)
    parser.add_argument("--completion_threshold", type=float, default=0.50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    proposal_checkpoint_path = resolve_path(args.proposal_checkpoint_path)
    rewrite_checkpoint_path = resolve_path(args.rewrite_checkpoint_path)
    completion_checkpoint_path = resolve_path(args.completion_checkpoint_path)
    data_path = resolve_path(args.data_path)
    artifact_dir = resolve_path(args.artifact_dir)
    device = get_device(args.device)

    _, loader = build_loader(
        data_path=str(data_path),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    proposal_model = load_proposal_model(proposal_checkpoint_path, device)
    rewrite_model, use_proposal_conditioning = load_rewrite_model(rewrite_checkpoint_path, device)
    completion_model = load_completion_model(completion_checkpoint_path, device)

    results = evaluate_rescue_frontier(
        proposal_model=proposal_model,
        rewrite_model=rewrite_model,
        completion_model=completion_model,
        loader=loader,
        device=device,
        node_threshold=args.node_threshold,
        edge_threshold=args.edge_threshold,
        completion_threshold=args.completion_threshold,
        rescue_budget_fractions=(FIXED_RESCUE_BUDGET_FRACTION,),
        use_proposal_conditioning=use_proposal_conditioning,
    )

    run_name = args.run_name or slugify(f"{args.split_name}_{rewrite_checkpoint_path.parent.name}")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifact_dir / f"{run_name}.json"
    csv_path = artifact_dir / f"{run_name}.csv"
    fixed_budget_mode = budget_mode_name(FIXED_RESCUE_BUDGET_FRACTION)
    payload = {
        "metadata": {
            "proposal_checkpoint_path": str(proposal_checkpoint_path),
            "rewrite_checkpoint_path": str(rewrite_checkpoint_path),
            "completion_checkpoint_path": str(completion_checkpoint_path),
            "data_path": str(data_path),
            "split_name": args.split_name,
            "node_threshold": args.node_threshold,
            "edge_threshold": args.edge_threshold,
            "completion_threshold": args.completion_threshold,
            "rescue_budget_fraction": FIXED_RESCUE_BUDGET_FRACTION,
            "fixed_budget_mode": fixed_budget_mode,
            "budget_definition": "per-sample top-k over internal candidates; k=floor(0.10 * candidate_count)",
            "candidate_definition": "both endpoints inside predicted node scope and base edge proposal did not select the edge",
            "reported_modes": [
                "off",
                "learned_default_threshold_0.5",
                fixed_budget_mode,
                "naive_induced_closure",
            ],
            "rewrite_uses_proposal_conditioning": use_proposal_conditioning,
            "notes": [
                "Step 9c is an operating-mode evaluation, not a training run.",
                "Node proposal, completion head, and rewrite checkpoint are unchanged.",
                "Only the proposal edge mask/probabilities are modified by fixed-budget internal completion.",
            ],
        },
        "results": results,
    }
    save_json(out_path, payload)
    write_summary_csv(csv_path, payload)

    compact = {
        "proposal_overall": {
            mode: metrics["overall"]
            for mode, metrics in results["proposal_side"].items()
        },
        "downstream_overall": {
            mode: metrics["overall"]
            for mode, metrics in results["downstream"].items()
        },
    }
    print(f"device: {device}")
    print(f"fixed rescue budget fraction: {FIXED_RESCUE_BUDGET_FRACTION}")
    print(f"saved json: {out_path}")
    print(f"saved csv: {csv_path}")
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
