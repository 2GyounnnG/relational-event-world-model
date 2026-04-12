from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train.eval_noisy_structured_observation import evaluate
from train.train_oracle_local_delta_proposal_conditioned_fp_keep import (
    get_device,
    require_oracle_scope,
    resolve_path,
    save_json,
    set_seed,
)
from train.train_step6_joint_noisy_finetune import (
    build_dataloaders,
    load_proposal_init,
    run_joint_epoch,
    save_checkpoint,
)
from train.train_step6_noisy_rewrite_finetune import build_rewrite_model


def safe_float(metrics: Dict[str, Any], key: str) -> float:
    value = metrics.get(key)
    return float(value) if value is not None else 0.0


def compute_coverage_emphasized_selection_score(results: Dict[str, Dict[str, Any]]) -> float:
    overall = results["overall"]
    # Fixed recipe-level selection: keep broad stability in the objective, but make
    # proposal edge recall visible so the selected checkpoint does not collapse back
    # to the conservative Step23/Step24 mode.
    return (
        0.25 * safe_float(overall, "full_edge_acc")
        + 0.25 * safe_float(overall, "context_edge_acc")
        + 0.20 * safe_float(overall, "changed_edge_acc")
        + 0.10 * safe_float(overall, "add")
        + 0.10 * safe_float(overall, "delete")
        + 0.10 * safe_float(overall, "proposal_edge_recall")
    )


def summarize_raw_samples(dataset) -> Dict[str, Any]:
    bucket_counts = Counter(str(sample.get("step5_dependency_bucket", "unknown")) for sample in dataset.samples)
    corruption_counts = Counter(str(sample.get("step6a_corruption_setting", "unknown")) for sample in dataset.samples)
    event_counts = Counter(
        str(sample.get("events", [{}])[0].get("event_type", "unknown"))
        for sample in dataset.samples
    )
    return {
        "sample_count": len(dataset),
        "dependency_bucket_counts": dict(sorted(bucket_counts.items())),
        "corruption_setting_counts": dict(sorted(corruption_counts.items())),
        "event_type_counts": dict(sorted(event_counts.items())),
    }


def write_training_summary(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="data/graph_event_step23_noisy_step5_train_transitions.pkl")
    parser.add_argument("--val_path", type=str, default="data/graph_event_step23_noisy_step5_val_transitions.pkl")
    parser.add_argument("--init_proposal_checkpoint", type=str, default="checkpoints/proposal_noisy_obs_p2/best.pt")
    parser.add_argument("--init_rewrite_checkpoint", type=str, default="checkpoints/step6_noisy_rewrite_rft1/best.pt")
    parser.add_argument("--save_dir", type=str, default="checkpoints/step26_noisy_interaction_joint_deeper")
    parser.add_argument("--node_threshold", type=float, default=0.15)
    parser.add_argument("--edge_threshold", type=float, default=0.10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--edge_loss_weight", type=float, default=1.0)
    parser.add_argument("--type_loss_weight", type=float, default=1.0)
    parser.add_argument("--state_loss_weight", type=float, default=1.0)
    parser.add_argument("--type_flip_weight", type=float, default=1.0)
    parser.add_argument("--delta_keep_weight", type=float, default=1.10)
    parser.add_argument("--delta_add_weight", type=float, default=1.0)
    parser.add_argument("--delta_delete_weight", type=float, default=3.0)
    parser.add_argument("--fp_edge_keep_loss_weight", type=float, default=0.12)
    parser.add_argument("--node_scope_loss_weight", type=float, default=1.0)
    parser.add_argument("--node_flip_weight", type=float, default=2.0)
    parser.add_argument("--edge_scope_loss_weight", type=float, default=1.0)
    parser.add_argument("--edge_scope_pos_weight", type=float, default=4.0)
    parser.add_argument("--joint_proposal_loss_weight", type=float, default=2.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    pin_memory = device.type == "cuda"

    init_proposal_checkpoint_path = resolve_path(args.init_proposal_checkpoint)
    init_rewrite_checkpoint_path = resolve_path(args.init_rewrite_checkpoint)
    save_dir = PROJECT_ROOT / args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset, train_loader, val_loader = build_dataloaders(
        train_path=str(resolve_path(args.train_path)),
        val_path=str(resolve_path(args.val_path)),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    sample = train_dataset[0]
    require_oracle_scope(sample)
    if "obs_node_feats" not in sample or "obs_adj" not in sample:
        raise KeyError("Step 26 expects noisy Step 22/23 transition samples with obs_node_feats/obs_adj.")

    proposal_model, proposal_config = load_proposal_init(init_proposal_checkpoint_path, device)
    rewrite_model, rewrite_config, use_proposal_conditioning = build_rewrite_model(
        sample=sample,
        init_rewrite_checkpoint_path=init_rewrite_checkpoint_path,
        device=device,
    )
    optimizer = torch.optim.Adam(
        list(proposal_model.parameters()) + list(rewrite_model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    proposal_best_path = save_dir / "proposal_best.pt"
    proposal_last_path = save_dir / "proposal_last.pt"
    rewrite_best_path = save_dir / "rewrite_best.pt"
    rewrite_last_path = save_dir / "rewrite_last.pt"
    best_metrics_path = save_dir / "best_metrics.json"
    training_summary_path = save_dir / "training_summary.json"

    best_score = float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    selection_metric = (
        "0.25*full_edge + 0.25*context_edge + 0.20*changed_edge + "
        "0.10*add + 0.10*delete + 0.10*proposal_edge_recall"
    )

    recipe = {
        "joint_recipe": "coverage_emphasized_joint",
        "coverage_emphasis": {
            "joint_proposal_loss_weight": args.joint_proposal_loss_weight,
            "edge_scope_pos_weight": args.edge_scope_pos_weight,
            "selection_metric": selection_metric,
        },
        "note": (
            "This is one recipe-level change: stronger proposal coverage pressure "
            "inside otherwise Step24-style joint fine-tuning."
        ),
    }

    print(f"device: {device}")
    print(f"train dataset size: {len(train_dataset)}")
    print(f"val dataset size: {len(val_dataset)}")
    print(f"train summary: {summarize_raw_samples(train_dataset)}")
    print(f"val summary: {summarize_raw_samples(val_dataset)}")
    print(f"proposal init checkpoint: {init_proposal_checkpoint_path}")
    print(f"rewrite init checkpoint: {init_rewrite_checkpoint_path}")
    print(f"proposal node threshold: {args.node_threshold}")
    print(f"proposal edge threshold: {args.edge_threshold}")
    print(f"proposal conditioning enabled: {use_proposal_conditioning}")
    print(f"joint recipe: {recipe}")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_joint_epoch(
            proposal_model=proposal_model,
            rewrite_model=rewrite_model,
            loader=train_loader,
            device=device,
            node_scope_loss_weight=args.node_scope_loss_weight,
            node_flip_weight=args.node_flip_weight,
            edge_scope_loss_weight=args.edge_scope_loss_weight,
            edge_scope_pos_weight=args.edge_scope_pos_weight,
            edge_loss_weight=args.edge_loss_weight,
            type_loss_weight=args.type_loss_weight,
            state_loss_weight=args.state_loss_weight,
            type_flip_weight=args.type_flip_weight,
            delta_keep_weight=args.delta_keep_weight,
            delta_add_weight=args.delta_add_weight,
            delta_delete_weight=args.delta_delete_weight,
            fp_edge_keep_loss_weight=args.fp_edge_keep_loss_weight,
            node_threshold=args.node_threshold,
            edge_threshold=args.edge_threshold,
            joint_proposal_loss_weight=args.joint_proposal_loss_weight,
            use_proposal_conditioning=use_proposal_conditioning,
            optimizer=optimizer,
            grad_clip=args.grad_clip,
        )
        val_metrics = run_joint_epoch(
            proposal_model=proposal_model,
            rewrite_model=rewrite_model,
            loader=val_loader,
            device=device,
            node_scope_loss_weight=args.node_scope_loss_weight,
            node_flip_weight=args.node_flip_weight,
            edge_scope_loss_weight=args.edge_scope_loss_weight,
            edge_scope_pos_weight=args.edge_scope_pos_weight,
            edge_loss_weight=args.edge_loss_weight,
            type_loss_weight=args.type_loss_weight,
            state_loss_weight=args.state_loss_weight,
            type_flip_weight=args.type_flip_weight,
            delta_keep_weight=args.delta_keep_weight,
            delta_add_weight=args.delta_add_weight,
            delta_delete_weight=args.delta_delete_weight,
            fp_edge_keep_loss_weight=args.fp_edge_keep_loss_weight,
            node_threshold=args.node_threshold,
            edge_threshold=args.edge_threshold,
            joint_proposal_loss_weight=args.joint_proposal_loss_weight,
            use_proposal_conditioning=use_proposal_conditioning,
            optimizer=None,
            grad_clip=None,
        )
        val_noisy_results = evaluate(
            proposal_model=proposal_model,
            rewrite_model=rewrite_model,
            loader=val_loader,
            device=device,
            node_threshold=args.node_threshold,
            edge_threshold=args.edge_threshold,
            use_proposal_conditioning=use_proposal_conditioning,
        )
        selection_score = compute_coverage_emphasized_selection_score(val_noisy_results)
        noisy_overall = val_noisy_results["overall"]
        history.append(
            {
                "epoch": epoch,
                "selection_score": selection_score,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "val_noisy_overall": noisy_overall,
            }
        )

        print(
            f"[Epoch {epoch:03d}] "
            f"train_rewrite_local_total={train_metrics['local_total_loss']:.6f} "
            f"train_proposal_total={train_metrics['proposal_total_loss']:.6f} "
            f"val_rewrite_local_total={val_metrics['local_total_loss']:.6f} "
            f"val_proposal_total={val_metrics['proposal_total_loss']:.6f} "
            f"val_prop_node_f1={val_metrics['proposal_node_f1']:.6f} "
            f"val_prop_edge_f1={val_metrics['proposal_edge_f1']:.6f} | "
            f"noisy_full_edge={safe_float(noisy_overall, 'full_edge_acc'):.6f} "
            f"noisy_changed_edge={safe_float(noisy_overall, 'changed_edge_acc'):.6f} "
            f"noisy_context_edge={safe_float(noisy_overall, 'context_edge_acc'):.6f} "
            f"noisy_delete={safe_float(noisy_overall, 'delete'):.6f} "
            f"noisy_add={safe_float(noisy_overall, 'add'):.6f} "
            f"noisy_prop_edge_recall={safe_float(noisy_overall, 'proposal_edge_recall'):.6f} "
            f"selection_score={selection_score:.6f}"
        )

        ckpt_args = vars(args)
        for path, model, model_config in (
            (proposal_last_path, proposal_model, proposal_config),
            (rewrite_last_path, rewrite_model, rewrite_config),
        ):
            save_checkpoint(
                path=path,
                epoch=epoch,
                model=model,
                model_config=model_config,
                optimizer_state_dict=optimizer.state_dict(),
                args=ckpt_args,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                val_noisy_results=val_noisy_results,
                selection_score=selection_score,
            )

        if selection_score > best_score:
            best_score = selection_score
            best_epoch = epoch
            epochs_without_improvement = 0
            for path, model, model_config in (
                (proposal_best_path, proposal_model, proposal_config),
                (rewrite_best_path, rewrite_model, rewrite_config),
            ):
                save_checkpoint(
                    path=path,
                    epoch=epoch,
                    model=model,
                    model_config=model_config,
                    optimizer_state_dict=optimizer.state_dict(),
                    args=ckpt_args,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    val_noisy_results=val_noisy_results,
                    selection_score=selection_score,
                )
            save_json(
                best_metrics_path,
                {
                    "epoch": epoch,
                    "best_selection_score": best_score,
                    "selection_metric": selection_metric,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "val_noisy_results": val_noisy_results,
                    "args": ckpt_args,
                    "recipe": recipe,
                    "proposal_init_checkpoint": str(init_proposal_checkpoint_path),
                    "rewrite_init_checkpoint": str(init_rewrite_checkpoint_path),
                    "proposal_best_checkpoint": str(proposal_best_path),
                    "rewrite_best_checkpoint": str(rewrite_best_path),
                    "proposal_conditioning_enabled": use_proposal_conditioning,
                    "train_dataset_summary": summarize_raw_samples(train_dataset),
                    "val_dataset_summary": summarize_raw_samples(val_dataset),
                },
            )
            print(f"  saved new best checkpoints -> {proposal_best_path} and {rewrite_best_path}")
        else:
            epochs_without_improvement += 1
            print(f"  no improvement for {epochs_without_improvement} epoch(s)")

        if epochs_without_improvement >= args.patience:
            print(f"early stopping triggered at epoch {epoch} (best epoch: {best_epoch})")
            break

    write_training_summary(
        training_summary_path,
        {
            "best_epoch": best_epoch,
            "best_selection_score": best_score,
            "selection_metric": selection_metric,
            "history": history,
            "args": vars(args),
            "recipe": recipe,
            "proposal_init_checkpoint": str(init_proposal_checkpoint_path),
            "rewrite_init_checkpoint": str(init_rewrite_checkpoint_path),
            "proposal_best_checkpoint": str(proposal_best_path),
            "rewrite_best_checkpoint": str(rewrite_best_path),
            "proposal_conditioning_enabled": use_proposal_conditioning,
            "train_dataset_summary": summarize_raw_samples(train_dataset),
            "val_dataset_summary": summarize_raw_samples(val_dataset),
        },
    )
    print(f"training complete. best epoch={best_epoch}, best selection score={best_score:.6f}")
    print(f"proposal best checkpoint: {proposal_best_path}")
    print(f"rewrite best checkpoint: {rewrite_best_path}")
    print(f"best metrics json: {best_metrics_path}")
    print(f"training summary json: {training_summary_path}")


if __name__ == "__main__":
    main()
