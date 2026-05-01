from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from common import (
    CANDIDATE_FEATURE_BASENAMES,
    CANDIDATE_K_VALUES,
    SequenceWindowDataset,
    compute_candidate_aware_metrics,
    compute_multitask_metrics,
    load_json,
    save_json,
)
from model import ModelConfig, MultitaskLstmPredictor


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train multitask LSTM handover predictor.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--target-loss-weight", type=float, default=1.5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--window-stride", type=int, default=1)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-val-batches", type=int, default=None)
    parser.add_argument("--target-mode", type=str, choices=("flat", "candidate"), default="flat")
    parser.add_argument("--candidate-top-k", type=int, choices=CANDIDATE_K_VALUES, default=None)
    parser.add_argument("--init-from-checkpoint", type=Path, default=None)
    return parser


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def create_loader(
    frame: pd.DataFrame,
    seq_len: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    window_stride: int,
    candidate_top_k: int | None,
    num_cells: int,
) -> DataLoader:
    dataset = SequenceWindowDataset(
        frame=frame,
        seq_len=seq_len,
        window_stride=window_stride,
        candidate_top_k=candidate_top_k,
        num_cells=num_cells,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def maybe_initialize_from_checkpoint(
    model: MultitaskLstmPredictor,
    checkpoint_path: Path | None,
) -> None:
    if checkpoint_path is None:
        return
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_state = checkpoint["model_state"]
    model_state = model.state_dict()
    filtered_state: dict[str, torch.Tensor] = {}
    partially_loaded: list[str] = []

    for key, value in checkpoint_state.items():
        if key not in model_state:
            continue
        if model_state[key].shape == value.shape:
            filtered_state[key] = value
            continue
        if key == "cell_embedding.weight" and model_state[key].shape[1] == value.shape[1]:
            rows = min(model_state[key].shape[0], value.shape[0])
            merged = model_state[key].clone()
            merged[:rows] = value[:rows]
            filtered_state[key] = merged
            partially_loaded.append(key)

    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    print(
        f"Initialized from {checkpoint_path} with missing_keys={len(missing)} "
        f"unexpected_keys={len(unexpected)} partial_keys={partially_loaded}",
        flush=True,
    )


def compute_target_loss(
    outputs: dict[str, torch.Tensor],
    trigger_target: torch.Tensor,
    target_target: torch.Tensor,
    target_mode: str,
    candidate_target_pos: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    positive_mask = (trigger_target > 0.5) & (target_target >= 0)
    positive_count = int(positive_mask.sum().item())
    if positive_count == 0:
        zero = (
            outputs["target_logits"].sum() * 0.0
            if target_mode == "flat"
            else outputs["global_target_logits"].sum() * 0.0
        )
        return zero, {
            "positive_count": 0.0,
            "candidate_loss_samples": 0.0,
            "fallback_loss_samples": 0.0,
        }

    if target_mode == "flat":
        target_loss = (
            F.cross_entropy(
                outputs["target_logits"][positive_mask],
                target_target[positive_mask],
                reduction="sum",
            )
            / positive_count
        )
        return target_loss, {
            "positive_count": float(positive_count),
            "candidate_loss_samples": 0.0,
            "fallback_loss_samples": float(positive_count),
        }

    if candidate_target_pos is None:
        raise ValueError("candidate_target_pos is required in candidate mode")

    candidate_loss_mask = positive_mask & (candidate_target_pos >= 0)
    fallback_loss_mask = positive_mask & ~candidate_loss_mask
    total_loss_sum = outputs["global_target_logits"].sum() * 0.0

    if candidate_loss_mask.any():
        total_loss_sum = total_loss_sum + F.cross_entropy(
            outputs["candidate_logits"][candidate_loss_mask],
            candidate_target_pos[candidate_loss_mask],
            reduction="sum",
        )
    if fallback_loss_mask.any():
        total_loss_sum = total_loss_sum + F.cross_entropy(
            outputs["global_target_logits"][fallback_loss_mask],
            target_target[fallback_loss_mask],
            reduction="sum",
        )

    return total_loss_sum / positive_count, {
        "positive_count": float(positive_count),
        "candidate_loss_samples": float(candidate_loss_mask.sum().item()),
        "fallback_loss_samples": float(fallback_loss_mask.sum().item()),
    }


def collect_target_predictions(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    target_mode: str,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    if target_mode == "flat":
        return (
            outputs["target_logits"].detach().cpu().numpy(),
            None,
            None,
        )

    candidate_mask = batch["candidate_mask"].cpu().numpy().astype(bool)
    candidate_cells = batch["candidate_cell"].cpu().numpy().astype(np.int64)
    candidate_logits = outputs["candidate_logits"].detach().cpu().numpy()
    global_target_logits = outputs["global_target_logits"].detach().cpu().numpy()

    has_candidate_data = candidate_mask.any(axis=1)
    candidate_choice = np.argmax(candidate_logits, axis=1)
    row_index = np.arange(candidate_choice.shape[0])
    candidate_prediction = candidate_cells[row_index, candidate_choice]
    global_prediction = np.argmax(global_target_logits, axis=1)
    final_prediction = np.where(has_candidate_data, candidate_prediction, global_prediction)
    return (
        final_prediction.astype(np.int64),
        batch["candidate_hit"].cpu().numpy().astype(bool),
        (~has_candidate_data).astype(bool),
    )


def evaluate_loop(
    model: MultitaskLstmPredictor,
    loader: DataLoader,
    trigger_criterion: nn.Module,
    device: torch.device,
    target_loss_weight: float,
    target_mode: str,
    limit_batches: int | None,
) -> tuple[dict[str, float], dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total_trigger_loss = 0.0
    total_target_loss = 0.0
    batch_count = 0

    trigger_logits_list: list[np.ndarray] = []
    trigger_targets_list: list[np.ndarray] = []
    flat_target_logits_list: list[np.ndarray] = []
    target_targets_list: list[np.ndarray] = []
    final_target_predictions_list: list[np.ndarray] = []
    candidate_hit_list: list[np.ndarray] = []
    fallback_used_list: list[np.ndarray] = []

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if limit_batches is not None and batch_index >= limit_batches:
                break

            numeric = batch["numeric"].to(device)
            serving_cell = batch["serving_cell"].to(device)
            trigger_target = batch["trigger"].to(device)
            target_target = batch["target"].to(device)

            if target_mode == "candidate":
                candidate_cell = batch["candidate_cell"].to(device)
                candidate_mask = batch["candidate_mask"].to(device)
                candidate_features = batch["candidate_features"].to(device)
                candidate_target_pos = batch["candidate_target_pos"].to(device)
                outputs = model(
                    numeric=numeric,
                    serving_cell=serving_cell,
                    candidate_cell=candidate_cell,
                    candidate_features=candidate_features,
                    candidate_mask=candidate_mask,
                )
            else:
                candidate_target_pos = None
                outputs = model(numeric=numeric, serving_cell=serving_cell)

            trigger_loss = trigger_criterion(outputs["trigger_logits"], trigger_target)
            target_loss, _ = compute_target_loss(
                outputs=outputs,
                trigger_target=trigger_target,
                target_target=target_target,
                target_mode=target_mode,
                candidate_target_pos=candidate_target_pos,
            )
            loss = trigger_loss + target_loss_weight * target_loss

            total_loss += float(loss.item())
            total_trigger_loss += float(trigger_loss.item())
            total_target_loss += float(target_loss.item())
            batch_count += 1

            trigger_logits_list.append(outputs["trigger_logits"].detach().cpu().numpy())
            trigger_targets_list.append(trigger_target.detach().cpu().numpy())
            target_targets_list.append(target_target.detach().cpu().numpy())

            if target_mode == "flat":
                flat_target_logits_list.append(outputs["target_logits"].detach().cpu().numpy())
            else:
                final_predictions, candidate_hit, fallback_used = collect_target_predictions(
                    outputs=outputs,
                    batch=batch,
                    target_mode=target_mode,
                )
                final_target_predictions_list.append(final_predictions)
                candidate_hit_list.append(candidate_hit)
                fallback_used_list.append(fallback_used)

    if batch_count == 0:
        raise RuntimeError("No batches were evaluated; dataset may be empty")

    if target_mode == "flat":
        metrics = compute_multitask_metrics(
            trigger_logits=np.concatenate(trigger_logits_list),
            trigger_targets=np.concatenate(trigger_targets_list).astype(np.int64),
            target_logits=np.concatenate(flat_target_logits_list),
            target_targets=np.concatenate(target_targets_list).astype(np.int64),
        )
    else:
        metrics = compute_candidate_aware_metrics(
            trigger_logits=np.concatenate(trigger_logits_list),
            trigger_targets=np.concatenate(trigger_targets_list).astype(np.int64),
            final_target_predictions=np.concatenate(final_target_predictions_list).astype(np.int64),
            target_targets=np.concatenate(target_targets_list).astype(np.int64),
            candidate_hit=np.concatenate(candidate_hit_list).astype(bool),
            fallback_used=np.concatenate(fallback_used_list).astype(bool),
        )

    losses = {
        "loss": total_loss / batch_count,
        "trigger_loss": total_trigger_loss / batch_count,
        "target_loss": total_target_loss / batch_count,
    }
    return metrics, losses


def main() -> None:
    args = build_argument_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision("high")
    set_seed(args.seed)

    if args.target_mode == "candidate" and args.candidate_top_k is None:
        raise SystemExit("--candidate-top-k is required when --target-mode candidate")

    metadata = load_json(args.dataset_dir / "metadata.json")
    seq_len = int(metadata["seq_len"])
    num_cells = len(metadata["cell_ids"])

    device = resolve_device(args.device)
    train_frame = pd.read_parquet(args.dataset_dir / "train_rows.parquet")
    val_frame = pd.read_parquet(args.dataset_dir / "val_rows.parquet")

    train_loader = create_loader(
        frame=train_frame,
        seq_len=seq_len,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        window_stride=args.window_stride,
        candidate_top_k=args.candidate_top_k if args.target_mode == "candidate" else None,
        num_cells=num_cells,
    )
    val_loader = create_loader(
        frame=val_frame,
        seq_len=seq_len,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        window_stride=args.window_stride,
        candidate_top_k=args.candidate_top_k if args.target_mode == "candidate" else None,
        num_cells=num_cells,
    )

    model = MultitaskLstmPredictor(
        ModelConfig(
            num_cells=num_cells,
            numeric_dim=len(metadata["numeric_feature_columns"]),
            target_mode=args.target_mode,
            candidate_feature_dim=len(CANDIDATE_FEATURE_BASENAMES),
            hidden_size=128,
            num_layers=1,
        )
    ).to(device)
    maybe_initialize_from_checkpoint(model, args.init_from_checkpoint)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    trigger_criterion = nn.BCEWithLogitsLoss()

    history: list[dict[str, float]] = []
    best_score = float("-inf")
    epochs_without_improvement = 0
    checkpoint_path = args.output_dir / "best_model.pt"
    serialized_training_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_trigger_loss = 0.0
        total_target_loss = 0.0
        batch_count = 0

        for batch_index, batch in enumerate(train_loader):
            if args.limit_train_batches is not None and batch_index >= args.limit_train_batches:
                break

            optimizer.zero_grad(set_to_none=True)
            numeric = batch["numeric"].to(device)
            serving_cell = batch["serving_cell"].to(device)
            trigger_target = batch["trigger"].to(device)
            target_target = batch["target"].to(device)

            if args.target_mode == "candidate":
                candidate_cell = batch["candidate_cell"].to(device)
                candidate_mask = batch["candidate_mask"].to(device)
                candidate_features = batch["candidate_features"].to(device)
                candidate_target_pos = batch["candidate_target_pos"].to(device)
                outputs = model(
                    numeric=numeric,
                    serving_cell=serving_cell,
                    candidate_cell=candidate_cell,
                    candidate_features=candidate_features,
                    candidate_mask=candidate_mask,
                )
            else:
                candidate_target_pos = None
                outputs = model(numeric=numeric, serving_cell=serving_cell)

            trigger_loss = trigger_criterion(outputs["trigger_logits"], trigger_target)
            target_loss, _ = compute_target_loss(
                outputs=outputs,
                trigger_target=trigger_target,
                target_target=target_target,
                target_mode=args.target_mode,
                candidate_target_pos=candidate_target_pos,
            )
            loss = trigger_loss + args.target_loss_weight * target_loss
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=args.gradient_clip)
            optimizer.step()

            total_loss += float(loss.item())
            total_trigger_loss += float(trigger_loss.item())
            total_target_loss += float(target_loss.item())
            batch_count += 1

        if batch_count == 0:
            raise RuntimeError("No training batches were processed; dataset may be empty")

        train_losses = {
            "loss": total_loss / batch_count,
            "trigger_loss": total_trigger_loss / batch_count,
            "target_loss": total_target_loss / batch_count,
        }
        val_metrics, val_losses = evaluate_loop(
            model=model,
            loader=val_loader,
            trigger_criterion=trigger_criterion,
            device=device,
            target_loss_weight=args.target_loss_weight,
            target_mode=args.target_mode,
            limit_batches=args.limit_val_batches,
        )

        val_target_metric = (
            val_metrics["target_macro_f1"]
            if args.target_mode == "flat"
            else val_metrics["candidate_macro_f1"]
        )
        val_score = val_metrics["trigger_f1"] + val_target_metric
        history_entry: dict[str, float] = {
            "epoch": float(epoch),
            "train_loss": train_losses["loss"],
            "train_trigger_loss": train_losses["trigger_loss"],
            "train_target_loss": train_losses["target_loss"],
            "val_loss": val_losses["loss"],
            "val_trigger_loss": val_losses["trigger_loss"],
            "val_target_loss": val_losses["target_loss"],
            **val_metrics,
            "val_score": float(val_score),
        }
        history.append(history_entry)
        if args.target_mode == "flat":
            print(
                f"epoch={epoch:02d} "
                f"train_loss={train_losses['loss']:.4f} "
                f"val_loss={val_losses['loss']:.4f} "
                f"trigger_f1={val_metrics['trigger_f1']:.4f} "
                f"target_macro_f1={val_metrics['target_macro_f1']:.4f} "
                f"target_top3={val_metrics['target_top3_accuracy']:.4f}",
                flush=True,
            )
        else:
            print(
                f"epoch={epoch:02d} "
                f"train_loss={train_losses['loss']:.4f} "
                f"val_loss={val_losses['loss']:.4f} "
                f"trigger_f1={val_metrics['trigger_f1']:.4f} "
                f"cand_acc={val_metrics['candidate_target_accuracy']:.4f} "
                f"cand_macro_f1={val_metrics['candidate_macro_f1']:.4f} "
                f"hit_rate={val_metrics['candidate_topk_hit_rate']:.4f} "
                f"fallback_rate={val_metrics['global_fallback_rate']:.4f}",
                flush=True,
            )

        if val_score > best_score:
            best_score = val_score
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": model.config.to_dict(),
                    "metadata": metadata,
                    "best_val_metrics": val_metrics,
                    "training_args": serialized_training_args,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping triggered at epoch {epoch}", flush=True)
                break

    save_json(args.output_dir / "history.json", {"epochs": history, "best_score": best_score})
    print(f"Best checkpoint: {checkpoint_path}", flush=True)


if __name__ == "__main__":
    main()
