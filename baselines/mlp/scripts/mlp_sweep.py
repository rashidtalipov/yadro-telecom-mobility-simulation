#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from mlp_baseline import (
    CandidateAwareMlp,
    CandidateRowDataset,
    MlpConfig,
    evaluate,
    load_split,
    make_eval_loader,
    make_train_loader,
    required_columns,
    resolve_device,
    save_json,
    set_seed,
    train_epoch,
)
from common import CANDIDATE_FEATURE_BASENAMES, NUMERIC_FEATURE_COLUMNS


@dataclass(frozen=True)
class VariantSpec:
    name: str
    cell_embedding_dim: int
    numeric_projection_dim: int
    hidden_size: int
    dropout: float
    lr: float
    weight_decay: float

    def to_config(self, num_cells: int) -> MlpConfig:
        return MlpConfig(
            num_cells=num_cells,
            numeric_dim=len(NUMERIC_FEATURE_COLUMNS),
            candidate_feature_dim=len(CANDIDATE_FEATURE_BASENAMES),
            cell_embedding_dim=self.cell_embedding_dim,
            numeric_projection_dim=self.numeric_projection_dim,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
        )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train several candidate-aware last-step MLP variants and select the best one.",
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-top-k", type=int, default=3, choices=(3, 5))
    parser.add_argument("--variant-set", choices=("smoke", "quick", "paper"), default="quick")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--train-samples-per-epoch", type=int, default=None)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-val-rows", type=int, default=None)
    parser.add_argument(
        "--max-test-rows",
        type=int,
        default=-1,
        help="Use -1 for full test, 0 to skip test, or a positive cap for a sampled test.",
    )
    parser.add_argument("--trigger-threshold", type=float, default=0.70)
    parser.add_argument("--target-loss-weight", type=float, default=1.5)
    parser.add_argument("--global-loss-weight", type=float, default=0.2)
    parser.add_argument(
        "--selection-metric",
        choices=(
            "validation_balanced_score",
            "validation_best_trigger_f1",
            "validation_candidate_target_accuracy",
            "test_balanced_score",
            "test_best_trigger_f1",
            "test_candidate_target_accuracy",
        ),
        default="validation_balanced_score",
    )
    parser.add_argument(
        "--evaluate-test-for-all",
        action="store_true",
        help=(
            "Evaluate every variant on the test set. Disabled by default so paper-grade runs "
            "select by validation and touch the test set only for the selected variant."
        ),
    )
    parser.add_argument("--force", action="store_true", help="Retrain variants even if outputs already exist.")
    return parser


def variants_for_set(name: str) -> list[VariantSpec]:
    if name == "smoke":
        return [
            VariantSpec("tiny_h32_d05_lr1e3", 8, 16, 32, 0.05, 1e-3, 1e-4),
            VariantSpec("small_h64_d10_lr1e3", 16, 32, 64, 0.10, 1e-3, 1e-4),
        ]
    if name == "quick":
        return [
            VariantSpec("small_h64_d10_lr1e3", 16, 32, 64, 0.10, 1e-3, 1e-4),
            VariantSpec("default_h128_d10_lr1e3", 16, 32, 128, 0.10, 1e-3, 1e-4),
            VariantSpec("default_h128_d20_lr1e3", 16, 32, 128, 0.20, 1e-3, 1e-4),
            VariantSpec("wide_h256_d15_lr5e4", 32, 64, 256, 0.15, 5e-4, 1e-4),
        ]
    return [
        VariantSpec("tiny_h32_d05_lr1e3", 8, 16, 32, 0.05, 1e-3, 1e-4),
        VariantSpec("small_h64_d10_lr1e3", 16, 32, 64, 0.10, 1e-3, 1e-4),
        VariantSpec("default_h128_d10_lr1e3", 16, 32, 128, 0.10, 1e-3, 1e-4),
        VariantSpec("default_h128_d20_lr1e3", 16, 32, 128, 0.20, 1e-3, 1e-4),
        VariantSpec("wide_h256_d15_lr5e4", 32, 64, 256, 0.15, 5e-4, 1e-4),
        VariantSpec("wide_h256_d30_lr5e4", 32, 64, 256, 0.30, 5e-4, 1e-4),
    ]


def defaults_for_variant_set(name: str) -> dict[str, int]:
    if name == "smoke":
        return {
            "epochs": 1,
            "patience": 1,
            "train_samples_per_epoch": 20000,
            "max_train_rows": 50000,
            "max_val_rows": 30000,
        }
    if name == "quick":
        return {
            "epochs": 3,
            "patience": 2,
            "train_samples_per_epoch": 600000,
            "max_train_rows": 800000,
            "max_val_rows": 250000,
        }
    return {
        "epochs": 6,
        "patience": 3,
        "train_samples_per_epoch": 1200000,
        "max_train_rows": 1500000,
        "max_val_rows": 500000,
    }


def ensure_value(user_value: int | None, default_value: int) -> int:
    return default_value if user_value is None else user_value


def load_sequence_eligible_test_frame(
    dataset_dir: Path,
    candidate_top_k: int,
    seq_len: int,
    max_test_rows: int,
    seed: int,
) -> pd.DataFrame | None:
    if max_test_rows == 0:
        return None

    columns = ["run_id", "imsi", "time", *required_columns(candidate_top_k)]
    columns = list(dict.fromkeys(columns))
    frame = pd.read_parquet(dataset_dir / "test_rows.parquet", columns=columns)
    frame = frame.sort_values(["run_id", "imsi", "time"], ignore_index=True)
    eligible = frame.groupby(["run_id", "imsi"], sort=False).cumcount() >= (seq_len - 1)
    frame = frame[eligible].reset_index(drop=True)
    if max_test_rows > 0 and len(frame) > max_test_rows:
        positives = frame[frame["trigger_label"] > 0.5]
        negatives = frame[frame["trigger_label"] <= 0.5]
        positive_budget = min(len(positives), max(1, int(round(max_test_rows * 0.35))))
        negative_budget = max(0, max_test_rows - positive_budget)
        frame = pd.concat(
            [
                positives.sample(n=positive_budget, random_state=seed),
                negatives.sample(n=min(negative_budget, len(negatives)), random_state=seed),
            ],
            ignore_index=True,
        ).sample(frac=1.0, random_state=seed)
    return frame.reset_index(drop=True)


def count_parameters(model: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def metric_scores(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    best_trigger = float(metrics.get("best_trigger_f1", 0.0))
    fixed_trigger = float(metrics.get("trigger_f1", 0.0))
    target = float(metrics.get("candidate_target_accuracy", 0.0))
    return {
        f"{prefix}_balanced_score": best_trigger + target,
        f"{prefix}_fixed_threshold_score": fixed_trigger + target,
        f"{prefix}_best_trigger_f1": best_trigger,
        f"{prefix}_candidate_target_accuracy": target,
    }


def train_one_variant(
    variant: VariantSpec,
    variant_index: int,
    args: argparse.Namespace,
    metadata: dict[str, Any],
    train_dataset: CandidateRowDataset,
    val_dataset: CandidateRowDataset,
    test_dataset: CandidateRowDataset | None,
    device: torch.device,
    epochs: int,
    patience: int,
    train_samples_per_epoch: int,
) -> dict[str, Any]:
    set_seed(int(args.seed) + variant_index)
    variant_dir = args.output_dir / variant.name
    variant_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = variant_dir / "best_model.pt"
    history_path = variant_dir / "history.json"
    test_metrics_path = variant_dir / "test_metrics.json"
    config_path = variant_dir / "variant_config.json"

    if (
        checkpoint_path.exists()
        and test_metrics_path.exists()
        and not args.force
        and bool(args.evaluate_test_for_all)
    ):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        validation_metrics = checkpoint.get("validation_metrics", {})
        test_metrics = json.loads(test_metrics_path.read_text(encoding="utf-8"))
        params = int(checkpoint.get("parameter_count", 0))
        if params <= 0:
            model_for_count = CandidateAwareMlp(MlpConfig(**checkpoint["model_config"]))
            params = count_parameters(model_for_count)
        row = {
            "variant": variant.name,
            "status": "reused",
            "parameter_count": params,
            "checkpoint_path": str(checkpoint_path),
            "history_path": str(history_path),
            "test_metrics_path": str(test_metrics_path),
            **asdict(variant),
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
            **metric_scores("validation", validation_metrics),
            **{f"test_{key}": value for key, value in test_metrics.items()},
            **metric_scores("test", test_metrics),
            "training_runtime_sec": 0.0,
            "test_eval_runtime_sec": 0.0,
            "total_runtime_sec": 0.0,
        }
        return row
    if checkpoint_path.exists() and not args.force:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        validation_metrics = checkpoint.get("validation_metrics", {})
        params = int(checkpoint.get("parameter_count", 0))
        if params <= 0:
            model_for_count = CandidateAwareMlp(MlpConfig(**checkpoint["model_config"]))
            params = count_parameters(model_for_count)
        row = {
            "variant": variant.name,
            "status": "reused_no_test",
            "parameter_count": params,
            "checkpoint_path": str(checkpoint_path),
            "history_path": str(history_path),
            "test_metrics_path": str(test_metrics_path),
            **asdict(variant),
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
            **metric_scores("validation", validation_metrics),
            "training_runtime_sec": 0.0,
            "test_eval_runtime_sec": 0.0,
            "total_runtime_sec": 0.0,
        }
        return row

    training_start = time.perf_counter()
    config = variant.to_config(num_cells=len(metadata["cell_ids"]))
    model = CandidateAwareMlp(config).to(device)
    params = count_parameters(model)

    train_loader = make_train_loader(
        train_dataset,
        batch_size=int(args.batch_size),
        train_samples_per_epoch=train_samples_per_epoch,
    )
    val_loader = make_eval_loader(val_dataset, batch_size=int(args.batch_size) * 2)
    test_loader = (
        make_eval_loader(test_dataset, batch_size=int(args.batch_size) * 4)
        if test_dataset is not None
        else None
    )

    positives = float((train_dataset.trigger > 0.5).sum())
    negatives = float(len(train_dataset) - positives)
    pos_weight = torch.tensor(negatives / max(1.0, positives), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=variant.lr, weight_decay=variant.weight_decay)

    best_score = -math.inf
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    config_payload = {
        "variant": asdict(variant),
        "model_config": config.to_dict(),
        "candidate_top_k": int(args.candidate_top_k),
        "seed": int(args.seed),
        "variant_seed": int(args.seed) + variant_index,
        "batch_size": int(args.batch_size),
        "epochs": epochs,
        "patience": patience,
        "train_samples_per_epoch": train_samples_per_epoch,
        "max_train_rows": int(args.max_train_rows),
        "max_val_rows": int(args.max_val_rows),
        "max_test_rows": int(args.max_test_rows),
        "trigger_threshold": float(args.trigger_threshold),
        "target_loss_weight": float(args.target_loss_weight),
        "global_loss_weight": float(args.global_loss_weight),
        "selection_metric": str(args.selection_metric),
        "evaluate_test_for_all": bool(args.evaluate_test_for_all),
        "parameter_count": params,
    }
    save_json(config_path, config_payload)

    print(
        f"[{variant_index + 1}] {variant.name}: params={params} "
        f"train={len(train_dataset)} val={len(val_dataset)} device={device}",
        flush=True,
    )
    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            target_loss_weight=float(args.target_loss_weight),
            global_loss_weight=float(args.global_loss_weight),
            pos_weight=pos_weight,
        )
        val_metrics = evaluate(model, val_loader, device=device, threshold=float(args.trigger_threshold))
        score = float(val_metrics["best_trigger_f1"]) + float(val_metrics["candidate_target_accuracy"])
        entry = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
            "validation_selection_score": score,
        }
        history.append(entry)
        print(
            f"  epoch={epoch} train_loss={train_metrics['loss']:.4f} "
            f"val_best_f1={val_metrics['best_trigger_f1']:.4f}@{val_metrics['best_trigger_threshold']:.2f} "
            f"val_cand_acc={val_metrics['candidate_target_accuracy']:.4f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            stale_epochs = 0
            torch.save(
                {
                    "model_type": "candidate_aware_last_step_mlp",
                    "model_config": config.to_dict(),
                    "training_args": {
                        **config_payload,
                        "dataset_dir": str(args.dataset_dir),
                        "output_dir": str(variant_dir),
                        "trigger_threshold": float(args.trigger_threshold),
                        "target_loss_weight": float(args.target_loss_weight),
                        "global_loss_weight": float(args.global_loss_weight),
                    },
                    "metadata": metadata,
                    "model_state": model.state_dict(),
                    "validation_metrics": val_metrics,
                    "parameter_count": params,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    save_json(history_path, {"epochs": history, "best_score": best_score})
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    training_runtime_sec = time.perf_counter() - training_start
    test_eval_runtime_sec = 0.0
    if test_loader is not None:
        test_eval_start = time.perf_counter()
        test_metrics = evaluate(model, test_loader, device=device, threshold=float(args.trigger_threshold))
        test_eval_runtime_sec = time.perf_counter() - test_eval_start
    else:
        test_metrics = {}
    if test_metrics:
        save_json(test_metrics_path, test_metrics)

    validation_metrics = checkpoint.get("validation_metrics", {})
    row = {
        "variant": variant.name,
        "status": "trained",
        "parameter_count": params,
        "checkpoint_path": str(checkpoint_path),
        "history_path": str(history_path),
        "test_metrics_path": str(test_metrics_path),
        **asdict(variant),
        **{f"validation_{key}": value for key, value in validation_metrics.items()},
        **metric_scores("validation", validation_metrics),
        "training_runtime_sec": training_runtime_sec,
        "test_eval_runtime_sec": test_eval_runtime_sec,
        "total_runtime_sec": training_runtime_sec + test_eval_runtime_sec,
    }
    if test_metrics:
        row.update({f"test_{key}": value for key, value in test_metrics.items()})
        row.update(metric_scores("test", test_metrics))
    return row


def evaluate_selected_variant_on_test(
    row: dict[str, Any],
    test_dataset: CandidateRowDataset,
    device: torch.device,
    batch_size: int,
    threshold: float,
) -> dict[str, Any]:
    test_eval_start = time.perf_counter()
    checkpoint_path = Path(str(row["checkpoint_path"]))
    test_metrics_path = Path(str(row["test_metrics_path"]))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = CandidateAwareMlp(MlpConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    test_loader = make_eval_loader(test_dataset, batch_size=batch_size * 4)
    test_metrics = evaluate(model, test_loader, device=device, threshold=threshold)
    test_eval_runtime_sec = time.perf_counter() - test_eval_start
    save_json(test_metrics_path, test_metrics)
    row.update({f"test_{key}": value for key, value in test_metrics.items()})
    row.update(metric_scores("test", test_metrics))
    row["test_eval_runtime_sec"] = test_eval_runtime_sec
    row["total_runtime_sec"] = float(row.get("training_runtime_sec", 0.0)) + test_eval_runtime_sec
    return row


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_empty_"
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        values = [str(row[column]) for column in frame.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    output_dir: Path,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    train_count: int,
    val_count: int,
    test_count: int,
    positive_test_count: int,
) -> Path:
    summary = pd.DataFrame(rows)
    if summary.empty:
        raise RuntimeError("No sweep rows were produced")
    summary = summary.sort_values(str(args.selection_metric), ascending=False, ignore_index=True)
    best = summary.iloc[0].to_dict()

    report_columns = [
        "variant",
        "status",
        "parameter_count",
        "hidden_size",
        "cell_embedding_dim",
        "numeric_projection_dim",
        "dropout",
        "lr",
        "weight_decay",
        "training_runtime_sec",
        "test_eval_runtime_sec",
        "validation_best_trigger_f1",
        "validation_best_trigger_threshold",
        "validation_candidate_target_accuracy",
        "validation_balanced_score",
        "test_best_trigger_f1",
        "test_best_trigger_threshold",
        "test_trigger_f1",
        "test_trigger_precision",
        "test_trigger_recall",
        "test_candidate_target_accuracy",
        "test_target_accuracy_positive",
        "test_balanced_score",
        "checkpoint_path",
    ]
    report_rows = summary.reindex(columns=report_columns).copy()
    for column in report_rows.columns:
        if report_rows[column].dtype.kind in {"f", "c"}:
            report_rows[column] = report_rows[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.4f}"
            )

    report_path = output_dir / "MLP_SWEEP_REPORT.md"
    lines = [
        "# MLP Architecture Sweep Report",
        "",
        "This report was generated by `results_night/oran_e2_lstm/mlp_sweep.py`.",
        "",
        "## Setup",
        "",
        f"- Dataset: `{args.dataset_dir}`",
        f"- Candidate K: `{args.candidate_top_k}`",
        f"- Variant set: `{args.variant_set}`",
        f"- Seed: `{args.seed}`",
        f"- Device requested/resolved: `{args.device}` / `{getattr(args, 'resolved_device', args.device)}`",
        f"- Epoch limit / patience: `{args.epochs}` / `{args.patience}`",
        f"- Batch size: `{args.batch_size}`",
        f"- Samples per epoch: `{args.train_samples_per_epoch}`",
        f"- Train rows loaded: `{train_count}`",
        f"- Validation rows loaded: `{val_count}`",
        f"- Test rows evaluated: `{test_count}`",
        f"- Test positives: `{positive_test_count}`",
        f"- Fixed trigger threshold used for fixed-threshold metrics: `{args.trigger_threshold}`",
        "- Online ns-3 thresholds: use the validation-selected trigger threshold from the Winner section,",
        "  with target confidence threshold `0.70` unless otherwise stated.",
        f"- Target/global loss weights: `{args.target_loss_weight}` / `{args.global_loss_weight}`",
        f"- Selection metric requested: `{args.selection_metric}`",
        f"- Test evaluated for all variants: `{bool(args.evaluate_test_for_all)}`",
        "",
        "Paper-grade protocol: the final architecture should be selected by validation metrics.",
        "The test set is then used once for the selected architecture unless `--evaluate-test-for-all`",
        "is explicitly enabled for exploratory screening.",
        "",
        "## Winner",
        "",
        f"- Best variant by `{args.selection_metric}`: `{best['variant']}`",
        f"- Best checkpoint: `{best['checkpoint_path']}`",
        f"- Parameters: `{int(best['parameter_count'])}`",
        f"- Validation best trigger F1: `{float(best.get('validation_best_trigger_f1', 0.0)):.4f}`",
        f"- Validation best trigger threshold: `{float(best.get('validation_best_trigger_threshold', 0.0)):.2f}`",
        f"- Validation candidate target accuracy: `{float(best.get('validation_candidate_target_accuracy', 0.0)):.4f}`",
        f"- Validation balanced score: `{float(best.get('validation_balanced_score', 0.0)):.4f}`",
        f"- Recommended online trigger threshold: `{float(best.get('validation_best_trigger_threshold', args.trigger_threshold)):.2f}`",
        "- Recommended online target confidence threshold: `0.70`",
        f"- Test best trigger F1: `{float(best.get('test_best_trigger_f1', 0.0)):.4f}`",
        f"- Test best trigger threshold: `{float(best.get('test_best_trigger_threshold', 0.0)):.2f}`",
        f"- Test candidate target accuracy: `{float(best.get('test_candidate_target_accuracy', 0.0)):.4f}`",
        f"- Test balanced score: `{float(best.get('test_balanced_score', 0.0)):.4f}`",
        "",
        "## Results",
        "",
        dataframe_to_markdown(report_rows),
        "",
        "## Interpretation",
        "",
        "The selected MLP should be treated as a lightweight non-recurrent baseline. It tests whether",
        "the candidate-aware formulation can work without an LSTM history. If the best MLP approaches",
        "the LSTM trigger and target metrics, the paper should emphasize that candidate-aware target",
        "restriction is a major contributor. If it underperforms the LSTM on trigger calibration or",
        "online stability, the LSTM remains justified as the main online controller.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    summary.to_csv(output_dir / "sweep_summary.csv", index=False)
    (output_dir / "sweep_summary.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report_path


def main() -> int:
    args = build_argument_parser().parse_args()
    defaults = defaults_for_variant_set(args.variant_set)
    args.epochs = ensure_value(args.epochs, defaults["epochs"])
    args.patience = ensure_value(args.patience, defaults["patience"])
    args.train_samples_per_epoch = ensure_value(
        args.train_samples_per_epoch,
        defaults["train_samples_per_epoch"],
    )
    args.max_train_rows = ensure_value(args.max_train_rows, defaults["max_train_rows"])
    args.max_val_rows = ensure_value(args.max_val_rows, defaults["max_val_rows"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(args.seed))
    torch.set_float32_matmul_precision("high")
    device = resolve_device(str(args.device))
    args.resolved_device = str(device)

    metadata = json.loads((args.dataset_dir / "metadata.json").read_text(encoding="utf-8"))
    num_cells = len(metadata["cell_ids"])
    seq_len = int(metadata.get("seq_len", 15))

    if str(args.selection_metric).startswith("test_") and int(args.max_test_rows) == 0:
        raise ValueError("Test-based selection requires a non-empty test set.")

    print("Loading train/validation data...", flush=True)
    train_frame = load_split(
        args.dataset_dir,
        "train",
        int(args.candidate_top_k),
        int(args.max_train_rows),
        int(args.seed),
    )
    val_frame = load_split(
        args.dataset_dir,
        "val",
        int(args.candidate_top_k),
        int(args.max_val_rows),
        int(args.seed) + 1,
    )

    train_dataset = CandidateRowDataset(train_frame, candidate_top_k=int(args.candidate_top_k), num_cells=num_cells)
    val_dataset = CandidateRowDataset(val_frame, candidate_top_k=int(args.candidate_top_k), num_cells=num_cells)
    test_dataset: CandidateRowDataset | None = None
    test_count = 0
    positive_test_count = 0
    if bool(args.evaluate_test_for_all):
        print("Loading test data for exploratory all-variant evaluation...", flush=True)
        test_frame = load_sequence_eligible_test_frame(
            args.dataset_dir,
            int(args.candidate_top_k),
            seq_len,
            int(args.max_test_rows),
            int(args.seed) + 2,
        )
        test_dataset = (
            CandidateRowDataset(test_frame, candidate_top_k=int(args.candidate_top_k), num_cells=num_cells)
            if test_frame is not None
            else None
        )
        test_count = len(test_dataset) if test_dataset is not None else 0
        positive_test_count = int(test_dataset.trigger.sum()) if test_dataset is not None else 0

    rows: list[dict[str, Any]] = []
    variants = variants_for_set(str(args.variant_set))
    if str(args.selection_metric).startswith("test_") and not bool(args.evaluate_test_for_all):
        raise ValueError("Test-based selection requires --evaluate-test-for-all.")
    for index, variant in enumerate(variants):
        row = train_one_variant(
            variant=variant,
            variant_index=index,
            args=args,
            metadata=metadata,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset if bool(args.evaluate_test_for_all) else None,
            device=device,
            epochs=int(args.epochs),
            patience=int(args.patience),
            train_samples_per_epoch=int(args.train_samples_per_epoch),
        )
        rows.append(row)

    if int(args.max_test_rows) != 0 and not bool(args.evaluate_test_for_all):
        selected_index = max(
            range(len(rows)),
            key=lambda index: float(rows[index].get(str(args.selection_metric), -math.inf)),
        )
        print(
            "Loading test data after validation selection...",
            flush=True,
        )
        test_frame = load_sequence_eligible_test_frame(
            args.dataset_dir,
            int(args.candidate_top_k),
            seq_len,
            int(args.max_test_rows),
            int(args.seed) + 2,
        )
        test_dataset = (
            CandidateRowDataset(test_frame, candidate_top_k=int(args.candidate_top_k), num_cells=num_cells)
            if test_frame is not None
            else None
        )
        test_count = len(test_dataset) if test_dataset is not None else 0
        positive_test_count = int(test_dataset.trigger.sum()) if test_dataset is not None else 0
        if test_dataset is not None:
            print(
                f"Evaluating selected variant on test once: {rows[selected_index]['variant']}",
                flush=True,
            )
            rows[selected_index] = evaluate_selected_variant_on_test(
                rows[selected_index],
                test_dataset=test_dataset,
                device=device,
                batch_size=int(args.batch_size),
                threshold=float(args.trigger_threshold),
            )

    report_path = write_report(
        output_dir=args.output_dir,
        args=args,
        rows=rows,
        train_count=len(train_dataset),
        val_count=len(val_dataset),
        test_count=test_count,
        positive_test_count=positive_test_count,
    )
    print(f"Report: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
