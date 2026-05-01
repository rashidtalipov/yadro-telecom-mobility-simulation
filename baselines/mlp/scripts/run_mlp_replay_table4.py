#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
NS3_ORAN_DIR = REPO_ROOT / "ns-allinone-3.46.1" / "ns-3.46.1" / "results_night" / "oran_e2_lstm"
if str(NS3_ORAN_DIR) not in sys.path:
    sys.path.insert(0, str(NS3_ORAN_DIR))

from common import save_json  # noqa: E402
from mlp_baseline import CandidateAwareMlp, CandidateRowDataset, MlpConfig  # noqa: E402
from replay import (  # noqa: E402
    MODEL_ORDER,
    ConservativePolicyConfig,
    attach_candidate_gain_metrics,
    build_actual_events,
    build_coverage_frame,
    build_replay_segments,
    evaluate_a3_policy,
    evaluate_candidate_policy,
    load_split_frame,
    make_policy_result,
    parse_float_grid,
    parse_int_grid,
    parse_str_grid,
    plot_core_metrics,
    plot_lead_time_bins,
    prepare_policy_groups,
    render_report,
    run_policy_sweep,
)


MLP_MODEL_ORDER = [*MODEL_ORDER, "mlp_current_k3", "mlp_conservative_k3", "mlp_hybrid_a3_k3"]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline replay for the validation-selected candidate-aware MLP baseline.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=NS3_ORAN_DIR / "processed_candidate_e2_100ms_full",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=REPO_ROOT / "ns-allinone-3.46.1" / "ns-3.46.1" / "results_night_teacher_100ms",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=REPO_ROOT / "MLP" / "checkpoints" / "wide_h256_d15_lr5e4_best_model.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=NS3_ORAN_DIR
        / "replay_outputs"
        / f"mlp_table4_replay_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--val-split", type=str, default="val", choices=("train", "val", "test"))
    parser.add_argument("--test-split", type=str, default="test", choices=("train", "val", "test"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--trigger-threshold-grid", type=str, default="0.50,0.55,0.60,0.65,0.70,0.85")
    parser.add_argument("--target-conf-grid", type=str, default="0.00,0.45,0.55,0.65,0.70")
    parser.add_argument("--score-margin-grid", type=str, default="0.00,0.05,0.10")
    parser.add_argument("--gain-grid-db", type=str, default="-999,0.0,1.0")
    parser.add_argument("--cooldown-grid-s", type=str, default="0.0,0.5,1.0")
    parser.add_argument("--anti-ping-pong-grid-s", type=str, default="0.0,5.0")
    parser.add_argument("--confirmation-grid", type=str, default="1,2")
    parser.add_argument("--topk-stage1", type=int, default=10)
    parser.add_argument("--a3-gate-mode-grid", type=str, default="assist,strict")
    parser.add_argument("--ping-pong-window-s", type=float, default=5.0)
    parser.add_argument("--stability-window-s", type=float, default=5.0)
    parser.add_argument("--early-lead-threshold-s", type=float, default=0.2)
    parser.add_argument("--a3-hysteresis-db", type=float, default=3.0)
    parser.add_argument("--a3-time-to-trigger-ms", type=float, default=256.0)
    parser.add_argument("--measurement-interval-ms", type=float, default=100.0)
    return parser


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_mlp_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[dict[str, Any], CandidateAwareMlp, int]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("model_type") != "candidate_aware_last_step_mlp":
        raise ValueError(f"{checkpoint_path} is not a candidate-aware MLP checkpoint")
    model = CandidateAwareMlp(MlpConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    candidate_top_k = int(checkpoint["training_args"]["candidate_top_k"])
    return checkpoint, model, candidate_top_k


def sequence_eligible_frame(frame: pd.DataFrame, seq_len: int) -> pd.DataFrame:
    eligible = frame.groupby(["run_id", "imsi"], sort=False).cumcount() >= (seq_len - 1)
    return frame[eligible].reset_index(drop=True)


def infer_mlp_predictions(
    frame: pd.DataFrame,
    checkpoint_path: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    seq_len: int,
) -> tuple[pd.DataFrame, dict[str, Any], int]:
    checkpoint, model, candidate_top_k = load_mlp_checkpoint(checkpoint_path, device)
    metadata = checkpoint["metadata"]
    scoring_frame = sequence_eligible_frame(frame, seq_len=seq_len)
    dataset = CandidateRowDataset(
        scoring_frame,
        candidate_top_k=candidate_top_k,
        num_cells=len(metadata["cell_ids"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    index_to_cell = {int(index): int(cell_id) for index, cell_id in metadata["index_to_cell"].items()}

    records: list[pd.DataFrame] = []
    offset = 0
    with torch.no_grad():
        for batch in loader:
            batch_size_actual = int(batch["numeric"].shape[0])
            row_slice = scoring_frame.iloc[offset : offset + batch_size_actual]
            offset += batch_size_actual

            numeric = batch["numeric"].to(device)
            serving_cell = batch["serving_cell"].to(device)
            candidate_cell = batch["candidate_cell"].to(device)
            candidate_mask = batch["candidate_mask"].to(device)
            candidate_features = batch["candidate_features"].to(device)

            outputs = model(
                numeric=numeric,
                serving_cell=serving_cell,
                candidate_cell=candidate_cell,
                candidate_features=candidate_features,
                candidate_mask=candidate_mask,
            )

            trigger_prob = torch.sigmoid(outputs["trigger_logits"])
            candidate_probs = torch.softmax(outputs["candidate_logits"], dim=1)
            global_probs = torch.softmax(outputs["global_target_logits"], dim=1)
            candidate_choice = torch.argmax(outputs["candidate_logits"], dim=1)
            global_choice = torch.argmax(outputs["global_target_logits"], dim=1)
            row_index = torch.arange(candidate_choice.shape[0], device=device)
            chosen_candidate_index = candidate_cell[row_index, candidate_choice]
            has_candidate = candidate_mask.any(dim=1)
            target_index = torch.where(has_candidate, chosen_candidate_index, global_choice)
            fallback_used = (~has_candidate)

            candidate_top_probs, _ = torch.topk(candidate_probs, k=min(2, candidate_probs.shape[1]), dim=1)
            global_top_probs, _ = torch.topk(global_probs, k=min(2, global_probs.shape[1]), dim=1)
            candidate_conf = candidate_top_probs[:, 0]
            global_conf = global_top_probs[:, 0]
            candidate_margin = candidate_top_probs[:, 0] - (
                candidate_top_probs[:, 1] if candidate_top_probs.shape[1] > 1 else 0.0
            )
            global_margin = global_top_probs[:, 0] - (
                global_top_probs[:, 1] if global_top_probs.shape[1] > 1 else 0.0
            )

            target_conf = torch.where(has_candidate, candidate_conf, global_conf).detach().cpu().numpy().astype(np.float32)
            score_margin = torch.where(has_candidate, candidate_margin, global_margin).detach().cpu().numpy().astype(np.float32)
            target_index_np = target_index.detach().cpu().numpy().astype(np.int64)
            target_cell_ids = np.asarray(
                [index_to_cell.get(int(idx), -1) for idx in target_index_np],
                dtype=np.int32,
            )

            records.append(
                pd.DataFrame(
                    {
                        "run_id": row_slice["run_id"].astype(str).to_numpy(copy=True),
                        "imsi": row_slice["imsi"].to_numpy(dtype=np.int64, copy=True),
                        "time": row_slice["time"].to_numpy(dtype=np.float32, copy=True),
                        "trigger_prob": trigger_prob.detach().cpu().numpy().astype(np.float32),
                        "target_confidence": target_conf,
                        "score_margin": score_margin,
                        "predicted_target_index": target_index_np,
                        "predicted_target_cell_id": target_cell_ids,
                        "chosen_candidate_rank": (candidate_choice.detach().cpu().numpy().astype(np.int16) + 1),
                        "fallback_used": fallback_used.detach().cpu().numpy().astype(np.int8),
                    }
                )
            )

    if records:
        predictions = pd.concat(records, ignore_index=True)
    else:
        predictions = pd.DataFrame(
            columns=[
                "run_id",
                "imsi",
                "time",
                "trigger_prob",
                "target_confidence",
                "score_margin",
                "predicted_target_index",
                "predicted_target_cell_id",
                "chosen_candidate_rank",
                "fallback_used",
            ]
        )
    return predictions.sort_values(["run_id", "imsi", "time"], ignore_index=True), checkpoint, candidate_top_k


def format_mlp_summary_table(summaries: pd.DataFrame) -> str:
    columns = [
        "model_name",
        "handover_count",
        "unnecessary_handover_count",
        "missed_useful_handover_count",
        "ping_pong_rate",
        "end_to_end_decision_success_rate",
        "early_prediction_rate",
        "mean_dwell_time_s",
    ]
    table = summaries[columns].copy()
    for column in columns:
        if column == "model_name":
            continue
        table[column] = table[column].map(lambda value: f"{float(value):.4f}")
    header = "| " + " | ".join(table.columns.tolist()) + " |"
    separator = "| " + " | ".join(["---"] * len(table.columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in table.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


def render_mlp_report(
    output_dir: Path,
    checkpoint_path: Path,
    candidate_top_k: int,
    seq_len: int,
    validation_current_summary: dict[str, Any],
    validation_conservative_summary: dict[str, Any],
    best_conservative_policy: ConservativePolicyConfig,
    best_hybrid_policy: ConservativePolicyConfig,
    test_summaries: pd.DataFrame,
    conservative_sweep_df: pd.DataFrame,
    hybrid_sweep_df: pd.DataFrame,
) -> str:
    mlp_current = test_summaries[test_summaries["model_name"] == "mlp_current_k3"].iloc[0]
    mlp_conservative = test_summaries[test_summaries["model_name"] == "mlp_conservative_k3"].iloc[0]
    mlp_hybrid = test_summaries[test_summaries["model_name"] == "mlp_hybrid_a3_k3"].iloc[0]
    return "\n".join(
        [
            "# MLP Offline Replay Report for Table IV",
            "",
            "## Setup",
            "",
            f"- Checkpoint: `{checkpoint_path}`",
            f"- Candidate K: `{candidate_top_k}`",
            f"- Sequence eligibility warm-up: `{seq_len}` rows per UE",
            f"- Current MLP validation success: `{validation_current_summary['end_to_end_decision_success_rate']:.4f}`",
            f"- Conservative MLP validation success: `{validation_conservative_summary['end_to_end_decision_success_rate']:.4f}`",
            "",
            "## Best Conservative MLP Policy",
            "",
            "```json",
            pd.Series(best_conservative_policy.to_dict()).to_json(indent=2),
            "```",
            "",
            "## Best Hybrid MLP Policy",
            "",
            "```json",
            pd.Series(best_hybrid_policy.to_dict()).to_json(indent=2),
            "```",
            "",
            "## Test Comparison",
            "",
            format_mlp_summary_table(test_summaries),
            "",
            "## Sweep Notes",
            "",
            f"- Conservative MLP sweep evaluated `{len(conservative_sweep_df)}` configs on validation replay.",
            f"- Hybrid MLP sweep evaluated `{len(hybrid_sweep_df)}` configs on validation replay.",
            "",
            "## Candidate Table IV Row",
            "",
            (
                f"- Recommended MLP replay row: `{mlp_conservative['model_name']}` with "
                f"HO `{int(mlp_conservative['handover_count'])}`, ping-pong rate "
                f"`{float(mlp_conservative['ping_pong_rate']):.4f}`, useful-decision success "
                f"`{float(mlp_conservative['end_to_end_decision_success_rate']):.4f}`, early rate "
                f"`{float(mlp_conservative['early_prediction_rate']):.4f}`, and mean dwell "
                f"`{float(mlp_conservative['mean_dwell_time_s']):.4f}`."
            ),
            "",
            "## Notes",
            "",
            (
                "This replay result is comparable to the existing Table IV replay rows because it uses "
                "the same logged trajectories, validation-first policy selection, and test-only final replay."
            ),
            (
                f"Raw MLP current policy: HO `{int(mlp_current['handover_count'])}`, ping-pong "
                f"`{float(mlp_current['ping_pong_rate']):.4f}`. Hybrid MLP+A3 policy: HO "
                f"`{int(mlp_hybrid['handover_count'])}`, ping-pong "
                f"`{float(mlp_hybrid['ping_pong_rate']):.4f}`."
            ),
            f"- Plots: `{(output_dir / 'replay_core_metrics.png').name}` and `{(output_dir / 'replay_lead_time.png').name}`",
            "",
        ]
    )


def main() -> None:
    args = build_argument_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    metadata, val_frame = load_split_frame(args.dataset_dir, args.val_split)
    _, test_frame = load_split_frame(args.dataset_dir, args.test_split)
    seq_len = int(metadata["seq_len"])

    print("Inferring MLP predictions on validation split...", flush=True)
    val_predictions, checkpoint, candidate_top_k = infer_mlp_predictions(
        frame=val_frame,
        checkpoint_path=args.checkpoint_path,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seq_len=seq_len,
    )
    print("Inferring MLP predictions on test split...", flush=True)
    test_predictions, _, _ = infer_mlp_predictions(
        frame=test_frame,
        checkpoint_path=args.checkpoint_path,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seq_len=seq_len,
    )

    val_decision_rows = attach_candidate_gain_metrics(val_frame, val_predictions, metadata)
    test_decision_rows = attach_candidate_gain_metrics(test_frame, test_predictions, metadata)

    val_coverage = build_coverage_frame(val_predictions)
    test_coverage = build_coverage_frame(test_predictions)
    dataset_root = args.dataset_root
    if not dataset_root.exists():
        fallback_root = Path(str(metadata["dataset_root"]).replace("<WORKSPACE>", str(REPO_ROOT)))
        dataset_root = fallback_root
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    val_events = build_actual_events(dataset_root=dataset_root, run_ids=list(metadata["splits"][args.val_split]))
    test_events = build_actual_events(dataset_root=dataset_root, run_ids=list(metadata["splits"][args.test_split]))

    val_segments, val_skipped = build_replay_segments(val_frame, val_events, val_coverage)
    test_segments, test_skipped = build_replay_segments(test_frame, test_events, test_coverage)
    prepared_val_groups = prepare_policy_groups(
        val_decision_rows,
        val_segments,
        a3_hysteresis_db=args.a3_hysteresis_db,
        a3_time_to_trigger_ms=args.a3_time_to_trigger_ms,
        measurement_interval_ms=args.measurement_interval_ms,
    )
    prepared_test_groups = prepare_policy_groups(
        test_decision_rows,
        test_segments,
        a3_hysteresis_db=args.a3_hysteresis_db,
        a3_time_to_trigger_ms=args.a3_time_to_trigger_ms,
        measurement_interval_ms=args.measurement_interval_ms,
    )

    current_policy = ConservativePolicyConfig(name="mlp_current_k3", a3_gate_mode="off")
    current_val = make_policy_result(
        model_name="mlp_current_k3",
        config=current_policy.to_dict(),
        segment_results=evaluate_candidate_policy(
            prepared_groups=prepared_val_groups,
            policy=current_policy,
            model_name="mlp_current_k3",
            stability_window_s=args.stability_window_s,
            early_lead_threshold_s=args.early_lead_threshold_s,
        ),
        skipped_positive_events=val_skipped,
    )

    print("Running conservative MLP validation policy sweep...", flush=True)
    conservative_sweep_df, best_conservative_policy, best_conservative_policy_row = run_policy_sweep(
        prepared_groups_val=prepared_val_groups,
        skipped_positive_events=val_skipped,
        baseline_summary=current_val.summary,
        early_lead_threshold_s=args.early_lead_threshold_s,
        stability_window_s=args.stability_window_s,
        topk_stage1=args.topk_stage1,
        model_name="mlp_conservative_k3",
        gate_modes=["off"],
        sweep_label="mlp_conservative",
        trigger_grid=parse_float_grid(args.trigger_threshold_grid),
        conf_grid=parse_float_grid(args.target_conf_grid),
        margin_grid=parse_float_grid(args.score_margin_grid),
        gain_grid=parse_float_grid(args.gain_grid_db),
        cooldown_grid=parse_float_grid(args.cooldown_grid_s),
        anti_ping_pong_grid=parse_float_grid(args.anti_ping_pong_grid_s),
        confirmation_grid=parse_int_grid(args.confirmation_grid),
    )
    conservative_val = make_policy_result(
        model_name="mlp_conservative_k3",
        config=best_conservative_policy.to_dict(),
        segment_results=evaluate_candidate_policy(
            prepared_groups=prepared_val_groups,
            policy=best_conservative_policy,
            model_name="mlp_conservative_k3",
            stability_window_s=args.stability_window_s,
            early_lead_threshold_s=args.early_lead_threshold_s,
        ),
        skipped_positive_events=val_skipped,
    )

    hybrid_modes = [mode for mode in parse_str_grid(args.a3_gate_mode_grid) if mode in {"assist", "strict"}]
    if not hybrid_modes:
        raise ValueError("--a3-gate-mode-grid must include assist or strict")

    print("Running hybrid MLP+A3 validation policy sweep...", flush=True)
    hybrid_sweep_df, best_hybrid_policy, best_hybrid_policy_row = run_policy_sweep(
        prepared_groups_val=prepared_val_groups,
        skipped_positive_events=val_skipped,
        baseline_summary=conservative_val.summary,
        early_lead_threshold_s=args.early_lead_threshold_s,
        stability_window_s=args.stability_window_s,
        topk_stage1=args.topk_stage1,
        model_name="mlp_hybrid_a3_k3",
        gate_modes=hybrid_modes,
        sweep_label="mlp_hybrid",
        trigger_grid=parse_float_grid(args.trigger_threshold_grid),
        conf_grid=parse_float_grid(args.target_conf_grid),
        margin_grid=parse_float_grid(args.score_margin_grid),
        gain_grid=parse_float_grid(args.gain_grid_db),
        cooldown_grid=parse_float_grid(args.cooldown_grid_s),
        anti_ping_pong_grid=parse_float_grid(args.anti_ping_pong_grid_s),
        confirmation_grid=parse_int_grid(args.confirmation_grid),
    )

    a3_test = make_policy_result(
        model_name="a3_original",
        config={"policy": "logged_a3"},
        segment_results=evaluate_a3_policy(
            segments=test_segments,
            model_name="a3_original",
            stability_window_s=args.stability_window_s,
            early_lead_threshold_s=args.early_lead_threshold_s,
        ),
        skipped_positive_events=test_skipped,
    )
    current_test = make_policy_result(
        model_name="mlp_current_k3",
        config=current_policy.to_dict(),
        segment_results=evaluate_candidate_policy(
            prepared_groups=prepared_test_groups,
            policy=current_policy,
            model_name="mlp_current_k3",
            stability_window_s=args.stability_window_s,
            early_lead_threshold_s=args.early_lead_threshold_s,
        ),
        skipped_positive_events=test_skipped,
    )
    conservative_test = make_policy_result(
        model_name="mlp_conservative_k3",
        config=best_conservative_policy.to_dict(),
        segment_results=evaluate_candidate_policy(
            prepared_groups=prepared_test_groups,
            policy=best_conservative_policy,
            model_name="mlp_conservative_k3",
            stability_window_s=args.stability_window_s,
            early_lead_threshold_s=args.early_lead_threshold_s,
        ),
        skipped_positive_events=test_skipped,
    )
    hybrid_test = make_policy_result(
        model_name="mlp_hybrid_a3_k3",
        config=best_hybrid_policy.to_dict(),
        segment_results=evaluate_candidate_policy(
            prepared_groups=prepared_test_groups,
            policy=best_hybrid_policy,
            model_name="mlp_hybrid_a3_k3",
            stability_window_s=args.stability_window_s,
            early_lead_threshold_s=args.early_lead_threshold_s,
        ),
        skipped_positive_events=test_skipped,
    )

    test_summary_frame = pd.DataFrame([a3_test.summary, current_test.summary, conservative_test.summary, hybrid_test.summary])
    test_summary_frame["model_name"] = pd.Categorical(
        test_summary_frame["model_name"],
        categories=["a3_original", "mlp_current_k3", "mlp_conservative_k3", "mlp_hybrid_a3_k3"],
        ordered=True,
    )
    test_summary_frame = test_summary_frame.sort_values("model_name", ignore_index=True)
    test_summary_frame["model_name"] = test_summary_frame["model_name"].astype(str)

    per_ue_frame = pd.concat([a3_test.per_ue, current_test.per_ue, conservative_test.per_ue, hybrid_test.per_ue], ignore_index=True)
    segment_frame = pd.concat(
        [a3_test.segment_results, current_test.segment_results, conservative_test.segment_results, hybrid_test.segment_results],
        ignore_index=True,
    )

    plot_frame = test_summary_frame.copy()
    plot_core_metrics(plot_frame, args.output_dir / "replay_core_metrics.png")
    plot_lead_time_bins(plot_frame, args.output_dir / "replay_lead_time.png")

    conservative_sweep_df.to_csv(args.output_dir / "mlp_policy_sweep_results.csv", index=False)
    hybrid_sweep_df.to_csv(args.output_dir / "mlp_hybrid_policy_sweep_results.csv", index=False)
    per_ue_frame.to_csv(args.output_dir / "mlp_per_ue_replay.csv", index=False)
    segment_frame.to_csv(args.output_dir / "mlp_segment_replay.csv", index=False)

    best_policy_payload = {
        "best_policy": best_conservative_policy.to_dict(),
        "best_policy_validation_row": best_conservative_policy_row.to_dict(),
        "baseline_validation_summary": current_val.summary,
        "search_space": {
            "trigger_threshold_grid": parse_float_grid(args.trigger_threshold_grid),
            "target_conf_grid": parse_float_grid(args.target_conf_grid),
            "score_margin_grid": parse_float_grid(args.score_margin_grid),
            "gain_grid_db": parse_float_grid(args.gain_grid_db),
            "cooldown_grid_s": parse_float_grid(args.cooldown_grid_s),
            "anti_ping_pong_grid_s": parse_float_grid(args.anti_ping_pong_grid_s),
            "confirmation_grid": parse_int_grid(args.confirmation_grid),
            "topk_stage1": args.topk_stage1,
        },
    }
    best_hybrid_payload = {
        "best_hybrid_policy": best_hybrid_policy.to_dict(),
        "best_hybrid_policy_validation_row": best_hybrid_policy_row.to_dict(),
        "baseline_validation_summary": conservative_val.summary,
        "search_space": {
            "a3_gate_mode_grid": hybrid_modes,
            "trigger_threshold_grid": parse_float_grid(args.trigger_threshold_grid),
            "target_conf_grid": parse_float_grid(args.target_conf_grid),
            "score_margin_grid": parse_float_grid(args.score_margin_grid),
            "gain_grid_db": parse_float_grid(args.gain_grid_db),
            "cooldown_grid_s": parse_float_grid(args.cooldown_grid_s),
            "anti_ping_pong_grid_s": parse_float_grid(args.anti_ping_pong_grid_s),
            "confirmation_grid": parse_int_grid(args.confirmation_grid),
            "topk_stage1": args.topk_stage1,
            "a3_hysteresis_db": args.a3_hysteresis_db,
            "a3_time_to_trigger_ms": args.a3_time_to_trigger_ms,
            "measurement_interval_ms": args.measurement_interval_ms,
        },
    }
    save_json(args.output_dir / "mlp_best_policy.json", best_policy_payload)
    save_json(args.output_dir / "mlp_best_hybrid_policy.json", best_hybrid_payload)

    replay_metrics = {
        "dataset_dir": str(args.dataset_dir.resolve()),
        "dataset_root": str(dataset_root.resolve()),
        "checkpoint_path": str(args.checkpoint_path.resolve()),
        "candidate_top_k": candidate_top_k,
        "seq_len": seq_len,
        "model_config": checkpoint["model_config"],
        "training_args": checkpoint["training_args"],
        "val_split": args.val_split,
        "test_split": args.test_split,
        "ping_pong_window_s": args.ping_pong_window_s,
        "stability_window_s": args.stability_window_s,
        "early_lead_threshold_s": args.early_lead_threshold_s,
        "a3_hysteresis_db": args.a3_hysteresis_db,
        "a3_time_to_trigger_ms": args.a3_time_to_trigger_ms,
        "measurement_interval_ms": args.measurement_interval_ms,
        "validation_baseline_summary": current_val.summary,
        "validation_conservative_summary": conservative_val.summary,
        "best_conservative_policy": best_conservative_policy.to_dict(),
        "best_hybrid_policy": best_hybrid_policy.to_dict(),
        "test_summaries": [a3_test.summary, current_test.summary, conservative_test.summary, hybrid_test.summary],
    }
    save_json(args.output_dir / "mlp_replay_metrics.json", replay_metrics)

    report = render_mlp_report(
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint_path,
        candidate_top_k=candidate_top_k,
        seq_len=seq_len,
        validation_current_summary=current_val.summary,
        validation_conservative_summary=conservative_val.summary,
        best_conservative_policy=best_conservative_policy,
        best_hybrid_policy=best_hybrid_policy,
        test_summaries=test_summary_frame,
        conservative_sweep_df=conservative_sweep_df,
        hybrid_sweep_df=hybrid_sweep_df,
    )
    (args.output_dir / "mlp_replay_report.md").write_text(report, encoding="utf-8")

    print(test_summary_frame.to_string(index=False), flush=True)
    print(f"Saved MLP replay artifacts to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
