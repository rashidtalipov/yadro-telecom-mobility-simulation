from __future__ import annotations

import argparse
import itertools
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support

from common import infer_run_dir_map, load_handover_trace_rows, load_json, save_json
from model import ModelConfig, MultitaskLstmPredictor
from train import create_loader, resolve_device


LEAD_TIME_BINS: tuple[tuple[float, float | None, str], ...] = (
    (0.0, 0.2, "0-0.2s"),
    (0.2, 0.5, "0.2-0.5s"),
    (0.5, 1.0, "0.5-1.0s"),
    (1.0, None, "1.0+s"),
)
MODEL_ORDER = ["a3_original", "current_k3", "conservative_k3", "hybrid_a3_k3"]
A3_GATE_MODES = ("off", "assist", "strict")


@dataclass(frozen=True)
class ConservativePolicyConfig:
    name: str
    a3_gate_mode: str = "off"
    trigger_threshold: float = 0.5
    target_conf_threshold: float = 0.0
    min_score_margin: float = 0.0
    min_gain_rsrp_db: float = -1e9
    cooldown_s: float = 0.0
    anti_ping_pong_window_s: float = 0.0
    consecutive_confirmation_steps: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PolicyReplayResult:
    model_name: str
    config: dict[str, Any]
    summary: dict[str, Any]
    segment_results: pd.DataFrame
    per_ue: pd.DataFrame


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline replay for conservative and hybrid A3-assisted candidate-aware K=3 handover policies.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-split", type=str, default="val", choices=("train", "val", "test"))
    parser.add_argument("--test-split", type=str, default="test", choices=("train", "val", "test"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--window-stride", type=int, default=1)
    parser.add_argument("--ping-pong-window-s", type=float, default=5.0)
    parser.add_argument("--stability-window-s", type=float, default=5.0)
    parser.add_argument("--early-lead-threshold-s", type=float, default=0.2)
    parser.add_argument("--topk-stage1", type=int, default=10)
    parser.add_argument("--a3-gate-mode-grid", type=str, default="assist,strict")
    parser.add_argument("--trigger-threshold-grid", type=str, default="0.50,0.55,0.60,0.65")
    parser.add_argument("--target-conf-grid", type=str, default="0.00,0.45,0.55,0.65")
    parser.add_argument("--score-margin-grid", type=str, default="0.00,0.05,0.10")
    parser.add_argument("--gain-grid-db", type=str, default="-999,0.0,1.0")
    parser.add_argument("--cooldown-grid-s", type=str, default="0.0,0.5,1.0")
    parser.add_argument("--anti-ping-pong-grid-s", type=str, default="0.0,5.0")
    parser.add_argument("--confirmation-grid", type=str, default="1,2")
    parser.add_argument("--a3-hysteresis-db", type=float, default=3.0)
    parser.add_argument("--a3-time-to-trigger-ms", type=float, default=256.0)
    parser.add_argument("--measurement-interval-ms", type=float, default=100.0)
    return parser


def parse_float_grid(raw: str) -> list[float]:
    return [float(token.strip()) for token in raw.split(",") if token.strip()]


def parse_int_grid(raw: str) -> list[int]:
    return [int(token.strip()) for token in raw.split(",") if token.strip()]


def parse_str_grid(raw: str) -> list[str]:
    return [token.strip() for token in raw.split(",") if token.strip()]


def load_split_frame(dataset_dir: Path, split: str) -> tuple[dict[str, Any], pd.DataFrame]:
    metadata = load_json(dataset_dir / "metadata.json")
    frame = pd.read_parquet(dataset_dir / f"{split}_rows.parquet")
    frame["run_id"] = frame["run_id"].astype(str)
    frame["imsi"] = pd.to_numeric(frame["imsi"], errors="coerce").astype(np.int64)
    frame["time"] = pd.to_numeric(frame["time"], errors="coerce").astype(np.float32)
    return metadata, frame.sort_values(["run_id", "imsi", "time"], ignore_index=True)


def load_candidate_checkpoint(
    checkpoint_path: Path,
    metadata: dict[str, Any],
) -> tuple[dict[str, Any], MultitaskLstmPredictor, int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = ModelConfig(**checkpoint["model_config"])
    if model_config.target_mode != "candidate":
        raise ValueError(f"{checkpoint_path} is not a candidate-aware checkpoint")

    candidate_top_k = checkpoint["training_args"].get("candidate_top_k")
    if candidate_top_k in ("None", "", None):
        raise ValueError(f"{checkpoint_path} does not expose candidate_top_k")
    candidate_top_k = int(candidate_top_k)

    model = MultitaskLstmPredictor(model_config)
    checkpoint_state = checkpoint["model_state"]
    model_state = model.state_dict()
    filtered_state: dict[str, torch.Tensor] = {}
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
    model.load_state_dict(filtered_state, strict=False)
    return checkpoint, model, candidate_top_k


def infer_candidate_predictions(
    frame: pd.DataFrame,
    metadata: dict[str, Any],
    checkpoint_path: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    window_stride: int,
) -> pd.DataFrame:
    checkpoint, model, candidate_top_k = load_candidate_checkpoint(checkpoint_path, metadata)
    loader = create_loader(
        frame=frame,
        seq_len=int(metadata["seq_len"]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        window_stride=window_stride,
        candidate_top_k=candidate_top_k,
        num_cells=len(metadata["cell_ids"]),
    )

    index_to_cell = {
        int(index): int(cell_id) for index, cell_id in metadata["index_to_cell"].items()
    }
    dataset = loader.dataset
    model = model.to(device)
    model.eval()

    records: list[pd.DataFrame] = []
    with torch.no_grad():
        for batch in loader:
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
                        "run_id": [dataset.run_names[int(code)] for code in batch["run_code"].cpu().numpy().tolist()],
                        "imsi": batch["imsi"].cpu().numpy().astype(np.int64),
                        "time": batch["time"].cpu().numpy().astype(np.float32),
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

    predictions = pd.concat(records, ignore_index=True)
    return predictions.sort_values(["run_id", "imsi", "time"], ignore_index=True)


def attach_candidate_gain_metrics(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    metadata: dict[str, Any],
) -> pd.DataFrame:
    merged = frame.merge(predictions, on=["run_id", "imsi", "time"], how="inner")
    candidate_scaler = metadata["candidate_scaler"]
    feature_scaler = metadata["scaler"]

    def unscale(series: pd.Series, column_name: str) -> pd.Series:
        mean = float(candidate_scaler["means"][column_name])
        std = float(candidate_scaler["stds"][column_name])
        return (pd.to_numeric(series, errors="coerce").astype(float) * std + mean).astype(np.float32)

    def unscale_feature(series: pd.Series, column_name: str) -> pd.Series:
        mean = float(feature_scaler["means"][column_name])
        std = float(feature_scaler["stds"][column_name])
        return (pd.to_numeric(series, errors="coerce").astype(float) * std + mean).astype(np.float32)

    gain_column = "predicted_target_gain_rsrp_db"
    merged[gain_column] = np.nan
    for rank in range(1, int(metadata["max_candidate_k"]) + 1):
        cell_column = f"candidate_cell_id_{rank}"
        diff_column = f"candidate_diff_rsrp_{rank}"
        raw_gain = unscale(merged[diff_column], diff_column)
        candidate_cells = pd.to_numeric(merged[cell_column], errors="coerce").fillna(-1).astype(np.int32)
        mask = candidate_cells == merged["predicted_target_cell_id"].astype(np.int32)
        merged.loc[mask, gain_column] = raw_gain.loc[mask]

    merged[gain_column] = pd.to_numeric(merged[gain_column], errors="coerce").astype(np.float32)
    merged["best_ngh_diff_rsrp_db"] = unscale_feature(merged["best_ngh_diff_rsrp"], "best_ngh_diff_rsrp")
    return merged.sort_values(["run_id", "imsi", "time"], ignore_index=True)


def build_actual_events(dataset_root: Path, run_ids: list[str]) -> pd.DataFrame:
    run_map = infer_run_dir_map(dataset_root)
    missing = sorted(set(run_ids) - set(run_map))
    if missing:
        raise FileNotFoundError(f"Missing run directories for replay: {missing}")

    events = pd.concat(
        [load_handover_trace_rows(run_map[run_id]) for run_id in run_ids],
        ignore_index=True,
    )
    events["run_id"] = events["run_id"].astype(str)
    events["imsi"] = pd.to_numeric(events["imsi"], errors="coerce").astype(np.int64)
    events["time"] = pd.to_numeric(events["time"], errors="coerce").astype(np.float32)
    events["target_cell_id"] = pd.to_numeric(events["target_cell_id"], errors="coerce").astype(np.int32)
    events["is_ping_pong"] = pd.to_numeric(events["is_ping_pong"], errors="coerce").fillna(0).astype(np.int8)
    return events.sort_values(["run_id", "imsi", "time"], ignore_index=True)


def build_coverage_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    return (
        predictions.groupby(["run_id", "imsi"], as_index=False)
        .agg(coverage_start_time=("time", "min"), coverage_end_time=("time", "max"))
        .sort_values(["run_id", "imsi"], ignore_index=True)
    )


def build_replay_segments(
    frame: pd.DataFrame,
    actual_events: pd.DataFrame,
    coverage: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    segments: list[dict[str, Any]] = []
    skipped_positive_events = 0

    frame_groups = {
        key: group.sort_values("time").copy()
        for key, group in frame.groupby(["run_id", "imsi"], sort=False)
    }
    coverage_map = {
        (str(row.run_id), int(row.imsi)): (float(row.coverage_start_time), float(row.coverage_end_time))
        for row in coverage.itertuples(index=False)
    }
    event_groups = {
        key: group.sort_values("time").copy()
        for key, group in actual_events.groupby(["run_id", "imsi"], sort=False)
    }

    for key, group in frame_groups.items():
        if key not in coverage_map:
            continue
        coverage_start, coverage_end = coverage_map[key]
        group = group[(group["time"] >= coverage_start) & (group["time"] <= coverage_end)].copy()
        if group.empty:
            continue

        events = event_groups.get(key)
        if events is None:
            events = pd.DataFrame(columns=["time", "target_cell_id", "is_ping_pong"])
        events = events[(events["time"] >= coverage_start) & (events["time"] <= coverage_end)].copy()
        event_records = list(events.itertuples(index=False))

        current_start = coverage_start
        current_source = int(group["serving_cell_id"].iloc[0])
        segment_index = 0
        for event_index, event in enumerate(event_records):
            segment_rows = group[(group["time"] >= current_start) & (group["time"] < float(event.time))]
            next_event_time = (
                float(event_records[event_index + 1].time)
                if event_index + 1 < len(event_records)
                else coverage_end
            )
            if segment_rows.empty:
                skipped_positive_events += 1
                current_start = float(event.time)
                current_source = int(event.target_cell_id)
                continue

            segments.append(
                {
                    "run_id": key[0],
                    "imsi": int(key[1]),
                    "segment_index": segment_index,
                    "segment_start_time": float(segment_rows["time"].min()),
                    "segment_end_time": float(event.time),
                    "source_cell_id": int(current_source),
                    "has_actual_ho": 1,
                    "actual_ho_time": float(event.time),
                    "actual_target_cell_id": int(event.target_cell_id),
                    "actual_dwell_s": float(max(0.0, next_event_time - float(event.time))),
                    "actual_is_ping_pong": int(event.is_ping_pong),
                    "segment_row_count": int(len(segment_rows)),
                }
            )
            current_start = float(event.time)
            current_source = int(event.target_cell_id)
            segment_index += 1

        terminal_rows = group[group["time"] >= current_start]
        if not terminal_rows.empty:
            segments.append(
                {
                    "run_id": key[0],
                    "imsi": int(key[1]),
                    "segment_index": segment_index,
                    "segment_start_time": float(terminal_rows["time"].min()),
                    "segment_end_time": float(coverage_end),
                    "source_cell_id": int(current_source),
                    "has_actual_ho": 0,
                    "actual_ho_time": math.nan,
                    "actual_target_cell_id": -1,
                    "actual_dwell_s": math.nan,
                    "actual_is_ping_pong": 0,
                    "segment_row_count": int(len(terminal_rows)),
                }
            )

    segment_frame = pd.DataFrame(segments).sort_values(
        ["run_id", "imsi", "segment_start_time"],
        ignore_index=True,
    )
    return segment_frame, skipped_positive_events


def prepare_policy_groups(
    decision_rows: pd.DataFrame,
    segments: pd.DataFrame,
    a3_hysteresis_db: float,
    a3_time_to_trigger_ms: float,
    measurement_interval_ms: float,
) -> dict[tuple[str, int], dict[str, Any]]:
    groups: dict[tuple[str, int], dict[str, Any]] = {}
    row_groups = {
        key: group.sort_values("time").copy()
        for key, group in decision_rows.groupby(["run_id", "imsi"], sort=False)
    }
    segment_groups = {
        key: group.sort_values("segment_start_time").copy()
        for key, group in segments.groupby(["run_id", "imsi"], sort=False)
    }

    for key, segment_group in segment_groups.items():
        row_group = row_groups.get(key)
        if row_group is None or row_group.empty:
            continue

        times = row_group["time"].to_numpy(dtype=np.float32, copy=True)
        best_ngh_diff_rsrp_db = row_group["best_ngh_diff_rsrp_db"].to_numpy(dtype=np.float32, copy=True)
        a3_entry = best_ngh_diff_rsrp_db >= float(a3_hysteresis_db)
        a3_hold_steps = np.zeros(len(best_ngh_diff_rsrp_db), dtype=np.int16)
        current_hold = 0
        for idx, condition_holds in enumerate(a3_entry):
            current_hold = current_hold + 1 if bool(condition_holds) else 0
            a3_hold_steps[idx] = current_hold
        ttt_steps = max(1, int(math.ceil(a3_time_to_trigger_ms / max(measurement_interval_ms, 1e-6))))
        near_steps = max(1, ttt_steps - 1)
        a3_true = a3_hold_steps >= ttt_steps
        a3_near = a3_hold_steps >= near_steps

        prepared_segments: list[dict[str, Any]] = []
        row_ptr = 0
        for segment in segment_group.itertuples(index=False):
            segment_start = float(segment.segment_start_time)
            segment_limit = float(segment.actual_ho_time) if int(segment.has_actual_ho) > 0 else float(segment.segment_end_time) + 1e-6
            while row_ptr < len(times) and times[row_ptr] < segment_start:
                row_ptr += 1
            start_idx = row_ptr
            while row_ptr < len(times) and times[row_ptr] < segment_limit:
                row_ptr += 1
            prepared_segments.append(
                {
                    "run_id": segment.run_id,
                    "imsi": int(segment.imsi),
                    "segment_index": int(segment.segment_index),
                    "segment_start_time": segment_start,
                    "segment_end_time": float(segment.segment_end_time),
                    "source_cell_id": int(segment.source_cell_id),
                    "has_actual_ho": int(segment.has_actual_ho),
                    "actual_ho_time": float(segment.actual_ho_time) if int(segment.has_actual_ho) > 0 else math.nan,
                    "actual_target_cell_id": int(segment.actual_target_cell_id),
                    "actual_dwell_s": float(segment.actual_dwell_s) if np.isfinite(segment.actual_dwell_s) else math.nan,
                    "actual_is_ping_pong": int(segment.actual_is_ping_pong),
                    "row_start_idx": int(start_idx),
                    "row_end_idx": int(row_ptr),
                }
            )

        groups[key] = {
            "times": times,
            "trigger_prob": row_group["trigger_prob"].to_numpy(dtype=np.float32, copy=True),
            "target_confidence": row_group["target_confidence"].to_numpy(dtype=np.float32, copy=True),
            "score_margin": row_group["score_margin"].to_numpy(dtype=np.float32, copy=True),
            "predicted_target_cell_id": row_group["predicted_target_cell_id"].to_numpy(dtype=np.int32, copy=True),
            "predicted_target_gain_rsrp_db": row_group["predicted_target_gain_rsrp_db"].to_numpy(dtype=np.float32, copy=True),
            "best_ngh_diff_rsrp_db": best_ngh_diff_rsrp_db,
            "a3_true": a3_true.astype(np.int8, copy=False),
            "a3_near": a3_near.astype(np.int8, copy=False),
            "segments": prepared_segments,
        }
    return groups


def evaluate_a3_policy(
    segments: pd.DataFrame,
    model_name: str,
    stability_window_s: float,
    early_lead_threshold_s: float,
) -> pd.DataFrame:
    results: list[dict[str, Any]] = []
    for segment in segments.itertuples(index=False):
        has_actual = int(segment.has_actual_ho)
        success = has_actual
        early_prediction = int(success > 0 and 0.0 >= early_lead_threshold_s)
        unstable_success = int(
            success > 0
            and (
                int(segment.actual_is_ping_pong) > 0
                or float(segment.actual_dwell_s) < stability_window_s
            )
        )
        results.append(
            {
                "model_name": model_name,
                "run_id": segment.run_id,
                "imsi": int(segment.imsi),
                "segment_index": int(segment.segment_index),
                "segment_start_time": float(segment.segment_start_time),
                "segment_end_time": float(segment.segment_end_time),
                "source_cell_id": int(segment.source_cell_id),
                "has_actual_ho": has_actual,
                "actual_ho_time": float(segment.actual_ho_time) if has_actual > 0 else math.nan,
                "actual_target_cell_id": int(segment.actual_target_cell_id),
                "actual_dwell_s": float(segment.actual_dwell_s) if np.isfinite(segment.actual_dwell_s) else math.nan,
                "actual_is_ping_pong": int(segment.actual_is_ping_pong),
                "predicted_positive": has_actual,
                "decision_time": float(segment.actual_ho_time) if has_actual > 0 else math.nan,
                "predicted_target_cell_id": int(segment.actual_target_cell_id) if has_actual > 0 else -1,
                "success": success,
                "wrong_target": 0,
                "negative_fp": 0,
                "unnecessary": unstable_success,
                "unstable_success": unstable_success,
                "missed_useful": 0,
                "lead_time_s": 0.0 if has_actual > 0 else math.nan,
                "early_prediction": early_prediction,
            }
        )
    return pd.DataFrame(results)


def evaluate_candidate_policy(
    prepared_groups: dict[tuple[str, int], dict[str, Any]],
    policy: ConservativePolicyConfig,
    model_name: str,
    stability_window_s: float,
    early_lead_threshold_s: float,
) -> pd.DataFrame:
    results: list[dict[str, Any]] = []

    for group in prepared_groups.values():
        cooldown_until = -np.inf
        blocked_returns: dict[int, float] = {}
        times = group["times"]
        trigger_prob = group["trigger_prob"]
        target_confidence = group["target_confidence"]
        score_margin = group["score_margin"]
        predicted_target_cell_id = group["predicted_target_cell_id"]
        target_gain = group["predicted_target_gain_rsrp_db"]
        a3_true = group.get("a3_true")
        a3_near = group.get("a3_near")

        for segment in group["segments"]:
            streak_target = -1
            streak_count = 0
            decision_time = math.nan
            decision_target = -1

            for row_index in range(segment["row_start_idx"], segment["row_end_idx"]):
                time_value = float(times[row_index])

                expired_cells = [
                    cell_id for cell_id, expiry in blocked_returns.items() if expiry <= time_value
                ]
                for cell_id in expired_cells:
                    del blocked_returns[cell_id]

                if time_value < cooldown_until:
                    streak_target = -1
                    streak_count = 0
                    continue

                proposal_target = int(predicted_target_cell_id[row_index])
                proposal_gain = float(target_gain[row_index]) if np.isfinite(target_gain[row_index]) else -np.inf
                base_passes = (
                    float(trigger_prob[row_index]) >= policy.trigger_threshold
                    and float(target_confidence[row_index]) >= policy.target_conf_threshold
                    and float(score_margin[row_index]) >= policy.min_score_margin
                    and proposal_gain >= policy.min_gain_rsrp_db
                    and proposal_target > 0
                    and proposal_target != int(segment["source_cell_id"])
                    and (
                        policy.anti_ping_pong_window_s <= 0.0
                        or blocked_returns.get(proposal_target, -np.inf) <= time_value
                    )
                )

                if not base_passes:
                    streak_target = -1
                    streak_count = 0
                    continue

                if proposal_target == streak_target:
                    streak_count += 1
                else:
                    streak_target = proposal_target
                    streak_count = 1

                a3_true_now = bool(a3_true[row_index]) if a3_true is not None else False
                a3_near_now = bool(a3_near[row_index]) if a3_near is not None else False
                consecutive_ready = streak_count >= policy.consecutive_confirmation_steps

                if policy.a3_gate_mode == "off":
                    gate_passes = consecutive_ready
                elif policy.a3_gate_mode == "assist":
                    gate_passes = a3_true_now or a3_near_now or consecutive_ready
                elif policy.a3_gate_mode == "strict":
                    gate_passes = a3_true_now or a3_near_now
                else:
                    raise ValueError(f"Unsupported a3_gate_mode={policy.a3_gate_mode}")

                if gate_passes:
                    decision_time = time_value
                    decision_target = proposal_target
                    cooldown_until = time_value + policy.cooldown_s
                    if policy.anti_ping_pong_window_s > 0.0:
                        blocked_returns[int(segment["source_cell_id"])] = time_value + policy.anti_ping_pong_window_s
                    break

            has_actual = int(segment["has_actual_ho"])
            predicted_positive = int(decision_target > 0 and np.isfinite(decision_time))
            success = int(
                has_actual > 0
                and predicted_positive > 0
                and decision_target == int(segment["actual_target_cell_id"])
            )
            wrong_target = int(
                has_actual > 0
                and predicted_positive > 0
                and decision_target != int(segment["actual_target_cell_id"])
            )
            negative_fp = int(has_actual == 0 and predicted_positive > 0)
            lead_time = (
                float(segment["actual_ho_time"]) - float(decision_time)
                if success > 0 and np.isfinite(decision_time)
                else math.nan
            )
            early_prediction = int(success > 0 and lead_time >= early_lead_threshold_s)
            unstable_success = int(
                success > 0
                and (
                    int(segment["actual_is_ping_pong"]) > 0
                    or float(segment["actual_dwell_s"]) < stability_window_s
                )
            )
            unnecessary = int((predicted_positive > 0 and success == 0) or unstable_success > 0)
            missed_useful = int(has_actual > 0 and success == 0)

            results.append(
                {
                    "model_name": model_name,
                    "run_id": segment["run_id"],
                    "imsi": int(segment["imsi"]),
                    "segment_index": int(segment["segment_index"]),
                    "segment_start_time": float(segment["segment_start_time"]),
                    "segment_end_time": float(segment["segment_end_time"]),
                    "source_cell_id": int(segment["source_cell_id"]),
                    "has_actual_ho": has_actual,
                    "actual_ho_time": float(segment["actual_ho_time"]) if has_actual > 0 else math.nan,
                    "actual_target_cell_id": int(segment["actual_target_cell_id"]),
                    "actual_dwell_s": float(segment["actual_dwell_s"]) if np.isfinite(segment["actual_dwell_s"]) else math.nan,
                    "actual_is_ping_pong": int(segment["actual_is_ping_pong"]),
                    "predicted_positive": predicted_positive,
                    "decision_time": float(decision_time) if np.isfinite(decision_time) else math.nan,
                    "predicted_target_cell_id": int(decision_target),
                    "success": success,
                    "wrong_target": wrong_target,
                    "negative_fp": negative_fp,
                    "unnecessary": unnecessary,
                    "unstable_success": unstable_success,
                    "missed_useful": missed_useful,
                    "lead_time_s": lead_time,
                    "early_prediction": early_prediction,
                }
            )

    return pd.DataFrame(results)


def compute_lead_time_summary(successful_segments: pd.DataFrame) -> dict[str, Any]:
    total = int(len(successful_segments))
    lead_times = successful_segments["lead_time_s"].dropna().to_numpy(dtype=float)
    summary: dict[str, Any] = {}
    for lower, upper, label in LEAD_TIME_BINS:
        if upper is None:
            mask = lead_times >= lower
        else:
            mask = (lead_times >= lower) & (lead_times < upper)
        count = int(mask.sum())
        summary[label] = {
            "count": count,
            "rate_over_successful": float(count / total) if total > 0 else 0.0,
        }
    return summary


def compute_policy_summary(
    segment_results: pd.DataFrame,
    skipped_positive_events: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    positive_segments = segment_results[segment_results["has_actual_ho"] > 0].copy()
    successful_segments = segment_results[segment_results["success"] > 0].copy()
    predicted_positive_segments = segment_results[segment_results["predicted_positive"] > 0].copy()

    trigger_true = segment_results["has_actual_ho"].to_numpy(dtype=np.int64)
    trigger_pred = segment_results["predicted_positive"].to_numpy(dtype=np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        trigger_true,
        trigger_pred,
        average="binary",
        zero_division=0,
    )

    per_ue = (
        segment_results.groupby(["model_name", "run_id", "imsi"], as_index=False)
        .agg(
            actual_ho_count=("has_actual_ho", "sum"),
            handover_count=("predicted_positive", "sum"),
            successful_ho_count=("success", "sum"),
            unnecessary_handover_count=("unnecessary", "sum"),
            missed_useful_handover_count=("missed_useful", "sum"),
            ping_pong_count=("actual_is_ping_pong", lambda x: int(x[segment_results.loc[x.index, "success"] > 0].sum())),
            mean_dwell_time_s=(
                "actual_dwell_s",
                lambda x: float(
                    np.nanmean(x[segment_results.loc[x.index, "success"] > 0].to_numpy(dtype=float))
                )
                if (segment_results.loc[x.index, "success"] > 0).any()
                else math.nan,
            ),
            early_prediction_count=("early_prediction", "sum"),
        )
        .sort_values(["model_name", "run_id", "imsi"], ignore_index=True)
    )
    per_ue["end_to_end_decision_success_rate"] = np.where(
        per_ue["actual_ho_count"] > 0,
        per_ue["successful_ho_count"] / per_ue["actual_ho_count"],
        np.nan,
    )
    per_ue["ping_pong_rate"] = np.where(
        per_ue["successful_ho_count"] > 0,
        per_ue["ping_pong_count"] / per_ue["successful_ho_count"],
        np.nan,
    )
    per_ue["early_prediction_rate"] = np.where(
        per_ue["actual_ho_count"] > 0,
        per_ue["early_prediction_count"] / per_ue["actual_ho_count"],
        np.nan,
    )

    summary = {
        "model_name": str(segment_results["model_name"].iloc[0]),
        "segment_count": int(len(segment_results)),
        "positive_segment_count": int(len(positive_segments)),
        "skipped_positive_events_due_to_no_context": int(skipped_positive_events),
        "handover_count": int(predicted_positive_segments["predicted_positive"].sum()),
        "successful_handover_count": int(successful_segments["success"].sum()),
        "unnecessary_handover_count": int(segment_results["unnecessary"].sum()),
        "missed_useful_handover_count": int(segment_results["missed_useful"].sum()),
        "ping_pong_count": int(successful_segments["actual_is_ping_pong"].sum()),
        "ping_pong_rate": float(successful_segments["actual_is_ping_pong"].mean()) if not successful_segments.empty else 0.0,
        "mean_dwell_time_s": float(successful_segments["actual_dwell_s"].mean()) if not successful_segments.empty else 0.0,
        "early_prediction_rate": float(positive_segments["early_prediction"].sum() / len(positive_segments)) if not positive_segments.empty else 0.0,
        "end_to_end_decision_success_rate": float(successful_segments["success"].sum() / len(positive_segments)) if not positive_segments.empty else 0.0,
        "decision_success_precision": float(successful_segments["success"].sum() / len(predicted_positive_segments)) if not predicted_positive_segments.empty else 0.0,
        "trigger_precision": float(precision),
        "trigger_recall": float(recall),
        "trigger_f1": float(f1),
        "mean_handover_count_per_ue": float(per_ue["handover_count"].mean()) if not per_ue.empty else 0.0,
        "std_handover_count_per_ue": float(per_ue["handover_count"].std(ddof=0)) if len(per_ue) > 0 else 0.0,
        "lead_time_bins": compute_lead_time_summary(successful_segments),
    }
    return summary, per_ue


def make_policy_result(
    model_name: str,
    config: dict[str, Any],
    segment_results: pd.DataFrame,
    skipped_positive_events: int,
) -> PolicyReplayResult:
    summary, per_ue = compute_policy_summary(segment_results, skipped_positive_events)
    return PolicyReplayResult(
        model_name=model_name,
        config=config,
        summary=summary,
        segment_results=segment_results,
        per_ue=per_ue,
    )


def score_policy_config(row: pd.Series, baseline: pd.Series) -> float:
    unnecessary_gain = (baseline["unnecessary_handover_count"] - row["unnecessary_handover_count"]) / max(
        baseline["unnecessary_handover_count"], 1.0
    )
    ping_pong_gain = (baseline["ping_pong_rate"] - row["ping_pong_rate"]) / max(baseline["ping_pong_rate"], 1e-6)
    success_delta = row["end_to_end_decision_success_rate"] - baseline["end_to_end_decision_success_rate"]
    early_delta = row["early_prediction_rate"] - baseline["early_prediction_rate"]
    precision_delta = row["decision_success_precision"] - baseline["decision_success_precision"]
    return (
        3.00 * success_delta
        + 2.00 * early_delta
        + 0.75 * unnecessary_gain
        + 0.50 * ping_pong_gain
        + 0.25 * precision_delta
    )


def select_best_policy(
    sweep_results: pd.DataFrame,
    baseline_summary: dict[str, Any],
) -> pd.Series:
    baseline = pd.Series(baseline_summary)
    ranked = sweep_results.copy()
    ranked["selection_score"] = ranked.apply(lambda row: score_policy_config(row, baseline), axis=1)
    ranked["improves_stability"] = (
        (ranked["unnecessary_handover_count"] < baseline["unnecessary_handover_count"])
        & (ranked["ping_pong_rate"] < baseline["ping_pong_rate"])
    )
    improved = ranked[ranked["improves_stability"]].copy()
    if not improved.empty:
        ranked = improved

    preserve_or_better = ranked[
        (ranked["end_to_end_decision_success_rate"] >= baseline["end_to_end_decision_success_rate"])
        & (ranked["early_prediction_rate"] >= baseline["early_prediction_rate"])
    ].copy()
    if not preserve_or_better.empty:
        ranked = preserve_or_better
    else:
        preserve = ranked[
            (ranked["end_to_end_decision_success_rate"] >= baseline["end_to_end_decision_success_rate"] - 0.05)
            & (ranked["early_prediction_rate"] >= baseline["early_prediction_rate"] - 0.05)
        ].copy()
        if not preserve.empty:
            ranked = preserve

    ranked = ranked.sort_values(
        [
            "selection_score",
            "end_to_end_decision_success_rate",
            "early_prediction_rate",
            "decision_success_precision",
            "unnecessary_handover_count",
            "ping_pong_rate",
            "cooldown_s",
            "anti_ping_pong_window_s",
            "consecutive_confirmation_steps",
            "min_score_margin",
            "target_conf_threshold",
            "trigger_threshold",
            "name",
        ],
        ascending=[False, False, False, False, True, True, True, True, True, True, True, True, True],
        ignore_index=True,
    )
    return ranked.iloc[0]


def stage1_configs(
    trigger_grid: list[float],
    conf_grid: list[float],
    margin_grid: list[float],
    gain_grid: list[float],
    gate_modes: list[str],
    name_prefix: str,
) -> list[ConservativePolicyConfig]:
    configs: list[ConservativePolicyConfig] = []
    for index, (a3_gate_mode, trigger_threshold, target_conf_threshold, min_score_margin, min_gain_rsrp_db) in enumerate(
        itertools.product(gate_modes, trigger_grid, conf_grid, margin_grid, gain_grid),
        start=1,
    ):
        configs.append(
            ConservativePolicyConfig(
                name=f"{name_prefix}_{index:03d}",
                a3_gate_mode=str(a3_gate_mode),
                trigger_threshold=float(trigger_threshold),
                target_conf_threshold=float(target_conf_threshold),
                min_score_margin=float(min_score_margin),
                min_gain_rsrp_db=float(min_gain_rsrp_db),
                cooldown_s=0.0,
                anti_ping_pong_window_s=0.0,
                consecutive_confirmation_steps=1,
            )
        )
    return configs


def expand_stage2_configs(
    seeds: pd.DataFrame,
    cooldown_grid: list[float],
    anti_ping_pong_grid: list[float],
    confirmation_grid: list[int],
    name_prefix: str,
) -> list[ConservativePolicyConfig]:
    configs: list[ConservativePolicyConfig] = []
    for seed_index, seed in enumerate(seeds.itertuples(index=False), start=1):
        for expansion_index, (cooldown_s, anti_ping_pong_window_s, consecutive_confirmation_steps) in enumerate(
            itertools.product(cooldown_grid, anti_ping_pong_grid, confirmation_grid),
            start=1,
        ):
            configs.append(
                ConservativePolicyConfig(
                    name=f"{name_prefix}_{seed_index:02d}_{expansion_index:02d}",
                    a3_gate_mode=str(seed.a3_gate_mode),
                    trigger_threshold=float(seed.trigger_threshold),
                    target_conf_threshold=float(seed.target_conf_threshold),
                    min_score_margin=float(seed.min_score_margin),
                    min_gain_rsrp_db=float(seed.min_gain_rsrp_db),
                    cooldown_s=float(cooldown_s),
                    anti_ping_pong_window_s=float(anti_ping_pong_window_s),
                    consecutive_confirmation_steps=int(consecutive_confirmation_steps),
                )
            )
    return configs


def run_policy_sweep(
    prepared_groups_val: dict[tuple[str, int], dict[str, Any]],
    skipped_positive_events: int,
    baseline_summary: dict[str, Any],
    early_lead_threshold_s: float,
    stability_window_s: float,
    topk_stage1: int,
    model_name: str,
    gate_modes: list[str],
    sweep_label: str,
    trigger_grid: list[float],
    conf_grid: list[float],
    margin_grid: list[float],
    gain_grid: list[float],
    cooldown_grid: list[float],
    anti_ping_pong_grid: list[float],
    confirmation_grid: list[int],
) -> tuple[pd.DataFrame, ConservativePolicyConfig, pd.Series]:
    stage1_rows: list[dict[str, Any]] = []
    for config in stage1_configs(
        trigger_grid=trigger_grid,
        conf_grid=conf_grid,
        margin_grid=margin_grid,
        gain_grid=gain_grid,
        gate_modes=gate_modes,
        name_prefix=f"{sweep_label}_stage1",
    ):
        result = make_policy_result(
            model_name=model_name,
            config=config.to_dict(),
            segment_results=evaluate_candidate_policy(
                prepared_groups=prepared_groups_val,
                policy=config,
                model_name=model_name,
                stability_window_s=stability_window_s,
                early_lead_threshold_s=early_lead_threshold_s,
            ),
            skipped_positive_events=skipped_positive_events,
        )
        stage1_rows.append({**config.to_dict(), **result.summary})

    stage1_df = pd.DataFrame(stage1_rows)
    stage1_df["selection_score"] = stage1_df.apply(lambda row: score_policy_config(row, pd.Series(baseline_summary)), axis=1)
    stage1_df = stage1_df.sort_values(
        ["selection_score", "end_to_end_decision_success_rate", "early_prediction_rate"],
        ascending=[False, False, False],
        ignore_index=True,
    )

    top_stage1 = stage1_df.head(topk_stage1).copy()
    stage2_rows: list[dict[str, Any]] = []
    for config in expand_stage2_configs(
        top_stage1,
        cooldown_grid,
        anti_ping_pong_grid,
        confirmation_grid,
        name_prefix=f"{sweep_label}_stage2",
    ):
        result = make_policy_result(
            model_name=model_name,
            config=config.to_dict(),
            segment_results=evaluate_candidate_policy(
                prepared_groups=prepared_groups_val,
                policy=config,
                model_name=model_name,
                stability_window_s=stability_window_s,
                early_lead_threshold_s=early_lead_threshold_s,
            ),
            skipped_positive_events=skipped_positive_events,
        )
        stage2_rows.append({**config.to_dict(), **result.summary})

    stage2_df = pd.DataFrame(stage2_rows)
    stage2_df["selection_score"] = stage2_df.apply(lambda row: score_policy_config(row, pd.Series(baseline_summary)), axis=1)

    sweep_df = pd.concat([stage1_df.assign(stage="stage1"), stage2_df.assign(stage="stage2")], ignore_index=True)
    best_row = select_best_policy(sweep_df, baseline_summary)
    best_policy = ConservativePolicyConfig(
        name=str(best_row["name"]),
        a3_gate_mode=str(best_row["a3_gate_mode"]),
        trigger_threshold=float(best_row["trigger_threshold"]),
        target_conf_threshold=float(best_row["target_conf_threshold"]),
        min_score_margin=float(best_row["min_score_margin"]),
        min_gain_rsrp_db=float(best_row["min_gain_rsrp_db"]),
        cooldown_s=float(best_row["cooldown_s"]),
        anti_ping_pong_window_s=float(best_row["anti_ping_pong_window_s"]),
        consecutive_confirmation_steps=int(best_row["consecutive_confirmation_steps"]),
    )
    return sweep_df.sort_values("selection_score", ascending=False, ignore_index=True), best_policy, best_row


def plot_core_metrics(summaries: pd.DataFrame, output_path: Path) -> None:
    metrics = [
        ("end_to_end_decision_success_rate", "E2E Success"),
        ("unnecessary_handover_count", "Unnecessary HO"),
        ("ping_pong_rate", "Ping-Pong Rate"),
        ("missed_useful_handover_count", "Missed Useful HO"),
        ("early_prediction_rate", "Early Prediction Rate"),
        ("mean_dwell_time_s", "Mean Dwell (s)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes_flat = axes.flatten()
    colors = ["#7a8fa6", "#d9a441", "#3f8f6b", "#8a5ba6"]
    for axis, (column, title) in zip(axes_flat, metrics):
        axis.bar(summaries["model_name"], summaries[column], color=colors[: len(summaries)])
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_lead_time_bins(summaries: pd.DataFrame, output_path: Path) -> None:
    labels = [label for _, _, label in LEAD_TIME_BINS]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(summaries))
    bottom = np.zeros(len(summaries), dtype=float)
    colors = ["#9bb7d4", "#6ea4bf", "#3b7a9e", "#204e67"]
    for color, label in zip(colors, labels):
        values = np.asarray(
            [float(summary["lead_time_bins"][label]["rate_over_successful"]) for summary in summaries.to_dict(orient="records")],
            dtype=float,
        )
        ax.bar(x, values, bottom=bottom, label=label, color=color)
        bottom += values
    ax.set_xticks(x)
    ax.set_xticklabels(summaries["model_name"], rotation=20)
    ax.set_ylabel("Rate over successful predicted HOs")
    ax.set_title("Lead-Time Distribution")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def format_summary_table(summaries: pd.DataFrame) -> str:
    table = summaries[
        [
            "model_name",
            "handover_count",
            "unnecessary_handover_count",
            "missed_useful_handover_count",
            "ping_pong_rate",
            "end_to_end_decision_success_rate",
            "early_prediction_rate",
            "mean_dwell_time_s",
        ]
    ].copy()
    numeric_cols = [column for column in table.columns if column != "model_name"]
    for column in numeric_cols:
        table[column] = table[column].map(lambda value: f"{float(value):.4f}")
    header = "| " + " | ".join(table.columns.tolist()) + " |"
    separator = "| " + " | ".join(["---"] * len(table.columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in table.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


def render_report(
    output_dir: Path,
    current_val: dict[str, Any],
    conservative_val: dict[str, Any],
    best_conservative_policy: ConservativePolicyConfig,
    best_hybrid_policy: ConservativePolicyConfig,
    test_summaries: pd.DataFrame,
    conservative_sweep_df: pd.DataFrame,
    hybrid_sweep_df: pd.DataFrame,
    ping_pong_window_s: float,
    early_lead_threshold_s: float,
    a3_hysteresis_db: float,
    a3_time_to_trigger_ms: float,
) -> str:
    conservative_row = test_summaries[test_summaries["model_name"] == "conservative_k3"].iloc[0]
    baseline_row = test_summaries[test_summaries["model_name"] == "current_k3"].iloc[0]
    hybrid_row = test_summaries[test_summaries["model_name"] == "hybrid_a3_k3"].iloc[0]
    a3_row = test_summaries[test_summaries["model_name"] == "a3_original"].iloc[0]
    report = "\n".join(
        [
            "# Hybrid A3-Assisted Replay Report",
            "",
            "## Setup",
            "",
            f"- Ping-pong window: `{ping_pong_window_s:.1f} s`",
            f"- Early lead threshold: `{early_lead_threshold_s:.1f} s`",
            f"- A3 hysteresis: `{a3_hysteresis_db:.1f} dB`",
            f"- A3 time-to-trigger: `{a3_time_to_trigger_ms:.0f} ms`",
            f"- Current K=3 validation end-to-end success: `{current_val['end_to_end_decision_success_rate']:.4f}`",
            f"- Conservative K=3 validation end-to-end success: `{conservative_val['end_to_end_decision_success_rate']:.4f}`",
            "",
            "## Best Conservative Policy",
            "",
            "```json",
            pd.Series(best_conservative_policy.to_dict()).to_json(indent=2),
            "```",
            "",
            "## Best Hybrid Policy",
            "",
            "```json",
            pd.Series(best_hybrid_policy.to_dict()).to_json(indent=2),
            "```",
            "",
            "## Test Comparison",
            "",
            format_summary_table(test_summaries),
            "",
            "## Sweep Notes",
            "",
            f"- Conservative sweep evaluated `{len(conservative_sweep_df)}` configs on validation replay.",
            f"- Hybrid sweep evaluated `{len(hybrid_sweep_df)}` configs on validation replay.",
            f"- Hybrid selection baseline was the best conservative validation policy, not the raw current K=3 policy.",
            "",
            "## Conclusion",
            "",
            f"- Current K=3 test unnecessary HOs: `{baseline_row['unnecessary_handover_count']}`",
            f"- Conservative K=3 test unnecessary HOs: `{conservative_row['unnecessary_handover_count']}`",
            f"- Hybrid A3 K=3 test unnecessary HOs: `{hybrid_row['unnecessary_handover_count']}`",
            f"- Logged A3 test unnecessary HOs: `{a3_row['unnecessary_handover_count']}`",
            f"- Current K=3 test ping-pong rate: `{baseline_row['ping_pong_rate']:.4f}`",
            f"- Conservative K=3 test ping-pong rate: `{conservative_row['ping_pong_rate']:.4f}`",
            f"- Hybrid A3 K=3 test ping-pong rate: `{hybrid_row['ping_pong_rate']:.4f}`",
            f"- Logged A3 test ping-pong rate: `{a3_row['ping_pong_rate']:.4f}`",
            f"- Current K=3 test end-to-end success: `{baseline_row['end_to_end_decision_success_rate']:.4f}`",
            f"- Conservative K=3 test end-to-end success: `{conservative_row['end_to_end_decision_success_rate']:.4f}`",
            f"- Hybrid A3 K=3 test end-to-end success: `{hybrid_row['end_to_end_decision_success_rate']:.4f}`",
            f"- Current K=3 test early rate: `{baseline_row['early_prediction_rate']:.4f}`",
            f"- Conservative K=3 test early rate: `{conservative_row['early_prediction_rate']:.4f}`",
            f"- Hybrid A3 K=3 test early rate: `{hybrid_row['early_prediction_rate']:.4f}`",
            f"- Plots: `{(output_dir / 'replay_core_metrics.png').name}` and `{(output_dir / 'replay_lead_time.png').name}`",
        ]
    )
    return report + "\n"


def main() -> None:
    args = build_argument_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    metadata, val_frame = load_split_frame(args.dataset_dir, args.val_split)
    _, test_frame = load_split_frame(args.dataset_dir, args.test_split)
    dataset_root = Path(metadata["dataset_root"])

    print("Inferring candidate predictions on validation split...", flush=True)
    val_predictions = infer_candidate_predictions(
        frame=val_frame,
        metadata=metadata,
        checkpoint_path=args.candidate_checkpoint,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        window_stride=args.window_stride,
    )
    print("Inferring candidate predictions on test split...", flush=True)
    test_predictions = infer_candidate_predictions(
        frame=test_frame,
        metadata=metadata,
        checkpoint_path=args.candidate_checkpoint,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        window_stride=args.window_stride,
    )

    val_decision_rows = attach_candidate_gain_metrics(val_frame, val_predictions, metadata)
    test_decision_rows = attach_candidate_gain_metrics(test_frame, test_predictions, metadata)

    val_coverage = build_coverage_frame(val_predictions)
    test_coverage = build_coverage_frame(test_predictions)
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

    current_policy = ConservativePolicyConfig(name="current_k3", a3_gate_mode="off")
    current_val = make_policy_result(
        model_name="current_k3",
        config=current_policy.to_dict(),
        segment_results=evaluate_candidate_policy(
            prepared_groups=prepared_val_groups,
            policy=current_policy,
            model_name="current_k3",
            stability_window_s=args.stability_window_s,
            early_lead_threshold_s=args.early_lead_threshold_s,
        ),
        skipped_positive_events=val_skipped,
    )

    print("Running conservative validation policy sweep...", flush=True)
    conservative_sweep_df, best_conservative_policy, best_conservative_policy_row = run_policy_sweep(
        prepared_groups_val=prepared_val_groups,
        skipped_positive_events=val_skipped,
        baseline_summary=current_val.summary,
        early_lead_threshold_s=args.early_lead_threshold_s,
        stability_window_s=args.stability_window_s,
        topk_stage1=args.topk_stage1,
        model_name="conservative_k3",
        gate_modes=["off"],
        sweep_label="conservative",
        trigger_grid=parse_float_grid(args.trigger_threshold_grid),
        conf_grid=parse_float_grid(args.target_conf_grid),
        margin_grid=parse_float_grid(args.score_margin_grid),
        gain_grid=parse_float_grid(args.gain_grid_db),
        cooldown_grid=parse_float_grid(args.cooldown_grid_s),
        anti_ping_pong_grid=parse_float_grid(args.anti_ping_pong_grid_s),
        confirmation_grid=parse_int_grid(args.confirmation_grid),
    )
    conservative_val = make_policy_result(
        model_name="conservative_k3",
        config=best_conservative_policy.to_dict(),
        segment_results=evaluate_candidate_policy(
            prepared_groups=prepared_val_groups,
            policy=best_conservative_policy,
            model_name="conservative_k3",
            stability_window_s=args.stability_window_s,
            early_lead_threshold_s=args.early_lead_threshold_s,
        ),
        skipped_positive_events=val_skipped,
    )

    hybrid_modes = [mode for mode in parse_str_grid(args.a3_gate_mode_grid) if mode in A3_GATE_MODES and mode != "off"]
    if not hybrid_modes:
        raise ValueError("--a3-gate-mode-grid must include at least one of assist or strict")

    print("Running hybrid A3-assisted validation policy sweep...", flush=True)
    hybrid_sweep_df, best_hybrid_policy, best_hybrid_policy_row = run_policy_sweep(
        prepared_groups_val=prepared_val_groups,
        skipped_positive_events=val_skipped,
        baseline_summary=conservative_val.summary,
        early_lead_threshold_s=args.early_lead_threshold_s,
        stability_window_s=args.stability_window_s,
        topk_stage1=args.topk_stage1,
        model_name="hybrid_a3_k3",
        gate_modes=hybrid_modes,
        sweep_label="hybrid",
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
        model_name="current_k3",
        config=current_policy.to_dict(),
        segment_results=evaluate_candidate_policy(
            prepared_groups=prepared_test_groups,
            policy=current_policy,
            model_name="current_k3",
            stability_window_s=args.stability_window_s,
            early_lead_threshold_s=args.early_lead_threshold_s,
        ),
        skipped_positive_events=test_skipped,
    )
    conservative_test = make_policy_result(
        model_name="conservative_k3",
        config=best_conservative_policy.to_dict(),
        segment_results=evaluate_candidate_policy(
            prepared_groups=prepared_test_groups,
            policy=best_conservative_policy,
            model_name="conservative_k3",
            stability_window_s=args.stability_window_s,
            early_lead_threshold_s=args.early_lead_threshold_s,
        ),
        skipped_positive_events=test_skipped,
    )
    hybrid_test = make_policy_result(
        model_name="hybrid_a3_k3",
        config=best_hybrid_policy.to_dict(),
        segment_results=evaluate_candidate_policy(
            prepared_groups=prepared_test_groups,
            policy=best_hybrid_policy,
            model_name="hybrid_a3_k3",
            stability_window_s=args.stability_window_s,
            early_lead_threshold_s=args.early_lead_threshold_s,
        ),
        skipped_positive_events=test_skipped,
    )

    test_summary_frame = pd.DataFrame(
        [a3_test.summary, current_test.summary, conservative_test.summary, hybrid_test.summary]
    )
    test_summary_frame["model_name"] = pd.Categorical(test_summary_frame["model_name"], categories=MODEL_ORDER, ordered=True)
    test_summary_frame = test_summary_frame.sort_values("model_name", ignore_index=True)
    test_summary_frame["model_name"] = test_summary_frame["model_name"].astype(str)
    per_ue_frame = pd.concat([a3_test.per_ue, current_test.per_ue, conservative_test.per_ue, hybrid_test.per_ue], ignore_index=True)
    segment_frame = pd.concat(
        [a3_test.segment_results, current_test.segment_results, conservative_test.segment_results, hybrid_test.segment_results],
        ignore_index=True,
    )

    plot_core_metrics(test_summary_frame, args.output_dir / "replay_core_metrics.png")
    plot_lead_time_bins(test_summary_frame, args.output_dir / "replay_lead_time.png")

    conservative_sweep_df.to_csv(args.output_dir / "policy_sweep_results.csv", index=False)
    hybrid_sweep_df.to_csv(args.output_dir / "hybrid_policy_sweep_results.csv", index=False)
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
    save_json(args.output_dir / "best_policy.json", best_policy_payload)
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
    save_json(args.output_dir / "best_hybrid_policy.json", best_hybrid_payload)

    replay_metrics = {
        "dataset_dir": str(args.dataset_dir.resolve()),
        "dataset_root": str(dataset_root.resolve()),
        "candidate_checkpoint": str(args.candidate_checkpoint.resolve()),
        "val_split": args.val_split,
        "test_split": args.test_split,
        "ping_pong_window_s": args.ping_pong_window_s,
        "stability_window_s": args.stability_window_s,
        "early_lead_threshold_s": args.early_lead_threshold_s,
        "a3_hysteresis_db": args.a3_hysteresis_db,
        "a3_time_to_trigger_ms": args.a3_time_to_trigger_ms,
        "measurement_interval_ms": args.measurement_interval_ms,
        "validation_baseline_summary": current_val.summary,
        "best_conservative_policy": best_conservative_policy.to_dict(),
        "best_hybrid_policy": best_hybrid_policy.to_dict(),
        "validation_conservative_summary": conservative_val.summary,
        "test_summaries": [a3_test.summary, current_test.summary, conservative_test.summary, hybrid_test.summary],
    }
    save_json(args.output_dir / "replay_metrics.json", replay_metrics)
    per_ue_frame.to_csv(args.output_dir / "per_ue_replay.csv", index=False)
    segment_frame.to_csv(args.output_dir / "segment_replay.csv", index=False)

    report = render_report(
        output_dir=args.output_dir,
        current_val=current_val.summary,
        conservative_val=conservative_val.summary,
        best_conservative_policy=best_conservative_policy,
        best_hybrid_policy=best_hybrid_policy,
        test_summaries=test_summary_frame,
        conservative_sweep_df=conservative_sweep_df,
        hybrid_sweep_df=hybrid_sweep_df,
        ping_pong_window_s=args.ping_pong_window_s,
        early_lead_threshold_s=args.early_lead_threshold_s,
        a3_hysteresis_db=args.a3_hysteresis_db,
        a3_time_to_trigger_ms=args.a3_time_to_trigger_ms,
    )
    (args.output_dir / "replay_report.md").write_text(report, encoding="utf-8")

    print(test_summary_frame.to_string(index=False), flush=True)
    print(f"Saved replay artifacts to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
