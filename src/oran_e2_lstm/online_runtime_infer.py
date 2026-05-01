#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (
    CANDIDATE_FEATURE_BASENAMES,
    MAX_CANDIDATE_K,
    NUMERIC_FEATURE_COLUMNS,
    ScalerState,
    apply_scaler,
    assemble_feature_frame,
    candidate_cell_id_column,
    candidate_feature_column,
    candidate_mask_column,
    derive_history_based_candidates,
    inspect_db_schema,
    load_lstm_feature_rows,
)
from model import ModelConfig, MultitaskLstmPredictor


@dataclass
class ConservativeOnlinePolicy:
    trigger_threshold: float = 0.65
    target_conf_threshold: float = 0.65
    min_score_margin: float = 0.0
    min_gain_rsrp_db: float = 1.0
    consecutive_confirmation_steps: int = 2
    cooldown_s: float = 0.0
    anti_ping_pong_window_s: float = 0.0


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Online runtime inference for candidate-aware conservative_k3 handover policy.",
    )
    parser.add_argument("--db-path", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=15)
    parser.add_argument("--trigger-threshold", type=float, default=0.65)
    parser.add_argument("--target-threshold", type=float, default=0.65)
    parser.add_argument(
        "--utility-threshold",
        type=float,
        default=0.0,
        help="Alias for minimum target score margin.",
    )
    parser.add_argument("--min-gain-rsrp-db", type=float, default=1.0)
    parser.add_argument("--consecutive-confirmation-steps", type=int, default=2)
    parser.add_argument("--cooldown-s", type=float, default=0.0)
    parser.add_argument("--anti-ping-pong-window-s", type=float, default=0.0)
    parser.add_argument(
        "--policy-mode",
        type=str,
        choices=("conservative", "raw"),
        default="conservative",
        help="Use conservative policy gating or emit raw scored decisions only.",
    )
    parser.add_argument("--prefer-non-serving-target", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--state-path", type=Path, default=None)
    return parser


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[dict[str, Any], MultitaskLstmPredictor, int]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = ModelConfig(**checkpoint["model_config"])
    if model_config.target_mode != "candidate":
        raise ValueError(f"{checkpoint_path} is not a candidate-aware checkpoint")

    candidate_top_k = int(checkpoint["training_args"]["candidate_top_k"])
    model = MultitaskLstmPredictor(model_config)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.to(device)
    model.eval()
    return checkpoint, model, candidate_top_k


def load_runtime_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"per_imsi": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_runtime_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def map_cell_indices(
    frame: pd.DataFrame,
    checkpoint_metadata: dict[str, Any],
) -> pd.DataFrame:
    mapped = frame.copy()
    cell_to_index = {int(cell_id): int(index) for cell_id, index in checkpoint_metadata["cell_to_index"].items()}
    mapped["serving_cell_index"] = [
        cell_to_index.get(int(cell_id), -1) if pd.notna(cell_id) else -1
        for cell_id in mapped["serving_cell_id"]
    ]

    for rank in range(1, MAX_CANDIDATE_K + 1):
        cell_column = candidate_cell_id_column(rank)
        index_column = f"candidate_cell_index_{rank}"
        candidate_ids = pd.to_numeric(mapped[cell_column], errors="coerce")
        mapped[index_column] = np.asarray(
            [
                cell_to_index.get(int(cell_id), -1) if pd.notna(cell_id) and int(cell_id) > 0 else -1
                for cell_id in candidate_ids
            ],
            dtype=np.int16,
        )
        mapped[candidate_mask_column(rank)] = (mapped[index_column] >= 0).astype(np.int8)
    return mapped


def prepare_runtime_frame(
    db_path: Path,
    checkpoint_metadata: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_dir = db_path.parent
    schema = inspect_db_schema(db_path)
    try:
        frame = assemble_feature_frame(
            run_dir=run_dir,
            schema=schema,
            rsrq_tolerance_s=float(checkpoint_metadata.get("rsrq_tolerance_s", 0.11)),
        )
    except ValueError as exc:
        if "serving RSRQ data" not in str(exc):
            raise
        frame = load_lstm_feature_rows(run_dir, schema)
        serving_rsrq_default = float(checkpoint_metadata["scaler"]["means"]["serving_rsrq"])
        frame["serving_rsrq"] = np.float32(serving_rsrq_default)
        frame["best_ngh_diff_rsrp"] = (
            pd.to_numeric(frame["best_ngh_rsrp"], errors="coerce")
            - pd.to_numeric(frame["serving_rsrp"], errors="coerce")
        ).astype(np.float32)
        frame["best_ngh_diff_rsrq"] = (
            pd.to_numeric(frame["best_ngh_rsrq"], errors="coerce")
            - frame["serving_rsrq"]
        ).astype(np.float32)

        for rank in range(1, MAX_CANDIDATE_K + 1):
            cell_column = candidate_cell_id_column(rank)
            rsrp_column = candidate_feature_column(rank, "candidate_rsrp")
            rsrq_column = candidate_feature_column(rank, "candidate_rsrq")
            diff_rsrp_column = candidate_feature_column(rank, "candidate_diff_rsrp")
            diff_rsrq_column = candidate_feature_column(rank, "candidate_diff_rsrq")
            rank_column = candidate_feature_column(rank, "candidate_rank_norm")
            mask_column = candidate_mask_column(rank)

            if rank == 1:
                frame[cell_column] = frame["best_ngh_cell_id"]
                frame[rsrp_column] = frame["best_ngh_rsrp"]
                frame[rsrq_column] = frame["best_ngh_rsrq"]
            elif rank == 2:
                frame[cell_column] = frame["second_ngh_cell_id"]
                frame[rsrp_column] = frame["second_ngh_rsrp"]
                frame[rsrq_column] = frame["second_ngh_rsrq"]
            else:
                frame[cell_column] = np.nan
                frame[rsrp_column] = np.nan
                frame[rsrq_column] = np.nan

            frame[diff_rsrp_column] = (
                pd.to_numeric(frame[rsrp_column], errors="coerce")
                - pd.to_numeric(frame["serving_rsrp"], errors="coerce")
            ).astype(np.float32)
            frame[diff_rsrq_column] = (
                pd.to_numeric(frame[rsrq_column], errors="coerce")
                - frame["serving_rsrq"]
            ).astype(np.float32)
            frame[mask_column] = pd.to_numeric(frame[cell_column], errors="coerce").notna().astype(np.int8)
            frame[rank_column] = np.where(
                frame[mask_column] > 0,
                np.float32(rank / MAX_CANDIDATE_K),
                np.nan,
            ).astype(np.float32)

    frame = derive_history_based_candidates(
        frame=frame,
        cell_to_index={int(k): int(v) for k, v in checkpoint_metadata["cell_to_index"].items()},
        history_len=int(checkpoint_metadata.get("candidate_history_len", checkpoint_metadata["seq_len"])),
        max_k=int(checkpoint_metadata.get("max_candidate_k", MAX_CANDIDATE_K)),
    )
    frame = map_cell_indices(frame, checkpoint_metadata)

    numeric_scaler = ScalerState(**checkpoint_metadata["scaler"])
    candidate_scaler = ScalerState(**checkpoint_metadata["candidate_scaler"])
    scaled = apply_scaler(frame, numeric_scaler, NUMERIC_FEATURE_COLUMNS)
    candidate_columns = [
        column
        for column in candidate_scaler.means.keys()
        if column in scaled.columns
    ]
    scaled = apply_scaler(scaled, candidate_scaler, candidate_columns)
    frame = frame.sort_values(["run_id", "imsi", "time"], ignore_index=True)
    scaled = scaled.sort_values(["run_id", "imsi", "time"], ignore_index=True)
    return frame, scaled


def build_latest_batches(
    raw_frame: pd.DataFrame,
    scaled_frame: pd.DataFrame,
    seq_len: int,
    candidate_top_k: int,
    num_cells: int,
) -> tuple[
    list[dict[str, Any]],
    list[pd.Series],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    records: list[dict[str, Any]] = []
    latest_raw_rows: list[pd.Series] = []
    numeric_batches: list[np.ndarray] = []
    serving_batches: list[np.ndarray] = []
    candidate_cell_batches: list[np.ndarray] = []
    candidate_mask_batches: list[np.ndarray] = []
    candidate_feature_batches: list[np.ndarray] = []

    grouped_raw = raw_frame.groupby(["run_id", "imsi"], sort=False)
    grouped_scaled = scaled_frame.groupby(["run_id", "imsi"], sort=False)

    for key, scaled_group in grouped_scaled:
        raw_group = grouped_raw.get_group(key)
        latest_raw = raw_group.iloc[-1]
        latest_scaled = scaled_group.iloc[-1]
        record = {
            "run_id": str(latest_raw["run_id"]),
            "imsi": int(latest_raw["imsi"]),
            "time": float(latest_raw["time"]),
            "serving_cell_id": int(latest_raw["serving_cell_id"]),
            "trigger_prob": 0.0,
            "target_confidence": 0.0,
            "score_margin": 0.0,
            "gain_rsrp_db": float("-inf"),
            "fallback_used": 1,
            "streak_count": 0,
            "policy_blocked": 1,
            "cooldown_remaining_s": 0.0,
            "anti_ping_remaining_s": 0.0,
            "status": "pending_inference",
            "reason": "pending_inference",
        }

        if len(scaled_group) < seq_len or int(latest_scaled["serving_cell_index"]) < 0:
            record.update(
                {
                    "target_cell_id": 0,
                    "confidence": 0.0,
                    "trigger_prob": 0.0,
                    "target_confidence": 0.0,
                    "score_margin": 0.0,
                    "gain_rsrp_db": float("-inf"),
                    "fallback_used": 1,
                    "status": "waiting",
                    "reason": "insufficient_history",
                }
            )
            records.append(record)
            continue

        window = scaled_group.tail(seq_len)
        candidate_indices = latest_scaled[
            [f"candidate_cell_index_{rank}" for rank in range(1, candidate_top_k + 1)]
        ].to_numpy(dtype=np.int64, copy=True)
        candidate_mask = latest_scaled[
            [candidate_mask_column(rank) for rank in range(1, candidate_top_k + 1)]
        ].to_numpy(dtype=np.int8, copy=True).astype(bool)
        valid_candidate_indices = candidate_mask & (candidate_indices >= 0) & (candidate_indices < num_cells)
        candidate_indices = np.where(valid_candidate_indices, candidate_indices, num_cells)
        candidate_mask = valid_candidate_indices

        feature_columns: list[str] = []
        for rank in range(1, candidate_top_k + 1):
            for base_name in CANDIDATE_FEATURE_BASENAMES:
                feature_columns.append(candidate_feature_column(rank, base_name))
        candidate_features = latest_scaled[feature_columns].to_numpy(dtype=np.float32, copy=True).reshape(
            candidate_top_k,
            len(CANDIDATE_FEATURE_BASENAMES),
        )

        serving_indices = window["serving_cell_index"].to_numpy(dtype=np.int64, copy=True)
        serving_indices = np.where(
            (serving_indices >= 0) & (serving_indices < num_cells),
            serving_indices,
            num_cells,
        )

        numeric_batches.append(window[NUMERIC_FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True))
        serving_batches.append(serving_indices)
        candidate_cell_batches.append(candidate_indices)
        candidate_mask_batches.append(candidate_mask)
        candidate_feature_batches.append(candidate_features)
        latest_raw_rows.append(latest_raw)
        records.append(record)

    if not numeric_batches:
        empty = torch.empty(0)
        return records, latest_raw_rows, empty, empty, empty, empty, empty

    return (
        records,
        latest_raw_rows,
        torch.from_numpy(np.stack(numeric_batches, axis=0)),
        torch.from_numpy(np.stack(serving_batches, axis=0)),
        torch.from_numpy(np.stack(candidate_cell_batches, axis=0)),
        torch.from_numpy(np.stack(candidate_mask_batches, axis=0)),
        torch.from_numpy(np.stack(candidate_feature_batches, axis=0)),
    )


def extract_gain_from_latest_row(
    row: pd.Series,
    target_cell_id: int,
) -> float:
    for rank in range(1, MAX_CANDIDATE_K + 1):
        cell_value = pd.to_numeric(row.get(candidate_cell_id_column(rank)), errors="coerce")
        if pd.notna(cell_value) and int(cell_value) == int(target_cell_id):
            gain_value = pd.to_numeric(
                row.get(candidate_feature_column(rank, "candidate_diff_rsrp")),
                errors="coerce",
            )
            if pd.notna(gain_value):
                return float(gain_value)
    return float("-inf")


def infer_latest_predictions(
    raw_frame: pd.DataFrame,
    scaled_frame: pd.DataFrame,
    model: MultitaskLstmPredictor,
    device: torch.device,
    seq_len: int,
    candidate_top_k: int,
    checkpoint_metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    index_to_cell = {int(index): int(cell_id) for index, cell_id in checkpoint_metadata["index_to_cell"].items()}
    num_cells = len(checkpoint_metadata["cell_ids"])
    (
        records,
        latest_raw_rows,
        numeric,
        serving_cell,
        candidate_cell,
        candidate_mask,
        candidate_features,
    ) = build_latest_batches(
        raw_frame=raw_frame,
        scaled_frame=scaled_frame,
        seq_len=seq_len,
        candidate_top_k=candidate_top_k,
        num_cells=num_cells,
    )

    valid_indices = [
        index
        for index, record in enumerate(records)
        if record["status"] == "pending_inference"
    ]
    if not valid_indices or numeric.numel() == 0:
        return records

    numeric = numeric.to(device)
    serving_cell = serving_cell.to(device)
    candidate_cell = candidate_cell.to(device)
    candidate_mask = candidate_mask.to(device)
    candidate_features = candidate_features.to(device)

    with torch.no_grad():
        outputs = model(
            numeric=numeric,
            serving_cell=serving_cell,
            candidate_cell=candidate_cell,
            candidate_features=candidate_features,
            candidate_mask=candidate_mask,
        )

    trigger_prob = torch.sigmoid(outputs["trigger_logits"]).detach().cpu().numpy().astype(np.float32)
    candidate_probs = torch.softmax(outputs["candidate_logits"], dim=1)
    global_probs = torch.softmax(outputs["global_target_logits"], dim=1)
    candidate_choice = torch.argmax(outputs["candidate_logits"], dim=1)
    global_choice = torch.argmax(outputs["global_target_logits"], dim=1)
    row_index = torch.arange(candidate_choice.shape[0], device=device)
    chosen_candidate_index = candidate_cell[row_index, candidate_choice]
    has_candidate = candidate_mask.any(dim=1)
    target_index = torch.where(has_candidate, chosen_candidate_index, global_choice)
    fallback_used = (~has_candidate).detach().cpu().numpy().astype(np.int8)

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
    target_indices = target_index.detach().cpu().numpy().astype(np.int64)
    target_cells = [index_to_cell.get(int(index), -1) for index in target_indices]

    for batch_index, record in enumerate([records[index] for index in valid_indices]):
        target_cell_id = int(target_cells[batch_index])
        raw_row = latest_raw_rows[batch_index]
        record.update(
            {
                "target_cell_id": target_cell_id,
                "confidence": float(target_conf[batch_index]),
                "trigger_prob": float(trigger_prob[batch_index]),
                "target_confidence": float(target_conf[batch_index]),
                "score_margin": float(score_margin[batch_index]),
                "gain_rsrp_db": extract_gain_from_latest_row(raw_row, target_cell_id),
                "fallback_used": int(fallback_used[batch_index]),
                "status": "scored",
                "reason": "candidate_model",
            }
        )
    return records


def apply_conservative_policy(
    predictions: list[dict[str, Any]],
    state: dict[str, Any],
    policy: ConservativeOnlinePolicy,
) -> list[dict[str, Any]]:
    per_imsi_state = state.setdefault("per_imsi", {})
    decisions: list[dict[str, Any]] = []

    for prediction in predictions:
        imsi_key = str(int(prediction["imsi"]))
        imsi_state = per_imsi_state.setdefault(
            imsi_key,
            {
                "last_serving_cell_id": 0,
                "streak_target": -1,
                "streak_count": 0,
                "cooldown_until": float("-inf"),
                "blocked_returns": {},
            },
        )
        current_serving = int(prediction["serving_cell_id"])
        previous_serving = int(imsi_state.get("last_serving_cell_id", 0))
        current_time = float(prediction.get("time", 0.0))
        blocked_returns = {
            int(cell_id): float(expiry)
            for cell_id, expiry in dict(imsi_state.get("blocked_returns", {})).items()
        }
        blocked_returns = {
            int(cell_id): float(expiry)
            for cell_id, expiry in blocked_returns.items()
            if float(expiry) > current_time
        }
        if previous_serving > 0 and previous_serving != current_serving:
            if policy.anti_ping_pong_window_s > 0.0:
                blocked_returns[previous_serving] = current_time + policy.anti_ping_pong_window_s
            imsi_state["streak_target"] = -1
            imsi_state["streak_count"] = 0

        status = prediction["status"]
        reason = prediction["reason"]
        target_cell_id = int(prediction.get("target_cell_id", 0))
        confidence = float(prediction.get("confidence", 0.0))
        cooldown_until = float(imsi_state.get("cooldown_until", float("-inf")))
        cooldown_remaining = max(0.0, cooldown_until - current_time)
        anti_ping_remaining = max(0.0, blocked_returns.get(target_cell_id, float("-inf")) - current_time)

        if status == "scored":
            if cooldown_remaining > 0.0:
                status = "hold"
                reason = "cooldown_active"
                imsi_state["streak_target"] = -1
                imsi_state["streak_count"] = 0
            elif anti_ping_remaining > 0.0:
                status = "hold"
                reason = "anti_ping_pong_guard"
                imsi_state["streak_target"] = -1
                imsi_state["streak_count"] = 0
            elif float(prediction["trigger_prob"]) < policy.trigger_threshold:
                status = "hold"
                reason = "trigger_below_threshold"
                imsi_state["streak_target"] = -1
                imsi_state["streak_count"] = 0
            elif float(prediction["target_confidence"]) < policy.target_conf_threshold:
                status = "hold"
                reason = "target_conf_below_threshold"
                imsi_state["streak_target"] = -1
                imsi_state["streak_count"] = 0
            elif float(prediction["score_margin"]) < policy.min_score_margin:
                status = "hold"
                reason = "score_margin_below_threshold"
                imsi_state["streak_target"] = -1
                imsi_state["streak_count"] = 0
            elif float(prediction["gain_rsrp_db"]) < policy.min_gain_rsrp_db:
                status = "hold"
                reason = "gain_below_threshold"
                imsi_state["streak_target"] = -1
                imsi_state["streak_count"] = 0
            elif target_cell_id <= 0:
                status = "hold"
                reason = "invalid_target"
                imsi_state["streak_target"] = -1
                imsi_state["streak_count"] = 0
            elif target_cell_id == current_serving:
                status = "hold"
                reason = "same_as_serving"
                imsi_state["streak_target"] = -1
                imsi_state["streak_count"] = 0
            else:
                if int(imsi_state.get("streak_target", -1)) == target_cell_id:
                    imsi_state["streak_count"] = int(imsi_state.get("streak_count", 0)) + 1
                else:
                    imsi_state["streak_target"] = target_cell_id
                    imsi_state["streak_count"] = 1

                if int(imsi_state["streak_count"]) >= int(policy.consecutive_confirmation_steps):
                    status = "ready"
                    reason = "conservative_k3_ready"
                    cooldown_until = current_time + policy.cooldown_s
                    if policy.anti_ping_pong_window_s > 0.0 and current_serving > 0:
                        blocked_returns[current_serving] = current_time + policy.anti_ping_pong_window_s
                else:
                    status = "hold"
                    reason = (
                        f"confirmation_{int(imsi_state['streak_count'])}"
                        f"_of_{int(policy.consecutive_confirmation_steps)}"
                    )

        prediction["status"] = status
        prediction["reason"] = reason
        prediction["confidence"] = confidence if status == "ready" else float(prediction.get("confidence", 0.0))
        prediction["streak_count"] = int(imsi_state.get("streak_count", 0))
        prediction["policy_blocked"] = int(status != "ready")
        prediction["cooldown_remaining_s"] = float(max(0.0, cooldown_until - current_time)) if status != "ready" else 0.0
        prediction["anti_ping_remaining_s"] = (
            float(max(0.0, blocked_returns.get(target_cell_id, float("-inf")) - current_time))
            if target_cell_id > 0 and status != "ready"
            else 0.0
        )
        imsi_state["cooldown_until"] = float(cooldown_until)
        imsi_state["last_serving_cell_id"] = current_serving
        imsi_state["blocked_returns"] = {
            str(int(cell_id)): float(expiry) for cell_id, expiry in blocked_returns.items()
        }
        decisions.append(prediction)

    return decisions


def write_decision_csv(path: Path, decisions: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "time": [float(item.get("time", 0.0)) for item in decisions],
            "imsi": [int(item["imsi"]) for item in decisions],
            "servingCellId": [int(item["serving_cell_id"]) for item in decisions],
            "targetCellId": [int(item.get("target_cell_id", 0)) for item in decisions],
            "confidence": [float(item.get("confidence", 0.0)) for item in decisions],
            "triggerProb": [float(item.get("trigger_prob", 0.0)) for item in decisions],
            "targetConfidence": [float(item.get("target_confidence", 0.0)) for item in decisions],
            "scoreMargin": [float(item.get("score_margin", 0.0)) for item in decisions],
            "gainRsrpDb": [float(item.get("gain_rsrp_db", float("-inf"))) for item in decisions],
            "fallbackUsed": [int(item.get("fallback_used", 0)) for item in decisions],
            "streakCount": [int(item.get("streak_count", 0)) for item in decisions],
            "policyBlocked": [int(item.get("policy_blocked", 0)) for item in decisions],
            "cooldownRemainingS": [float(item.get("cooldown_remaining_s", 0.0)) for item in decisions],
            "antiPingRemainingS": [float(item.get("anti_ping_remaining_s", 0.0)) for item in decisions],
            "status": [str(item.get("status", "hold")) for item in decisions],
            "reason": [str(item.get("reason", "unknown")).replace(",", "_") for item in decisions],
        }
    )
    frame.to_csv(path, index=False)


def main() -> int:
    args = build_argument_parser().parse_args()
    state_path = args.state_path or (args.output_path.parent / "lstm-runtime-state.json")
    state = load_runtime_state(state_path)

    device = torch.device(args.device)
    checkpoint, model, candidate_top_k = load_checkpoint(args.checkpoint_path, device)
    checkpoint_metadata = checkpoint["metadata"]

    raw_frame, scaled_frame = prepare_runtime_frame(args.db_path, checkpoint_metadata)

    predictions = infer_latest_predictions(
        raw_frame=raw_frame,
        scaled_frame=scaled_frame,
        model=model,
        device=device,
        seq_len=args.seq_len,
        candidate_top_k=candidate_top_k,
        checkpoint_metadata=checkpoint_metadata,
    )

    policy = ConservativeOnlinePolicy(
        trigger_threshold=float(args.trigger_threshold),
        target_conf_threshold=float(args.target_threshold),
        min_score_margin=float(args.utility_threshold),
        min_gain_rsrp_db=float(args.min_gain_rsrp_db),
        consecutive_confirmation_steps=int(args.consecutive_confirmation_steps),
        cooldown_s=float(args.cooldown_s),
        anti_ping_pong_window_s=float(args.anti_ping_pong_window_s),
    )
    if args.policy_mode == "raw":
        decisions = predictions
        for item in decisions:
            item["policy_blocked"] = int(str(item.get("status", "")) != "scored")
            item["streak_count"] = int(item.get("streak_count", 0))
            item["cooldown_remaining_s"] = float(item.get("cooldown_remaining_s", 0.0))
            item["anti_ping_remaining_s"] = float(item.get("anti_ping_remaining_s", 0.0))
    else:
        decisions = apply_conservative_policy(predictions, state, policy)
        save_runtime_state(state_path, state)
    write_decision_csv(args.output_path, decisions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
