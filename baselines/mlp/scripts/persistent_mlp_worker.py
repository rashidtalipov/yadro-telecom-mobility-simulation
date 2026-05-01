#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    CANDIDATE_FEATURE_BASENAMES,
    MAX_CANDIDATE_K,
    NUMERIC_FEATURE_COLUMNS,
    ScalerState,
    apply_scaler,
    candidate_cell_id_column,
    candidate_feature_column,
    candidate_mask_column,
    candidate_numeric_columns,
    derive_history_based_candidates,
)
from mlp_baseline import CandidateAwareMlp, MlpConfig  # noqa: E402
from online_runtime_infer import extract_gain_from_latest_row, map_cell_indices  # noqa: E402
from persistent_inference_worker import (  # noqa: E402
    build_candidate_seed_frame,
    build_raw_frame,
    emit_error_response,
    emit_ok_response,
    parse_request,
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Persistent stdio worker for last-step MLP inference.")
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=15)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--min-history",
        type=int,
        default=None,
        help="Minimum rows per UE before scoring. Defaults to --seq-len for parity with LSTM runs.",
    )
    return parser


def load_mlp_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[dict[str, Any], CandidateAwareMlp, int]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("model_type") != "candidate_aware_last_step_mlp":
        raise ValueError(f"{checkpoint_path} is not a candidate-aware MLP checkpoint")
    config = MlpConfig(**checkpoint["model_config"])
    model = CandidateAwareMlp(config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    candidate_top_k = int(checkpoint["training_args"]["candidate_top_k"])
    return checkpoint, model, candidate_top_k


def prepare_frames(
    items: list[dict[str, Any]],
    run_id: str,
    seq_len: int,
    numeric_scaler: ScalerState,
    candidate_scaler: ScalerState,
    checkpoint_metadata: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = build_raw_frame(items, run_id)
    if raw.empty:
        return raw.copy(), raw.copy()

    seeded = build_candidate_seed_frame(raw)
    seeded["best_ngh_diff_rsrp"] = (
        pd.to_numeric(seeded["best_ngh_rsrp"], errors="coerce")
        - pd.to_numeric(seeded["serving_rsrp"], errors="coerce")
    ).astype(np.float32)
    seeded["best_ngh_diff_rsrq"] = (
        pd.to_numeric(seeded["best_ngh_rsrq"], errors="coerce")
        - pd.to_numeric(seeded["serving_rsrq"], errors="coerce")
    ).astype(np.float32)
    cell_to_index = {
        int(cell_id): int(index) for cell_id, index in checkpoint_metadata["cell_to_index"].items()
    }
    with_history = derive_history_based_candidates(
        seeded,
        cell_to_index=cell_to_index,
        history_len=seq_len,
        max_k=MAX_CANDIDATE_K,
    )
    mapped = map_cell_indices(with_history, checkpoint_metadata)
    scaled = apply_scaler(mapped, scaler=numeric_scaler, numeric_columns=NUMERIC_FEATURE_COLUMNS)
    candidate_columns = [column for column in candidate_numeric_columns() if column in scaled.columns]
    scaled = apply_scaler(scaled, scaler=candidate_scaler, numeric_columns=candidate_columns)
    return with_history, scaled


def infer_latest_predictions(
    raw_frame: pd.DataFrame,
    scaled_frame: pd.DataFrame,
    model: CandidateAwareMlp,
    device: torch.device,
    candidate_top_k: int,
    checkpoint_metadata: dict[str, Any],
    min_history: int,
) -> list[dict[str, Any]]:
    index_to_cell = {int(index): int(cell_id) for index, cell_id in checkpoint_metadata["index_to_cell"].items()}
    num_cells = len(checkpoint_metadata["cell_ids"])

    records: list[dict[str, Any]] = []
    latest_rows: list[pd.Series] = []
    numeric_batches: list[np.ndarray] = []
    serving_batches: list[int] = []
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
            "target_cell_id": 0,
            "confidence": 0.0,
            "trigger_prob": 0.0,
            "target_confidence": 0.0,
            "score_margin": 0.0,
            "gain_rsrp_db": float("-inf"),
            "fallback_used": 1,
            "status": "pending_inference",
            "reason": "pending_inference",
        }
        if len(scaled_group) < min_history or int(latest_scaled["serving_cell_index"]) < 0:
            record.update({"status": "waiting", "reason": "insufficient_history"})
            records.append(record)
            continue

        candidate_indices = latest_scaled[
            [f"candidate_cell_index_{rank}" for rank in range(1, candidate_top_k + 1)]
        ].to_numpy(dtype=np.int64, copy=True)
        candidate_mask = latest_scaled[
            [candidate_mask_column(rank) for rank in range(1, candidate_top_k + 1)]
        ].to_numpy(dtype=np.int8, copy=True).astype(bool)
        valid_candidate = candidate_mask & (candidate_indices >= 0) & (candidate_indices < num_cells)
        candidate_indices = np.where(valid_candidate, candidate_indices, num_cells)
        candidate_mask = valid_candidate

        feature_columns: list[str] = []
        for rank in range(1, candidate_top_k + 1):
            for basename in CANDIDATE_FEATURE_BASENAMES:
                feature_columns.append(candidate_feature_column(rank, basename))

        candidate_features = latest_scaled[feature_columns].to_numpy(dtype=np.float32, copy=True).reshape(
            candidate_top_k,
            len(CANDIDATE_FEATURE_BASENAMES),
        )
        serving_cell = int(latest_scaled["serving_cell_index"])
        serving_cell = serving_cell if 0 <= serving_cell < num_cells else num_cells

        records.append(record)
        latest_rows.append(latest_raw)
        numeric_batches.append(latest_scaled[NUMERIC_FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True))
        serving_batches.append(serving_cell)
        candidate_cell_batches.append(candidate_indices)
        candidate_mask_batches.append(candidate_mask)
        candidate_feature_batches.append(candidate_features)

    valid_records = [record for record in records if record["status"] == "pending_inference"]
    if not valid_records:
        return records

    numeric = torch.from_numpy(np.stack(numeric_batches, axis=0)).to(device)
    serving_cell = torch.from_numpy(np.asarray(serving_batches, dtype=np.int64)).to(device)
    candidate_cell = torch.from_numpy(np.stack(candidate_cell_batches, axis=0)).to(device)
    candidate_mask = torch.from_numpy(np.stack(candidate_mask_batches, axis=0)).to(device)
    candidate_features = torch.from_numpy(np.stack(candidate_feature_batches, axis=0)).to(device)

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

    batch_index = 0
    for record in records:
        if record["status"] != "pending_inference":
            continue
        target_cell_id = int(target_cells[batch_index])
        raw_row = latest_rows[batch_index]
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
                "reason": "mlp_candidate_model",
            }
        )
        batch_index += 1
    return records


def main() -> int:
    args = build_argument_parser().parse_args()
    device = torch.device(args.device)
    checkpoint, model, candidate_top_k = load_mlp_checkpoint(args.checkpoint_path, device)
    metadata = checkpoint["metadata"]
    numeric_scaler = ScalerState(**metadata["scaler"])
    candidate_scaler = ScalerState(**metadata["candidate_scaler"])
    min_history = int(args.min_history if args.min_history is not None else args.seq_len)

    sys.stdout.write(f"READY {args.seq_len} {candidate_top_k}\n")
    sys.stdout.flush()

    request_counter = 0
    while True:
        request = parse_request(sys.stdin)
        if request is None:
            return 0
        if request["type"] == "ping":
            sys.stdout.write("PONG\n")
            sys.stdout.flush()
            continue
        if request["type"] == "shutdown":
            sys.stdout.write("BYE\n")
            sys.stdout.flush()
            return 0

        request_id = str(request["request_id"])
        request_counter += 1
        started_at = time.perf_counter()
        try:
            raw_frame, scaled_frame = prepare_frames(
                items=request["items"],
                run_id=f"mlp_worker_req_{request_counter}",
                seq_len=int(request["seq_len"]),
                numeric_scaler=numeric_scaler,
                candidate_scaler=candidate_scaler,
                checkpoint_metadata=metadata,
            )
            predictions = infer_latest_predictions(
                raw_frame=raw_frame,
                scaled_frame=scaled_frame,
                model=model,
                device=device,
                candidate_top_k=candidate_top_k,
                checkpoint_metadata=metadata,
                min_history=min_history,
            )
            worker_latency_ms = (time.perf_counter() - started_at) * 1000.0
            emit_ok_response(request_id, predictions, worker_latency_ms)
        except Exception as exc:  # noqa: BLE001
            worker_latency_ms = (time.perf_counter() - started_at) * 1000.0
            print(f"mlp_worker_error request_id={request_id} error={exc}", file=sys.stderr, flush=True)
            emit_error_response(request_id, f"{type(exc).__name__}:{exc}", worker_latency_ms)


if __name__ == "__main__":
    raise SystemExit(main())
