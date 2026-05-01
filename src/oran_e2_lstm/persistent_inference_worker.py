#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import pandas as pd
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (
    MAX_CANDIDATE_K,
    NUMERIC_FEATURE_COLUMNS,
    ScalerState,
    apply_scaler,
    candidate_cell_id_column,
    candidate_feature_column,
    candidate_numeric_columns,
    derive_history_based_candidates,
)
from online_runtime_infer import infer_latest_predictions, load_checkpoint, map_cell_indices


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Persistent stdio worker for online raw LSTM inference.",
    )
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=15)
    parser.add_argument("--device", type=str, default="cpu")
    return parser


def build_candidate_seed_frame(frame: pd.DataFrame) -> pd.DataFrame:
    seeded = frame.copy()
    for rank in range(1, MAX_CANDIDATE_K + 1):
        seeded[candidate_cell_id_column(rank)] = np.nan
        seeded[candidate_feature_column(rank, "candidate_rsrp")] = np.nan
        seeded[candidate_feature_column(rank, "candidate_rsrq")] = np.nan

    seeded[candidate_cell_id_column(1)] = seeded["best_ngh_cell_id"]
    seeded[candidate_feature_column(1, "candidate_rsrp")] = seeded["best_ngh_rsrp"]
    seeded[candidate_feature_column(1, "candidate_rsrq")] = seeded["best_ngh_rsrq"]
    seeded[candidate_cell_id_column(2)] = seeded["second_ngh_cell_id"]
    seeded[candidate_feature_column(2, "candidate_rsrp")] = seeded["second_ngh_rsrp"]
    seeded[candidate_feature_column(2, "candidate_rsrq")] = seeded["second_ngh_rsrq"]
    return seeded


def build_raw_frame(items: list[dict[str, Any]], run_id: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in items:
        imsi = int(item["imsi"])
        ue_id = int(item["ue_id"])
        nodeid = int(item["nodeid"])
        for row in item["window"]:
            rows.append(
                {
                    "run_id": run_id,
                    "time": float(row["time"]),
                    "imsi": imsi,
                    "ue_id": ue_id,
                    "nodeid": nodeid,
                    "serving_cell_id": int(row["serving_cell_id"]),
                    "serving_rsrp": float(row["serving_rsrp"]),
                    "serving_rsrq": float(row["serving_rsrq"]),
                    "serving_sinr": float(row["serving_sinr"]),
                    "best_ngh_cell_id": int(row["best_ngh_cell_id"]),
                    "best_ngh_rsrp": float(row["best_ngh_rsrp"]),
                    "best_ngh_rsrq": float(row["best_ngh_rsrq"]),
                    "second_ngh_cell_id": int(row["second_ngh_cell_id"]),
                    "second_ngh_rsrp": float(row["second_ngh_rsrp"]),
                    "second_ngh_rsrq": float(row["second_ngh_rsrq"]),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "run_id",
                "time",
                "imsi",
                "ue_id",
                "nodeid",
                "serving_cell_id",
                "serving_rsrp",
                "serving_rsrq",
                "serving_sinr",
                "best_ngh_cell_id",
                "best_ngh_rsrp",
                "best_ngh_rsrq",
                "second_ngh_cell_id",
                "second_ngh_rsrp",
                "second_ngh_rsrq",
            ]
        )

    frame = pd.DataFrame(rows)
    int_columns = ["imsi", "ue_id", "nodeid", "serving_cell_id", "best_ngh_cell_id", "second_ngh_cell_id"]
    for column in int_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0).astype(np.int32)
    float_columns = [
        "time",
        "serving_rsrp",
        "serving_rsrq",
        "serving_sinr",
        "best_ngh_rsrp",
        "best_ngh_rsrq",
        "second_ngh_rsrp",
        "second_ngh_rsrq",
    ]
    for column in float_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(np.float32)
    return frame.sort_values(["run_id", "imsi", "time"], ignore_index=True)


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


def parse_request(stdin: TextIO) -> dict[str, Any] | None:
    header = stdin.readline()
    if not header:
        return None
    header = header.strip()
    if not header:
        return None
    parts = header.split()
    command = parts[0]
    if command == "PING":
        return {"type": "ping"}
    if command == "SHUTDOWN":
        return {"type": "shutdown"}
    if command != "INFER" or len(parts) < 4:
        raise ValueError(f"Unsupported worker command: {header}")

    request_id = parts[1]
    ue_count = int(parts[2])
    seq_len = int(parts[3])
    items: list[dict[str, Any]] = []
    for _ in range(ue_count):
        ue_line = stdin.readline()
        if not ue_line:
            raise ValueError("Unexpected EOF while reading UE header")
        ue_parts = ue_line.strip().split()
        if len(ue_parts) != 7 or ue_parts[0] != "UE":
            raise ValueError(f"Malformed UE line: {ue_line.strip()}")
        imsi = int(ue_parts[1])
        ue_id = int(ue_parts[2])
        nodeid = int(ue_parts[3])
        latest_time = float(ue_parts[4])
        serving_cell_id = int(ue_parts[5])
        window_len = int(ue_parts[6])
        window: list[dict[str, Any]] = []
        for _ in range(window_len):
            row_line = stdin.readline()
            if not row_line:
                raise ValueError("Unexpected EOF while reading ROW line")
            row_parts = row_line.strip().split()
            if len(row_parts) != 12 or row_parts[0] != "ROW":
                raise ValueError(f"Malformed ROW line: {row_line.strip()}")
            window.append(
                {
                    "time": float(row_parts[1]),
                    "serving_cell_id": int(row_parts[2]),
                    "serving_rsrp": float(row_parts[3]),
                    "serving_rsrq": float(row_parts[4]),
                    "serving_sinr": float(row_parts[5]),
                    "best_ngh_cell_id": int(row_parts[6]),
                    "best_ngh_rsrp": float(row_parts[7]),
                    "best_ngh_rsrq": float(row_parts[8]),
                    "second_ngh_cell_id": int(row_parts[9]),
                    "second_ngh_rsrp": float(row_parts[10]),
                    "second_ngh_rsrq": float(row_parts[11]),
                }
            )
        items.append(
            {
                "imsi": imsi,
                "ue_id": ue_id,
                "nodeid": nodeid,
                "latest_time": latest_time,
                "serving_cell_id": serving_cell_id,
                "window": window,
            }
        )

    end_line = stdin.readline()
    if not end_line or end_line.strip() != "END":
        raise ValueError("Missing END marker after INFER request")

    return {
        "type": "infer",
        "request_id": request_id,
        "ue_count": ue_count,
        "seq_len": seq_len,
        "items": items,
    }


def emit_ok_response(
    request_id: str,
    predictions: list[dict[str, Any]],
    worker_latency_ms: float,
) -> None:
    sys.stdout.write(f"RESULT {request_id} {len(predictions)} {worker_latency_ms:.3f} OK\n")
    for prediction in predictions:
        status = str(prediction.get("status", "unknown")).replace(" ", "_")
        reason = str(prediction.get("reason", "unknown")).replace(" ", "_")
        sys.stdout.write(
            "PRED "
            f"{int(prediction['imsi'])} "
            f"{float(prediction.get('time', 0.0)):.6f} "
            f"{int(prediction.get('serving_cell_id', 0))} "
            f"{int(prediction.get('target_cell_id', 0))} "
            f"{float(prediction.get('confidence', 0.0)):.6f} "
            f"{float(prediction.get('trigger_prob', 0.0)):.6f} "
            f"{float(prediction.get('target_confidence', 0.0)):.6f} "
            f"{float(prediction.get('score_margin', 0.0)):.6f} "
            f"{float(prediction.get('gain_rsrp_db', float('-inf'))):.6f} "
            f"{int(prediction.get('fallback_used', 0))} "
            f"{status} "
            f"{reason}\n"
        )
    sys.stdout.write("END\n")
    sys.stdout.flush()


def emit_error_response(request_id: str, message: str, worker_latency_ms: float = 0.0) -> None:
    safe_message = message.replace(" ", "_")
    sys.stdout.write(f"RESULT {request_id} 0 {worker_latency_ms:.3f} ERROR {safe_message}\n")
    sys.stdout.write("END\n")
    sys.stdout.flush()


def main() -> int:
    args = build_argument_parser().parse_args()
    device = torch.device(args.device)
    checkpoint, model, candidate_top_k = load_checkpoint(args.checkpoint_path, device)
    numeric_scaler = ScalerState(**checkpoint["metadata"]["scaler"])
    candidate_scaler = ScalerState(**checkpoint["metadata"]["candidate_scaler"])

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
                run_id=f"worker_req_{request_counter}",
                seq_len=int(request["seq_len"]),
                numeric_scaler=numeric_scaler,
                candidate_scaler=candidate_scaler,
                checkpoint_metadata=checkpoint["metadata"],
            )
            predictions = infer_latest_predictions(
                raw_frame=raw_frame,
                scaled_frame=scaled_frame,
                model=model,
                device=device,
                seq_len=int(request["seq_len"]),
                candidate_top_k=candidate_top_k,
                checkpoint_metadata=checkpoint["metadata"],
            )
            worker_latency_ms = (time.perf_counter() - started_at) * 1000.0
            emit_ok_response(request_id, predictions, worker_latency_ms)
        except Exception as exc:  # noqa: BLE001
            worker_latency_ms = (time.perf_counter() - started_at) * 1000.0
            print(f"worker_error request_id={request_id} error={exc}", file=sys.stderr, flush=True)
            emit_error_response(request_id, f"{type(exc).__name__}:{exc}", worker_latency_ms)


if __name__ == "__main__":
    raise SystemExit(main())
