#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(os.environ.get("NS3_ROOT", "/path/to/ns-allinone-3.46.1/ns-3.46.1"))
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results_night/oran_e2_lstm/results_final_900s"
DEFAULT_NS3_BINARY = PROJECT_ROOT / "build/optimized/scratch/ns3.46.1-lte-oran-helper-lstm-only-hex7-optimized"
DEFAULT_BASELINE_ROOT = PROJECT_ROOT / "results_night_teacher_100ms"
DEFAULT_WORKER_PYTHON = PROJECT_ROOT / "results_night/.venv/bin/python"
DEFAULT_WORKER_SCRIPT = PROJECT_ROOT / "results_night/oran_e2_lstm/persistent_inference_worker.py"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "results_night/oran_e2_lstm/runs/candidate_history_k3_20ep/best_model.pt"

PDCP_RLC_COLUMNS = [
    "start",
    "end",
    "CellId",
    "IMSI",
    "RNTI",
    "LCID",
    "nTxPDUs",
    "TxBytes",
    "nRxPDUs",
    "RxBytes",
    "delay",
    "delayStdDev",
    "delayMin",
    "delayMax",
    "PduSize",
    "PduSizeStdDev",
    "PduSizeMin",
    "PduSizeMax",
]

POLICY_PROFILE_A = {
    "trigger_threshold": 0.70,
    "target_conf_threshold": 0.70,
    "min_gain_rsrp_db": 1.0,
    "consecutive_confirmation_steps": 2,
    "cooldown_s": 1.0,
    "anti_ping_pong_window_s": 2.0,
}

MODE_LABELS = {
    "a3": "A3",
    "lstm_only": "LSTM-only",
    "lstm_hybrid": "LSTM+A3 hybrid",
}


@dataclass(frozen=True)
class ModeSpec:
    name: str
    label: str
    use_lte_handover: bool
    description: str


@dataclass
class Job:
    mode: ModeSpec
    seed: int
    run: int
    output_root: Path
    log_path: Path
    run_dir: Path
    command: list[str]
    process: subprocess.Popen[str] | None = None
    status: str = "pending"
    exit_code: int | None = None
    launch_time_wall: float | None = None
    end_time_wall: float | None = None


FINAL_MODES = [
    ModeSpec(
        name="lstm_only",
        label="LSTM-only",
        use_lte_handover=False,
        description="Pure candidate-aware LSTM controller without LTE A3 handover safety net.",
    ),
    ModeSpec(
        name="lstm_hybrid",
        label="LSTM+A3 hybrid",
        use_lte_handover=True,
        description="Candidate-aware LSTM controller with LTE A3 fallback enabled.",
    ),
]


def detect_default_parallelism() -> int:
    cpu_count = os.cpu_count() or 8
    return max(1, min(4, cpu_count // 8 if cpu_count >= 8 else 1))


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run final matched 900 s ns-3 experiments for A3, LSTM-only, and LSTM+A3 hybrid.",
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--ns3-binary", type=Path, default=DEFAULT_NS3_BINARY)
    parser.add_argument("--worker-python", type=Path, default=DEFAULT_WORKER_PYTHON)
    parser.add_argument("--worker-script", type=Path, default=DEFAULT_WORKER_SCRIPT)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--runs", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--sim-time", type=float, default=900.0)
    parser.add_argument("--max-parallel", type=int, default=detect_default_parallelism())
    parser.add_argument("--poll-interval-s", type=float, default=60.0)
    parser.add_argument("--delivery-margin-abs", type=float, default=0.03)
    parser.add_argument("--throughput-margin-rel", type=float, default=0.10)
    parser.add_argument("--skip-cleanup", action="store_true")
    parser.add_argument("--keep-heavy-traces", action="store_true")
    parser.add_argument("--refresh-only", action="store_true")
    return parser


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


def latex_escape(value: str) -> str:
    replacements = {
        "\\": "\\textbackslash{}",
        "_": "\\_",
        "%": "\\%",
        "&": "\\&",
        "#": "\\#",
        "{": "\\{",
        "}": "\\}",
        "$": "\\$",
    }
    escaped = value
    for old, new in replacements.items():
        escaped = escaped.replace(old, new)
    return escaped


def log_line(log_path: Path, message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def read_run_info(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def read_trace(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path, sep=r"\s+", engine="python")


def filter_time(frame: pd.DataFrame, time_limit_s: float | None) -> pd.DataFrame:
    if frame.empty or time_limit_s is None or "time" not in frame.columns:
        return frame
    return frame[pd.to_numeric(frame["time"], errors="coerce") <= float(time_limit_s)].copy()


def read_bearer_stats(path: Path, time_limit_s: float | None) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=PDCP_RLC_COLUMNS)
    frame = pd.read_csv(
        path,
        sep=r"\s+",
        engine="python",
        comment="%",
        names=PDCP_RLC_COLUMNS,
    )
    if frame.empty:
        return frame

    for column in ["start", "end", "nTxPDUs", "TxBytes", "nRxPDUs", "RxBytes", "delay", "delayStdDev"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)
    if time_limit_s is not None:
        frame = frame[frame["start"] < float(time_limit_s)].copy()
    return frame


def weighted_average(values: pd.Series, weights: pd.Series) -> float:
    valid = np.isfinite(values.to_numpy(dtype=float)) & np.isfinite(weights.to_numpy(dtype=float))
    if not valid.any():
        return float("nan")
    filtered_values = values.to_numpy(dtype=float)[valid]
    filtered_weights = weights.to_numpy(dtype=float)[valid]
    weight_sum = filtered_weights.sum()
    if weight_sum <= 0:
        return float("nan")
    return float(np.average(filtered_values, weights=filtered_weights))


def summarize_bearer_file(path: Path, time_limit_s: float | None) -> dict[str, Any]:
    frame = read_bearer_stats(path, time_limit_s)
    if frame.empty:
        return {
            "present": path.exists(),
            "tx_bytes": 0,
            "rx_bytes": 0,
            "delay_mean_ms": float("nan"),
            "delay_std_ms": float("nan"),
            "delay_samples": 0.0,
        }

    rx_weight = frame["nRxPDUs"].clip(lower=0)
    delay_mean_ms = weighted_average(frame["delay"], rx_weight)
    delay_std_ms = weighted_average(frame["delayStdDev"], rx_weight)
    if math.isfinite(delay_mean_ms):
        delay_mean_ms *= 1000.0
    if math.isfinite(delay_std_ms):
        delay_std_ms *= 1000.0

    return {
        "present": True,
        "tx_bytes": int(frame["TxBytes"].sum()),
        "rx_bytes": int(frame["RxBytes"].sum()),
        "delay_mean_ms": delay_mean_ms,
        "delay_std_ms": delay_std_ms,
        "delay_samples": float(rx_weight.sum()),
    }


def compute_transport_metrics(run_dir: Path, time_limit_s: float | None) -> dict[str, Any]:
    dl_pdcp = summarize_bearer_file(run_dir / "DlPdcpStats.txt", time_limit_s)
    ul_pdcp = summarize_bearer_file(run_dir / "UlPdcpStats.txt", time_limit_s)
    dl_rlc = summarize_bearer_file(run_dir / "DlRlcStats.txt", time_limit_s)
    ul_rlc = summarize_bearer_file(run_dir / "UlRlcStats.txt", time_limit_s)

    use_pdcp = dl_pdcp["present"] or ul_pdcp["present"]
    dl_stats = dl_pdcp if use_pdcp else dl_rlc
    ul_stats = ul_pdcp if use_pdcp else ul_rlc
    source = "pdcp" if use_pdcp else ("rlc" if (dl_rlc["present"] or ul_rlc["present"]) else "none")

    dl_tx_bytes = int(dl_stats["tx_bytes"])
    dl_rx_bytes = int(dl_stats["rx_bytes"])
    ul_tx_bytes = int(ul_stats["tx_bytes"])
    ul_rx_bytes = int(ul_stats["rx_bytes"])

    return {
        "traffic_metric_source": source,
        "dl_tx_bytes": dl_tx_bytes,
        "dl_rx_bytes": dl_rx_bytes,
        "ul_tx_bytes": ul_tx_bytes,
        "ul_rx_bytes": ul_rx_bytes,
        "dl_delay_mean_ms": float(dl_stats["delay_mean_ms"]),
        "dl_delay_std_ms": float(dl_stats["delay_std_ms"]),
        "ul_delay_mean_ms": float(ul_stats["delay_mean_ms"]),
        "ul_delay_std_ms": float(ul_stats["delay_std_ms"]),
    }


def compute_dwell_time(handover_end: pd.DataFrame, sim_time_s: float) -> float:
    if handover_end.empty:
        return float(sim_time_s)
    events = handover_end.sort_values(["imsi", "time"]).copy()
    dwell_values: list[float] = []
    for _, group in events.groupby("imsi", sort=False):
        times = pd.to_numeric(group["time"], errors="coerce").to_numpy(dtype=float)
        if len(times) == 0:
            continue
        next_times = np.concatenate([times[1:], np.asarray([sim_time_s], dtype=float)])
        dwell_values.extend((next_times - times).tolist())
    return float(np.mean(dwell_values)) if dwell_values else float(sim_time_s)


def find_repeated_requests(decision_trace: pd.DataFrame, window_s: float) -> pd.DataFrame:
    columns = ["imsi", "prev_time", "time", "delta_s", "prev_target_cell", "target_cell", "same_target"]
    if decision_trace.empty:
        return pd.DataFrame(columns=columns)
    requests = decision_trace[decision_trace["actualOrRequested"] == "REQUEST"].copy()
    if requests.empty:
        return pd.DataFrame(columns=columns)
    requests["time"] = pd.to_numeric(requests["time"], errors="coerce")
    requests["targetCellId"] = pd.to_numeric(requests["targetCellId"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for imsi, group in requests.sort_values(["imsi", "time"]).groupby("imsi", sort=False):
        previous_row: pd.Series | None = None
        for _, row in group.iterrows():
            if previous_row is not None:
                delta_s = float(row["time"]) - float(previous_row["time"])
                if delta_s <= window_s:
                    rows.append(
                        {
                            "imsi": int(imsi),
                            "prev_time": float(previous_row["time"]),
                            "time": float(row["time"]),
                            "delta_s": delta_s,
                            "prev_target_cell": int(previous_row["targetCellId"]),
                            "target_cell": int(row["targetCellId"]),
                            "same_target": int(int(previous_row["targetCellId"]) == int(row["targetCellId"])),
                        }
                    )
            previous_row = row
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=columns)


def find_rapid_returns(handover_start: pd.DataFrame, window_s: float) -> pd.DataFrame:
    columns = ["imsi", "first_time", "return_time", "delta_s", "path"]
    if handover_start.empty:
        return pd.DataFrame(columns=columns)
    events = handover_start.copy()
    events["time"] = pd.to_numeric(events["time"], errors="coerce")
    events["sourceCellId"] = pd.to_numeric(events["sourceCellId"], errors="coerce")
    events["targetCellId"] = pd.to_numeric(events["targetCellId"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for imsi, group in events.sort_values(["imsi", "time"]).groupby("imsi", sort=False):
        group = group.reset_index(drop=True)
        for index in range(len(group) - 1):
            first = group.iloc[index]
            second = group.iloc[index + 1]
            if int(first["targetCellId"]) == int(second["sourceCellId"]) and int(second["targetCellId"]) == int(first["sourceCellId"]):
                delta_s = float(second["time"]) - float(first["time"])
                if delta_s <= window_s:
                    rows.append(
                        {
                            "imsi": int(imsi),
                            "first_time": float(first["time"]),
                            "return_time": float(second["time"]),
                            "delta_s": delta_s,
                            "path": f"{int(first['sourceCellId'])}->{int(first['targetCellId'])}->{int(second['targetCellId'])}",
                        }
                    )
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=columns)


def summarize_worker_rtt(run_dir: Path, time_limit_s: float | None) -> dict[str, Any]:
    worker_trace = filter_time(read_trace(run_dir / "lstm-worker-state.tr"), time_limit_s)
    if worker_trace.empty or "status" not in worker_trace.columns:
        return {
            "worker_failed_requests": 0.0,
            "worker_mean_rtt_ms": float("nan"),
            "worker_median_rtt_ms": float("nan"),
            "worker_max_rtt_ms": float("nan"),
            "worker_inner_mean_ms": float("nan"),
            "worker_inner_median_ms": float("nan"),
            "worker_inner_max_ms": float("nan"),
            "worker_ok_count": 0.0,
        }
    ok = worker_trace[worker_trace["status"] == "OK"].copy()
    failures = worker_trace[~worker_trace["status"].isin(["OK", "STARTED", "STOPPED"])].copy()
    if ok.empty:
        return {
            "worker_failed_requests": float(len(failures)),
            "worker_mean_rtt_ms": float("nan"),
            "worker_median_rtt_ms": float("nan"),
            "worker_max_rtt_ms": float("nan"),
            "worker_inner_mean_ms": float("nan"),
            "worker_inner_median_ms": float("nan"),
            "worker_inner_max_ms": float("nan"),
            "worker_ok_count": 0.0,
        }
    ok["latencyMs"] = pd.to_numeric(ok["latencyMs"], errors="coerce")
    ok["workerLatencyMs"] = pd.to_numeric(ok["workerLatencyMs"], errors="coerce")
    return {
        "worker_failed_requests": float(len(failures)),
        "worker_mean_rtt_ms": float(ok["latencyMs"].mean()),
        "worker_median_rtt_ms": float(ok["latencyMs"].median()),
        "worker_max_rtt_ms": float(ok["latencyMs"].max()),
        "worker_inner_mean_ms": float(ok["workerLatencyMs"].mean()),
        "worker_inner_median_ms": float(ok["workerLatencyMs"].median()),
        "worker_inner_max_ms": float(ok["workerLatencyMs"].max()),
        "worker_ok_count": float(len(ok)),
    }


def summarize_controller_behavior(run_dir: Path, time_limit_s: float | None) -> dict[str, Any]:
    decision_debug = filter_time(read_trace(run_dir / "lstm-decision-debug.tr"), time_limit_s)
    if decision_debug.empty or "runtimeReason" not in decision_debug.columns:
        return {
            "controller_decision_points": 0,
            "executed_lstm_decisions": 0,
            "blocked_by_threshold": 0,
            "blocked_by_cooldown": 0,
            "blocked_by_confirmation": 0,
            "blocked_by_anti_ping_pong": 0,
            "blocked_by_handover_in_progress": 0,
            "blocked_by_insufficient_history": 0,
            "blocked_by_other": 0,
        }

    reasons = decision_debug["runtimeReason"].fillna("").astype(str)
    executed = pd.to_numeric(decision_debug.get("executed", pd.Series(dtype=float)), errors="coerce").fillna(0)

    threshold_block = reasons.str.endswith("_below_threshold") | reasons.isin(
        ["trigger_below_threshold", "target_conf_below_threshold", "gain_below_threshold", "score_margin_below_threshold"]
    )
    cooldown_block = reasons == "cooldown_active"
    confirmation_block = reasons.str.startswith("confirmation_")
    anti_ping_pong_block = reasons == "anti_ping_pong_guard"
    handover_progress_block = reasons == "handover_in_progress"
    history_block = reasons == "insufficient_history"
    known_block = (
        threshold_block
        | cooldown_block
        | confirmation_block
        | anti_ping_pong_block
        | handover_progress_block
        | history_block
        | (reasons == "missing_serving_cell")
    )

    return {
        "controller_decision_points": int(len(decision_debug)),
        "executed_lstm_decisions": int(executed.sum()),
        "blocked_by_threshold": int(threshold_block.sum()),
        "blocked_by_cooldown": int(cooldown_block.sum()),
        "blocked_by_confirmation": int(confirmation_block.sum()),
        "blocked_by_anti_ping_pong": int(anti_ping_pong_block.sum()),
        "blocked_by_handover_in_progress": int(handover_progress_block.sum()),
        "blocked_by_insufficient_history": int(history_block.sum()),
        "blocked_by_other": int((reasons.ne("") & ~known_block & (executed == 0)).sum()),
    }


def summarize_run_metrics(mode: str, run_dir: Path, time_limit_s: float) -> dict[str, Any]:
    run_info = read_run_info(run_dir / "run-info.txt")
    handover_end = filter_time(read_trace(run_dir / "handover-end.tr"), time_limit_s)
    handover_start = filter_time(read_trace(run_dir / "handover-start.tr"), time_limit_s)
    decision_trace = filter_time(read_trace(run_dir / "handover-decision-source.tr"), time_limit_s)

    transport = compute_transport_metrics(run_dir, time_limit_s)
    worker = summarize_worker_rtt(run_dir, time_limit_s)
    controller = summarize_controller_behavior(run_dir, time_limit_s)
    repeated_requests = find_repeated_requests(decision_trace, window_s=1.0)
    rapid_returns = find_rapid_returns(handover_start, window_s=5.0)

    handover_count = int(len(handover_end))
    ping_pong_count = int(
        pd.to_numeric(handover_end.get("isPingPong", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
    )
    dl_tx_bytes = int(transport["dl_tx_bytes"])
    dl_rx_bytes = int(transport["dl_rx_bytes"])
    ul_tx_bytes = int(transport["ul_tx_bytes"])
    ul_rx_bytes = int(transport["ul_rx_bytes"])

    metrics: dict[str, Any] = {
        "mode": mode,
        "mode_label": MODE_LABELS.get(mode, mode),
        "run_dir": str(run_dir),
        "seed": int(run_info.get("seed", 0)),
        "run": int(run_info.get("run", 0)),
        "sim_time_s": float(time_limit_s),
        "handover_scenario": run_info.get("handoverScenario", ""),
        "a3_safety_net_enabled": int(float(run_info.get("a3SafetyNetEnabled", 0) or 0)),
        "handover_count": handover_count,
        "handovers_per_ue": float(handover_count / max(1.0, float(run_info.get("numberOfUes", 30)))),
        "ping_pong_count": ping_pong_count,
        "ping_pong_rate": float(ping_pong_count / handover_count) if handover_count > 0 else 0.0,
        "mean_dwell_time_s": compute_dwell_time(handover_end, time_limit_s),
        "rapid_repeat_requests": int(len(repeated_requests)),
        "rapid_cell_returns": int(len(rapid_returns)),
        "same_target_repeat_requests": int(
            pd.to_numeric(repeated_requests.get("same_target", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
        ),
        "dl_tx_bytes": dl_tx_bytes,
        "dl_rx_bytes": dl_rx_bytes,
        "ul_tx_bytes": ul_tx_bytes,
        "ul_rx_bytes": ul_rx_bytes,
        "dl_delivery_ratio": float(dl_rx_bytes / dl_tx_bytes) if dl_tx_bytes > 0 else 0.0,
        "ul_delivery_ratio": float(ul_rx_bytes / ul_tx_bytes) if ul_tx_bytes > 0 else 0.0,
        "dl_packet_loss_proxy": float(1.0 - (dl_rx_bytes / dl_tx_bytes)) if dl_tx_bytes > 0 else 0.0,
        "ul_packet_loss_proxy": float(1.0 - (ul_rx_bytes / ul_tx_bytes)) if ul_tx_bytes > 0 else 0.0,
        "dl_mean_throughput_mbps": float((dl_rx_bytes * 8.0) / max(time_limit_s, 1e-9) / 1e6),
        "ul_mean_throughput_mbps": float((ul_rx_bytes * 8.0) / max(time_limit_s, 1e-9) / 1e6),
        "dl_delay_mean_ms": float(transport["dl_delay_mean_ms"]),
        "dl_delay_std_ms": float(transport["dl_delay_std_ms"]),
        "ul_delay_mean_ms": float(transport["ul_delay_mean_ms"]),
        "ul_delay_std_ms": float(transport["ul_delay_std_ms"]),
        "traffic_metric_source": transport["traffic_metric_source"],
    }
    metrics.update(worker)
    metrics.update(controller)
    return metrics


def list_project_processes() -> list[dict[str, Any]]:
    output = subprocess.check_output(["ps", "-eo", "pid=,ppid=,command="], text=True)
    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        rows.append({"pid": int(parts[0]), "ppid": int(parts[1]), "cmd": parts[2]})
    return rows


def cleanup_old_processes(log_path: Path) -> list[dict[str, Any]]:
    patterns = [
        "ns3.46.1-lte-oran-helper-lstm-only-hex7-optimized",
        "persistent_inference_worker.py",
        "final_900s_experiments.py",
        "overnight_online_batch.py",
    ]
    current_pid = os.getpid()
    candidates = [
        row
        for row in list_project_processes()
        if row["pid"] != current_pid and any(pattern in row["cmd"] for pattern in patterns)
    ]
    killed: list[dict[str, Any]] = []
    for row in candidates:
        try:
            os.kill(row["pid"], signal.SIGTERM)
            killed.append({"pid": row["pid"], "signal": "TERM", "cmd": row["cmd"]})
        except ProcessLookupError:
            continue
    if killed:
        time.sleep(2.0)
    for row in candidates:
        try:
            os.kill(row["pid"], 0)
        except ProcessLookupError:
            continue
        try:
            os.kill(row["pid"], signal.SIGKILL)
            killed.append({"pid": row["pid"], "signal": "KILL", "cmd": row["cmd"]})
        except ProcessLookupError:
            continue
    log_line(log_path, f"cleanup_actions={json.dumps(killed)}")
    return killed


def verify_baseline_runs(baseline_root: Path, seed: int, runs: list[int], sim_time: float) -> dict[int, Path]:
    mapping: dict[int, Path] = {}
    for run in runs:
        pattern = f"seed{seed}-run{run}-*"
        matches = sorted(baseline_root.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No matched A3 baseline run found for {pattern} under {baseline_root}")
        baseline_dir = matches[0]
        run_info = read_run_info(baseline_dir / "run-info.txt")
        sim_time_sec = float(run_info.get("simTimeSec", 0.0) or 0.0)
        if sim_time_sec + 1e-9 < sim_time:
            raise RuntimeError(
                f"Matched baseline {baseline_dir} has simTimeSec={sim_time_sec}, which is shorter than requested {sim_time}"
            )
        mapping[run] = baseline_dir
    return mapping


def prepare_results_layout(results_root: Path, baseline_runs: dict[int, Path]) -> dict[str, Path]:
    if results_root.exists():
        backup_root = results_root.with_name(results_root.name + "_backup_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        shutil.move(str(results_root), str(backup_root))
    results_root.mkdir(parents=True, exist_ok=True)

    a3_root = results_root / "a3"
    lstm_only_root = results_root / "lstm_only"
    lstm_hybrid_root = results_root / "lstm_hybrid"
    reports_root = results_root / "reports"
    for path in [a3_root, lstm_only_root, lstm_hybrid_root, reports_root]:
        path.mkdir(parents=True, exist_ok=True)

    for run, src in baseline_runs.items():
        dst = a3_root / src.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst, target_is_directory=True)

    return {
        "root": results_root,
        "a3": a3_root,
        "lstm_only": lstm_only_root,
        "lstm_hybrid": lstm_hybrid_root,
        "reports": reports_root,
    }


def build_ns3_command(args: argparse.Namespace, mode: ModeSpec, output_root: Path, run: int) -> list[str]:
    command = [
        str(args.ns3_binary),
        f"--seed={args.seed}",
        f"--run={run}",
        f"--sim-time={args.sim_time}",
        f"--outputRoot={output_root}",
        "--enableLstmController=1",
        f"--useLteHandover={1 if mode.use_lte_handover else 0}",
        "--useOran=1",
        "--lstmDecisionIntervalSec=0.1",
        "--lstmSeqLen=15",
        f"--lstmMinConfidence={POLICY_PROFILE_A['target_conf_threshold']:.2f}",
        f"--lstmCooldownSec={POLICY_PROFILE_A['cooldown_s']:.2f}",
        f"--lstmAntiPingPongWindowSec={POLICY_PROFILE_A['anti_ping_pong_window_s']:.2f}",
        f"--lstmTriggerThreshold={POLICY_PROFILE_A['trigger_threshold']:.2f}",
        f"--lstmTargetThreshold={POLICY_PROFILE_A['target_conf_threshold']:.2f}",
        "--lstmUtilityThreshold=0.0",
        f"--lstmMinGainRsrpDb={POLICY_PROFILE_A['min_gain_rsrp_db']:.2f}",
        f"--lstmConsecutiveConfirmationSteps={POLICY_PROFILE_A['consecutive_confirmation_steps']}",
        "--lstmTargetDistanceTopK=0",
        "--lstmPreferNonServingTarget=0",
        f"--lstmPythonPath={args.worker_python}",
        f"--lstmInferenceScript={args.worker_script}",
        f"--lstmCheckpointPath={args.checkpoint_path}",
    ]
    if args.keep_heavy_traces:
        command.extend(
            [
                "--enablePacketByteTraces=1",
                "--enableRadioDebugTraces=1",
                "--enablePhyTraceFiles=1",
                "--enableMacTraceFiles=1",
                "--enableRlcTraceFiles=1",
                "--enablePdcpTraceFiles=1",
            ]
        )
    else:
        command.extend(
            [
                "--enablePacketByteTraces=0",
                "--enableRadioDebugTraces=0",
                "--enablePhyTraceFiles=0",
                "--enableMacTraceFiles=0",
                "--enableRlcTraceFiles=1",
                "--enablePdcpTraceFiles=1",
            ]
        )
    return command


def build_jobs(args: argparse.Namespace, layout: dict[str, Path]) -> list[Job]:
    jobs: list[Job] = []
    for mode in FINAL_MODES:
        output_root = layout[mode.name]
        for run in args.runs:
            run_dir = output_root / f"seed{args.seed}-run{run}-00001"
            log_path = output_root / f"seed{args.seed}-run{run}.log"
            jobs.append(
                Job(
                    mode=mode,
                    seed=args.seed,
                    run=run,
                    output_root=output_root,
                    log_path=log_path,
                    run_dir=run_dir,
                    command=build_ns3_command(args, mode, output_root, run),
                )
            )
    return jobs


def write_state(jobs: list[Job], state_path: Path) -> None:
    rows = []
    for job in jobs:
        rows.append(
            {
                "mode": job.mode.name,
                "seed": job.seed,
                "run": job.run,
                "status": job.status,
                "exit_code": job.exit_code,
                "log_path": str(job.log_path),
                "run_dir": str(job.run_dir),
                "launch_time_wall": job.launch_time_wall,
                "end_time_wall": job.end_time_wall,
            }
        )
    state_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def relative_delta(value: float, baseline: float) -> float:
    if baseline == 0.0:
        return 0.0 if value == 0.0 else math.inf
    return (value - baseline) / baseline


def collect_per_run_rows(
    args: argparse.Namespace,
    baseline_runs: dict[int, Path],
    jobs: list[Job],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    baseline_rows_by_run: dict[int, dict[str, Any]] = {}

    for run, run_dir in baseline_runs.items():
        row = summarize_run_metrics("a3", run_dir, float(args.sim_time))
        baseline_rows_by_run[run] = row
        rows.append(row)

    for job in jobs:
        if job.status != "completed" or job.exit_code != 0 or not job.run_dir.exists():
            continue
        row = summarize_run_metrics(job.mode.name, job.run_dir, float(args.sim_time))
        baseline = baseline_rows_by_run[job.run]
        for key, value in baseline.items():
            if key in {"mode", "mode_label", "run_dir", "seed", "run", "sim_time_s"}:
                continue
            if key in row and isinstance(row[key], (int, float)) and isinstance(value, (int, float)):
                row[f"{key}_delta_vs_a3"] = float(row[key]) - float(value)
        row["dl_throughput_rel_delta_vs_a3"] = relative_delta(
            float(row["dl_mean_throughput_mbps"]),
            float(baseline["dl_mean_throughput_mbps"]),
        )
        row["ul_throughput_rel_delta_vs_a3"] = relative_delta(
            float(row["ul_mean_throughput_mbps"]),
            float(baseline["ul_mean_throughput_mbps"]),
        )
        rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["run", "mode"]).reset_index(drop=True)


def build_summary_frame(per_run: pd.DataFrame) -> pd.DataFrame:
    if per_run.empty:
        return pd.DataFrame()

    metric_columns = [
        "handover_count",
        "handovers_per_ue",
        "ping_pong_count",
        "ping_pong_rate",
        "mean_dwell_time_s",
        "rapid_repeat_requests",
        "rapid_cell_returns",
        "same_target_repeat_requests",
        "dl_delivery_ratio",
        "ul_delivery_ratio",
        "dl_packet_loss_proxy",
        "ul_packet_loss_proxy",
        "dl_mean_throughput_mbps",
        "ul_mean_throughput_mbps",
        "dl_delay_mean_ms",
        "ul_delay_mean_ms",
        "worker_failed_requests",
        "worker_mean_rtt_ms",
        "worker_median_rtt_ms",
        "worker_max_rtt_ms",
        "executed_lstm_decisions",
        "blocked_by_threshold",
        "blocked_by_cooldown",
        "blocked_by_confirmation",
        "blocked_by_anti_ping_pong",
        "blocked_by_handover_in_progress",
        "blocked_by_insufficient_history",
        "blocked_by_other",
        "controller_decision_points",
    ]
    delta_columns = [column for column in per_run.columns if column.endswith("_delta_vs_a3")]
    delta_columns.extend(["dl_throughput_rel_delta_vs_a3", "ul_throughput_rel_delta_vs_a3"])
    delta_columns = [column for column in delta_columns if column in per_run.columns]

    grouped_rows: list[dict[str, Any]] = []
    for mode, group in per_run.groupby("mode", sort=False):
        row: dict[str, Any] = {
            "mode": mode,
            "mode_label": MODE_LABELS.get(mode, mode),
            "completed_runs": int(group["run"].nunique()),
        }
        for column in metric_columns:
            if column in group.columns:
                row[f"{column}_mean"] = float(pd.to_numeric(group[column], errors="coerce").mean())
                if column in {
                    "rapid_repeat_requests",
                    "rapid_cell_returns",
                    "same_target_repeat_requests",
                    "worker_failed_requests",
                    "executed_lstm_decisions",
                    "blocked_by_threshold",
                    "blocked_by_cooldown",
                    "blocked_by_confirmation",
                    "blocked_by_anti_ping_pong",
                    "blocked_by_handover_in_progress",
                    "blocked_by_insufficient_history",
                    "blocked_by_other",
                    "controller_decision_points",
                }:
                    row[f"{column}_total"] = float(pd.to_numeric(group[column], errors="coerce").sum())
        for column in delta_columns:
            row[f"{column}_mean"] = float(
                pd.to_numeric(group[column], errors="coerce").replace([np.inf, -np.inf], np.nan).mean()
            )
        grouped_rows.append(row)

    summary = pd.DataFrame(grouped_rows)
    return summary.sort_values("mode").reset_index(drop=True)


def rank_predictive_modes(
    summary: pd.DataFrame,
    delivery_margin_abs: float,
    throughput_margin_rel: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if summary.empty or "a3" not in summary["mode"].tolist():
        return pd.DataFrame(), {}

    a3_row = summary[summary["mode"] == "a3"].iloc[0]
    predictive = summary[summary["mode"] != "a3"].copy()
    if predictive.empty:
        return pd.DataFrame(), {}

    predictive["stability_win_vs_a3"] = (
        (predictive["ping_pong_rate_mean"] <= float(a3_row["ping_pong_rate_mean"]) + 1e-12)
        & (predictive["mean_dwell_time_s_mean"] >= float(a3_row["mean_dwell_time_s_mean"]) - 1e-12)
        & (predictive["rapid_repeat_requests_total"] == 0)
    ).astype(int)
    predictive["qos_within_margin"] = (
        (predictive["dl_delivery_ratio_delta_vs_a3_mean"] >= -delivery_margin_abs)
        & (predictive["ul_delivery_ratio_delta_vs_a3_mean"] >= -delivery_margin_abs)
        & (predictive["dl_throughput_rel_delta_vs_a3_mean"] >= -throughput_margin_rel)
        & (predictive["ul_throughput_rel_delta_vs_a3_mean"] >= -throughput_margin_rel)
    ).astype(int)
    predictive["balanced_score"] = (
        120.0 * predictive["stability_win_vs_a3"]
        + 40.0 * predictive["qos_within_margin"]
        - 400.0 * predictive["ping_pong_rate_mean"]
        - 5.0 * predictive["rapid_repeat_requests_total"]
        - 1.0 * predictive["worker_failed_requests_total"]
        + 0.4 * predictive["mean_dwell_time_s_delta_vs_a3_mean"]
        + 10.0 * predictive["dl_delivery_ratio_delta_vs_a3_mean"]
        + 10.0 * predictive["ul_delivery_ratio_delta_vs_a3_mean"]
        + 6.0 * predictive["dl_throughput_rel_delta_vs_a3_mean"]
        + 6.0 * predictive["ul_throughput_rel_delta_vs_a3_mean"]
    )
    predictive["stability_score"] = (
        -500.0 * predictive["ping_pong_rate_mean"]
        - 10.0 * predictive["rapid_repeat_requests_total"]
        - 0.1 * predictive["handover_count_mean"]
        + 0.8 * predictive["mean_dwell_time_s_delta_vs_a3_mean"]
        - 1.0 * predictive["worker_failed_requests_total"]
    )
    predictive["qos_score"] = (
        12.0 * predictive["dl_delivery_ratio_delta_vs_a3_mean"]
        + 12.0 * predictive["ul_delivery_ratio_delta_vs_a3_mean"]
        + 8.0 * predictive["dl_throughput_rel_delta_vs_a3_mean"]
        + 8.0 * predictive["ul_throughput_rel_delta_vs_a3_mean"]
        - 150.0 * predictive["ping_pong_rate_mean"]
        - 1.0 * predictive["worker_failed_requests_total"]
    )
    predictive = predictive.sort_values(
        ["balanced_score", "stability_score", "qos_score", "mode"],
        ascending=[False, False, False, True],
        ignore_index=True,
    )
    selection = {
        "main_result": predictive.iloc[0].to_dict(),
        "stability_first_variant": predictive.sort_values(
            ["stability_score", "balanced_score", "mode"],
            ascending=[False, False, True],
            ignore_index=True,
        ).iloc[0].to_dict(),
    }
    return predictive, selection


def make_plots(summary: pd.DataFrame, output_path: Path) -> None:
    if summary.empty:
        return
    ordered = summary.copy()
    ordered["mode_label"] = ordered["mode"].map(MODE_LABELS).fillna(ordered["mode"])
    x = np.arange(len(ordered))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].bar(x, ordered["ping_pong_rate_mean"], color="#c44e52")
    axes[0, 0].set_title("Ping-Pong Rate")
    axes[0, 0].set_xticks(x, ordered["mode_label"], rotation=15)

    axes[0, 1].bar(x, ordered["mean_dwell_time_s_mean"], color="#4c72b0")
    axes[0, 1].set_title("Mean Dwell Time (s)")
    axes[0, 1].set_xticks(x, ordered["mode_label"], rotation=15)

    axes[1, 0].bar(x - 0.18, ordered["dl_delivery_ratio_mean"], width=0.36, label="DL delivery")
    axes[1, 0].bar(x + 0.18, ordered["ul_delivery_ratio_mean"], width=0.36, label="UL delivery")
    axes[1, 0].set_title("Delivery Ratio")
    axes[1, 0].set_xticks(x, ordered["mode_label"], rotation=15)
    axes[1, 0].legend()

    axes[1, 1].bar(x - 0.18, ordered["dl_mean_throughput_mbps_mean"], width=0.36, label="DL throughput")
    axes[1, 1].bar(x + 0.18, ordered["ul_mean_throughput_mbps_mean"], width=0.36, label="UL throughput")
    axes[1, 1].set_title("Mean Throughput (Mbps)")
    axes[1, 1].set_xticks(x, ordered["mode_label"], rotation=15)
    axes[1, 1].legend()

    for axis in axes.flat:
        axis.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def render_directional_table_markdown(
    frame: pd.DataFrame,
    columns: list[tuple[str, str, str]],
) -> str:
    if frame.empty:
        return "_empty_"
    df = frame.copy()
    best_lookup: dict[str, set[int]] = {}
    for column, _, direction in columns:
        numeric = pd.to_numeric(df[column], errors="coerce")
        valid = numeric.dropna()
        if valid.empty:
            best_lookup[column] = set()
            continue
        best_value = valid.min() if direction == "min" else valid.max()
        best_lookup[column] = set(df.index[numeric == best_value].tolist())

    header = [label for _, label, _ in columns]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for idx, row in df.iterrows():
        values: list[str] = []
        for column, _, _ in columns:
            value = row[column]
            if isinstance(value, str):
                text = value
            elif pd.isna(value):
                text = "-"
            elif float(value).is_integer():
                text = str(int(round(float(value))))
            else:
                text = f"{float(value):.4f}"
            if idx in best_lookup[column]:
                text = f"**{text}**"
            values.append(text)
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def render_directional_table_latex(
    frame: pd.DataFrame,
    columns: list[tuple[str, str, str]],
) -> str:
    if frame.empty:
        return "% empty table\n"
    df = frame.copy()
    best_lookup: dict[str, set[int]] = {}
    for column, _, direction in columns:
        numeric = pd.to_numeric(df[column], errors="coerce")
        valid = numeric.dropna()
        if valid.empty:
            best_lookup[column] = set()
            continue
        best_value = valid.min() if direction == "min" else valid.max()
        best_lookup[column] = set(df.index[numeric == best_value].tolist())

    lines = [
        "\\begin{tabular}{" + "l" * len(columns) + "}",
        "\\hline",
        " & ".join(latex_escape(label) for _, label, _ in columns) + " \\\\",
        "\\hline",
    ]
    for idx, row in df.iterrows():
        values: list[str] = []
        for column, _, _ in columns:
            value = row[column]
            if isinstance(value, str):
                text = latex_escape(value)
            elif pd.isna(value):
                text = "-"
            elif float(value).is_integer():
                text = str(int(round(float(value))))
            else:
                text = f"{float(value):.4f}"
            if idx in best_lookup[column]:
                text = f"\\textbf{{{text}}}"
            values.append(text)
        lines.append(" & ".join(values) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def write_result_tables(summary: pd.DataFrame, reports_root: Path) -> None:
    ordered_modes = ["a3", "lstm_only", "lstm_hybrid"]
    summary_indexed = summary.set_index("mode", drop=False) if not summary.empty else pd.DataFrame()
    available = [mode for mode in ordered_modes if not summary.empty and mode in summary_indexed.index]
    main_rows = []
    runtime_rows = []
    for mode in available:
        row = summary_indexed.loc[mode]
        main_rows.append(
            {
                "mode": MODE_LABELS[mode],
                "handover_count": row["handover_count_mean"],
                "ping_pong_rate": row["ping_pong_rate_mean"],
                "mean_dwell_time_s": row["mean_dwell_time_s_mean"],
                "dl_delivery_ratio": row["dl_delivery_ratio_mean"],
                "ul_delivery_ratio": row["ul_delivery_ratio_mean"],
                "dl_mean_throughput_mbps": row["dl_mean_throughput_mbps_mean"],
                "ul_mean_throughput_mbps": row["ul_mean_throughput_mbps_mean"],
            }
        )
        runtime_rows.append(
            {
                "mode": MODE_LABELS[mode],
                "worker_failed_requests": row.get("worker_failed_requests_total", np.nan) if mode != "a3" else np.nan,
                "worker_mean_rtt_ms": row.get("worker_mean_rtt_ms_mean", np.nan) if mode != "a3" else np.nan,
                "worker_median_rtt_ms": row.get("worker_median_rtt_ms_mean", np.nan) if mode != "a3" else np.nan,
                "worker_max_rtt_ms": row.get("worker_max_rtt_ms_mean", np.nan) if mode != "a3" else np.nan,
                "rapid_repeat_requests": row["rapid_repeat_requests_total"],
                "rapid_cell_returns": row["rapid_cell_returns_total"],
                "blocked_by_threshold": row.get("blocked_by_threshold_total", np.nan) if mode != "a3" else np.nan,
                "blocked_by_cooldown": row.get("blocked_by_cooldown_total", np.nan) if mode != "a3" else np.nan,
                "blocked_by_confirmation": row.get("blocked_by_confirmation_total", np.nan) if mode != "a3" else np.nan,
                "blocked_by_anti_ping_pong": row.get("blocked_by_anti_ping_pong_total", np.nan) if mode != "a3" else np.nan,
            }
        )

    main_frame = pd.DataFrame(main_rows)
    runtime_frame = pd.DataFrame(runtime_rows)

    main_columns = [
        ("mode", "Mode", "max"),
        ("handover_count", "HO Count", "min"),
        ("ping_pong_rate", "Ping-Pong Rate", "min"),
        ("mean_dwell_time_s", "Mean Dwell (s)", "max"),
        ("dl_delivery_ratio", "DL Delivery", "max"),
        ("ul_delivery_ratio", "UL Delivery", "max"),
        ("dl_mean_throughput_mbps", "DL Throughput (Mbps)", "max"),
        ("ul_mean_throughput_mbps", "UL Throughput (Mbps)", "max"),
    ]
    runtime_columns = [
        ("mode", "Mode", "max"),
        ("worker_failed_requests", "Worker Fail", "min"),
        ("worker_mean_rtt_ms", "Mean RTT (ms)", "min"),
        ("worker_median_rtt_ms", "Median RTT (ms)", "min"),
        ("worker_max_rtt_ms", "Max RTT (ms)", "min"),
        ("rapid_repeat_requests", "Repeat Req", "min"),
        ("rapid_cell_returns", "Rapid Returns", "min"),
        ("blocked_by_threshold", "Blocked Threshold", "min"),
        ("blocked_by_cooldown", "Blocked Cooldown", "min"),
        ("blocked_by_confirmation", "Blocked Confirm", "min"),
        ("blocked_by_anti_ping_pong", "Blocked Anti-PP", "min"),
    ]

    (reports_root / "final_results_table.md").write_text(
        render_directional_table_markdown(main_frame, main_columns),
        encoding="utf-8",
    )
    (reports_root / "final_results_table.tex").write_text(
        render_directional_table_latex(main_frame, main_columns),
        encoding="utf-8",
    )
    (reports_root / "runtime_table.md").write_text(
        render_directional_table_markdown(runtime_frame, runtime_columns),
        encoding="utf-8",
    )
    (reports_root / "runtime_table.tex").write_text(
        render_directional_table_latex(runtime_frame, runtime_columns),
        encoding="utf-8",
    )


def write_docs_bundle(
    args: argparse.Namespace,
    reports_root: Path,
    summary: pd.DataFrame,
    selection: dict[str, Any],
    baseline_reused: bool,
) -> None:
    if not summary.empty and "a3" in summary["mode"].tolist():
        a3_row = summary[summary["mode"] == "a3"].iloc[0]
    else:
        a3_row = None

    predictive = summary[summary["mode"] != "a3"] if not summary.empty else pd.DataFrame()
    main_result = selection.get("main_result", {})
    stability_result = selection.get("stability_first_variant", {})
    main_label = MODE_LABELS.get(main_result.get("mode", ""), "pending") if main_result else "pending"
    stability_label = MODE_LABELS.get(stability_result.get("mode", ""), "pending") if stability_result else "pending"

    methodology_notes = f"""# Methodology Notes

## Research Objective

The study evaluates whether a candidate-aware LSTM mobility controller can improve closed-loop LTE handover behavior in an ns-3 + O-RAN scenario relative to the logged A3 baseline. The final long experiments use matched `900 s` runs and compare three operating modes:

- `A3` baseline reused from existing matched simulation outputs
- `LSTM-only` online controller
- `LSTM+A3 hybrid` online controller

## Final Experiment Logic

The final report-quality comparison uses a matched run design. All compared modes share the same topology, traffic, UE population, mobility configuration, radio setup, and simulation duration. Only the mobility-control path changes.

- Matched runs selected for the final long study: `{", ".join(str(run) for run in args.runs)}`
- Simulation duration: `{args.sim_time:.0f} s`
- A3 baseline reused: `{baseline_reused}`
- Online long-run policy profile: `profile_A`

## Selection Philosophy

The final selection is not based on a single KPI. The main result should favor balanced mobility stability and QoS/QoE preservation, while the stability-first variant should emphasize low ping-pong and long dwell time. The report generated from the final long runs is the authoritative source for the final operating-point choice.

- Current main operating point: `{main_label}`
- Current stability-first operating point: `{stability_label}`
"""

    experiment_setup = f"""# Experiment Setup

## Simulation Scenario

- Simulator: `ns-3.46.x`
- Radio stack: LTE + O-RAN instrumentation
- Cell layout: 7-site hexagonal deployment
- Sectorization: 3 sectors per site
- Total cells: 21
- UE count: 30
- Propagation setup: `ns3::ThreeGppUmaPropagationLossModel`
- Carrier frequency: `2.1 GHz`
- eNB bandwidth: `50 RB` DL / `50 RB` UL
- Mobility speed configured in the scenario: `10 m/s`
- Baseline LTE handover algorithm: `A3 RSRP`
- O-RAN enabled: yes
- Measurement periodicity used by the online controller: `100 ms`
- Final closed-loop simulation duration: `{args.sim_time:.0f} s`

## Output Layout

The final long-run results are stored in a dedicated root:

- `results_final_900s/a3/`
- `results_final_900s/lstm_only/`
- `results_final_900s/lstm_hybrid/`
- `results_final_900s/reports/`

## Matched Comparison Policy

The final report experiments reuse matched `900 s` A3 baselines if they already exist. This avoids unnecessary reruns and preserves direct comparability with the new LSTM-based online runs. In the current setup the matched A3 baselines were reused, not rerun.
"""

    dataset_training = """# Dataset and Training

## Training Data Sources

The predictive controller was trained offline using the `results_night_teacher_100ms` dataset. Training data came from the following simulation artifacts:

- `oran-repository.db`
- serving and neighbor radio observations persisted through the O-RAN path
- LTE handover traces and serving-cell state traces
- optional PDCP/RLC transport statistics

`oran-repository.db` serves as the primary repository-backed source for aligned per-UE mobility and radio context. Trace files complement the repository and are used for labels, traffic proxies, and consistency checks.

## Sequence Construction

- Feature sampling step: `100 ms`
- Sequence length: `15`
- Prediction horizon for the trigger task: `1.0 s`
- Binary trigger label: whether a successful handover occurs within the next `1.0 s`
- Target label: next serving cell, trained only on positive handover windows

## E2-Friendly Feature Space

The final online model uses compact E2-friendly features:

- `servingCellId`
- `servingRsrp`
- `servingRsrq`
- `servingSinr`
- `bestNghRsrp`
- `bestNghRsrq`
- `bestNghDiffRsrp`
- `bestNghDiffRsrq`

## Candidate Construction

The target-cell head is candidate-aware. Candidate sets are derived from historical strongest-neighbor observations rather than a flat 21-class output at inference time. Historical best-neighbor and second-neighbor observations are used to build realistic candidate sets for the classifier.

## Split and Leakage Control

- Split strategy: by run, not by random row shuffle
- Normalization: fit on train split only
- Target loss: masked to positive handover samples only
- Leakage prevention: no future radio samples, labels, or future serving-cell transitions are used in features
"""

    model_policy = f"""# Paper Model and Policy

## Model Architecture

The deployed online controller uses the trained candidate-aware LSTM model:

- `servingCellId` embedding
- numeric feature projection layer
- 1-layer LSTM backbone
- hidden size `128`
- shared temporal representation
- trigger head for near-term handover prediction
- candidate-aware target head for target-cell selection

## Online Deployment Path

The online ns-3 controller does not execute policy gating inside Python. Python performs raw model inference only. All conservative gating is kept in the ns-3 controller path.

The current final online profile is:

- `trigger_threshold = {POLICY_PROFILE_A['trigger_threshold']:.2f}`
- `target_conf_threshold = {POLICY_PROFILE_A['target_conf_threshold']:.2f}`
- `min_gain_rsrp_db = {POLICY_PROFILE_A['min_gain_rsrp_db']:.1f}`
- `consecutive_confirmation_steps = {POLICY_PROFILE_A['consecutive_confirmation_steps']}`
- `cooldown_s = {POLICY_PROFILE_A['cooldown_s']:.1f}`
- `anti_ping_pong_window_s = {POLICY_PROFILE_A['anti_ping_pong_window_s']:.1f}`

## Why Profile_A Was Selected

The `profile_A` operating point was selected from the completed `300 s` online validation batch because it gave the best practical balance between mobility stability and QoS preservation. It consistently reduced ping-pong and extended dwell time without causing worker failures, while keeping throughput and delivery changes within a manageable trade-off range.

## Persistent Worker Design

Per-cycle Python process spawning was removed because it was too expensive for closed-loop `100 ms` control. The final deployment path uses a persistent Python inference worker connected to the ns-3 controller through a local stdin/stdout protocol. The worker returns raw trigger probability, target prediction, target confidence, and timing data. This reduced runtime instability and made longer closed-loop experiments practical.
"""

    if a3_row is not None and not predictive.empty:
        result_lines = [
            "# Results Interpretation",
            "",
            "## Current Final-Run Interpretation",
            "",
            f"- Reused A3 baseline aggregate ping-pong rate: `{float(a3_row['ping_pong_rate_mean']):.4f}`",
            f"- Reused A3 baseline mean dwell time: `{float(a3_row['mean_dwell_time_s_mean']):.2f} s`",
            f"- Current main result candidate: `{main_label}`",
            f"- Current stability-first candidate: `{stability_label}`",
            "",
            "## Interpretation Guidance",
            "",
            "- Flat 21-class target prediction was not sufficient for robust online handover behavior.",
            "- Candidate-aware target prediction improved target selection and reduced top-1 ambiguity.",
            "- Conservative controller-side gating was essential for suppressing unnecessary handovers and ping-pong.",
            "- Closed-loop online results are more demanding than replay because controller decisions alter future radio and serving-cell trajectories.",
            "- The final report should explicitly discuss the trade-off between stronger mobility stability and any remaining QoS/QoE deviation from A3.",
            "- The difference between `LSTM-only` and `LSTM+A3 hybrid` should be framed as a policy trade-off: pure predictive control versus predictive control with an A3 safety net.",
            "",
            "The detailed, metric-level interpretation should be taken from `final_900s_vs_a3_report.md` once all final long runs finish.",
        ]
        results_interpretation = "\n".join(result_lines) + "\n"
    else:
        results_interpretation = """# Results Interpretation

The final `900 s` long-run interpretation will be written from the generated `final_900s_vs_a3_report.md` once the matched LSTM-only and LSTM+A3 runs complete. Until then, the interpretation should rely on the already validated short-run and `300 s` findings:

- candidate-aware prediction is stronger than flat target classification
- controller-side conservative gating is necessary for stability
- closed-loop online behavior must be judged by both mobility stability and QoS/QoE
- persistent-worker deployment is required for practical long closed-loop runs
"""

    (reports_root / "methodology_notes.md").write_text(methodology_notes, encoding="utf-8")
    (reports_root / "experiment_setup.md").write_text(experiment_setup, encoding="utf-8")
    (reports_root / "dataset_and_training.md").write_text(dataset_training, encoding="utf-8")
    (reports_root / "model_and_policy.md").write_text(model_policy, encoding="utf-8")
    (reports_root / "results_interpretation.md").write_text(results_interpretation, encoding="utf-8")


def build_final_report(
    args: argparse.Namespace,
    reports_root: Path,
    per_run: pd.DataFrame,
    summary: pd.DataFrame,
    predictive_ranking: pd.DataFrame,
    selection: dict[str, Any],
    jobs: list[Job],
    baseline_reused: bool,
) -> None:
    status_lines = [
        f"- `{job.mode.label}` run `{job.run}`: `{job.status}`" + (f" (exit `{job.exit_code}`)" if job.exit_code is not None else "")
        for job in jobs
    ]
    completed_jobs = [job for job in jobs if job.status == "completed" and job.exit_code == 0]
    failed_jobs = [job for job in jobs if job.status == "failed"]

    concise_summary = summary[
        [
            "mode_label",
            "completed_runs",
            "handover_count_mean",
            "ping_pong_rate_mean",
            "mean_dwell_time_s_mean",
            "dl_delivery_ratio_mean",
            "ul_delivery_ratio_mean",
            "dl_mean_throughput_mbps_mean",
            "ul_mean_throughput_mbps_mean",
        ]
    ].copy() if not summary.empty else pd.DataFrame()

    comparative_view = summary[
        [
            "mode_label",
            "rapid_repeat_requests_total",
            "rapid_cell_returns_total",
            "worker_failed_requests_total",
            "executed_lstm_decisions_total",
            "blocked_by_threshold_total",
            "blocked_by_cooldown_total",
            "blocked_by_confirmation_total",
            "blocked_by_anti_ping_pong_total",
            "dl_delay_mean_ms_mean",
            "ul_delay_mean_ms_mean",
        ]
    ].copy() if not summary.empty else pd.DataFrame()

    consistency_lines: list[str] = []
    predictive_modes = [mode for mode in ["lstm_only", "lstm_hybrid"] if mode in per_run["mode"].tolist()] if not per_run.empty else []
    for mode in predictive_modes:
        frame = per_run[per_run["mode"] == mode].copy()
        if frame.empty:
            continue
        stability_wins = (
            (frame["ping_pong_rate_delta_vs_a3"] <= 0.0)
            & (frame["mean_dwell_time_s_delta_vs_a3"] >= 0.0)
            & (frame["rapid_repeat_requests"] == 0)
        ).sum()
        qos_preserved = (
            (frame["dl_delivery_ratio_delta_vs_a3"] >= -args.delivery_margin_abs)
            & (frame["ul_delivery_ratio_delta_vs_a3"] >= -args.delivery_margin_abs)
            & (frame["dl_throughput_rel_delta_vs_a3"] >= -args.throughput_margin_rel)
            & (frame["ul_throughput_rel_delta_vs_a3"] >= -args.throughput_margin_rel)
        ).sum()
        consistency_lines.append(
            f"- `{MODE_LABELS[mode]}` stability wins in `{int(stability_wins)}/{len(frame)}` matched runs; "
            f"QoS stays within the configured margin in `{int(qos_preserved)}/{len(frame)}` matched runs."
        )

    if selection:
        main_label = MODE_LABELS.get(selection["main_result"]["mode"], selection["main_result"]["mode"])
        stability_label = MODE_LABELS.get(
            selection["stability_first_variant"]["mode"],
            selection["stability_first_variant"]["mode"],
        )
    else:
        main_label = "pending"
        stability_label = "pending"

    consistency_section = consistency_lines if consistency_lines else ["- Pending completion of predictive modes."]

    lines = [
        "# Final 900s Closed-Loop Results vs A3",
        "",
        f"- Results root: `{reports_root.parent}`",
        f"- Matched runs: `{', '.join(str(run) for run in args.runs)}`",
        f"- Simulation duration: `{args.sim_time:.0f} s`",
        f"- A3 baseline reused: `{baseline_reused}`",
        f"- Long-run online policy: `profile_A`",
        f"- Completed predictive jobs: `{len(completed_jobs)}/{len(jobs)}`",
        f"- Failed predictive jobs: `{len(failed_jobs)}`",
        "",
        "## Aggregate Summary",
        "",
        dataframe_to_markdown(concise_summary),
        "",
        "## Controller / Runtime Appendix",
        "",
        dataframe_to_markdown(comparative_view),
        "",
        "## Predictive Mode Ranking",
        "",
        dataframe_to_markdown(
            predictive_ranking[
                [
                    "mode_label",
                    "stability_win_vs_a3",
                    "qos_within_margin",
                    "balanced_score",
                    "stability_score",
                    "qos_score",
                ]
            ]
        ) if not predictive_ranking.empty else "_empty_",
        "",
        "## Per-Run Detail",
        "",
        dataframe_to_markdown(
            per_run[
                [
                    "mode_label",
                    "run",
                    "handover_count",
                    "ping_pong_rate",
                    "mean_dwell_time_s",
                    "dl_delivery_ratio",
                    "ul_delivery_ratio",
                    "dl_mean_throughput_mbps",
                    "ul_mean_throughput_mbps",
                    "rapid_repeat_requests",
                    "rapid_cell_returns",
                ]
            ]
        ) if not per_run.empty else "_empty_",
        "",
        "## Consistency Across Matched Runs",
        "",
        *consistency_section,
        "",
        "## Main Conclusions",
        "",
        "- The final report should state explicitly whether `LSTM-only` beats `A3` on mobility stability.",
        "- The final report should state explicitly whether `LSTM+A3 hybrid` beats `A3` on mobility stability.",
        "- QoS/QoE conclusions should separate `DL` and `UL` effects instead of collapsing them into one score.",
        f"- Current main result: `{main_label}`",
        f"- Current stability-first variant: `{stability_label}`",
        "",
        "## Job Status",
        "",
        *status_lines,
    ]
    (reports_root / "final_900s_vs_a3_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_best_policy_json(
    reports_root: Path,
    args: argparse.Namespace,
    selection: dict[str, Any],
    baseline_reused: bool,
) -> None:
    payload = {
        "policy_name": "profile_A",
        "policy": POLICY_PROFILE_A,
        "sim_time_s": float(args.sim_time),
        "matched_runs": [int(run) for run in args.runs],
        "a3_reused": bool(baseline_reused),
        "selection": selection,
    }
    (reports_root / "final_best_policy.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def generate_reports(
    args: argparse.Namespace,
    layout: dict[str, Path],
    baseline_runs: dict[int, Path],
    jobs: list[Job],
    log_path: Path,
) -> None:
    reports_root = layout["reports"]
    per_run = collect_per_run_rows(args, baseline_runs, jobs)
    summary = build_summary_frame(per_run)
    predictive_ranking, selection = rank_predictive_modes(
        summary,
        delivery_margin_abs=float(args.delivery_margin_abs),
        throughput_margin_rel=float(args.throughput_margin_rel),
    )

    per_run.to_csv(reports_root / "final_900s_per_run.csv", index=False)
    summary.to_csv(reports_root / "final_900s_summary.csv", index=False)

    if not summary.empty:
        debug_appendix = summary[
            [
                "mode_label",
                "worker_failed_requests_total",
                "worker_mean_rtt_ms_mean",
                "worker_median_rtt_ms_mean",
                "worker_max_rtt_ms_mean",
                "rapid_repeat_requests_total",
                "rapid_cell_returns_total",
                "executed_lstm_decisions_total",
                "blocked_by_threshold_total",
                "blocked_by_cooldown_total",
                "blocked_by_confirmation_total",
                "blocked_by_anti_ping_pong_total",
                "blocked_by_handover_in_progress_total",
                "blocked_by_insufficient_history_total",
                "blocked_by_other_total",
            ]
        ].copy()
    else:
        debug_appendix = pd.DataFrame()
    debug_appendix.to_csv(reports_root / "final_debug_appendix.csv", index=False)

    make_plots(summary, reports_root / "final_900s_plots.png")
    build_final_report(args, reports_root, per_run, summary, predictive_ranking, selection, jobs, baseline_reused=True)
    write_result_tables(summary, reports_root)
    write_docs_bundle(args, reports_root, summary, selection, baseline_reused=True)
    write_best_policy_json(reports_root, args, selection, baseline_reused=True)
    log_line(log_path, f"reports_refreshed completed_jobs={len([job for job in jobs if job.status == 'completed' and job.exit_code == 0])}")


def launch_jobs(
    args: argparse.Namespace,
    jobs: list[Job],
    layout: dict[str, Path],
    baseline_runs: dict[int, Path],
    log_path: Path,
) -> None:
    state_path = layout["reports"] / "final_900s_state.json"
    pending = jobs[:]
    running: list[Job] = []

    while pending or running:
        while pending and len(running) < args.max_parallel:
            job = pending.pop(0)
            job.log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = job.log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                job.command,
                cwd=PROJECT_ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            handle.close()
            job.process = process
            job.launch_time_wall = time.time()
            job.status = "running"
            running.append(job)
            log_line(log_path, f"launched mode={job.mode.name} run={job.run} pid={process.pid}")
            write_state(jobs, state_path)

        time.sleep(args.poll_interval_s)
        still_running: list[Job] = []
        for job in running:
            assert job.process is not None
            exit_code = job.process.poll()
            if exit_code is None:
                still_running.append(job)
                continue
            job.exit_code = int(exit_code)
            job.end_time_wall = time.time()
            job.status = "completed" if exit_code == 0 else "failed"
            log_line(
                log_path,
                f"finished mode={job.mode.name} run={job.run} status={job.status} exit_code={job.exit_code}",
            )
            write_state(jobs, state_path)
            generate_reports(args, layout, baseline_runs, jobs, log_path)
        running = still_running

    log_line(log_path, f"all_jobs_finished total_jobs={len(jobs)}")


def main() -> int:
    args = build_argument_parser().parse_args()
    results_root = args.results_root.resolve()

    baseline_runs = verify_baseline_runs(args.baseline_root, args.seed, args.runs, args.sim_time)
    layout = prepare_results_layout(results_root, baseline_runs)
    reports_root = layout["reports"]
    log_path = reports_root / "final_900s_launcher.log"

    manifest = {
        "timestamp": datetime.now().isoformat(),
        "results_root": str(results_root),
        "seed": int(args.seed),
        "runs": [int(run) for run in args.runs],
        "sim_time_s": float(args.sim_time),
        "baseline_root": str(args.baseline_root),
        "baseline_reused": True,
        "policy": POLICY_PROFILE_A,
        "modes": [asdict(mode) for mode in FINAL_MODES],
    }

    jobs = build_jobs(args, layout)
    manifest["commands"] = [
        {
            "mode": job.mode.name,
            "run": job.run,
            "command": job.command,
            "output_root": str(job.output_root),
        }
        for job in jobs
    ]
    (reports_root / "launch_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (reports_root / "baseline_reuse.json").write_text(
        json.dumps({str(run): str(path) for run, path in baseline_runs.items()}, indent=2),
        encoding="utf-8",
    )

    if not args.refresh_only:
        if not args.skip_cleanup:
            cleanup_old_processes(log_path)
        else:
            log_line(log_path, "cleanup skipped by request")

    write_state(jobs, reports_root / "final_900s_state.json")
    generate_reports(args, layout, baseline_runs, jobs, log_path)

    if args.refresh_only:
        log_line(log_path, "refresh_only completed")
        return 0

    log_line(
        log_path,
        f"prepared_jobs={len(jobs)} runs={args.runs} max_parallel={args.max_parallel} results_root={results_root}",
    )
    launch_jobs(args, jobs, layout, baseline_runs, log_path)
    generate_reports(args, layout, baseline_runs, jobs, log_path)
    log_line(log_path, "final 900s batch finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
