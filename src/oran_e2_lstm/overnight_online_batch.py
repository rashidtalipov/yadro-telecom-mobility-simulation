#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(os.environ.get("NS3_ROOT", "/path/to/ns-allinone-3.46.1/ns-3.46.1"))
DEFAULT_NS3_BINARY = PROJECT_ROOT / "build/optimized/scratch/ns3.46.1-lte-oran-helper-lstm-only-hex7-optimized"
DEFAULT_WORKER_PYTHON = PROJECT_ROOT / "results_night/.venv/bin/python"
DEFAULT_WORKER_SCRIPT = PROJECT_ROOT / "results_night/oran_e2_lstm/persistent_inference_worker.py"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "results_night/oran_e2_lstm/runs/candidate_history_k3_20ep/best_model.pt"
DEFAULT_BASELINE_ROOT = PROJECT_ROOT / "results_night_teacher_100ms"
DEFAULT_BATCH_ROOT = PROJECT_ROOT / "results_night/oran_e2_lstm/overnight_batches"

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


@dataclass(frozen=True)
class PolicyProfile:
    name: str
    trigger_threshold: float
    target_conf_threshold: float
    min_gain_rsrp_db: float
    consecutive_confirmation_steps: int
    cooldown_s: float
    anti_ping_pong_window_s: float


@dataclass
class BatchJob:
    profile: PolicyProfile
    seed: int
    run: int
    output_root: Path
    log_path: Path
    run_dir: Path
    command: list[str]
    process: subprocess.Popen[str] | None = None
    launch_time_wall: float | None = None
    end_time_wall: float | None = None
    exit_code: int | None = None
    status: str = "pending"


REQUIRED_PROFILES: list[PolicyProfile] = [
    PolicyProfile("profile_A", 0.70, 0.70, 1.0, 2, 1.0, 2.0),
    PolicyProfile("profile_B", 0.70, 0.68, 1.0, 2, 1.0, 2.0),
    PolicyProfile("profile_C", 0.72, 0.70, 1.0, 2, 1.0, 2.0),
    PolicyProfile("profile_D", 0.70, 0.70, 1.5, 2, 1.0, 2.0),
]


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


def detect_default_parallelism() -> int:
    cpu_count = os.cpu_count() or 8
    return max(1, min(4, cpu_count // 8 if cpu_count >= 8 else 1))


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an overnight online-only ns-3 LSTM handover batch and compare it to existing A3 baselines."
    )
    parser.add_argument("--batch-root", type=Path, default=None)
    parser.add_argument("--ns3-binary", type=Path, default=DEFAULT_NS3_BINARY)
    parser.add_argument("--ns3-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--worker-python", type=Path, default=DEFAULT_WORKER_PYTHON)
    parser.add_argument("--worker-script", type=Path, default=DEFAULT_WORKER_SCRIPT)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--runs", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--sim-time", type=float, default=900.0)
    parser.add_argument("--max-parallel", type=int, default=detect_default_parallelism())
    parser.add_argument("--poll-interval-s", type=float, default=30.0)
    parser.add_argument("--delivery-margin-abs", type=float, default=0.03)
    parser.add_argument("--throughput-margin-rel", type=float, default=0.10)
    parser.add_argument("--current-conservative-pingpong-baseline", type=float, default=0.0)
    parser.add_argument("--keep-heavy-traces", action="store_true")
    parser.add_argument("--skip-cleanup", action="store_true")
    return parser


def make_batch_root(user_value: Path | None) -> Path:
    if user_value is not None:
        return user_value.resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (DEFAULT_BATCH_ROOT / f"batch_{stamp}").resolve()


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
    frame["start"] = pd.to_numeric(frame["start"], errors="coerce")
    frame["end"] = pd.to_numeric(frame["end"], errors="coerce")
    frame["TxBytes"] = pd.to_numeric(frame["TxBytes"], errors="coerce").fillna(0)
    frame["RxBytes"] = pd.to_numeric(frame["RxBytes"], errors="coerce").fillna(0)
    if time_limit_s is not None:
        frame = frame[frame["start"] < float(time_limit_s)].copy()
    return frame


def packet_trace_bytes(run_dir: Path, prefix: str, time_limit_s: float | None) -> tuple[int, int]:
    tx = filter_time(read_trace(run_dir / f"{prefix}-tx.tr"), time_limit_s)
    rx = filter_time(read_trace(run_dir / f"{prefix}-rx.tr"), time_limit_s)
    tx_bytes = int(pd.to_numeric(tx.get("bytes", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    rx_bytes = int(pd.to_numeric(rx.get("bytes", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    return tx_bytes, rx_bytes


def bearer_trace_bytes(run_dir: Path, prefix: str, time_limit_s: float | None) -> tuple[int, int, bool]:
    stats = read_bearer_stats(run_dir / f"{prefix}PdcpStats.txt", time_limit_s)
    used_stats = (run_dir / f"{prefix}PdcpStats.txt").exists()
    if stats.empty:
        stats = read_bearer_stats(run_dir / f"{prefix}RlcStats.txt", time_limit_s)
        used_stats = used_stats or (run_dir / f"{prefix}RlcStats.txt").exists()
    if stats.empty:
        return 0, 0, used_stats
    return int(stats["TxBytes"].sum()), int(stats["RxBytes"].sum()), True


def compute_transport_bytes(run_dir: Path, time_limit_s: float | None) -> tuple[int, int, int, int, str]:
    dl_tx_bytes, dl_rx_bytes, dl_has_stats = bearer_trace_bytes(run_dir, "Dl", time_limit_s)
    ul_tx_bytes, ul_rx_bytes, ul_has_stats = bearer_trace_bytes(run_dir, "Ul", time_limit_s)
    source = "pdcp_rlc"
    if not (dl_has_stats or ul_has_stats):
        dl_tx_bytes, dl_rx_bytes = packet_trace_bytes(run_dir, "dl", time_limit_s)
        ul_tx_bytes, ul_rx_bytes = packet_trace_bytes(run_dir, "ul", time_limit_s)
        source = "packet_trace"
    return dl_tx_bytes, dl_rx_bytes, ul_tx_bytes, ul_rx_bytes, source


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


def summarize_worker_rtt(run_dir: Path) -> dict[str, float]:
    worker_trace = read_trace(run_dir / "lstm-worker-state.tr")
    if worker_trace.empty or "status" not in worker_trace.columns:
        return {
            "worker_request_failures": 0.0,
            "worker_rtt_mean_ms": np.nan,
            "worker_rtt_median_ms": np.nan,
            "worker_rtt_max_ms": np.nan,
            "worker_inner_mean_ms": np.nan,
            "worker_inner_median_ms": np.nan,
            "worker_inner_max_ms": np.nan,
            "worker_ok_count": 0.0,
        }

    ok = worker_trace[worker_trace["status"] == "OK"].copy()
    failures = worker_trace[~worker_trace["status"].isin(["OK", "STARTED", "STOPPED"])].copy()
    if ok.empty:
        return {
            "worker_request_failures": float(len(failures)),
            "worker_rtt_mean_ms": np.nan,
            "worker_rtt_median_ms": np.nan,
            "worker_rtt_max_ms": np.nan,
            "worker_inner_mean_ms": np.nan,
            "worker_inner_median_ms": np.nan,
            "worker_inner_max_ms": np.nan,
            "worker_ok_count": 0.0,
        }

    ok["latencyMs"] = pd.to_numeric(ok["latencyMs"], errors="coerce")
    ok["workerLatencyMs"] = pd.to_numeric(ok["workerLatencyMs"], errors="coerce")
    return {
        "worker_request_failures": float(len(failures)),
        "worker_rtt_mean_ms": float(ok["latencyMs"].mean()),
        "worker_rtt_median_ms": float(ok["latencyMs"].median()),
        "worker_rtt_max_ms": float(ok["latencyMs"].max()),
        "worker_inner_mean_ms": float(ok["workerLatencyMs"].mean()),
        "worker_inner_median_ms": float(ok["workerLatencyMs"].median()),
        "worker_inner_max_ms": float(ok["workerLatencyMs"].max()),
        "worker_ok_count": float(len(ok)),
    }


def summarize_run_metrics(run_dir: Path, time_limit_s: float) -> dict[str, Any]:
    run_info = read_run_info(run_dir / "run-info.txt")
    handover_end = filter_time(read_trace(run_dir / "handover-end.tr"), time_limit_s)
    handover_start = filter_time(read_trace(run_dir / "handover-start.tr"), time_limit_s)
    decision_trace = filter_time(read_trace(run_dir / "handover-decision-source.tr"), time_limit_s)
    dl_tx_bytes, dl_rx_bytes, ul_tx_bytes, ul_rx_bytes, traffic_source = compute_transport_bytes(run_dir, time_limit_s)

    handover_count = int(len(handover_end))
    ping_pong_count = int(
        pd.to_numeric(handover_end.get("isPingPong", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
    )
    repeated_requests = find_repeated_requests(decision_trace, window_s=1.0)
    rapid_returns = find_rapid_returns(handover_start, window_s=5.0)
    worker_summary = summarize_worker_rtt(run_dir)

    metrics = {
        "seed": int(run_info.get("seed", 0)),
        "run": int(run_info.get("run", 0)),
        "sim_time_s": float(time_limit_s),
        "handover_count": handover_count,
        "handovers_per_ue": float(handover_count / max(1.0, float(run_info.get("numberOfUes", 30)))),
        "ping_pong_count": ping_pong_count,
        "ping_pong_rate": float(ping_pong_count / handover_count) if handover_count > 0 else 0.0,
        "mean_dwell_time_s": compute_dwell_time(handover_end, time_limit_s),
        "rapid_repeat_requests": int(len(repeated_requests)),
        "rapid_cell_returns": int(len(rapid_returns)),
        "dl_tx_bytes": int(dl_tx_bytes),
        "dl_rx_bytes": int(dl_rx_bytes),
        "ul_tx_bytes": int(ul_tx_bytes),
        "ul_rx_bytes": int(ul_rx_bytes),
        "dl_delivery_ratio": float(dl_rx_bytes / dl_tx_bytes) if dl_tx_bytes > 0 else 0.0,
        "ul_delivery_ratio": float(ul_rx_bytes / ul_tx_bytes) if ul_tx_bytes > 0 else 0.0,
        "dl_mean_throughput_mbps": float((dl_rx_bytes * 8.0) / max(time_limit_s, 1e-9) / 1e6),
        "ul_mean_throughput_mbps": float((ul_rx_bytes * 8.0) / max(time_limit_s, 1e-9) / 1e6),
        "traffic_metric_source": traffic_source,
    }
    metrics.update(worker_summary)
    return metrics


def find_matching_baseline(baseline_root: Path, seed: int, run: int) -> Path:
    pattern = f"seed{seed}-run{run}-*"
    matches = sorted(baseline_root.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No baseline run matched {pattern} under {baseline_root}")
    return matches[0]


def list_project_processes() -> list[dict[str, Any]]:
    output = subprocess.check_output(["ps", "-eo", "pid=,ppid=,command="], text=True)
    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        pid = int(parts[0])
        ppid = int(parts[1])
        cmd = parts[2]
        rows.append({"pid": pid, "ppid": ppid, "cmd": cmd})
    return rows


def cleanup_old_processes(log_path: Path) -> list[dict[str, Any]]:
    patterns = [
        "ns3.46.1-lte-oran-helper-lstm-only-hex7-optimized",
        "persistent_inference_worker.py",
        "overnight_online_batch.py",
    ]
    current_pid = os.getpid()
    processes = [
        row
        for row in list_project_processes()
        if row["pid"] != current_pid and any(pattern in row["cmd"] for pattern in patterns)
    ]
    killed: list[dict[str, Any]] = []
    for row in processes:
        try:
            os.kill(row["pid"], signal.SIGTERM)
            killed.append({"pid": row["pid"], "signal": "TERM", "cmd": row["cmd"]})
        except ProcessLookupError:
            continue
    if killed:
        time.sleep(2.0)
    for row in processes:
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


def build_ns3_command(
    args: argparse.Namespace,
    profile: PolicyProfile,
    run: int,
    output_root: Path,
) -> list[str]:
    command = [
        str(args.ns3_binary),
        f"--seed={args.seed}",
        f"--run={run}",
        f"--sim-time={args.sim_time}",
        f"--outputRoot={output_root}",
        "--enableLstmController=1",
        "--useLteHandover=0",
        "--lstmDecisionIntervalSec=0.1",
        "--lstmSeqLen=15",
        f"--lstmMinConfidence={profile.target_conf_threshold:.2f}",
        f"--lstmCooldownSec={profile.cooldown_s:.2f}",
        f"--lstmAntiPingPongWindowSec={profile.anti_ping_pong_window_s:.2f}",
        f"--lstmTriggerThreshold={profile.trigger_threshold:.2f}",
        f"--lstmTargetThreshold={profile.target_conf_threshold:.2f}",
        "--lstmUtilityThreshold=0.0",
        f"--lstmMinGainRsrpDb={profile.min_gain_rsrp_db:.2f}",
        f"--lstmConsecutiveConfirmationSteps={profile.consecutive_confirmation_steps}",
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


def build_jobs(args: argparse.Namespace, batch_root: Path) -> list[BatchJob]:
    jobs: list[BatchJob] = []
    runs_root = batch_root / "runs"
    for profile in REQUIRED_PROFILES:
        profile_root = runs_root / profile.name
        profile_root.mkdir(parents=True, exist_ok=True)
        for run in args.runs:
            run_dir = profile_root / f"seed{args.seed}-run{run}-00001"
            log_path = profile_root / f"seed{args.seed}-run{run}.log"
            command = build_ns3_command(args, profile, run, profile_root)
            jobs.append(
                BatchJob(
                    profile=profile,
                    seed=args.seed,
                    run=run,
                    output_root=profile_root,
                    log_path=log_path,
                    run_dir=run_dir,
                    command=command,
                )
            )
    return jobs


def write_state(jobs: list[BatchJob], state_path: Path) -> None:
    rows = []
    for job in jobs:
        rows.append(
            {
                "profile": job.profile.name,
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


def launch_jobs(args: argparse.Namespace, jobs: list[BatchJob], batch_root: Path, log_path: Path) -> None:
    state_path = batch_root / "batch_state.json"
    pending = jobs[:]
    running: list[BatchJob] = []
    completed: list[BatchJob] = []

    while pending or running:
        while pending and len(running) < args.max_parallel:
            job = pending.pop(0)
            job.log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = job.log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                job.command,
                cwd=args.ns3_root,
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
            log_line(log_path, f"launched profile={job.profile.name} run={job.run} pid={process.pid}")
            write_state(jobs, state_path)

        time.sleep(args.poll_interval_s)
        still_running: list[BatchJob] = []
        for job in running:
            assert job.process is not None
            exit_code = job.process.poll()
            if exit_code is None:
                still_running.append(job)
                continue
            job.exit_code = int(exit_code)
            job.end_time_wall = time.time()
            job.status = "completed" if exit_code == 0 else "failed"
            completed.append(job)
            log_line(
                log_path,
                f"finished profile={job.profile.name} run={job.run} status={job.status} exit_code={job.exit_code}",
            )
            write_state(jobs, state_path)
            generate_reports(args, jobs, batch_root, log_path)
        running = still_running

    log_line(log_path, f"all_jobs_finished total={len(completed)}")


def relative_delta(value: float, baseline: float) -> float:
    if baseline == 0.0:
        return 0.0 if value == 0.0 else math.inf
    return (value - baseline) / baseline


def rank_profiles(
    summary: pd.DataFrame,
    delivery_margin_abs: float,
    throughput_margin_rel: float,
    current_conservative_pingpong_baseline: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if summary.empty:
        return pd.DataFrame(), {}

    ranked = summary.copy()
    ranked["stability_clean"] = (
        (ranked["ping_pong_rate_online_mean"] <= current_conservative_pingpong_baseline + 1e-9)
        & (ranked["rapid_repeat_requests_total"] == 0)
        & (ranked["rapid_cell_returns_total"] == 0)
        & (ranked["worker_request_failures_total"] == 0)
    ).astype(int)
    ranked["qos_within_margin"] = (
        (ranked["dl_delivery_ratio_delta_mean"] >= -delivery_margin_abs)
        & (ranked["ul_delivery_ratio_delta_mean"] >= -delivery_margin_abs)
        & (ranked["dl_throughput_rel_delta_mean"] >= -throughput_margin_rel)
        & (ranked["ul_throughput_rel_delta_mean"] >= -throughput_margin_rel)
    ).astype(int)
    ranked["balanced_score"] = (
        100.0 * ranked["stability_clean"]
        + 40.0 * ranked["qos_within_margin"]
        - 400.0 * ranked["ping_pong_rate_online_mean"]
        - 10.0 * ranked["rapid_repeat_requests_total"]
        - 10.0 * ranked["rapid_cell_returns_total"]
        - 1.0 * ranked["worker_request_failures_total"]
        + 8.0 * ranked["dl_delivery_ratio_delta_mean"]
        + 8.0 * ranked["ul_delivery_ratio_delta_mean"]
        + 4.0 * ranked["dl_throughput_rel_delta_mean"]
        + 4.0 * ranked["ul_throughput_rel_delta_mean"]
        + 0.2 * ranked["mean_dwell_time_s_delta_mean"]
    )
    ranked["stability_score"] = (
        -500.0 * ranked["ping_pong_rate_online_mean"]
        - 20.0 * ranked["rapid_repeat_requests_total"]
        - 20.0 * ranked["rapid_cell_returns_total"]
        - 2.0 * ranked["worker_request_failures_total"]
        + 0.5 * ranked["mean_dwell_time_s_delta_mean"]
        - 0.05 * ranked["handover_count_online_mean"]
    )
    ranked["qos_score"] = (
        12.0 * ranked["dl_delivery_ratio_delta_mean"]
        + 12.0 * ranked["ul_delivery_ratio_delta_mean"]
        + 8.0 * ranked["dl_throughput_rel_delta_mean"]
        + 8.0 * ranked["ul_throughput_rel_delta_mean"]
        - 100.0 * ranked["ping_pong_rate_online_mean"]
        - 1.0 * ranked["worker_request_failures_total"]
    )
    ranked = ranked.sort_values(
        ["balanced_score", "stability_score", "qos_score", "profile"],
        ascending=[False, False, False, True],
        ignore_index=True,
    )

    best_balanced = ranked.iloc[0].to_dict()
    best_stability = ranked.sort_values(
        ["stability_score", "balanced_score", "profile"],
        ascending=[False, False, True],
        ignore_index=True,
    ).iloc[0].to_dict()
    best_qos = ranked.sort_values(
        ["qos_score", "balanced_score", "profile"],
        ascending=[False, False, True],
        ignore_index=True,
    ).iloc[0].to_dict()

    selection = {
        "best_balanced": best_balanced,
        "best_stability_first": best_stability,
        "best_qos_first": best_qos,
    }
    return ranked, selection


def make_plots(summary: pd.DataFrame, output_path: Path) -> None:
    if summary.empty:
        return
    profiles = summary["profile"].tolist()
    x = np.arange(len(profiles))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].bar(x, summary["ping_pong_rate_online_mean"], color="#c44e52")
    axes[0, 0].set_title("Ping-Pong Rate")
    axes[0, 0].set_xticks(x, profiles, rotation=15)

    axes[0, 1].bar(x, summary["mean_dwell_time_s_delta_mean"], color="#4c72b0")
    axes[0, 1].set_title("Mean Dwell Delta vs A3 (s)")
    axes[0, 1].set_xticks(x, profiles, rotation=15)

    axes[1, 0].bar(x - 0.18, summary["dl_delivery_ratio_delta_mean"], width=0.36, label="DL delivery")
    axes[1, 0].bar(x + 0.18, summary["ul_delivery_ratio_delta_mean"], width=0.36, label="UL delivery")
    axes[1, 0].set_title("Delivery Ratio Delta vs A3")
    axes[1, 0].set_xticks(x, profiles, rotation=15)
    axes[1, 0].legend()

    axes[1, 1].bar(x - 0.18, summary["dl_throughput_rel_delta_mean"], width=0.36, label="DL throughput")
    axes[1, 1].bar(x + 0.18, summary["ul_throughput_rel_delta_mean"], width=0.36, label="UL throughput")
    axes[1, 1].set_title("Relative Throughput Delta vs A3")
    axes[1, 1].set_xticks(x, profiles, rotation=15)
    axes[1, 1].legend()

    for axis in axes.flat:
        axis.grid(True, alpha=0.25)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def collect_completed_rows(args: argparse.Namespace, jobs: list[BatchJob]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for job in jobs:
        if job.status != "completed" or job.exit_code != 0 or not job.run_dir.exists():
            continue
        baseline_run_dir = find_matching_baseline(args.baseline_root, seed=job.seed, run=job.run)
        online_metrics = summarize_run_metrics(job.run_dir, float(args.sim_time))
        baseline_metrics = summarize_run_metrics(baseline_run_dir, float(args.sim_time))
        row: dict[str, Any] = {
            "profile": job.profile.name,
            "seed": job.seed,
            "run": job.run,
            "online_run_dir": str(job.run_dir),
            "baseline_run_dir": str(baseline_run_dir),
            "launch_time_wall": job.launch_time_wall,
            "end_time_wall": job.end_time_wall,
            "wall_runtime_s": float(job.end_time_wall - job.launch_time_wall) if job.end_time_wall and job.launch_time_wall else np.nan,
        }
        for key, value in online_metrics.items():
            row[f"{key}_online"] = value
        for key, value in baseline_metrics.items():
            row[f"{key}_a3"] = value
            if isinstance(row.get(f"{key}_online"), (int, float)) and isinstance(value, (int, float)):
                row[f"{key}_delta"] = float(row[f"{key}_online"]) - float(value)
        row["dl_throughput_rel_delta"] = relative_delta(
            float(row["dl_mean_throughput_mbps_online"]),
            float(row["dl_mean_throughput_mbps_a3"]),
        )
        row["ul_throughput_rel_delta"] = relative_delta(
            float(row["ul_mean_throughput_mbps_online"]),
            float(row["ul_mean_throughput_mbps_a3"]),
        )
        rows.append(row)
    return rows


def build_summary(per_run: pd.DataFrame) -> pd.DataFrame:
    if per_run.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for profile, frame in per_run.groupby("profile", sort=False):
        rows.append(
            {
                "profile": profile,
                "completed_runs": int(len(frame)),
                "handover_count_online_mean": float(frame["handover_count_online"].mean()),
                "ping_pong_rate_online_mean": float(frame["ping_pong_rate_online"].mean()),
                "mean_dwell_time_s_online_mean": float(frame["mean_dwell_time_s_online"].mean()),
                "rapid_repeat_requests_total": int(frame["rapid_repeat_requests_online"].sum()),
                "rapid_cell_returns_total": int(frame["rapid_cell_returns_online"].sum()),
                "dl_delivery_ratio_online_mean": float(frame["dl_delivery_ratio_online"].mean()),
                "ul_delivery_ratio_online_mean": float(frame["ul_delivery_ratio_online"].mean()),
                "dl_mean_throughput_mbps_online_mean": float(frame["dl_mean_throughput_mbps_online"].mean()),
                "ul_mean_throughput_mbps_online_mean": float(frame["ul_mean_throughput_mbps_online"].mean()),
                "worker_request_failures_total": float(frame["worker_request_failures_online"].sum()),
                "worker_rtt_mean_ms_mean": float(frame["worker_rtt_mean_ms_online"].mean()),
                "worker_rtt_median_ms_mean": float(frame["worker_rtt_median_ms_online"].mean()),
                "worker_rtt_max_ms_mean": float(frame["worker_rtt_max_ms_online"].mean()),
                "handover_count_a3_mean": float(frame["handover_count_a3"].mean()),
                "ping_pong_rate_a3_mean": float(frame["ping_pong_rate_a3"].mean()),
                "mean_dwell_time_s_a3_mean": float(frame["mean_dwell_time_s_a3"].mean()),
                "dl_delivery_ratio_a3_mean": float(frame["dl_delivery_ratio_a3"].mean()),
                "ul_delivery_ratio_a3_mean": float(frame["ul_delivery_ratio_a3"].mean()),
                "dl_mean_throughput_mbps_a3_mean": float(frame["dl_mean_throughput_mbps_a3"].mean()),
                "ul_mean_throughput_mbps_a3_mean": float(frame["ul_mean_throughput_mbps_a3"].mean()),
                "handover_count_delta_mean": float(frame["handover_count_delta"].mean()),
                "ping_pong_rate_delta_mean": float(frame["ping_pong_rate_delta"].mean()),
                "mean_dwell_time_s_delta_mean": float(frame["mean_dwell_time_s_delta"].mean()),
                "dl_delivery_ratio_delta_mean": float(frame["dl_delivery_ratio_delta"].mean()),
                "ul_delivery_ratio_delta_mean": float(frame["ul_delivery_ratio_delta"].mean()),
                "dl_throughput_rel_delta_mean": float(frame["dl_throughput_rel_delta"].replace([np.inf, -np.inf], np.nan).mean()),
                "ul_throughput_rel_delta_mean": float(frame["ul_throughput_rel_delta"].replace([np.inf, -np.inf], np.nan).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("profile").reset_index(drop=True)


def write_best_policy_json(
    best_selection: dict[str, Any],
    output_path: Path,
) -> None:
    output_path.write_text(json.dumps(best_selection, indent=2), encoding="utf-8")


def generate_reports(args: argparse.Namespace, jobs: list[BatchJob], batch_root: Path, log_path: Path) -> None:
    reports_root = batch_root / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)

    per_run_rows = collect_completed_rows(args, jobs)
    per_run = pd.DataFrame(per_run_rows)
    summary = build_summary(per_run)
    ranking, selection = rank_profiles(
        summary,
        delivery_margin_abs=float(args.delivery_margin_abs),
        throughput_margin_rel=float(args.throughput_margin_rel),
        current_conservative_pingpong_baseline=float(args.current_conservative_pingpong_baseline),
    )

    completed_jobs = [job for job in jobs if job.status == "completed" and job.exit_code == 0]
    failed_jobs = [job for job in jobs if job.status == "failed"]

    per_run_path = reports_root / "overnight_per_run_results.csv"
    summary_path = reports_root / "overnight_batch_summary.csv"
    ranking_path = reports_root / "overnight_policy_ranking.csv"
    best_policy_path = reports_root / "best_overnight_policy.json"
    report_path = reports_root / "overnight_vs_a3_report.md"
    plots_path = reports_root / "overnight_plots.png"

    per_run.to_csv(per_run_path, index=False)
    summary.to_csv(summary_path, index=False)
    ranking.to_csv(ranking_path, index=False)
    write_best_policy_json(selection, best_policy_path)
    make_plots(summary, plots_path)

    profile_table = summary[
        [
            "profile",
            "completed_runs",
            "handover_count_online_mean",
            "ping_pong_rate_online_mean",
            "mean_dwell_time_s_online_mean",
            "dl_delivery_ratio_online_mean",
            "ul_delivery_ratio_online_mean",
            "dl_mean_throughput_mbps_online_mean",
            "ul_mean_throughput_mbps_online_mean",
            "worker_request_failures_total",
        ]
    ].copy() if not summary.empty else pd.DataFrame()

    ranking_table = ranking[
        [
            "profile",
            "stability_clean",
            "qos_within_margin",
            "balanced_score",
            "stability_score",
            "qos_score",
            "ping_pong_rate_online_mean",
            "dl_delivery_ratio_delta_mean",
            "ul_delivery_ratio_delta_mean",
            "dl_throughput_rel_delta_mean",
            "ul_throughput_rel_delta_mean",
        ]
    ].copy() if not ranking.empty else pd.DataFrame()

    summary_lines = [
        "# Overnight Online LSTM-Only Batch vs Existing A3",
        "",
        f"- Batch root: `{batch_root}`",
        f"- Matched runs: `{', '.join(str(run) for run in args.runs)}`",
        f"- Sim time: `{args.sim_time} s`",
        f"- Max parallel jobs: `{args.max_parallel}`",
        f"- Heavy traces enabled: `{bool(args.keep_heavy_traces)}`",
        "",
        "## Profiles",
        "",
    ]
    for profile in REQUIRED_PROFILES:
        summary_lines.append(
            f"- `{profile.name}`: trigger `{profile.trigger_threshold:.2f}`, "
            f"target `{profile.target_conf_threshold:.2f}`, gain `{profile.min_gain_rsrp_db:.1f}`, "
            f"confirm `{profile.consecutive_confirmation_steps}`, cooldown `{profile.cooldown_s:.1f}`, "
            f"anti-ping-pong `{profile.anti_ping_pong_window_s:.1f}`"
        )

    summary_lines.extend(
        [
            "",
            f"Completed jobs: `{len(completed_jobs)}/{len(jobs)}`",
            f"Failed jobs: `{len(failed_jobs)}`",
            "",
            "## Aggregate Summary",
            "",
            dataframe_to_markdown(profile_table),
            "",
            "## Ranking",
            "",
            dataframe_to_markdown(ranking_table),
            "",
        ]
    )

    if selection:
        summary_lines.extend(
            [
                "## Selected Policies",
                "",
                f"- Best balanced: `{selection['best_balanced']['profile']}`",
                f"- Best stability-first: `{selection['best_stability_first']['profile']}`",
                f"- Best QoS-first: `{selection['best_qos_first']['profile']}`",
                "",
            ]
        )

    summary_lines.extend(
        [
            "## Mobility Stability Conclusions",
            "",
            "- Priority rule is strict: no rapid repeat requests, no rapid cell returns, and ping-pong should stay at or below the current conservative online baseline.",
            "- Lower ping-pong is always preferred, but not if it comes from starving useful handovers so hard that QoS/QoE collapses.",
            "",
            "## QoS / QoE Conclusions",
            "",
            f"- Delivery ratio margin: `{args.delivery_margin_abs:.3f}` absolute against A3.",
            f"- Throughput margin: `{args.throughput_margin_rel:.3f}` relative against A3.",
            "- QoS-first ranking favors profiles that preserve or improve delivery/throughput while still keeping stability risk controlled.",
            "",
            "## Artifacts",
            "",
            f"- [overnight_batch_summary.csv]({summary_path})",
            f"- [overnight_per_run_results.csv]({per_run_path})",
            f"- [overnight_policy_ranking.csv]({ranking_path})",
            f"- [best_overnight_policy.json]({best_policy_path})",
            f"- [overnight_plots.png]({plots_path})",
            "",
            "## Job Status",
            "",
        ]
    )

    for job in jobs:
        summary_lines.append(
            f"- `{job.profile.name}` run `{job.run}`: `{job.status}`"
            + (f" (exit `{job.exit_code}`)" if job.exit_code is not None else "")
        )

    report_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    log_line(log_path, f"reports_refreshed completed_jobs={len(completed_jobs)}")


def main() -> int:
    args = build_argument_parser().parse_args()
    batch_root = make_batch_root(args.batch_root)
    batch_root.mkdir(parents=True, exist_ok=True)
    log_path = batch_root / "launcher.log"
    profiles_path = batch_root / "profiles.json"
    profiles_path.write_text(
        json.dumps([asdict(profile) for profile in REQUIRED_PROFILES], indent=2),
        encoding="utf-8",
    )

    if not args.skip_cleanup:
        cleanup_old_processes(log_path)
    else:
        log_line(log_path, "cleanup skipped by request")

    jobs = build_jobs(args, batch_root)
    log_line(
        log_path,
        f"prepared_jobs={len(jobs)} runs={args.runs} max_parallel={args.max_parallel} batch_root={batch_root}",
    )

    generate_reports(args, jobs, batch_root, log_path)
    launch_jobs(args, jobs, batch_root, log_path)
    generate_reports(args, jobs, batch_root, log_path)
    log_line(log_path, "overnight batch finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
