#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from final_900s_experiments import (
    MODE_LABELS,
    build_summary_frame,
    dataframe_to_markdown,
    summarize_run_metrics,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
MODE_NAME = "mlp_val_selected"
MODE_LABELS[MODE_NAME] = "MLP-only val-selected"

DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "online_runs/mlp_val_selected_matched_900s"
DEFAULT_NS3_BINARY = PROJECT_ROOT / "build/scratch/ns3.46.1-lte-oran-helper-lstm-only-hex7-optimized"
DEFAULT_WORKER_PYTHON = PROJECT_ROOT / "results_night/.venv/bin/python"
DEFAULT_WORKER_SCRIPT = SCRIPT_DIR / "persistent_mlp_worker.py"
DEFAULT_CHECKPOINT = (
    SCRIPT_DIR
    / "mlp_sweeps/paper_val_selected_20260419_010927/wide_h256_d15_lr5e4/best_model.pt"
)

ARTICLE_REFERENCE_ROWS = [
    {
        "policy": "A3 article mean",
        "runs": "1,3,4,5",
        "handover_count": 2329.50,
        "ping_pong_rate": 0.2673,
        "rapid_cell_returns": 630.75,
        "mean_dwell_time_s": 10.9621,
        "dl_delivery_ratio": 0.7401,
        "ul_delivery_ratio": 0.5222,
        "dl_mean_throughput_mbps": 33.8246,
        "ul_mean_throughput_mbps": 8.2520,
        "dl_delay_mean_ms": 422.4,
        "ul_delay_mean_ms": 2782.2,
    },
    {
        "policy": "LSTM-only article mean",
        "runs": "1,3,4,5",
        "handover_count": 1477.50,
        "ping_pong_rate": 0.1863,
        "rapid_cell_returns": 274.50,
        "mean_dwell_time_s": 17.1526,
        "dl_delivery_ratio": 0.6884,
        "ul_delivery_ratio": 0.4927,
        "dl_mean_throughput_mbps": 31.4682,
        "ul_mean_throughput_mbps": 7.7875,
        "dl_delay_mean_ms": 722.9,
        "ul_delay_mean_ms": 4341.0,
    },
    {
        "policy": "Hybrid LSTM+A3 article mean",
        "runs": "1,3,4,5",
        "handover_count": 2441.75,
        "ping_pong_rate": 0.2539,
        "rapid_cell_returns": 623.00,
        "mean_dwell_time_s": 10.4093,
        "dl_delivery_ratio": 0.7456,
        "ul_delivery_ratio": 0.5243,
        "dl_mean_throughput_mbps": 34.0699,
        "ul_mean_throughput_mbps": 8.2844,
        "dl_delay_mean_ms": 401.2,
        "ul_delay_mean_ms": 2880.7,
    },
]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run matched 900 s ns-3 online experiments for the validation-selected MLP baseline.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--ns3-binary", type=Path, default=DEFAULT_NS3_BINARY)
    parser.add_argument("--worker-python", type=Path, default=DEFAULT_WORKER_PYTHON)
    parser.add_argument("--worker-script", type=Path, default=DEFAULT_WORKER_SCRIPT)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--runs", type=int, nargs="+", default=[1, 3, 4, 5])
    parser.add_argument("--sim-time", type=float, default=900.0)
    parser.add_argument("--poll-interval-s", type=float, default=60.0)
    parser.add_argument("--trigger-threshold", type=float, default=0.85)
    parser.add_argument("--target-threshold", type=float, default=0.70)
    parser.add_argument("--min-gain-rsrp-db", type=float, default=1.0)
    parser.add_argument("--confirmation-steps", type=int, default=2)
    parser.add_argument("--cooldown-s", type=float, default=1.0)
    parser.add_argument("--anti-ping-pong-window-s", type=float, default=2.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-heavy-traces", action="store_true")
    return parser


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def latest_run_dir(output_root: Path, seed: int, run: int) -> Path | None:
    matches = sorted(
        output_root.glob(f"seed{seed}-run{run}-*"),
        key=lambda path: (path.stat().st_mtime, path.name),
    )
    return matches[-1] if matches else None


def run_is_complete(run_dir: Path | None, sim_time: float) -> bool:
    if run_dir is None or not run_dir.exists():
        return False
    run_info_path = run_dir / "run-info.txt"
    handover_path = run_dir / "handover-end.tr"
    if not run_info_path.exists() or not handover_path.exists():
        return False
    values: dict[str, str] = {}
    for line in run_info_path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return float(values.get("simTimeSec", 0.0) or 0.0) + 1e-9 >= float(sim_time)


def build_ns3_command(args: argparse.Namespace, run: int) -> list[str]:
    command = [
        str(args.ns3_binary),
        f"--seed={args.seed}",
        f"--run={run}",
        f"--sim-time={args.sim_time}",
        f"--outputRoot={args.output_root}",
        "--useOran=1",
        "--enableLstmController=1",
        "--useLteHandover=0",
        "--lstmDecisionIntervalSec=0.1",
        "--lstmSeqLen=15",
        f"--lstmMinConfidence={args.target_threshold:.2f}",
        f"--lstmCooldownSec={args.cooldown_s:.2f}",
        f"--lstmAntiPingPongWindowSec={args.anti_ping_pong_window_s:.2f}",
        f"--lstmTriggerThreshold={args.trigger_threshold:.2f}",
        f"--lstmTargetThreshold={args.target_threshold:.2f}",
        "--lstmUtilityThreshold=0.0",
        f"--lstmMinGainRsrpDb={args.min_gain_rsrp_db:.2f}",
        f"--lstmConsecutiveConfirmationSteps={args.confirmation_steps}",
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


def build_env() -> dict[str, str]:
    env = os.environ.copy()
    build_lib = str(PROJECT_ROOT / "build/lib")
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = build_lib if not existing else f"{build_lib}:{existing}"
    return env


def run_one(args: argparse.Namespace, run: int) -> dict[str, Any]:
    existing_dir = latest_run_dir(args.output_root, int(args.seed), int(run))
    if args.resume and run_is_complete(existing_dir, float(args.sim_time)):
        print(f"[run {run}] reusing completed run_dir={existing_dir}", flush=True)
        return {
            "run": int(run),
            "status": "reused",
            "exit_code": 0,
            "run_dir": str(existing_dir),
            "wall_runtime_s": 0.0,
        }

    command = build_ns3_command(args, run)
    log_path = args.output_root / f"seed{args.seed}-run{run}.stdout.log"
    command_path = args.output_root / f"seed{args.seed}-run{run}.command.txt"
    command_path.write_text(" ".join(command) + "\n", encoding="utf-8")

    started = time.time()
    print(
        f"[run {run}] starting at {datetime.now().isoformat(timespec='seconds')} "
        f"trigger={args.trigger_threshold:.2f} target={args.target_threshold:.2f}",
        flush=True,
    )
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=build_env(),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        while process.poll() is None:
            elapsed = time.time() - started
            print(f"[run {run}] still running elapsed={elapsed / 60.0:.1f} min", flush=True)
            time.sleep(float(args.poll_interval_s))
        exit_code = int(process.returncode)

    ended = time.time()
    run_dir = latest_run_dir(args.output_root, int(args.seed), int(run))
    print(
        f"[run {run}] finished exit={exit_code} elapsed={(ended - started) / 60.0:.1f} min "
        f"run_dir={run_dir}",
        flush=True,
    )
    return {
        "run": int(run),
        "status": "completed" if exit_code == 0 else "failed",
        "exit_code": exit_code,
        "run_dir": str(run_dir) if run_dir is not None else "",
        "log_path": str(log_path),
        "command_path": str(command_path),
        "launch_time_wall": started,
        "end_time_wall": ended,
        "wall_runtime_s": float(ended - started),
    }


def article_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    rows = list(ARTICLE_REFERENCE_ROWS)
    if not summary.empty:
        mlp = summary[summary["mode"] == MODE_NAME]
        if not mlp.empty:
            row = mlp.iloc[0]
            rows.append(
                {
                    "policy": "MLP-only val-selected online",
                    "runs": int(row["completed_runs"]),
                    "handover_count": float(row["handover_count_mean"]),
                    "ping_pong_rate": float(row["ping_pong_rate_mean"]),
                    "rapid_cell_returns": float(row["rapid_cell_returns_total"])
                    / max(1.0, float(row["completed_runs"])),
                    "mean_dwell_time_s": float(row["mean_dwell_time_s_mean"]),
                    "dl_delivery_ratio": float(row["dl_delivery_ratio_mean"]),
                    "ul_delivery_ratio": float(row["ul_delivery_ratio_mean"]),
                    "dl_mean_throughput_mbps": float(row["dl_mean_throughput_mbps_mean"]),
                    "ul_mean_throughput_mbps": float(row["ul_mean_throughput_mbps_mean"]),
                    "dl_delay_mean_ms": float(row["dl_delay_mean_ms_mean"]),
                    "ul_delay_mean_ms": float(row["ul_delay_mean_ms_mean"]),
                }
            )
    frame = pd.DataFrame(rows)
    for column in frame.columns:
        if column in {"policy", "runs"}:
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        frame[column] = numeric.map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")
    return frame


def summarize(args: argparse.Namespace, run_states: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    runtime_by_run = {int(row["run"]): row for row in run_states}
    for run in args.runs:
        state = runtime_by_run.get(int(run), {})
        run_dir_text = state.get("run_dir") or str(latest_run_dir(args.output_root, int(args.seed), int(run)) or "")
        run_dir = Path(run_dir_text)
        if not run_is_complete(run_dir, float(args.sim_time)):
            continue
        row = summarize_run_metrics(MODE_NAME, run_dir, float(args.sim_time))
        row["wall_runtime_s"] = float(state.get("wall_runtime_s", np.nan))
        rows.append(row)

    per_run = pd.DataFrame(rows)
    if not per_run.empty:
        per_run = per_run.sort_values("run").reset_index(drop=True)
    summary = build_summary_frame(per_run)
    return per_run, summary


def write_report(
    args: argparse.Namespace,
    run_states: list[dict[str, Any]],
    per_run: pd.DataFrame,
    summary: pd.DataFrame,
) -> Path:
    reports_root = args.output_root / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)

    per_run_path = reports_root / "mlp_matched_per_run.csv"
    summary_path = reports_root / "mlp_matched_summary.csv"
    comparison_path = reports_root / "mlp_vs_article_reference.csv"
    state_path = reports_root / "run_state.json"
    report_path = reports_root / "MLP_MATCHED_ONLINE_900S_REPORT.md"

    per_run.to_csv(per_run_path, index=False)
    summary.to_csv(summary_path, index=False)
    save_json(state_path, run_states)
    comparison = article_comparison(summary)
    comparison.to_csv(comparison_path, index=False)

    per_run_columns = [
        "run",
        "handover_count",
        "ping_pong_rate",
        "rapid_cell_returns",
        "mean_dwell_time_s",
        "dl_delivery_ratio",
        "ul_delivery_ratio",
        "dl_mean_throughput_mbps",
        "ul_mean_throughput_mbps",
        "dl_delay_mean_ms",
        "ul_delay_mean_ms",
        "worker_failed_requests",
        "worker_mean_rtt_ms",
        "wall_runtime_s",
        "run_dir",
    ]
    per_run_table = per_run.reindex(columns=per_run_columns) if not per_run.empty else pd.DataFrame()

    summary_columns = [
        "mode_label",
        "completed_runs",
        "handover_count_mean",
        "ping_pong_rate_mean",
        "rapid_cell_returns_total",
        "mean_dwell_time_s_mean",
        "dl_delivery_ratio_mean",
        "ul_delivery_ratio_mean",
        "dl_mean_throughput_mbps_mean",
        "ul_mean_throughput_mbps_mean",
        "dl_delay_mean_ms_mean",
        "ul_delay_mean_ms_mean",
        "worker_failed_requests_total",
        "worker_mean_rtt_ms_mean",
        "worker_max_rtt_ms_mean",
    ]
    summary_table = summary.reindex(columns=summary_columns) if not summary.empty else pd.DataFrame()

    lines = [
        "# MLP Matched Online 900 s Report",
        "",
        "## Setup",
        "",
        f"- Output root: `{args.output_root}`",
        f"- ns-3 binary: `{args.ns3_binary}`",
        f"- Worker python: `{args.worker_python}`",
        f"- Worker script: `{args.worker_script}`",
        f"- Checkpoint: `{args.checkpoint_path}`",
        f"- Seed: `{args.seed}`",
        f"- Runs: `{', '.join(str(run) for run in args.runs)}`",
        f"- Sim time: `{args.sim_time}` s",
        f"- Trigger threshold: `{args.trigger_threshold:.2f}`",
        f"- Target confidence threshold: `{args.target_threshold:.2f}`",
        f"- Min RSRP gain: `{args.min_gain_rsrp_db:.2f}` dB",
        f"- Confirmation steps: `{args.confirmation_steps}`",
        f"- Cooldown: `{args.cooldown_s:.2f}` s",
        f"- Anti-ping-pong window: `{args.anti_ping_pong_window_s:.2f}` s",
        f"- Heavy traces enabled: `{bool(args.keep_heavy_traces)}`",
        "",
        "## Aggregate",
        "",
        dataframe_to_markdown(summary_table),
        "",
        "## Per Run",
        "",
        dataframe_to_markdown(per_run_table),
        "",
        "## Article Reference Comparison",
        "",
        dataframe_to_markdown(comparison),
        "",
        "## Files",
        "",
        f"- Per-run CSV: `{per_run_path}`",
        f"- Summary CSV: `{summary_path}`",
        f"- Article comparison CSV: `{comparison_path}`",
        f"- State JSON: `{state_path}`",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> int:
    args = build_argument_parser().parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    # Keep the venv interpreter path itself. Path.resolve() follows the
    # venv symlink to /usr/bin/python and loses the installed packages.
    args.output_root = args.output_root.absolute()
    args.ns3_binary = args.ns3_binary.absolute()
    args.worker_python = args.worker_python.absolute()
    args.worker_script = args.worker_script.absolute()
    args.checkpoint_path = args.checkpoint_path.absolute()

    missing = [
        str(path)
        for path in [args.ns3_binary, args.worker_python, args.worker_script, args.checkpoint_path]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing required files: " + ", ".join(missing))

    save_json(
        args.output_root / "mlp_matched_args.json",
        {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    )

    run_states: list[dict[str, Any]] = []
    for run in args.runs:
        state = run_one(args, int(run))
        run_states.append(state)
        per_run, summary = summarize(args, run_states)
        report_path = write_report(args, run_states, per_run, summary)
        print(f"[run {run}] updated report: {report_path}", flush=True)
        if int(state.get("exit_code", 1)) != 0:
            return int(state.get("exit_code", 1))

    per_run, summary = summarize(args, run_states)
    report_path = write_report(args, run_states, per_run, summary)
    print(f"Final report: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
