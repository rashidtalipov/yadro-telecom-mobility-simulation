#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare online LSTM-only ns-3 runs against existing A3 baseline outputs.",
    )
    parser.add_argument("--online-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--time-limit-s", type=float, default=None)
    return parser


def read_run_info(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def read_trace(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep=r"\s+", engine="python")


def filter_time(frame: pd.DataFrame, time_limit_s: float | None) -> pd.DataFrame:
    if frame.empty or time_limit_s is None or "time" not in frame.columns:
        return frame
    return frame[pd.to_numeric(frame["time"], errors="coerce") <= float(time_limit_s)].copy()


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
    if not dwell_values:
        return float(sim_time_s)
    return float(np.mean(dwell_values))


def summarize_run(run_dir: Path, policy_name: str, time_limit_s: float | None) -> dict[str, Any]:
    run_info = read_run_info(run_dir / "run-info.txt")
    sim_time_s = float(time_limit_s if time_limit_s is not None else run_info.get("simTimeSec", 0.0))

    handover_end = filter_time(read_trace(run_dir / "handover-end.tr"), sim_time_s)
    dl_tx = filter_time(read_trace(run_dir / "dl-tx.tr"), sim_time_s)
    dl_rx = filter_time(read_trace(run_dir / "dl-rx.tr"), sim_time_s)
    ul_tx = filter_time(read_trace(run_dir / "ul-tx.tr"), sim_time_s)
    ul_rx = filter_time(read_trace(run_dir / "ul-rx.tr"), sim_time_s)

    handover_count = int(len(handover_end))
    ping_pong_count = int(pd.to_numeric(handover_end.get("isPingPong", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    ping_pong_rate = float(ping_pong_count / handover_count) if handover_count > 0 else 0.0
    mean_dwell_time_s = compute_dwell_time(handover_end, sim_time_s)
    handovers_per_ue = float(handover_count / max(1.0, float(run_info.get("numberOfUes", 30))))

    dl_tx_bytes = int(pd.to_numeric(dl_tx.get("bytes", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    dl_rx_bytes = int(pd.to_numeric(dl_rx.get("bytes", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    ul_tx_bytes = int(pd.to_numeric(ul_tx.get("bytes", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    ul_rx_bytes = int(pd.to_numeric(ul_rx.get("bytes", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())

    return {
        "policy_name": policy_name,
        "run_dir": str(run_dir),
        "seed": int(run_info.get("seed", 0)),
        "run": int(run_info.get("run", 0)),
        "sim_time_s": sim_time_s,
        "handover_count": handover_count,
        "handovers_per_ue": handovers_per_ue,
        "ping_pong_count": ping_pong_count,
        "ping_pong_rate": ping_pong_rate,
        "mean_dwell_time_s": mean_dwell_time_s,
        "dl_tx_bytes": dl_tx_bytes,
        "dl_rx_bytes": dl_rx_bytes,
        "ul_tx_bytes": ul_tx_bytes,
        "ul_rx_bytes": ul_rx_bytes,
        "dl_delivery_ratio": float(dl_rx_bytes / dl_tx_bytes) if dl_tx_bytes > 0 else 0.0,
        "ul_delivery_ratio": float(ul_rx_bytes / ul_tx_bytes) if ul_tx_bytes > 0 else 0.0,
        "dl_mean_throughput_mbps": float((dl_rx_bytes * 8.0) / max(sim_time_s, 1e-9) / 1e6),
        "ul_mean_throughput_mbps": float((ul_rx_bytes * 8.0) / max(sim_time_s, 1e-9) / 1e6),
    }


def find_online_runs(root: Path) -> list[Path]:
    return sorted(path.parent for path in root.rglob("run-info.txt"))


def find_matching_baseline(baseline_root: Path, seed: int, run: int) -> Path:
    pattern = f"seed{seed}-run{run}-*"
    matches = sorted(baseline_root.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No baseline run matched {pattern} under {baseline_root}")
    return matches[0]


def main() -> int:
    args = build_argument_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    online_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    skipped_runs: list[str] = []
    for online_run_dir in find_online_runs(args.online_root):
        run_info = read_run_info(online_run_dir / "run-info.txt")
        seed = int(run_info.get("seed", 0))
        run = int(run_info.get("run", 0))
        sim_time_s = float(args.time_limit_s if args.time_limit_s is not None else run_info.get("simTimeSec", 0.0))

        try:
            baseline_run_dir = find_matching_baseline(args.baseline_root, seed=seed, run=run)
        except FileNotFoundError:
            skipped_runs.append(f"seed={seed}, run={run}")
            continue
        online_rows.append(summarize_run(online_run_dir, "conservative_k3_online", sim_time_s))
        baseline_rows.append(summarize_run(baseline_run_dir, "a3_existing_baseline", sim_time_s))

    detail = pd.DataFrame(online_rows + baseline_rows).sort_values(["run", "policy_name"]).reset_index(drop=True)
    summary = (
        detail.groupby("policy_name", as_index=False)
        .agg(
            runs=("run", "nunique"),
            handover_count_mean=("handover_count", "mean"),
            handovers_per_ue_mean=("handovers_per_ue", "mean"),
            ping_pong_count_mean=("ping_pong_count", "mean"),
            ping_pong_rate_mean=("ping_pong_rate", "mean"),
            mean_dwell_time_s_mean=("mean_dwell_time_s", "mean"),
            dl_delivery_ratio_mean=("dl_delivery_ratio", "mean"),
            ul_delivery_ratio_mean=("ul_delivery_ratio", "mean"),
            dl_mean_throughput_mbps_mean=("dl_mean_throughput_mbps", "mean"),
            ul_mean_throughput_mbps_mean=("ul_mean_throughput_mbps", "mean"),
        )
        .reset_index(drop=True)
    )

    detail_path = args.output_dir / "online_vs_a3_detail.csv"
    summary_path = args.output_dir / "online_vs_a3_summary.csv"
    report_path = args.output_dir / "online_vs_a3_report.md"
    json_path = args.output_dir / "online_vs_a3_summary.json"

    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    json_path.write_text(json.dumps(summary.to_dict(orient="records"), indent=2), encoding="utf-8")

    report_lines = [
        "# Online Conservative K=3 vs Existing A3 Baseline",
        "",
        "This comparison uses only pre-existing A3 baseline outputs. No baseline reruns were performed.",
        "",
    ]
    if skipped_runs:
        report_lines.extend(
            [
                "Skipped online runs without matching existing baseline:",
                "",
                *[f"- {item}" for item in skipped_runs],
                "",
            ]
        )
    report_lines.extend(
        [
        dataframe_to_markdown(summary),
        "",
        "Per-run details:",
        "",
        dataframe_to_markdown(detail),
        ]
    )
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
