#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
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
        description="Aggregate multi-run online LSTM-only validation results against existing A3 baselines.",
    )
    parser.add_argument(
        "--online-root",
        type=Path,
        action="append",
        required=True,
        help="Online run root. Can be passed multiple times.",
    )
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--time-limit-s", type=float, default=60.0)
    parser.add_argument("--rapid-return-window-s", type=float, default=5.0)
    parser.add_argument("--repeated-request-window-s", type=float, default=1.0)
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
    if not path.exists() or path.stat().st_size == 0:
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
    return float(np.mean(dwell_values)) if dwell_values else float(sim_time_s)


def summarize_run(run_dir: Path, policy_name: str, time_limit_s: float) -> dict[str, Any]:
    run_info = read_run_info(run_dir / "run-info.txt")
    handover_end = filter_time(read_trace(run_dir / "handover-end.tr"), time_limit_s)
    dl_tx = filter_time(read_trace(run_dir / "dl-tx.tr"), time_limit_s)
    dl_rx = filter_time(read_trace(run_dir / "dl-rx.tr"), time_limit_s)
    ul_tx = filter_time(read_trace(run_dir / "ul-tx.tr"), time_limit_s)
    ul_rx = filter_time(read_trace(run_dir / "ul-rx.tr"), time_limit_s)

    handover_count = int(len(handover_end))
    ping_pong_count = int(
        pd.to_numeric(handover_end.get("isPingPong", pd.Series(dtype=float)), errors="coerce")
        .fillna(0)
        .sum()
    )
    dl_tx_bytes = int(pd.to_numeric(dl_tx.get("bytes", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    dl_rx_bytes = int(pd.to_numeric(dl_rx.get("bytes", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    ul_tx_bytes = int(pd.to_numeric(ul_tx.get("bytes", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    ul_rx_bytes = int(pd.to_numeric(ul_rx.get("bytes", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())

    return {
        "policy_name": policy_name,
        "run_dir": str(run_dir),
        "seed": int(run_info.get("seed", 0)),
        "run": int(run_info.get("run", 0)),
        "time_limit_s": float(time_limit_s),
        "handover_count": handover_count,
        "handovers_per_ue": float(handover_count / max(1.0, float(run_info.get("numberOfUes", 30)))),
        "ping_pong_count": ping_pong_count,
        "ping_pong_rate": float(ping_pong_count / handover_count) if handover_count > 0 else 0.0,
        "mean_dwell_time_s": compute_dwell_time(handover_end, time_limit_s),
        "dl_delivery_ratio": float(dl_rx_bytes / dl_tx_bytes) if dl_tx_bytes > 0 else 0.0,
        "ul_delivery_ratio": float(ul_rx_bytes / ul_tx_bytes) if ul_tx_bytes > 0 else 0.0,
        "dl_mean_throughput_mbps": float((dl_rx_bytes * 8.0) / max(time_limit_s, 1e-9) / 1e6),
        "ul_mean_throughput_mbps": float((ul_rx_bytes * 8.0) / max(time_limit_s, 1e-9) / 1e6),
    }


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
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)


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
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)


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


def find_online_runs(roots: list[Path]) -> list[Path]:
    seen: dict[str, Path] = {}
    for root in roots:
        for run_info in sorted(root.rglob("run-info.txt")):
            run_dir = run_info.parent
            key = str(run_dir.resolve())
            seen[key] = run_dir
    return sorted(seen.values())


def find_matching_baseline(baseline_root: Path, seed: int, run: int) -> Path:
    pattern = f"seed{seed}-run{run}-*"
    matches = sorted(baseline_root.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No baseline run matched {pattern} under {baseline_root}")
    return matches[0]


def collect_per_run_row(
    online_run_dir: Path,
    baseline_root: Path,
    time_limit_s: float,
    repeated_request_window_s: float,
    rapid_return_window_s: float,
) -> dict[str, Any]:
    online_run_info = read_run_info(online_run_dir / "run-info.txt")
    seed = int(online_run_info.get("seed", 0))
    run = int(online_run_info.get("run", 0))
    baseline_run_dir = find_matching_baseline(baseline_root, seed=seed, run=run)

    online_summary = summarize_run(online_run_dir, "conservative_k3_online", time_limit_s)
    baseline_summary = summarize_run(baseline_run_dir, "a3_existing_baseline", time_limit_s)

    decision_trace = filter_time(read_trace(online_run_dir / "handover-decision-source.tr"), time_limit_s)
    handover_start = filter_time(read_trace(online_run_dir / "handover-start.tr"), time_limit_s)
    repeated_requests = find_repeated_requests(decision_trace, repeated_request_window_s)
    rapid_returns = find_rapid_returns(handover_start, rapid_return_window_s)
    rtt_summary = summarize_worker_rtt(online_run_dir)

    row: dict[str, Any] = {
        "seed": seed,
        "run": run,
        "online_run_dir": str(online_run_dir),
        "baseline_run_dir": str(baseline_run_dir),
        "time_limit_s": float(time_limit_s),
        "rapid_repeat_requests": int(len(repeated_requests)),
        "same_target_repeat_requests": int(
            pd.to_numeric(repeated_requests.get("same_target", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
        ),
        "rapid_cell_returns": int(len(rapid_returns)),
    }

    for key, value in online_summary.items():
        if key in {"policy_name", "run_dir", "seed", "run", "time_limit_s"}:
            continue
        row[f"{key}_online"] = value
    for key, value in baseline_summary.items():
        if key in {"policy_name", "run_dir", "seed", "run", "time_limit_s"}:
            continue
        row[f"{key}_a3"] = value
        if f"{key}_online" in row and isinstance(row[f"{key}_online"], (int, float)) and isinstance(value, (int, float)):
            row[f"{key}_delta"] = float(row[f"{key}_online"]) - float(value)

    row.update(rtt_summary)
    row["stability_win"] = int(
        row["ping_pong_rate_online"] <= row["ping_pong_rate_a3"]
        and row["mean_dwell_time_s_online"] >= row["mean_dwell_time_s_a3"]
        and row["rapid_repeat_requests"] == 0
        and row["rapid_cell_returns"] == 0
    )
    row["throughput_win"] = int(
        row["dl_mean_throughput_mbps_online"] >= row["dl_mean_throughput_mbps_a3"]
        and row["ul_mean_throughput_mbps_online"] >= row["ul_mean_throughput_mbps_a3"]
    )
    row["delivery_win"] = int(
        row["dl_delivery_ratio_online"] >= row["dl_delivery_ratio_a3"]
        and row["ul_delivery_ratio_online"] >= row["ul_delivery_ratio_a3"]
    )
    row["degraded_badly"] = int(
        row["ping_pong_rate_online"] > row["ping_pong_rate_a3"]
        or row["rapid_repeat_requests"] > 0
        or row["rapid_cell_returns"] > 0
        or row["worker_request_failures"] > 0
        or (
            row["dl_mean_throughput_mbps_online"] < row["dl_mean_throughput_mbps_a3"]
            and row["ul_mean_throughput_mbps_online"] < row["ul_mean_throughput_mbps_a3"]
        )
    )
    return row


def build_summary_frame(per_run: pd.DataFrame) -> pd.DataFrame:
    online_metrics = [
        "handover_count_online",
        "ping_pong_rate_online",
        "mean_dwell_time_s_online",
        "dl_delivery_ratio_online",
        "ul_delivery_ratio_online",
        "dl_mean_throughput_mbps_online",
        "ul_mean_throughput_mbps_online",
        "rapid_repeat_requests",
        "rapid_cell_returns",
        "worker_request_failures",
        "worker_rtt_mean_ms",
        "worker_rtt_median_ms",
        "worker_rtt_max_ms",
    ]
    baseline_metrics = [
        "handover_count_a3",
        "ping_pong_rate_a3",
        "mean_dwell_time_s_a3",
        "dl_delivery_ratio_a3",
        "ul_delivery_ratio_a3",
        "dl_mean_throughput_mbps_a3",
        "ul_mean_throughput_mbps_a3",
    ]

    rows: list[dict[str, Any]] = []
    for label, metrics in (
        ("conservative_k3_online", online_metrics),
        ("a3_existing_baseline", baseline_metrics),
    ):
        row: dict[str, Any] = {"policy_name": label, "runs": int(len(per_run))}
        for metric in metrics:
            row[f"{metric}_mean"] = float(per_run[metric].mean())
            row[f"{metric}_std"] = float(per_run[metric].std(ddof=0))
        rows.append(row)

    delta_metrics = [
        "handover_count_delta",
        "ping_pong_rate_delta",
        "mean_dwell_time_s_delta",
        "dl_delivery_ratio_delta",
        "ul_delivery_ratio_delta",
        "dl_mean_throughput_mbps_delta",
        "ul_mean_throughput_mbps_delta",
    ]
    delta_row: dict[str, Any] = {"policy_name": "online_minus_a3", "runs": int(len(per_run))}
    for metric in delta_metrics:
        delta_row[f"{metric}_mean"] = float(per_run[metric].mean())
        delta_row[f"{metric}_std"] = float(per_run[metric].std(ddof=0))
    delta_row["stability_win_rate"] = float(per_run["stability_win"].mean())
    delta_row["throughput_win_rate"] = float(per_run["throughput_win"].mean())
    delta_row["delivery_win_rate"] = float(per_run["delivery_win"].mean())
    delta_row["degraded_badly_count"] = int(per_run["degraded_badly"].sum())
    rows.append(delta_row)
    return pd.DataFrame(rows)


def build_report_tables(per_run: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    aggregate_table = pd.DataFrame(
        [
            {
                "policy": "A3 baseline",
                "runs": int(len(per_run)),
                "handover_count_mean": round(float(per_run["handover_count_a3"].mean()), 4),
                "ping_pong_rate_mean": round(float(per_run["ping_pong_rate_a3"].mean()), 4),
                "mean_dwell_time_s_mean": round(float(per_run["mean_dwell_time_s_a3"].mean()), 4),
                "dl_delivery_ratio_mean": round(float(per_run["dl_delivery_ratio_a3"].mean()), 4),
                "ul_delivery_ratio_mean": round(float(per_run["ul_delivery_ratio_a3"].mean()), 4),
                "dl_mean_throughput_mbps_mean": round(float(per_run["dl_mean_throughput_mbps_a3"].mean()), 4),
                "ul_mean_throughput_mbps_mean": round(float(per_run["ul_mean_throughput_mbps_a3"].mean()), 4),
            },
            {
                "policy": "Online conservative_k3",
                "runs": int(len(per_run)),
                "handover_count_mean": round(float(per_run["handover_count_online"].mean()), 4),
                "ping_pong_rate_mean": round(float(per_run["ping_pong_rate_online"].mean()), 4),
                "mean_dwell_time_s_mean": round(float(per_run["mean_dwell_time_s_online"].mean()), 4),
                "dl_delivery_ratio_mean": round(float(per_run["dl_delivery_ratio_online"].mean()), 4),
                "ul_delivery_ratio_mean": round(float(per_run["ul_delivery_ratio_online"].mean()), 4),
                "dl_mean_throughput_mbps_mean": round(float(per_run["dl_mean_throughput_mbps_online"].mean()), 4),
                "ul_mean_throughput_mbps_mean": round(float(per_run["ul_mean_throughput_mbps_online"].mean()), 4),
            },
        ]
    )

    per_run_table = per_run[
        [
            "seed",
            "run",
            "handover_count_online",
            "handover_count_a3",
            "ping_pong_rate_online",
            "ping_pong_rate_a3",
            "mean_dwell_time_s_online",
            "mean_dwell_time_s_a3",
            "dl_delivery_ratio_online",
            "dl_delivery_ratio_a3",
            "ul_delivery_ratio_online",
            "ul_delivery_ratio_a3",
            "dl_mean_throughput_mbps_online",
            "dl_mean_throughput_mbps_a3",
            "ul_mean_throughput_mbps_online",
            "ul_mean_throughput_mbps_a3",
            "rapid_repeat_requests",
            "rapid_cell_returns",
            "worker_request_failures",
            "worker_rtt_mean_ms",
            "worker_rtt_median_ms",
            "worker_rtt_max_ms",
            "stability_win",
            "throughput_win",
            "delivery_win",
            "degraded_badly",
        ]
    ].copy()
    return aggregate_table, per_run_table


def make_plots(per_run: pd.DataFrame, output_path: Path) -> None:
    runs = [f"run{int(run)}" for run in per_run["run"]]
    x = np.arange(len(runs))
    width = 0.38

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].bar(x - width / 2, per_run["handover_count_a3"], width, label="A3")
    axes[0, 0].bar(x + width / 2, per_run["handover_count_online"], width, label="Online")
    axes[0, 0].set_title("Handover Count")
    axes[0, 0].set_xticks(x, runs)
    axes[0, 0].legend()

    axes[0, 1].bar(x - width / 2, per_run["ping_pong_rate_a3"], width, label="A3")
    axes[0, 1].bar(x + width / 2, per_run["ping_pong_rate_online"], width, label="Online")
    axes[0, 1].set_title("Ping-Pong Rate")
    axes[0, 1].set_xticks(x, runs)

    axes[1, 0].bar(x - width / 2, per_run["mean_dwell_time_s_a3"], width, label="A3")
    axes[1, 0].bar(x + width / 2, per_run["mean_dwell_time_s_online"], width, label="Online")
    axes[1, 0].set_title("Mean Dwell Time (s)")
    axes[1, 0].set_xticks(x, runs)

    axes[1, 1].plot(x, per_run["dl_mean_throughput_mbps_a3"], marker="o", label="DL A3")
    axes[1, 1].plot(x, per_run["dl_mean_throughput_mbps_online"], marker="o", label="DL Online")
    axes[1, 1].plot(x, per_run["ul_mean_throughput_mbps_a3"], marker="s", label="UL A3")
    axes[1, 1].plot(x, per_run["ul_mean_throughput_mbps_online"], marker="s", label="UL Online")
    axes[1, 1].set_title("Throughput (Mbps)")
    axes[1, 1].set_xticks(x, runs)
    axes[1, 1].legend()

    for axis in axes.flat:
        axis.grid(True, alpha=0.25)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> int:
    args = build_argument_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    online_run_dirs = find_online_runs(args.online_root)
    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for run_dir in online_run_dirs:
        try:
            row = collect_per_run_row(
                online_run_dir=run_dir,
                baseline_root=args.baseline_root,
                time_limit_s=float(args.time_limit_s),
                repeated_request_window_s=float(args.repeated_request_window_s),
                rapid_return_window_s=float(args.rapid_return_window_s),
            )
            rows.append(row)
        except FileNotFoundError as exc:
            skipped.append(str(exc))

    if not rows:
        raise SystemExit("No matched online runs were found for batch validation.")

    per_run = pd.DataFrame(rows).sort_values(["seed", "run"]).reset_index(drop=True)
    summary = build_summary_frame(per_run)
    aggregate_table, per_run_table = build_report_tables(per_run)

    consistency = (
        "Online gains remain consistent across the batch."
        if int(per_run["degraded_badly"].sum()) == 0
        else "One or more runs degrade materially relative to A3; see the failure analysis section."
    )

    failure_rows = per_run[per_run["degraded_badly"] > 0].copy()
    if failure_rows.empty:
        failure_text = "No runs met the degraded-badly criterion."
    else:
        failure_lines = []
        for _, row in failure_rows.iterrows():
            issues: list[str] = []
            if row["ping_pong_rate_online"] > row["ping_pong_rate_a3"]:
                issues.append("higher ping-pong")
            if row["rapid_repeat_requests"] > 0:
                issues.append("rapid repeat requests")
            if row["rapid_cell_returns"] > 0:
                issues.append("rapid cell returns")
            if row["worker_request_failures"] > 0:
                issues.append(f"{int(row['worker_request_failures'])} worker request failures")
            if (
                row["dl_mean_throughput_mbps_online"] < row["dl_mean_throughput_mbps_a3"]
                and row["ul_mean_throughput_mbps_online"] < row["ul_mean_throughput_mbps_a3"]
            ):
                issues.append("both DL and UL throughput below baseline")
            failure_lines.append(f"- run={int(row['run'])}: " + ", ".join(issues))
        failure_text = "\n".join(failure_lines)

    summary_path = args.output_dir / "online_batch_summary.csv"
    per_run_path = args.output_dir / "per_run_online_vs_a3.csv"
    report_path = args.output_dir / "online_vs_a3_batch_report.md"
    plots_path = args.output_dir / "comparison_plots.png"
    json_path = args.output_dir / "online_batch_summary.json"

    summary.to_csv(summary_path, index=False)
    per_run.to_csv(per_run_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "consistency": consistency,
                "skipped": skipped,
                "summary": summary.to_dict(orient="records"),
                "per_run": per_run.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    make_plots(per_run, plots_path)

    report_lines = [
        "# Online LSTM-Only Batch Validation vs Existing A3",
        "",
        f"- Time window: `{float(args.time_limit_s):.1f} s`",
        f"- Online roots: {', '.join(str(path) for path in args.online_root)}",
        f"- Baseline root: `{args.baseline_root}`",
        "",
        "## Aggregate Comparison",
        "",
        dataframe_to_markdown(aggregate_table),
        "",
        "## Per-Run Comparison",
        "",
        dataframe_to_markdown(per_run_table),
        "",
        "## Consistency Assessment",
        "",
        consistency,
        "",
        f"- Stability win rate: `{float(per_run['stability_win'].mean()):.3f}`",
        f"- Throughput win rate: `{float(per_run['throughput_win'].mean()):.3f}`",
        f"- Delivery win rate: `{float(per_run['delivery_win'].mean()):.3f}`",
        "",
        "## Failure Analysis",
        "",
        failure_text,
    ]
    if skipped:
        report_lines.extend(["", "## Skipped Runs", "", *[f"- {item}" for item in skipped]])
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
