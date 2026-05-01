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
        description="Analyze short online LSTM-only runs and explain replay-vs-online policy gaps.",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--baseline-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--time-limit-s", type=float, default=73.0)
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


def summarize_reason_counts(decision_trace: pd.DataFrame) -> pd.DataFrame:
    if decision_trace.empty:
        return pd.DataFrame(columns=["action", "reason", "count"])
    summary = (
        decision_trace.groupby(["actualOrRequested", "reason"], as_index=False)
        .size()
        .rename(columns={"actualOrRequested": "action", "size": "count"})
        .sort_values(["count", "action", "reason"], ascending=[False, True, True], ignore_index=True)
    )
    return summary


def summarize_runtime_debug(debug_trace: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if debug_trace.empty:
        return (
            pd.DataFrame(columns=["runtimeAction", "runtimeReason", "count"]),
            pd.DataFrame(columns=["executed", "count", "avgTriggerProb", "avgTargetConfidence", "avgGainRsrpDb"]),
        )

    runtime_summary = (
        debug_trace.groupby(["runtimeAction", "runtimeReason"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["count", "runtimeAction", "runtimeReason"], ascending=[False, True, True], ignore_index=True)
    )
    executed_summary = (
        debug_trace.groupby("executed", as_index=False)
        .agg(
            count=("executed", "size"),
            avgTriggerProb=("triggerProb", "mean"),
            avgTargetConfidence=("targetConfidence", "mean"),
            avgGainRsrpDb=("gainRsrpDb", "mean"),
        )
        .sort_values("executed", ascending=False, ignore_index=True)
    )
    return runtime_summary, executed_summary


def find_repeated_requests(
    decision_trace: pd.DataFrame,
    window_s: float,
) -> pd.DataFrame:
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
    return pd.DataFrame(rows).sort_values(["delta_s", "imsi"], ascending=[True, True], ignore_index=True)


def find_rapid_returns(
    handover_start: pd.DataFrame,
    window_s: float,
) -> pd.DataFrame:
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
    return pd.DataFrame(rows).sort_values(["delta_s", "imsi"], ascending=[True, True], ignore_index=True)


def top_ues_by_handover_count(handover_end: pd.DataFrame) -> pd.DataFrame:
    if handover_end.empty:
        return pd.DataFrame(columns=["imsi", "handover_count", "ping_pong_count", "ping_pong_rate"])
    grouped = (
        handover_end.groupby("imsi", as_index=False)
        .agg(
            handover_count=("imsi", "size"),
            ping_pong_count=("isPingPong", lambda series: int(pd.to_numeric(series, errors="coerce").fillna(0).sum())),
        )
        .sort_values(["handover_count", "ping_pong_count", "imsi"], ascending=[False, False, True], ignore_index=True)
    )
    grouped["ping_pong_rate"] = grouped["ping_pong_count"] / grouped["handover_count"].clip(lower=1)
    return grouped


def build_policy_alignment_notes() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "area": "Trigger/target thresholds",
                "replay_policy": "Applied in replay candidate policy before a decision is accepted.",
                "old_online_behavior": "Applied inside Python runtime script.",
                "current_fix": "Online path now emits raw scores to ns-3 and applies the policy in the controller path.",
            },
            {
                "area": "Consecutive confirmation",
                "replay_policy": "Per-UE streak on the same proposed target.",
                "old_online_behavior": "Tracked in Python state file only.",
                "current_fix": "Per-UE streak moved into ns-3 controller state to mirror replay timing.",
            },
            {
                "area": "Cooldown",
                "replay_policy": "Starts at decision time via cooldown_until.",
                "old_online_behavior": "Only a post-HO guard in ns-3; not part of Python decision gating.",
                "current_fix": "Controller now applies decision-time cooldown before sending new requests.",
            },
            {
                "area": "Anti-ping-pong guard",
                "replay_policy": "blocked_returns[source_cell] prevents rapid return to the previous cell.",
                "old_online_behavior": "Missing in the online controller path.",
                "current_fix": "Controller now tracks blocked return cells and can enforce a 5 s guard.",
            },
            {
                "area": "Closed-loop effect",
                "replay_policy": "Offline open-loop evaluation; it does not change the future serving-cell timeline.",
                "old_online_behavior": "Every accepted HO changes the future online state and can create extra HO opportunities.",
                "current_fix": "Short smoke tests plus detailed logs identify which UEs drift into repeated HO loops.",
            },
        ]
    )


def main() -> int:
    args = build_argument_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    online_summary = summarize_run(args.run_dir, "online_lstm_only", args.time_limit_s)
    baseline_summary = summarize_run(args.baseline_run_dir, "existing_a3_baseline", args.time_limit_s)
    summary_frame = pd.DataFrame([online_summary, baseline_summary])

    decision_trace = filter_time(read_trace(args.run_dir / "handover-decision-source.tr"), args.time_limit_s)
    handover_start = filter_time(read_trace(args.run_dir / "handover-start.tr"), args.time_limit_s)
    handover_end = filter_time(read_trace(args.run_dir / "handover-end.tr"), args.time_limit_s)
    debug_trace = filter_time(read_trace(args.run_dir / "lstm-decision-debug.tr"), args.time_limit_s)

    reason_counts = summarize_reason_counts(decision_trace)
    runtime_reason_counts, executed_summary = summarize_runtime_debug(debug_trace)
    repeated_requests = find_repeated_requests(decision_trace, args.repeated_request_window_s)
    rapid_returns = find_rapid_returns(handover_start, args.rapid_return_window_s)
    top_ue_counts = top_ues_by_handover_count(handover_end)
    alignment_notes = build_policy_alignment_notes()

    repeated_summary = pd.DataFrame(
        [
            {
                "metric": "rapid_repeat_requests_within_window",
                "value": int(len(repeated_requests)),
            },
            {
                "metric": "same_target_repeat_requests_within_window",
                "value": int(repeated_requests["same_target"].sum()) if not repeated_requests.empty else 0,
            },
            {
                "metric": "rapid_cell_returns_within_window",
                "value": int(len(rapid_returns)),
            },
        ]
    )

    summary_frame.to_csv(args.output_dir / "run_vs_baseline_summary.csv", index=False)
    reason_counts.to_csv(args.output_dir / "policy_reason_counts.csv", index=False)
    runtime_reason_counts.to_csv(args.output_dir / "runtime_reason_counts.csv", index=False)
    executed_summary.to_csv(args.output_dir / "executed_vs_blocked_summary.csv", index=False)
    repeated_requests.to_csv(args.output_dir / "repeated_requests.csv", index=False)
    rapid_returns.to_csv(args.output_dir / "rapid_returns.csv", index=False)
    top_ue_counts.to_csv(args.output_dir / "top_ue_handover_counts.csv", index=False)
    alignment_notes.to_csv(args.output_dir / "policy_alignment_notes.csv", index=False)
    repeated_summary.to_csv(args.output_dir / "repeated_event_summary.csv", index=False)

    report_lines = [
        "# Online Conservative Policy Debug Report",
        "",
        f"- Online run: `{args.run_dir}`",
        f"- Baseline run: `{args.baseline_run_dir}`",
        f"- Time window: `{args.time_limit_s:.1f} s`",
        "",
        "## Run vs Baseline",
        "",
        dataframe_to_markdown(summary_frame),
        "",
        "## Policy Alignment Notes",
        "",
        dataframe_to_markdown(alignment_notes),
        "",
        "## Repeated Event Summary",
        "",
        dataframe_to_markdown(repeated_summary),
        "",
        "## Top Policy Reasons",
        "",
        dataframe_to_markdown(reason_counts.head(15)),
        "",
        "## Runtime Debug Reasons",
        "",
        dataframe_to_markdown(runtime_reason_counts.head(15)),
        "",
        "## Executed vs Blocked Decisions",
        "",
        dataframe_to_markdown(executed_summary),
        "",
        "## UEs with Highest HO Counts",
        "",
        dataframe_to_markdown(top_ue_counts.head(10)),
        "",
        "## Repeated HO Requests",
        "",
        dataframe_to_markdown(repeated_requests.head(20)),
        "",
        "## Rapid Cell Returns",
        "",
        dataframe_to_markdown(rapid_returns.head(20)),
    ]
    (args.output_dir / "debug_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "online_summary": online_summary,
                "baseline_summary": baseline_summary,
                "repeated_event_summary": repeated_summary.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
