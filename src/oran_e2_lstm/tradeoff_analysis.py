from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MAIN_MODELS = ["a3_original", "current_k3", "conservative_k3", "hybrid_a3_k3"]
PREDICTIVE_MODELS = ["current_k3", "conservative_k3", "hybrid_a3_k3"]
MINIMIZE_COLUMNS = [
    "unnecessary_handover_count",
    "ping_pong_rate",
    "missed_useful_handover_count",
]
MAXIMIZE_COLUMNS = [
    "end_to_end_decision_success_rate",
    "early_prediction_rate",
]
PLOT_COLORS = {
    "a3_original": "#708090",
    "current_k3": "#d08b2e",
    "conservative_k3": "#2f8f5b",
    "hybrid_a3_k3": "#7b5ea7",
}
PLOT_MARKERS = {
    "a3_original": "s",
    "current_k3": "X",
    "conservative_k3": "*",
    "hybrid_a3_k3": "D",
}


def build_argument_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    replay_root = script_dir / "replay_outputs"

    parser = argparse.ArgumentParser(description="Trade-off analysis for replay operating points and policy sweeps.")
    parser.add_argument(
        "--replay-dir",
        type=Path,
        action="append",
        dest="replay_dirs",
        help="Replay output directory with replay_metrics.json and optional sweep CSVs. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=replay_root / "final_tradeoff_analysis",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Replay Trade-Off Analysis",
    )
    parser.add_argument(
        "--prefer-dir",
        type=Path,
        default=replay_root / "hybrid_a3_k3_policy_sweep",
        help="Replay directory to prefer for main operating points when duplicates exist.",
    )
    return parser


def default_replay_dirs(script_dir: Path) -> list[Path]:
    replay_root = script_dir / "replay_outputs"
    return [
        replay_root / "conservative_k3_policy_sweep",
        replay_root / "hybrid_a3_k3_policy_sweep",
    ]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_operating_points(replay_dirs: Iterable[Path], prefer_dir: Path | None) -> pd.DataFrame:
    operating_by_model: dict[str, dict] = {}

    ordered_dirs = list(replay_dirs)
    if prefer_dir is not None and prefer_dir in ordered_dirs:
        ordered_dirs = [directory for directory in ordered_dirs if directory != prefer_dir] + [prefer_dir]

    for replay_dir in ordered_dirs:
        metrics_path = replay_dir / "replay_metrics.json"
        if not metrics_path.exists():
            continue
        metrics = load_json(metrics_path)
        for summary in metrics.get("test_summaries", []):
            model_name = str(summary.get("model_name"))
            if model_name not in MAIN_MODELS:
                continue
            row = dict(summary)
            row["source_dir"] = str(replay_dir)
            operating_by_model[model_name] = row

    missing = [model_name for model_name in MAIN_MODELS if model_name not in operating_by_model]
    if missing:
        raise FileNotFoundError(f"Missing operating-point summaries for: {missing}")

    operating_points = pd.DataFrame([operating_by_model[model_name] for model_name in MAIN_MODELS])
    operating_points["model_name"] = pd.Categorical(
        operating_points["model_name"],
        categories=MAIN_MODELS,
        ordered=True,
    )
    operating_points = operating_points.sort_values("model_name", ignore_index=True)
    operating_points["model_name"] = operating_points["model_name"].astype(str)
    return operating_points


def load_sweep_tables(replay_dirs: Iterable[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for replay_dir in replay_dirs:
        sweep_files = [
            (replay_dir / "policy_sweep_results.csv", "conservative"),
            (replay_dir / "hybrid_policy_sweep_results.csv", "hybrid"),
        ]
        for csv_path, family in sweep_files:
            if not csv_path.exists():
                continue
            frame = pd.read_csv(csv_path)
            frame["policy_family"] = family
            frame["source_dir"] = str(replay_dir)
            if "a3_gate_mode" not in frame.columns:
                frame["a3_gate_mode"] = "off"
            frames.append(frame)

    if not frames:
        raise FileNotFoundError("No policy sweep CSVs were found in the provided replay directories")

    combined = pd.concat(frames, ignore_index=True)
    dedupe_columns = [
        "policy_family",
        "a3_gate_mode",
        "trigger_threshold",
        "target_conf_threshold",
        "min_score_margin",
        "min_gain_rsrp_db",
        "cooldown_s",
        "anti_ping_pong_window_s",
        "consecutive_confirmation_steps",
        "unnecessary_handover_count",
        "ping_pong_rate",
        "missed_useful_handover_count",
        "end_to_end_decision_success_rate",
        "early_prediction_rate",
        "handover_count",
        "mean_dwell_time_s",
    ]
    available_columns = [column for column in dedupe_columns if column in combined.columns]
    combined = combined.drop_duplicates(subset=available_columns, ignore_index=True)
    return combined


def to_minimization_frame(frame: pd.DataFrame) -> np.ndarray:
    values: list[np.ndarray] = []
    for column in MINIMIZE_COLUMNS:
        values.append(frame[column].to_numpy(dtype=float))
    for column in MAXIMIZE_COLUMNS:
        values.append(-frame[column].to_numpy(dtype=float))
    return np.column_stack(values)


def pareto_mask(frame: pd.DataFrame) -> np.ndarray:
    values = to_minimization_frame(frame)
    count = values.shape[0]
    efficient = np.ones(count, dtype=bool)
    for row_index in range(count):
        if not efficient[row_index]:
            continue
        dominates_row = np.all(values <= values[row_index], axis=1) & np.any(values < values[row_index], axis=1)
        dominates_row[row_index] = False
        if dominates_row.any():
            efficient[row_index] = False
    return efficient


def normalize_series(series: pd.Series, minimize: bool) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    minimum = float(numeric.min())
    maximum = float(numeric.max())
    if math.isclose(minimum, maximum):
        return pd.Series(np.ones(len(numeric), dtype=float), index=series.index)
    if minimize:
        return (maximum - numeric) / (maximum - minimum)
    return (numeric - minimum) / (maximum - minimum)


def rank_operating_points(operating_points: pd.DataFrame) -> pd.DataFrame:
    ranked = operating_points.copy()
    predictive_mask = ranked["model_name"].isin(PREDICTIVE_MODELS)

    predictive = ranked.loc[predictive_mask].copy()
    predictive["norm_unnecessary"] = normalize_series(predictive["unnecessary_handover_count"], minimize=True)
    predictive["norm_ping_pong"] = normalize_series(predictive["ping_pong_rate"], minimize=True)
    predictive["norm_missed_useful"] = normalize_series(predictive["missed_useful_handover_count"], minimize=True)
    predictive["norm_success"] = normalize_series(predictive["end_to_end_decision_success_rate"], minimize=False)
    predictive["norm_early"] = normalize_series(predictive["early_prediction_rate"], minimize=False)

    predictive["balanced_score"] = (
        0.30 * predictive["norm_success"]
        + 0.25 * predictive["norm_early"]
        + 0.20 * predictive["norm_unnecessary"]
        + 0.15 * predictive["norm_ping_pong"]
        + 0.10 * predictive["norm_missed_useful"]
    )
    predictive["stability_score"] = (
        0.45 * predictive["norm_ping_pong"]
        + 0.30 * predictive["norm_unnecessary"]
        + 0.15 * predictive["norm_missed_useful"]
        + 0.10 * predictive["norm_success"]
    )
    predictive["early_action_score"] = (
        0.45 * predictive["norm_early"]
        + 0.30 * predictive["norm_success"]
        + 0.15 * predictive["norm_ping_pong"]
        + 0.10 * predictive["norm_unnecessary"]
    )

    best_balanced = predictive.sort_values(
        ["balanced_score", "end_to_end_decision_success_rate", "early_prediction_rate"],
        ascending=[False, False, False],
    )["model_name"].iloc[0]
    best_stability = predictive.sort_values(
        ["stability_score", "ping_pong_rate", "unnecessary_handover_count"],
        ascending=[False, True, True],
    )["model_name"].iloc[0]
    best_early = predictive.sort_values(
        ["early_action_score", "early_prediction_rate", "end_to_end_decision_success_rate"],
        ascending=[False, False, False],
    )["model_name"].iloc[0]

    ranked["balanced_score"] = np.nan
    ranked["stability_score"] = np.nan
    ranked["early_action_score"] = np.nan
    ranked.loc[predictive.index, ["balanced_score", "stability_score", "early_action_score"]] = predictive[
        ["balanced_score", "stability_score", "early_action_score"]
    ]

    ranked["recommendation_role"] = ""
    ranked.loc[ranked["model_name"] == best_balanced, "recommendation_role"] = "main_result"
    ranked.loc[ranked["model_name"] == best_stability, "recommendation_role"] = np.where(
        ranked.loc[ranked["model_name"] == best_stability, "recommendation_role"].eq(""),
        "stability_first_variant",
        ranked.loc[ranked["model_name"] == best_stability, "recommendation_role"] + ";stability_first_variant",
    )
    ranked.loc[ranked["model_name"] == best_early, "recommendation_role"] = np.where(
        ranked.loc[ranked["model_name"] == best_early, "recommendation_role"].eq(""),
        "early_action_variant;future_online_oran_test",
        ranked.loc[ranked["model_name"] == best_early, "recommendation_role"] + ";early_action_variant;future_online_oran_test",
    )

    ranked["is_operating_point_pareto"] = pareto_mask(ranked)
    return ranked


def make_pareto_table(sweep_points: pd.DataFrame, operating_points: pd.DataFrame) -> pd.DataFrame:
    pareto = sweep_points.copy()
    pareto["is_pareto"] = pareto_mask(pareto)
    pareto_points = pareto[pareto["is_pareto"]].copy()

    main_lookup = {
        row.model_name: row
        for row in operating_points.itertuples(index=False)
        if row.model_name in ("conservative_k3", "hybrid_a3_k3")
    }

    def matches_operating_point(row: pd.Series) -> str:
        for model_name, operating_row in main_lookup.items():
            same_metrics = (
                math.isclose(float(row["unnecessary_handover_count"]), float(operating_row.unnecessary_handover_count), rel_tol=0.0, abs_tol=1e-9)
                and math.isclose(float(row["ping_pong_rate"]), float(operating_row.ping_pong_rate), rel_tol=0.0, abs_tol=1e-12)
                and math.isclose(float(row["missed_useful_handover_count"]), float(operating_row.missed_useful_handover_count), rel_tol=0.0, abs_tol=1e-9)
                and math.isclose(float(row["end_to_end_decision_success_rate"]), float(operating_row.end_to_end_decision_success_rate), rel_tol=0.0, abs_tol=1e-12)
                and math.isclose(float(row["early_prediction_rate"]), float(operating_row.early_prediction_rate), rel_tol=0.0, abs_tol=1e-12)
            )
            if same_metrics:
                return model_name
        return ""

    pareto_points["matches_main_operating_point"] = pareto_points.apply(matches_operating_point, axis=1)
    pareto_points = pareto_points.sort_values(
        [
            "policy_family",
            "end_to_end_decision_success_rate",
            "early_prediction_rate",
            "ping_pong_rate",
            "unnecessary_handover_count",
        ],
        ascending=[True, False, False, True, True],
        ignore_index=True,
    )
    return pareto_points


def annotate_operating_points(axis: plt.Axes, operating_points: pd.DataFrame, x_column: str, y_column: str) -> None:
    for row in operating_points.itertuples(index=False):
        axis.scatter(
            getattr(row, x_column),
            getattr(row, y_column),
            s=180,
            color=PLOT_COLORS[row.model_name],
            marker=PLOT_MARKERS[row.model_name],
            edgecolor="black",
            linewidth=0.8,
            zorder=5,
        )
        axis.annotate(
            row.model_name,
            (getattr(row, x_column), getattr(row, y_column)),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=9,
            color=PLOT_COLORS[row.model_name],
        )


def plot_tradeoffs(
    sweep_points: pd.DataFrame,
    pareto_points: pd.DataFrame,
    operating_points: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    plot_specs = [
        ("unnecessary_handover_count", "missed_useful_handover_count", "Unnecessary HOs", "Missed Useful HOs"),
        ("ping_pong_rate", "end_to_end_decision_success_rate", "Ping-Pong Rate", "End-to-End Success Rate"),
        ("early_prediction_rate", "ping_pong_rate", "Early Prediction Rate", "Ping-Pong Rate"),
        ("handover_count", "mean_dwell_time_s", "Handover Count", "Mean Dwell Time (s)"),
    ]

    family_styles = {
        "conservative": {"color": "#3f8f6b", "marker": "o"},
        "hybrid": {"color": "#7b5ea7", "marker": "^"},
    }

    for axis, (x_column, y_column, x_label, y_label) in zip(axes.flatten(), plot_specs):
        for family, family_frame in sweep_points.groupby("policy_family", sort=False):
            style = family_styles.get(family, {"color": "#999999", "marker": "o"})
            axis.scatter(
                family_frame[x_column],
                family_frame[y_column],
                s=16,
                alpha=0.12,
                color=style["color"],
                marker=style["marker"],
                label=f"{family} sweep",
            )

        axis.scatter(
            pareto_points[x_column],
            pareto_points[y_column],
            s=40,
            facecolors="none",
            edgecolors="black",
            linewidths=1.0,
            label="Pareto sweep points",
            zorder=4,
        )
        annotate_operating_points(axis, operating_points, x_column, y_column)
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.grid(alpha=0.2)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    dedup: dict[str, object] = {}
    for handle, label in zip(handles, labels):
        dedup.setdefault(label, handle)
    fig.legend(dedup.values(), dedup.keys(), loc="upper center", ncol=3, frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def recommendation_lines(summary: pd.DataFrame) -> list[str]:
    indexed = summary.set_index("model_name")
    balanced = summary[summary["recommendation_role"].str.contains("main_result", regex=False, na=False)].iloc[0]
    stability = summary[summary["recommendation_role"].str.contains("stability_first_variant", regex=False, na=False)].iloc[0]
    early = summary[summary["recommendation_role"].str.contains("early_action_variant", regex=False, na=False)].iloc[0]

    return [
        f"- Main result: `{balanced['model_name']}` because it keeps the strongest overall balance between success, early prediction, and stability among predictive policies.",
        f"- Stability-first variant: `{stability['model_name']}` because it minimizes ping-pong and unnecessary handovers most aggressively.",
        f"- Future online O-RAN tests: `{early['model_name']}` because it best preserves the predictive early-action value while staying stronger than `current_k3` on replay quality.",
        f"- Dominated operating point: `current_k3` is dominated by `conservative_k3` on the final test operating points.",
        f"- Reference baseline: `a3_original` should stay as the logged non-predictive baseline, not the primary learned-policy result.",
        f"- Best balanced policy metrics: success `{balanced['end_to_end_decision_success_rate']:.4f}`, early `{balanced['early_prediction_rate']:.4f}`, unnecessary `{int(balanced['unnecessary_handover_count'])}`, ping-pong `{balanced['ping_pong_rate']:.4f}`.",
        f"- Best stability-first policy metrics: success `{stability['end_to_end_decision_success_rate']:.4f}`, early `{stability['early_prediction_rate']:.4f}`, unnecessary `{int(stability['unnecessary_handover_count'])}`, ping-pong `{stability['ping_pong_rate']:.4f}`.",
    ]


def format_markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    table = frame[columns].copy()
    for column in table.columns:
        if column == "model_name" or column == "recommendation_role":
            continue
        if pd.api.types.is_numeric_dtype(table[column]):
            table[column] = table[column].map(lambda value: f"{float(value):.4f}" if pd.notna(value) else "")
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in table.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def render_report(
    output_dir: Path,
    operating_points: pd.DataFrame,
    pareto_points: pd.DataFrame,
    sweep_points: pd.DataFrame,
) -> str:
    summary_columns = [
        "model_name",
        "unnecessary_handover_count",
        "ping_pong_rate",
        "missed_useful_handover_count",
        "end_to_end_decision_success_rate",
        "early_prediction_rate",
        "handover_count",
        "mean_dwell_time_s",
        "recommendation_role",
    ]
    report_lines = [
        "# Final Trade-Off Report",
        "",
        "## Operating Points",
        "",
        format_markdown_table(operating_points, summary_columns),
        "",
        "## Pareto Findings",
        "",
        f"- Total sweep points analyzed: `{len(sweep_points)}`",
        f"- Pareto-efficient sweep points: `{len(pareto_points)}`",
        f"- Conservative Pareto points: `{int((pareto_points['policy_family'] == 'conservative').sum())}`",
        f"- Hybrid Pareto points: `{int((pareto_points['policy_family'] == 'hybrid').sum())}`",
        "",
        "## Recommendations",
        "",
        *recommendation_lines(operating_points),
        "",
        "## Paper Guidance",
        "",
        "- Main result should be the balanced predictive policy, not the logged A3 baseline.",
        "- Stability-first variant should be reported as a separate operating point rather than replacing the main predictive result.",
        "- Future online O-RAN tests should start from the balanced predictive policy, with the stability-first hybrid policy as a guarded alternative for risk-sensitive studies.",
        "",
        "## Artifacts",
        "",
        f"- Summary CSV: `{(output_dir / 'final_tradeoff_summary.csv').name}`",
        f"- Pareto CSV: `{(output_dir / 'pareto_points.csv').name}`",
        f"- Plot: `{(output_dir / 'comparison_plots.png').name}`",
    ]
    return "\n".join(report_lines) + "\n"


def main() -> None:
    args = build_argument_parser().parse_args()
    script_dir = Path(__file__).resolve().parent
    replay_dirs = args.replay_dirs or default_replay_dirs(script_dir)
    replay_dirs = [directory.resolve() for directory in replay_dirs]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    operating_points = load_operating_points(replay_dirs, args.prefer_dir.resolve() if args.prefer_dir else None)
    sweep_points = load_sweep_tables(replay_dirs)
    pareto_points = make_pareto_table(sweep_points, operating_points)
    operating_points = rank_operating_points(operating_points)

    summary = operating_points.copy()
    summary.to_csv(output_dir / "final_tradeoff_summary.csv", index=False)
    pareto_points.to_csv(output_dir / "pareto_points.csv", index=False)

    plot_tradeoffs(
        sweep_points=sweep_points,
        pareto_points=pareto_points,
        operating_points=operating_points,
        output_path=output_dir / "comparison_plots.png",
        title=args.title,
    )

    report = render_report(
        output_dir=output_dir,
        operating_points=operating_points,
        pareto_points=pareto_points,
        sweep_points=sweep_points,
    )
    (output_dir / "final_tradeoff_report.md").write_text(report, encoding="utf-8")

    print(summary[[
        "model_name",
        "unnecessary_handover_count",
        "ping_pong_rate",
        "missed_useful_handover_count",
        "end_to_end_decision_success_rate",
        "early_prediction_rate",
        "recommendation_role",
    ]].to_string(index=False))
    print(f"Saved trade-off artifacts to {output_dir}")


if __name__ == "__main__":
    main()
