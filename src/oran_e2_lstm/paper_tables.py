from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


MODEL_ORDER = ["a3_original", "current_k3", "conservative_k3", "hybrid_a3_k3"]
DISPLAY_COLUMNS = [
    ("unnecessary_handover_count", "Unnecessary HO", "min"),
    ("ping_pong_rate", "Ping-Pong Rate", "min"),
    ("missed_useful_handover_count", "Missed Useful HO", "min"),
    ("end_to_end_decision_success_rate", "E2E Success", "max"),
    ("early_prediction_rate", "Early Prediction", "max"),
    ("handover_count", "HO Count", "none"),
    ("mean_dwell_time_s", "Mean Dwell (s)", "max"),
]


def build_argument_parser() -> argparse.ArgumentParser:
    base_dir = Path(__file__).resolve().parent / "replay_outputs" / "final_tradeoff_analysis"
    parser = argparse.ArgumentParser(description="Generate publication-ready markdown and LaTeX summary tables.")
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=base_dir / "final_tradeoff_summary.csv",
    )
    parser.add_argument(
        "--pareto-csv",
        type=Path,
        default=base_dir / "pareto_points.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base_dir,
    )
    return parser


def load_summary(summary_csv: Path) -> pd.DataFrame:
    frame = pd.read_csv(summary_csv)
    frame["model_name"] = pd.Categorical(frame["model_name"], categories=MODEL_ORDER, ordered=True)
    frame = frame.sort_values("model_name", ignore_index=True)
    frame["model_name"] = frame["model_name"].astype(str)
    return frame


def select_policy(summary: pd.DataFrame, role_token: str, score_column: str) -> str:
    with_role = summary[summary["recommendation_role"].fillna("").str.contains(role_token, regex=False)].copy()
    if not with_role.empty:
        return str(with_role.iloc[0]["model_name"])
    predictive = summary[summary["model_name"] != "a3_original"].copy()
    predictive = predictive.sort_values(score_column, ascending=False, ignore_index=True)
    return str(predictive.iloc[0]["model_name"])


def format_value(column: str, value: Any) -> str:
    if pd.isna(value):
        return ""
    if column in {"unnecessary_handover_count", "missed_useful_handover_count", "handover_count"}:
        return f"{int(round(float(value)))}"
    return f"{float(value):.4f}"


def best_value_map(frame: pd.DataFrame) -> dict[str, float]:
    best: dict[str, float] = {}
    for column, _, direction in DISPLAY_COLUMNS:
        numeric = pd.to_numeric(frame[column], errors="coerce").astype(float)
        if direction == "min":
            best[column] = float(numeric.min())
        elif direction == "max":
            best[column] = float(numeric.max())
    return best


def is_best(column: str, value: Any, best_values: dict[str, float]) -> bool:
    if column not in best_values or pd.isna(value):
        return False
    return abs(float(value) - best_values[column]) <= 1e-12


def model_note(model_name: str, main_model: str, stability_model: str) -> str:
    if model_name == "a3_original":
        return "Logged non-predictive baseline"
    if model_name == main_model:
        return "Main paper result"
    if model_name == stability_model:
        return "Stability-first variant"
    if model_name == "current_k3":
        return "Flat predictive baseline"
    return ""


def render_markdown_table(frame: pd.DataFrame, main_model: str, stability_model: str) -> str:
    best_values = best_value_map(frame)
    headers = ["Policy", *[label for _, label, _ in DISPLAY_COLUMNS], "Note"]
    rows = []
    for row in frame.itertuples(index=False):
        rendered = [str(row.model_name)]
        for column, _, _ in DISPLAY_COLUMNS:
            value = getattr(row, column)
            text = format_value(column, value)
            if is_best(column, value, best_values):
                text = f"**{text}**"
            rendered.append(text)
        rendered.append(model_note(str(row.model_name), main_model, stability_model))
        rows.append(rendered)

    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(values) + " |" for values in rows]
    return "\n".join([header, separator, *body]) + "\n"


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def render_latex_table(
    frame: pd.DataFrame,
    caption: str,
    label: str,
    main_model: str,
    stability_model: str,
) -> str:
    best_values = best_value_map(frame)
    headers = ["Policy", *[label_name for _, label_name, _ in DISPLAY_COLUMNS], "Note"]
    alignment = "l" + "r" * (len(headers) - 2) + "l"
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        f"\\caption{{{latex_escape(caption)}}}",
        f"\\label{{{latex_escape(label)}}}",
        f"\\begin{{tabular}}{{{alignment}}}",
        "\\toprule",
        " & ".join(latex_escape(header) for header in headers) + " \\\\",
        "\\midrule",
    ]

    for row in frame.itertuples(index=False):
        values = [latex_escape(str(row.model_name))]
        for column, _, _ in DISPLAY_COLUMNS:
            raw_value = getattr(row, column)
            text = format_value(column, raw_value)
            if is_best(column, raw_value, best_values):
                text = f"\\textbf{{{text}}}"
            values.append(text)
        values.append(latex_escape(model_note(str(row.model_name), main_model, stability_model)))
        lines.append(" & ".join(values) + " \\\\")

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_summary(summary: pd.DataFrame, pareto: pd.DataFrame, main_model: str, stability_model: str) -> str:
    main_row = summary.set_index("model_name").loc[main_model]
    stability_row = summary.set_index("model_name").loc[stability_model]
    a3_row = summary.set_index("model_name").loc["a3_original"]

    pareto_models = sorted(set(pareto["policy_family"].astype(str).tolist()))
    return "\n".join(
        [
            "# Short Results Summary",
            "",
            f"- `conservative_k3` is the main paper result because it provides the best overall balance between replay success, early action, and stability.",
            f"- `hybrid_a3_k3` is the stability-first variant because it achieves the lowest unnecessary handovers and ping-pong rate among the learned policies.",
            f"- `a3_original` is the logged non-predictive baseline and should remain the reference non-ML mobility controller in the paper.",
            f"- Main result metrics: success `{main_row['end_to_end_decision_success_rate']:.4f}`, early `{main_row['early_prediction_rate']:.4f}`, unnecessary `{int(main_row['unnecessary_handover_count'])}`, ping-pong `{main_row['ping_pong_rate']:.4f}`.",
            f"- Stability-first metrics: success `{stability_row['end_to_end_decision_success_rate']:.4f}`, early `{stability_row['early_prediction_rate']:.4f}`, unnecessary `{int(stability_row['unnecessary_handover_count'])}`, ping-pong `{stability_row['ping_pong_rate']:.4f}`.",
            f"- Logged A3 baseline metrics: success `{a3_row['end_to_end_decision_success_rate']:.4f}`, early `{a3_row['early_prediction_rate']:.4f}`, unnecessary `{int(a3_row['unnecessary_handover_count'])}`, ping-pong `{a3_row['ping_pong_rate']:.4f}`.",
            f"- Pareto sweep families present in the final frontier export: `{', '.join(pareto_models)}`.",
            "",
        ]
    )


def main() -> None:
    args = build_argument_parser().parse_args()
    summary = load_summary(args.summary_csv.resolve())
    pareto = pd.read_csv(args.pareto_csv.resolve())

    main_model = select_policy(summary, "main_paper_result", "balanced_score")
    stability_model = select_policy(summary, "stability_first_variant", "stability_score")

    main_table_frame = summary[summary["model_name"].isin(["a3_original", main_model, stability_model])].copy()
    main_table_frame["model_name"] = pd.Categorical(
        main_table_frame["model_name"],
        categories=["a3_original", main_model, stability_model],
        ordered=True,
    )
    main_table_frame = main_table_frame.sort_values("model_name", ignore_index=True)
    main_table_frame["model_name"] = main_table_frame["model_name"].astype(str)

    ablation_table_frame = summary[summary["model_name"].isin(["current_k3", main_model, stability_model])].copy()
    ablation_table_frame["model_name"] = pd.Categorical(
        ablation_table_frame["model_name"],
        categories=["current_k3", main_model, stability_model],
        ordered=True,
    )
    ablation_table_frame = ablation_table_frame.sort_values("model_name", ignore_index=True)
    ablation_table_frame["model_name"] = ablation_table_frame["model_name"].astype(str)

    main_md = render_markdown_table(main_table_frame, main_model, stability_model)
    main_tex = render_latex_table(
        main_table_frame,
        caption="Main replay comparison table: logged baseline versus the selected balanced and stability-first predictive policies.",
        label="tab:main_replay_results",
        main_model=main_model,
        stability_model=stability_model,
    )
    ablation_md = render_markdown_table(ablation_table_frame, main_model, stability_model)
    ablation_tex = render_latex_table(
        ablation_table_frame,
        caption="Predictive policy evolution from the current candidate-aware model to the conservative and hybrid A3-assisted variants.",
        label="tab:ablation_replay_results",
        main_model=main_model,
        stability_model=stability_model,
    )
    summary_md = build_summary(summary, pareto, main_model, stability_model)

    output_dir = args.output_dir.resolve()
    write_text(output_dir / "paper_results_table.md", main_md)
    write_text(output_dir / "paper_results_table.tex", main_tex)
    write_text(output_dir / "paper_ablation_table.md", ablation_md)
    write_text(output_dir / "paper_ablation_table.tex", ablation_tex)
    write_text(output_dir / "short_results_summary.md", summary_md)

    print(f"Main paper result: {main_model}")
    print(f"Stability-first variant: {stability_model}")
    print(f"Saved paper tables to {output_dir}")


if __name__ == "__main__":
    main()
