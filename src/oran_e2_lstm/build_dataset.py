from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    BASE_FEATURE_COLUMNS,
    CANDIDATE_FEATURE_BASENAMES,
    CANDIDATE_K_VALUES,
    MAX_CANDIDATE_K,
    NUMERIC_FEATURE_COLUMNS,
    apply_scaler,
    annotate_candidate_supervision,
    assemble_feature_frame,
    assign_future_handover_labels,
    candidate_hit_column,
    candidate_feature_columns_for_k,
    candidate_numeric_columns,
    db_path_for_run,
    derive_history_based_candidates,
    fit_scaler,
    infer_run_dirs,
    inspect_db_schema,
    load_cell_ids,
    load_handover_events,
    run_name,
    save_json,
    split_runs,
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build row-level dataset for E2-friendly LSTM handover prediction.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=15)
    parser.add_argument("--horizon-s", type=float, default=1.0)
    parser.add_argument("--rsrq-tolerance-s", type=float, default=0.25)
    parser.add_argument(
        "--candidate-history-len",
        type=int,
        default=None,
        help="History length used to derive candidate sets. Defaults to seq-len.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument(
        "--max-missing-features",
        type=int,
        default=2,
        help="Drop rows with more than this many missing numeric features before scaling.",
    )
    return parser


def estimate_window_count(frame: pd.DataFrame, seq_len: int) -> int:
    total = 0
    for _, group in frame.groupby(["run_id", "imsi"], sort=False, observed=True):
        total += max(0, len(group) - seq_len + 1)
    return total


def main() -> None:
    args = build_argument_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_history_len = args.candidate_history_len or args.seq_len

    run_dirs = infer_run_dirs(args.dataset_root, max_runs=args.max_runs)
    if not run_dirs:
        raise SystemExit(f"No run directories found under {args.dataset_root}")

    all_features: list[pd.DataFrame] = []
    all_events: list[pd.DataFrame] = []
    all_cell_ids: set[int] = set()
    schema_summary: dict[str, dict[str, list[str]]] = {}

    for run_dir in run_dirs:
        schema = inspect_db_schema(db_path_for_run(run_dir))
        schema_summary[run_name(run_dir)] = schema
        feature_frame = assemble_feature_frame(
            run_dir=run_dir,
            schema=schema,
            rsrq_tolerance_s=args.rsrq_tolerance_s,
        )
        event_frame = load_handover_events(run_dir)
        all_features.append(feature_frame)
        all_events.append(event_frame)
        all_cell_ids.update(load_cell_ids(run_dir, schema))
        all_cell_ids.update(
            int(cell_id)
            for cell_id in feature_frame["serving_cell_id"].dropna().unique().tolist()
            if int(cell_id) > 0
        )
        all_cell_ids.update(
            int(cell_id)
            for cell_id in event_frame["target_cell_id"].dropna().unique().tolist()
            if int(cell_id) > 0
        )

    features = pd.concat(all_features, ignore_index=True)
    events = pd.concat(all_events, ignore_index=True)
    cell_ids = sorted(all_cell_ids)
    if not cell_ids:
        raise SystemExit("No cell IDs detected in dataset")
    cell_to_index = {cell_id: idx for idx, cell_id in enumerate(cell_ids)}

    labeled = assign_future_handover_labels(
        features=features,
        events=events,
        horizon_s=args.horizon_s,
        cell_to_index=cell_to_index,
    )
    labeled = derive_history_based_candidates(
        labeled,
        cell_to_index=cell_to_index,
        history_len=candidate_history_len,
        max_k=MAX_CANDIDATE_K,
    )
    labeled = annotate_candidate_supervision(labeled, cell_to_index=cell_to_index)
    labeled["serving_cell_index"] = labeled["serving_cell_id"].map(cell_to_index)
    labeled["missing_numeric_features"] = labeled[NUMERIC_FEATURE_COLUMNS].isna().sum(axis=1)
    labeled = labeled[
        labeled["serving_cell_index"].notna()
        & (labeled["missing_numeric_features"] <= args.max_missing_features)
    ].copy()
    labeled["serving_cell_index"] = labeled["serving_cell_index"].astype(np.int16)
    labeled["target_cell_index"] = labeled["target_cell_index"].astype(np.int16)
    labeled["trigger_label"] = labeled["trigger_label"].astype(np.int8)
    labeled["run_id"] = labeled["run_id"].astype(str)

    splits = split_runs(
        run_ids=sorted(labeled["run_id"].unique().tolist()),
        seed=args.seed,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
    )

    train_frame = labeled[labeled["run_id"].isin(splits["train"])].copy()
    scaler = fit_scaler(train_frame, NUMERIC_FEATURE_COLUMNS)
    candidate_scaler = fit_scaler(train_frame, candidate_numeric_columns())

    split_stats: dict[str, dict[str, int]] = {}
    for split_name, split_runs_list in splits.items():
        split_frame = labeled[labeled["run_id"].isin(split_runs_list)].copy()
        split_frame = apply_scaler(split_frame, scaler, NUMERIC_FEATURE_COLUMNS)
        split_frame = apply_scaler(split_frame, candidate_scaler, candidate_numeric_columns())
        split_frame = split_frame.sort_values(["run_id", "imsi", "time"], ignore_index=True)
        split_file = args.output_dir / f"{split_name}_rows.parquet"
        split_frame.to_parquet(split_file, index=False)
        split_stats[split_name] = {
            "rows": int(len(split_frame)),
            "positive_trigger_rows": int(split_frame["trigger_label"].sum()),
            "estimated_windows": estimate_window_count(split_frame, args.seq_len),
            "runs": int(len(split_runs_list)),
            **{
                f"candidate_hit_top{k}": int(split_frame[candidate_hit_column(k)].sum())
                for k in CANDIDATE_K_VALUES
            },
        }

    metadata = {
        "dataset_root": str(args.dataset_root.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "seq_len": args.seq_len,
        "horizon_s": args.horizon_s,
        "rsrq_tolerance_s": args.rsrq_tolerance_s,
        "candidate_history_len": candidate_history_len,
        "base_feature_columns": BASE_FEATURE_COLUMNS,
        "numeric_feature_columns": NUMERIC_FEATURE_COLUMNS,
        "categorical_feature_column": "serving_cell_index",
        "cell_ids": cell_ids,
        "cell_to_index": {str(key): value for key, value in cell_to_index.items()},
        "index_to_cell": {str(value): key for key, value in cell_to_index.items()},
        "max_candidate_k": MAX_CANDIDATE_K,
        "candidate_k_values": list(CANDIDATE_K_VALUES),
        "candidate_feature_basenames": CANDIDATE_FEATURE_BASENAMES,
        "candidate_numeric_columns": candidate_numeric_columns(),
        "candidate_feature_columns": {
            str(k): candidate_feature_columns_for_k(k) for k in CANDIDATE_K_VALUES
        },
        "candidate_source": "history_best_second_neighbor_with_instant_fallback",
        "splits": splits,
        "scaler": scaler.to_dict(),
        "candidate_scaler": candidate_scaler.to_dict(),
        "split_stats": split_stats,
        "schema_summary": schema_summary,
        "row_filter": {
            "max_missing_features": args.max_missing_features,
        },
    }
    save_json(args.output_dir / "metadata.json", metadata)

    print("Saved processed dataset to", args.output_dir)
    for split_name, stats in split_stats.items():
        print(
            f"{split_name}: rows={stats['rows']}, positive_rows={stats['positive_trigger_rows']}, "
            f"estimated_windows={stats['estimated_windows']}, runs={stats['runs']}"
        )


if __name__ == "__main__":
    main()
