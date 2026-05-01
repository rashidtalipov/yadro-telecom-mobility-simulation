from __future__ import annotations

import argparse
from pathlib import Path

from common import db_path_for_run, infer_run_dirs, inspect_db_schema, read_sql_query, trace_path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect ns-3 O-RAN SQLite schemas and traces.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root directory that contains run folders with oran-repository.db",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=3,
        help="Inspect at most this many runs",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    run_dirs = infer_run_dirs(args.dataset_root, max_runs=args.max_runs)
    if not run_dirs:
        raise SystemExit(f"No run directories with oran-repository.db found under {args.dataset_root}")

    for run_dir in run_dirs:
        db_path = db_path_for_run(run_dir)
        schema = inspect_db_schema(db_path)
        print(f"Run: {run_dir.name}")
        print(f"DB:  {db_path}")
        for table_name, columns in schema.items():
            count = read_sql_query(db_path, f"SELECT COUNT(*) AS n FROM {table_name}")["n"].iloc[0]
            print(f"  - {table_name}: rows={count}, columns={', '.join(columns)}")

        for trace_name in ("lstm-features.tr", "handover-end.tr", "ue-cell-state.tr", "rsrp-sinr.tr"):
            trace_file = trace_path(run_dir, trace_name)
            status = "present" if trace_file.exists() else "missing"
            print(f"  - trace {trace_name}: {status}")
        print()


if __name__ == "__main__":
    main()
