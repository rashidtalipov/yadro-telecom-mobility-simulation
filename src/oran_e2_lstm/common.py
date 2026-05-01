from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch.utils.data import Dataset


RUN_DB_NAME = "oran-repository.db"
DEFAULT_HO_TRACE = "handover-end.tr"
DEFAULT_LSTM_TRACE = "lstm-features.tr"
MAX_CANDIDATE_K = 5
CANDIDATE_K_VALUES = (3, 5)
CANDIDATE_FEATURE_BASENAMES = [
    "candidate_rsrp",
    "candidate_rsrq",
    "candidate_diff_rsrp",
    "candidate_diff_rsrq",
    "candidate_rank_norm",
]

BASE_FEATURE_COLUMNS = [
    "serving_cell_id",
    "serving_rsrp",
    "serving_rsrq",
    "serving_sinr",
    "best_ngh_rsrp",
    "best_ngh_rsrq",
    "best_ngh_diff_rsrp",
    "best_ngh_diff_rsrq",
]

NUMERIC_FEATURE_COLUMNS = [
    "serving_rsrp",
    "serving_rsrq",
    "serving_sinr",
    "best_ngh_rsrp",
    "best_ngh_rsrq",
    "best_ngh_diff_rsrp",
    "best_ngh_diff_rsrq",
]

LSTM_REQUIRED_COLUMNS = {
    "simulationtime",
    "imsi",
    "ueid",
    "nodeid",
    "servingcellid",
    "servingrsrp",
    "servingsinr",
    "bestneighborcellid",
    "bestneighborrsrp",
    "bestneighborrsrq",
    "secondneighborcellid",
    "secondneighborrsrp",
    "secondneighborrsrq",
}

RSRQ_REQUIRED_COLUMNS = {
    "simulationtime",
    "nodeid",
    "cellid",
    "rsrq",
    "serving",
}


def candidate_cell_id_column(rank: int) -> str:
    return f"candidate_cell_id_{rank}"


def candidate_cell_index_column(rank: int) -> str:
    return f"candidate_cell_index_{rank}"


def candidate_mask_column(rank: int) -> str:
    return f"candidate_mask_{rank}"


def candidate_feature_column(rank: int, basename: str) -> str:
    return f"{basename}_{rank}"


def candidate_target_pos_column(k: int) -> str:
    return f"candidate_target_pos_top{k}"


def candidate_hit_column(k: int) -> str:
    return f"candidate_hit_top{k}"


def candidate_feature_columns_for_k(k: int) -> list[str]:
    columns: list[str] = []
    for rank in range(1, k + 1):
        for basename in CANDIDATE_FEATURE_BASENAMES:
            columns.append(candidate_feature_column(rank, basename))
    return columns


def candidate_numeric_columns(max_k: int = MAX_CANDIDATE_K) -> list[str]:
    return candidate_feature_columns_for_k(max_k)


@dataclass
class ScalerState:
    means: dict[str, float]
    stds: dict[str, float]

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {"means": self.means, "stds": self.stds}


def infer_run_dirs(dataset_root: Path, max_runs: int | None = None) -> list[Path]:
    run_dirs = sorted(path.parent for path in dataset_root.rglob(RUN_DB_NAME))
    if max_runs is not None:
        return run_dirs[:max_runs]
    return run_dirs


def infer_run_dir_map(dataset_root: Path, max_runs: int | None = None) -> dict[str, Path]:
    return {run_name(run_dir): run_dir for run_dir in infer_run_dirs(dataset_root, max_runs=max_runs)}


def run_name(run_dir: Path) -> str:
    return run_dir.name


def db_path_for_run(run_dir: Path) -> Path:
    return run_dir / RUN_DB_NAME


def trace_path(run_dir: Path, filename: str) -> Path:
    return run_dir / filename


def inspect_db_schema(db_path: Path) -> dict[str, list[str]]:
    schema: dict[str, list[str]] = {}
    with sqlite3.connect(db_path) as connection:
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",
            connection,
        )
        for table_name in tables["name"].tolist():
            pragma = pd.read_sql_query(
                f"PRAGMA table_info({table_name})",
                connection,
            )
            schema[table_name] = pragma["name"].astype(str).tolist()
    return schema


def schema_has_columns(
    schema: dict[str, list[str]],
    table_name: str,
    required_columns: Iterable[str],
) -> bool:
    available = set(schema.get(table_name, []))
    return set(required_columns).issubset(available)


def read_sql_query(db_path: Path, query: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as connection:
        return pd.read_sql_query(query, connection)


def read_whitespace_trace(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", engine="python")


def linear_power_to_dbm(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    output = pd.Series(np.nan, index=series.index, dtype=np.float32)
    mask = np.isfinite(values) & (values > 0)
    if mask.any():
        output.loc[mask] = (10.0 * np.log10(values.loc[mask]) + 30.0).astype(np.float32)
    return output


def linear_ratio_to_db(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    output = pd.Series(np.nan, index=series.index, dtype=np.float32)
    mask = np.isfinite(values) & (values > 0)
    if mask.any():
        output.loc[mask] = (10.0 * np.log10(values.loc[mask])).astype(np.float32)
    return output


def sanitize_radio_values(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return values.astype(np.float32)


def load_lstm_feature_rows(run_dir: Path, schema: dict[str, list[str]]) -> pd.DataFrame:
    db_path = db_path_for_run(run_dir)
    if schema_has_columns(schema, "lstm_features", LSTM_REQUIRED_COLUMNS):
        query = """
        SELECT
            simulationtime AS time,
            imsi,
            ueid AS ue_id,
            nodeid,
            servingcellid AS serving_cell_id,
            servingrsrp,
            servingsinr,
            bestneighborcellid AS best_ngh_cell_id,
            bestneighborrsrp AS best_ngh_rsrp,
            bestneighborrsrq AS best_ngh_rsrq,
            secondneighborcellid AS second_ngh_cell_id,
            secondneighborrsrp AS second_ngh_rsrp,
            secondneighborrsrq AS second_ngh_rsrq
        FROM lstm_features
        ORDER BY imsi, simulationtime
        """
        features = read_sql_query(db_path, query)
    else:
        trace_file = trace_path(run_dir, DEFAULT_LSTM_TRACE)
        if not trace_file.exists():
            raise FileNotFoundError(
                f"Missing both DB table lstm_features and trace file {trace_file}"
            )
        features = read_whitespace_trace(trace_file).rename(
            columns={
                "ueId": "ue_id",
                "nodeId": "nodeid",
                "servingCellId": "serving_cell_id",
                "servingRsrp": "servingrsrp",
                "servingSinr": "servingsinr",
                "bestNeighborCellId": "best_ngh_cell_id",
                "bestNeighborRsrp": "best_ngh_rsrp",
                "bestNeighborRsrq": "best_ngh_rsrq",
                "secondNeighborCellId": "second_ngh_cell_id",
                "secondNeighborRsrp": "second_ngh_rsrp",
                "secondNeighborRsrq": "second_ngh_rsrq",
            }
        )
        features = features[
            [
                "time",
                "imsi",
                "ue_id",
                "nodeid",
                "serving_cell_id",
                "servingrsrp",
                "servingsinr",
                "best_ngh_cell_id",
                "best_ngh_rsrp",
                "best_ngh_rsrq",
                "second_ngh_cell_id",
                "second_ngh_rsrp",
                "second_ngh_rsrq",
            ]
        ]

    features["run_id"] = run_name(run_dir)
    features["time"] = pd.to_numeric(features["time"], errors="coerce").astype(np.float32)
    features["imsi"] = pd.to_numeric(features["imsi"], errors="coerce").astype(np.int16)
    features["ue_id"] = pd.to_numeric(features["ue_id"], errors="coerce").astype(np.int16)
    features["nodeid"] = pd.to_numeric(features["nodeid"], errors="coerce").astype(np.int16)
    features["serving_cell_id"] = (
        pd.to_numeric(features["serving_cell_id"], errors="coerce").astype(np.int16)
    )

    features["serving_rsrp"] = linear_power_to_dbm(features["servingrsrp"])
    features["serving_sinr"] = linear_ratio_to_db(features["servingsinr"])
    features["best_ngh_cell_id"] = pd.to_numeric(
        features["best_ngh_cell_id"], errors="coerce"
    ).replace({0: np.nan})
    features["best_ngh_rsrp"] = sanitize_radio_values(features["best_ngh_rsrp"])
    features["best_ngh_rsrq"] = sanitize_radio_values(features["best_ngh_rsrq"])
    features["second_ngh_cell_id"] = pd.to_numeric(
        features["second_ngh_cell_id"], errors="coerce"
    ).replace({0: np.nan})
    features["second_ngh_rsrp"] = sanitize_radio_values(features["second_ngh_rsrp"])
    features["second_ngh_rsrq"] = sanitize_radio_values(features["second_ngh_rsrq"])

    return features[
        [
            "run_id",
            "time",
            "imsi",
            "ue_id",
            "nodeid",
            "serving_cell_id",
            "serving_rsrp",
            "serving_sinr",
            "best_ngh_cell_id",
            "best_ngh_rsrp",
            "best_ngh_rsrq",
            "second_ngh_cell_id",
            "second_ngh_rsrp",
            "second_ngh_rsrq",
        ]
    ].sort_values(["imsi", "time"], ignore_index=True)


def load_serving_rsrq_rows(run_dir: Path, schema: dict[str, list[str]]) -> pd.DataFrame:
    if not schema_has_columns(schema, "lteuersrprsrq", RSRQ_REQUIRED_COLUMNS):
        raise ValueError("DB schema does not expose lteuersrprsrq with serving RSRQ data")

    query = """
    SELECT
        simulationtime / 1e9 AS time,
        nodeid,
        cellid AS serving_cell_id,
        rsrq AS serving_rsrq
    FROM lteuersrprsrq
    WHERE serving = 1
    ORDER BY nodeid, cellid, simulationtime
    """
    rsrq = read_sql_query(db_path_for_run(run_dir), query)
    rsrq["time"] = pd.to_numeric(rsrq["time"], errors="coerce").astype(np.float32)
    rsrq["nodeid"] = pd.to_numeric(rsrq["nodeid"], errors="coerce").astype(np.int16)
    rsrq["serving_cell_id"] = pd.to_numeric(
        rsrq["serving_cell_id"], errors="coerce"
    ).astype(np.int16)
    rsrq["serving_rsrq"] = sanitize_radio_values(rsrq["serving_rsrq"])
    return rsrq.sort_values(["nodeid", "serving_cell_id", "time"], ignore_index=True)


def derive_best_neighbor_rows(run_dir: Path, schema: dict[str, list[str]]) -> pd.DataFrame:
    if not schema_has_columns(schema, "lteuersrprsrq", RSRQ_REQUIRED_COLUMNS):
        return pd.DataFrame(
            columns=["time", "nodeid", "best_ngh_rsrp_fallback", "best_ngh_rsrq_fallback"]
        )

    query = """
    SELECT
        simulationtime / 1e9 AS time,
        nodeid,
        cellid,
        rsrp,
        rsrq
    FROM lteuersrprsrq
    WHERE serving = 0
    ORDER BY nodeid, simulationtime, rsrp DESC
    """
    neighbors = read_sql_query(db_path_for_run(run_dir), query)
    if neighbors.empty:
        return pd.DataFrame(
            columns=["time", "nodeid", "best_ngh_rsrp_fallback", "best_ngh_rsrq_fallback"]
        )

    neighbors["time"] = pd.to_numeric(neighbors["time"], errors="coerce").astype(np.float32)
    neighbors["nodeid"] = pd.to_numeric(neighbors["nodeid"], errors="coerce").astype(np.int16)
    neighbors["rsrp"] = sanitize_radio_values(neighbors["rsrp"])
    neighbors["rsrq"] = sanitize_radio_values(neighbors["rsrq"])
    neighbors = neighbors.sort_values(
        ["nodeid", "time", "rsrp"], ascending=[True, True, False], ignore_index=True
    )
    best = neighbors.groupby(["nodeid", "time"], as_index=False).first()
    return best.rename(
        columns={
            "rsrp": "best_ngh_rsrp_fallback",
            "rsrq": "best_ngh_rsrq_fallback",
        }
    )[
        ["nodeid", "time", "best_ngh_rsrp_fallback", "best_ngh_rsrq_fallback"]
    ]


def derive_topk_candidate_rows(
    run_dir: Path,
    schema: dict[str, list[str]],
    max_k: int = MAX_CANDIDATE_K,
) -> pd.DataFrame:
    output_columns = ["nodeid", "time"]
    for rank in range(1, max_k + 1):
        output_columns.extend(
            [
                candidate_cell_id_column(rank),
                candidate_feature_column(rank, "candidate_rsrp"),
                candidate_feature_column(rank, "candidate_rsrq"),
            ]
        )

    if not schema_has_columns(schema, "lteuersrprsrq", RSRQ_REQUIRED_COLUMNS):
        return pd.DataFrame(columns=output_columns)

    query = """
    SELECT
        simulationtime / 1e9 AS time,
        nodeid,
        cellid,
        rsrp,
        rsrq
    FROM lteuersrprsrq
    WHERE serving = 0
    ORDER BY nodeid, simulationtime, rsrp DESC, rsrq DESC
    """
    neighbors = read_sql_query(db_path_for_run(run_dir), query)
    if neighbors.empty:
        return pd.DataFrame(columns=output_columns)

    neighbors["time"] = pd.to_numeric(neighbors["time"], errors="coerce").astype(np.float32)
    neighbors["nodeid"] = pd.to_numeric(neighbors["nodeid"], errors="coerce").astype(np.int16)
    neighbors["cellid"] = pd.to_numeric(neighbors["cellid"], errors="coerce").astype(np.int16)
    neighbors["rsrp"] = sanitize_radio_values(neighbors["rsrp"])
    neighbors["rsrq"] = sanitize_radio_values(neighbors["rsrq"])
    neighbors = neighbors.sort_values(
        ["nodeid", "time", "cellid", "rsrp", "rsrq"],
        ascending=[True, True, True, False, False],
        ignore_index=True,
    )
    neighbors = neighbors.groupby(["nodeid", "time", "cellid"], as_index=False).first()
    neighbors = neighbors.sort_values(
        ["nodeid", "time", "rsrp", "rsrq"],
        ascending=[True, True, False, False],
        ignore_index=True,
    )
    neighbors["rank"] = neighbors.groupby(["nodeid", "time"]).cumcount() + 1
    top = neighbors[neighbors["rank"] <= max_k].copy()
    if top.empty:
        return pd.DataFrame(columns=output_columns)

    wide_parts: list[pd.DataFrame] = []
    for source, prefix in (
        ("cellid", candidate_cell_id_column),
        ("rsrp", lambda rank: candidate_feature_column(rank, "candidate_rsrp")),
        ("rsrq", lambda rank: candidate_feature_column(rank, "candidate_rsrq")),
    ):
        pivot = top.pivot(index=["nodeid", "time"], columns="rank", values=source)
        if pivot.empty:
            continue
        pivot = pivot.rename(columns={rank: prefix(int(rank)) for rank in pivot.columns})
        wide_parts.append(pivot)

    wide = pd.concat(wide_parts, axis=1).reset_index()
    for column in output_columns:
        if column not in wide.columns:
            wide[column] = np.nan
    return wide[output_columns].sort_values(["nodeid", "time"], ignore_index=True)


def map_cell_ids_to_indices(
    cell_ids: np.ndarray,
    cell_to_index: dict[int, int],
) -> np.ndarray:
    return np.asarray(
        [cell_to_index.get(int(cell_id), -1) if int(cell_id) > 0 else -1 for cell_id in cell_ids],
        dtype=np.int16,
    )


def derive_history_based_candidates(
    frame: pd.DataFrame,
    cell_to_index: dict[int, int],
    history_len: int,
    max_k: int = MAX_CANDIDATE_K,
) -> pd.DataFrame:
    if history_len <= 0:
        raise ValueError("history_len must be positive")

    enriched = frame.sort_values(["run_id", "imsi", "time"], ignore_index=True).copy()
    index_to_cell = np.asarray(
        [cell_id for cell_id, _ in sorted(cell_to_index.items(), key=lambda item: item[1])],
        dtype=np.int32,
    )
    num_cells = len(index_to_cell)
    row_count = len(enriched)

    candidate_cell_ids = np.full((row_count, max_k), -1, dtype=np.int32)
    candidate_rsrp = np.full((row_count, max_k), np.nan, dtype=np.float32)
    candidate_rsrq = np.full((row_count, max_k), np.nan, dtype=np.float32)
    candidate_mask = np.zeros((row_count, max_k), dtype=bool)

    current_candidate_ids = enriched[
        [candidate_cell_id_column(rank) for rank in range(1, max_k + 1)]
    ].to_numpy(dtype=np.float32, copy=True)
    current_candidate_rsrp = enriched[
        [candidate_feature_column(rank, "candidate_rsrp") for rank in range(1, max_k + 1)]
    ].to_numpy(dtype=np.float32, copy=True)
    current_candidate_rsrq = enriched[
        [candidate_feature_column(rank, "candidate_rsrq") for rank in range(1, max_k + 1)]
    ].to_numpy(dtype=np.float32, copy=True)

    grouped = enriched.groupby(["run_id", "imsi"], sort=False, observed=True).indices
    for _, indices in grouped.items():
        row_indices = np.asarray(indices, dtype=np.int64)
        group = enriched.iloc[row_indices]
        length = len(group)
        if length == 0:
            continue

        serving_cell_ids = pd.to_numeric(group["serving_cell_id"], errors="coerce").fillna(-1).to_numpy(
            dtype=np.int32
        )
        serving_cell_indices = map_cell_ids_to_indices(serving_cell_ids, cell_to_index)

        best_cell_ids = pd.to_numeric(group["best_ngh_cell_id"], errors="coerce").fillna(-1).to_numpy(
            dtype=np.int32
        )
        second_cell_ids = pd.to_numeric(group["second_ngh_cell_id"], errors="coerce").fillna(-1).to_numpy(
            dtype=np.int32
        )
        best_cell_indices = map_cell_ids_to_indices(best_cell_ids, cell_to_index)
        second_cell_indices = map_cell_ids_to_indices(second_cell_ids, cell_to_index)

        best_rsrp = sanitize_radio_values(group["best_ngh_rsrp"]).to_numpy(dtype=np.float32, copy=True)
        best_rsrq = sanitize_radio_values(group["best_ngh_rsrq"]).to_numpy(dtype=np.float32, copy=True)
        second_rsrp = sanitize_radio_values(group["second_ngh_rsrp"]).to_numpy(
            dtype=np.float32,
            copy=True,
        )
        second_rsrq = sanitize_radio_values(group["second_ngh_rsrq"]).to_numpy(
            dtype=np.float32,
            copy=True,
        )

        source_rows = np.arange(length, dtype=np.int32)
        cell_indices = np.arange(num_cells, dtype=np.int16)
        best_onehot = best_cell_indices[:, None] == cell_indices[None, :]
        second_onehot = second_cell_indices[:, None] == cell_indices[None, :]

        best_counts = np.cumsum(best_onehot.astype(np.int16), axis=0)
        second_counts = np.cumsum(second_onehot.astype(np.int16), axis=0)
        if history_len < length:
            best_counts[history_len:] = best_counts[history_len:] - best_counts[:-history_len]
            second_counts[history_len:] = second_counts[history_len:] - second_counts[:-history_len]

        absent_marker = np.int32(-(history_len + 1))
        best_occurrences = np.where(best_onehot, source_rows[:, None], absent_marker)
        second_occurrences = np.where(second_onehot, source_rows[:, None], absent_marker)
        best_latest = np.maximum.accumulate(best_occurrences, axis=0)
        second_latest = np.maximum.accumulate(second_occurrences, axis=0)
        latest_occurrence = np.maximum(best_latest, second_latest)

        window_start = source_rows[:, None] - history_len + 1
        recent_mask = latest_occurrence >= window_start

        safe_best_latest = np.clip(best_latest, 0, max(length - 1, 0))
        safe_second_latest = np.clip(second_latest, 0, max(length - 1, 0))
        latest_best_rsrp = best_rsrp[safe_best_latest]
        latest_best_rsrq = best_rsrq[safe_best_latest]
        latest_second_rsrp = second_rsrp[safe_second_latest]
        latest_second_rsrq = second_rsrq[safe_second_latest]

        choose_best = best_latest >= second_latest
        latest_rsrp = np.where(choose_best, latest_best_rsrp, latest_second_rsrp).astype(np.float32)
        latest_rsrq = np.where(choose_best, latest_best_rsrq, latest_second_rsrq).astype(np.float32)
        latest_rsrp = np.where(recent_mask, latest_rsrp, np.nan).astype(np.float32)
        latest_rsrq = np.where(recent_mask, latest_rsrq, np.nan).astype(np.float32)

        age = source_rows[:, None] - latest_occurrence
        recency_score = np.clip((history_len - age) / max(history_len, 1), 0.0, 1.0).astype(np.float32)
        rsrp_score = np.clip((np.nan_to_num(latest_rsrp, nan=-140.0) + 140.0) / 60.0, 0.0, 1.5)
        rsrq_score = np.clip((np.nan_to_num(latest_rsrq, nan=-30.0) + 30.0) / 30.0, 0.0, 1.0)
        candidate_score = (
            (2.0 * best_counts.astype(np.float32))
            + second_counts.astype(np.float32)
            + (1.25 * recency_score)
            + (0.75 * rsrp_score.astype(np.float32))
            + (0.25 * rsrq_score.astype(np.float32))
        )
        candidate_score = np.where(recent_mask, candidate_score, -np.inf).astype(np.float32)

        valid_serving = serving_cell_indices >= 0
        if valid_serving.any():
            candidate_score[valid_serving, serving_cell_indices[valid_serving]] = -np.inf

        ranked_indices = np.argsort(candidate_score, axis=1)[:, ::-1][:, :max_k]
        ranked_scores = np.take_along_axis(candidate_score, ranked_indices, axis=1)
        ranked_rsrp = np.take_along_axis(latest_rsrp, ranked_indices, axis=1)
        ranked_rsrq = np.take_along_axis(latest_rsrq, ranked_indices, axis=1)
        ranked_valid = np.isfinite(ranked_scores)

        group_candidate_ids = np.full((length, max_k), -1, dtype=np.int32)
        group_candidate_rsrp = np.full((length, max_k), np.nan, dtype=np.float32)
        group_candidate_rsrq = np.full((length, max_k), np.nan, dtype=np.float32)

        if ranked_valid.any():
            group_candidate_ids[ranked_valid] = index_to_cell[ranked_indices[ranked_valid]]
            group_candidate_rsrp[ranked_valid] = ranked_rsrp[ranked_valid]
            group_candidate_rsrq[ranked_valid] = ranked_rsrq[ranked_valid]

        current_ids = current_candidate_ids[row_indices]
        current_rsrp = current_candidate_rsrp[row_indices]
        current_rsrq = current_candidate_rsrq[row_indices]
        for row_offset in range(length):
            next_slot = int(ranked_valid[row_offset].sum())
            if next_slot >= max_k:
                continue

            seen_candidates = {
                int(cell_id)
                for cell_id in group_candidate_ids[row_offset]
                if int(cell_id) > 0
            }
            serving_cell_id = int(serving_cell_ids[row_offset])
            for current_rank in range(max_k):
                candidate_value = current_ids[row_offset, current_rank]
                if not np.isfinite(candidate_value):
                    continue
                candidate_cell_id = int(candidate_value)
                if candidate_cell_id <= 0:
                    continue
                if candidate_cell_id == serving_cell_id or candidate_cell_id in seen_candidates:
                    continue
                group_candidate_ids[row_offset, next_slot] = candidate_cell_id
                group_candidate_rsrp[row_offset, next_slot] = current_rsrp[row_offset, current_rank]
                group_candidate_rsrq[row_offset, next_slot] = current_rsrq[row_offset, current_rank]
                ranked_valid[row_offset, next_slot] = True
                seen_candidates.add(candidate_cell_id)
                next_slot += 1
                if next_slot >= max_k:
                    break

        candidate_cell_ids[row_indices] = group_candidate_ids
        candidate_rsrp[row_indices] = group_candidate_rsrp
        candidate_rsrq[row_indices] = group_candidate_rsrq
        candidate_mask[row_indices] = ranked_valid

    for rank in range(1, max_k + 1):
        rank_index = rank - 1
        cell_column = candidate_cell_id_column(rank)
        rsrp_column = candidate_feature_column(rank, "candidate_rsrp")
        rsrq_column = candidate_feature_column(rank, "candidate_rsrq")
        diff_rsrp_column = candidate_feature_column(rank, "candidate_diff_rsrp")
        diff_rsrq_column = candidate_feature_column(rank, "candidate_diff_rsrq")
        rank_column = candidate_feature_column(rank, "candidate_rank_norm")
        mask_column = candidate_mask_column(rank)

        cell_values = np.where(candidate_mask[:, rank_index], candidate_cell_ids[:, rank_index], np.nan)
        enriched[cell_column] = pd.Series(cell_values, index=enriched.index, dtype=np.float32)
        enriched[rsrp_column] = pd.Series(candidate_rsrp[:, rank_index], index=enriched.index, dtype=np.float32)
        enriched[rsrq_column] = pd.Series(candidate_rsrq[:, rank_index], index=enriched.index, dtype=np.float32)
        enriched[diff_rsrp_column] = (
            enriched[rsrp_column].astype(np.float32) - enriched["serving_rsrp"].astype(np.float32)
        ).astype(np.float32)
        enriched[diff_rsrq_column] = (
            enriched[rsrq_column].astype(np.float32) - enriched["serving_rsrq"].astype(np.float32)
        ).astype(np.float32)
        enriched[rank_column] = np.where(
            candidate_mask[:, rank_index],
            np.float32(rank / max_k),
            np.nan,
        ).astype(np.float32)
        enriched[mask_column] = candidate_mask[:, rank_index].astype(np.int8)

    return enriched


def load_cell_ids(run_dir: Path, schema: dict[str, list[str]]) -> list[int]:
    if schema_has_columns(schema, "lteenb", {"cellid"}):
        frame = read_sql_query(
            db_path_for_run(run_dir),
            "SELECT DISTINCT cellid FROM lteenb ORDER BY cellid",
        )
        return sorted(pd.to_numeric(frame["cellid"], errors="coerce").dropna().astype(int))
    return []


def grouped_merge_asof(
    left: pd.DataFrame,
    right: pd.DataFrame,
    by_columns: list[str],
    tolerance: float,
) -> pd.DataFrame:
    right_only_columns = [
        column for column in right.columns if column not in set(by_columns + ["time"])
    ]
    merged_groups: list[pd.DataFrame] = []
    right_grouped = {key: group.copy() for key, group in right.groupby(by_columns, sort=False)}

    for key, left_group in left.groupby(by_columns, sort=False):
        left_sorted = left_group.sort_values("time").copy()
        right_group = right_grouped.get(key)
        if right_group is None or right_group.empty:
            for column in right_only_columns:
                dtype = right[column].dtype if column in right.columns else np.float32
                fill_dtype = np.float32 if np.issubdtype(dtype, np.integer) else dtype
                left_sorted[column] = pd.Series(
                    np.full(len(left_sorted), np.nan, dtype=fill_dtype),
                    index=left_sorted.index,
                )
            merged_groups.append(left_sorted)
            continue

        right_sorted = right_group[["time", *right_only_columns]].sort_values("time").copy()
        merged = pd.merge_asof(
            left_sorted,
            right_sorted,
            on="time",
            direction="backward",
            tolerance=tolerance,
        )
        merged_groups.append(merged)

    if not merged_groups:
        return left.copy()
    return pd.concat(merged_groups, ignore_index=True)


def assemble_feature_frame(
    run_dir: Path,
    schema: dict[str, list[str]],
    rsrq_tolerance_s: float,
) -> pd.DataFrame:
    base = load_lstm_feature_rows(run_dir, schema)
    serving_rsrq = load_serving_rsrq_rows(run_dir, schema)

    merged = grouped_merge_asof(
        left=base,
        right=serving_rsrq,
        by_columns=["nodeid", "serving_cell_id"],
        tolerance=rsrq_tolerance_s,
    )

    fallback_best = derive_best_neighbor_rows(run_dir, schema)
    if not fallback_best.empty:
        merged = grouped_merge_asof(
            left=merged,
            right=fallback_best,
            by_columns=["nodeid"],
            tolerance=rsrq_tolerance_s,
        )
        merged["best_ngh_rsrp"] = merged["best_ngh_rsrp"].fillna(
            merged["best_ngh_rsrp_fallback"]
        )
        merged["best_ngh_rsrq"] = merged["best_ngh_rsrq"].fillna(
            merged["best_ngh_rsrq_fallback"]
        )
        merged = merged.drop(columns=["best_ngh_rsrp_fallback", "best_ngh_rsrq_fallback"])

    candidate_rows = derive_topk_candidate_rows(run_dir, schema, max_k=MAX_CANDIDATE_K)
    if not candidate_rows.empty:
        merged = grouped_merge_asof(
            left=merged,
            right=candidate_rows,
            by_columns=["nodeid"],
            tolerance=rsrq_tolerance_s,
        )
    else:
        for rank in range(1, MAX_CANDIDATE_K + 1):
            merged[candidate_cell_id_column(rank)] = np.nan
            merged[candidate_feature_column(rank, "candidate_rsrp")] = np.nan
            merged[candidate_feature_column(rank, "candidate_rsrq")] = np.nan

    merged[candidate_cell_id_column(1)] = merged[candidate_cell_id_column(1)].fillna(
        merged["best_ngh_cell_id"]
    )
    merged[candidate_feature_column(1, "candidate_rsrp")] = merged[
        candidate_feature_column(1, "candidate_rsrp")
    ].fillna(merged["best_ngh_rsrp"])
    merged[candidate_feature_column(1, "candidate_rsrq")] = merged[
        candidate_feature_column(1, "candidate_rsrq")
    ].fillna(merged["best_ngh_rsrq"])

    merged["best_ngh_diff_rsrp"] = merged["best_ngh_rsrp"] - merged["serving_rsrp"]
    merged["best_ngh_diff_rsrq"] = merged["best_ngh_rsrq"] - merged["serving_rsrq"]

    candidate_columns: list[str] = []
    for rank in range(1, MAX_CANDIDATE_K + 1):
        cell_column = candidate_cell_id_column(rank)
        rsrp_column = candidate_feature_column(rank, "candidate_rsrp")
        rsrq_column = candidate_feature_column(rank, "candidate_rsrq")
        diff_rsrp_column = candidate_feature_column(rank, "candidate_diff_rsrp")
        diff_rsrq_column = candidate_feature_column(rank, "candidate_diff_rsrq")
        rank_column = candidate_feature_column(rank, "candidate_rank_norm")
        mask_column = candidate_mask_column(rank)

        merged[cell_column] = pd.to_numeric(merged[cell_column], errors="coerce")
        merged[rsrp_column] = sanitize_radio_values(merged[rsrp_column])
        merged[rsrq_column] = sanitize_radio_values(merged[rsrq_column])
        merged[diff_rsrp_column] = merged[rsrp_column] - merged["serving_rsrp"]
        merged[diff_rsrq_column] = merged[rsrq_column] - merged["serving_rsrq"]
        merged[mask_column] = merged[cell_column].notna().astype(np.int8)
        merged[rank_column] = np.where(
            merged[mask_column] > 0,
            np.float32(rank / MAX_CANDIDATE_K),
            np.nan,
        )
        candidate_columns.extend(
            [
                cell_column,
                rsrp_column,
                rsrq_column,
                diff_rsrp_column,
                diff_rsrq_column,
                rank_column,
                mask_column,
            ]
        )

    merged = merged[
        [
            "run_id",
            "time",
            "imsi",
            "ue_id",
            "nodeid",
            "serving_cell_id",
            "serving_rsrp",
            "serving_rsrq",
            "serving_sinr",
            "best_ngh_cell_id",
            "best_ngh_rsrp",
            "best_ngh_rsrq",
            "second_ngh_cell_id",
            "second_ngh_rsrp",
            "second_ngh_rsrq",
            "best_ngh_diff_rsrp",
            "best_ngh_diff_rsrq",
            *candidate_columns,
        ]
    ].sort_values(["run_id", "imsi", "time"], ignore_index=True)

    for column in NUMERIC_FEATURE_COLUMNS:
        merged[column] = sanitize_radio_values(merged[column])
    return merged


def load_handover_trace_rows(run_dir: Path) -> pd.DataFrame:
    ho_trace = trace_path(run_dir, DEFAULT_HO_TRACE)
    if not ho_trace.exists():
        raise FileNotFoundError(f"Missing required trace file {ho_trace}")
    events = read_whitespace_trace(ho_trace).rename(
        columns={
            "targetCellId": "target_cell_id",
            "successfulHoCount": "successful_ho_count",
            "pingPongCount": "ping_pong_count",
            "isPingPong": "is_ping_pong",
        }
    )
    events["time"] = pd.to_numeric(events["time"], errors="coerce").astype(np.float32)
    events["imsi"] = pd.to_numeric(events["imsi"], errors="coerce").astype(np.int16)
    events["target_cell_id"] = pd.to_numeric(
        events["target_cell_id"], errors="coerce"
    ).astype(np.int16)
    if "successful_ho_count" in events.columns:
        events["successful_ho_count"] = pd.to_numeric(
            events["successful_ho_count"], errors="coerce"
        ).fillna(0).astype(np.int32)
    else:
        events["successful_ho_count"] = np.arange(1, len(events) + 1, dtype=np.int32)
    if "ping_pong_count" in events.columns:
        events["ping_pong_count"] = pd.to_numeric(
            events["ping_pong_count"], errors="coerce"
        ).fillna(0).astype(np.int32)
    else:
        events["ping_pong_count"] = np.zeros(len(events), dtype=np.int32)
    if "is_ping_pong" in events.columns:
        events["is_ping_pong"] = pd.to_numeric(events["is_ping_pong"], errors="coerce").fillna(0).astype(np.int8)
    else:
        events["is_ping_pong"] = np.zeros(len(events), dtype=np.int8)
    events["run_id"] = run_name(run_dir)
    return events[
        [
            "run_id",
            "time",
            "imsi",
            "target_cell_id",
            "successful_ho_count",
            "ping_pong_count",
            "is_ping_pong",
        ]
    ].sort_values(
        ["run_id", "imsi", "time"],
        ignore_index=True,
    )


def load_handover_events(run_dir: Path) -> pd.DataFrame:
    events = load_handover_trace_rows(run_dir)
    return events[["run_id", "time", "imsi", "target_cell_id"]].copy()


def assign_future_handover_labels(
    features: pd.DataFrame,
    events: pd.DataFrame,
    horizon_s: float,
    cell_to_index: dict[int, int],
) -> pd.DataFrame:
    labeled = features.copy()
    labeled["trigger_label"] = np.int8(0)
    labeled["target_cell_id"] = np.int16(-1)
    labeled["target_cell_index"] = np.int16(-1)

    event_groups = {
        key: group[["time", "target_cell_id"]].to_numpy()
        for key, group in events.groupby(["run_id", "imsi"], sort=False)
    }
    feature_groups = labeled.groupby(["run_id", "imsi"], sort=False).indices

    for key, indices in feature_groups.items():
        future_events = event_groups.get(key)
        if future_events is None or len(future_events) == 0:
            continue

        row_indices = np.asarray(indices, dtype=np.int64)
        times = labeled.loc[row_indices, "time"].to_numpy(dtype=np.float32)
        future_times = future_events[:, 0].astype(np.float32)
        future_targets = future_events[:, 1].astype(np.int16)

        next_idx = np.searchsorted(future_times, times, side="right")
        valid = next_idx < len(future_times)
        within_horizon = np.zeros(len(times), dtype=bool)
        chosen_targets = np.full(len(times), -1, dtype=np.int16)

        valid_rows = np.where(valid)[0]
        if len(valid_rows) > 0:
            candidate_times = future_times[next_idx[valid_rows]]
            good = candidate_times <= (times[valid_rows] + horizon_s)
            within_horizon[valid_rows] = good
            chosen_targets[valid_rows[good]] = future_targets[next_idx[valid_rows[good]]]

        labeled.loc[row_indices, "trigger_label"] = within_horizon.astype(np.int8)
        labeled.loc[row_indices, "target_cell_id"] = chosen_targets.astype(np.int16)
        labeled.loc[row_indices, "target_cell_index"] = np.asarray(
            [
            cell_to_index.get(int(cell_id), -1) if cell_id > 0 else -1
            for cell_id in chosen_targets
            ],
            dtype=np.int16,
        )

    return labeled


def annotate_candidate_supervision(
    frame: pd.DataFrame,
    cell_to_index: dict[int, int],
    candidate_k_values: tuple[int, ...] = CANDIDATE_K_VALUES,
) -> pd.DataFrame:
    labeled = frame.copy()
    for rank in range(1, MAX_CANDIDATE_K + 1):
        cell_id_column = candidate_cell_id_column(rank)
        cell_index_column = candidate_cell_index_column(rank)
        mask_column = candidate_mask_column(rank)
        cell_ids = pd.to_numeric(labeled[cell_id_column], errors="coerce")
        cell_indices = [
            cell_to_index.get(int(cell_id), -1) if pd.notna(cell_id) else -1 for cell_id in cell_ids
        ]
        labeled[cell_index_column] = np.asarray(cell_indices, dtype=np.int16)
        labeled[mask_column] = (labeled[cell_index_column] >= 0).astype(np.int8)

    target_cell_ids = pd.to_numeric(labeled["target_cell_id"], errors="coerce").fillna(-1).astype(int)
    for k in candidate_k_values:
        candidate_index_columns = [candidate_cell_index_column(rank) for rank in range(1, k + 1)]
        candidate_values = labeled[candidate_index_columns].to_numpy(dtype=np.int16, copy=True)
        target_indices = np.asarray(
            [cell_to_index.get(int(cell_id), -1) if int(cell_id) > 0 else -1 for cell_id in target_cell_ids],
            dtype=np.int16,
        )
        matches = candidate_values == target_indices[:, None]
        hit = (target_indices >= 0) & matches.any(axis=1)
        target_pos = np.full(len(labeled), -1, dtype=np.int16)
        if hit.any():
            target_pos[hit] = matches[hit].argmax(axis=1).astype(np.int16)
        labeled[candidate_hit_column(k)] = hit.astype(np.int8)
        labeled[candidate_target_pos_column(k)] = target_pos

    return labeled


def split_runs(
    run_ids: list[str],
    seed: int,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
) -> dict[str, list[str]]:
    unique_runs = sorted(set(run_ids))
    rng = np.random.default_rng(seed)
    shuffled = unique_runs.copy()
    rng.shuffle(shuffled)

    n_runs = len(shuffled)
    train_count = max(1, int(round(n_runs * train_fraction)))
    val_count = max(1, int(round(n_runs * val_fraction))) if n_runs >= 3 else 1
    if train_count + val_count >= n_runs:
        val_count = max(1, n_runs - train_count - 1)
    test_count = n_runs - train_count - val_count
    if test_count <= 0:
        test_count = 1
        if train_count > val_count:
            train_count -= 1
        else:
            val_count -= 1

    train_runs = shuffled[:train_count]
    val_runs = shuffled[train_count : train_count + val_count]
    test_runs = shuffled[train_count + val_count :]

    return {"train": train_runs, "val": val_runs, "test": test_runs}


def fit_scaler(frame: pd.DataFrame, numeric_columns: list[str]) -> ScalerState:
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for column in numeric_columns:
        numeric = pd.to_numeric(frame[column], errors="coerce").astype(float)
        mean = float(np.nanmean(numeric))
        std = float(np.nanstd(numeric))
        if not np.isfinite(std) or std < 1e-6:
            std = 1.0
        if not np.isfinite(mean):
            mean = 0.0
        means[column] = mean
        stds[column] = std
    return ScalerState(means=means, stds=stds)


def apply_scaler(frame: pd.DataFrame, scaler: ScalerState, numeric_columns: list[str]) -> pd.DataFrame:
    scaled = frame.copy()
    for column in numeric_columns:
        values = pd.to_numeric(scaled[column], errors="coerce").astype(float)
        values = values.fillna(scaler.means[column])
        scaled[column] = ((values - scaler.means[column]) / scaler.stds[column]).astype(
            np.float32
        )
    return scaled


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def top_k_accuracy(logits: np.ndarray, targets: np.ndarray, k: int) -> float:
    if len(targets) == 0:
        return 0.0
    k = min(k, logits.shape[1])
    topk = np.argpartition(logits, -k, axis=1)[:, -k:]
    hits = (topk == targets[:, None]).any(axis=1)
    return float(np.mean(hits))


def compute_multitask_metrics(
    trigger_logits: np.ndarray,
    trigger_targets: np.ndarray,
    target_logits: np.ndarray,
    target_targets: np.ndarray,
) -> dict[str, float]:
    trigger_pred = (trigger_logits >= 0.0).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        trigger_targets,
        trigger_pred,
        average="binary",
        zero_division=0,
    )

    positive_mask = target_targets >= 0
    metrics = {
        "trigger_precision": float(precision),
        "trigger_recall": float(recall),
        "trigger_f1": float(f1),
        "target_accuracy": 0.0,
        "target_macro_f1": 0.0,
        "target_top3_accuracy": 0.0,
    }
    if positive_mask.any():
        target_true = target_targets[positive_mask]
        target_score = target_logits[positive_mask]
        target_pred = np.argmax(target_score, axis=1)
        metrics["target_accuracy"] = float(accuracy_score(target_true, target_pred))
        metrics["target_macro_f1"] = float(
            f1_score(target_true, target_pred, average="macro", zero_division=0)
        )
        metrics["target_top3_accuracy"] = float(top_k_accuracy(target_score, target_true, k=3))
    return metrics


def compute_candidate_aware_metrics(
    trigger_logits: np.ndarray,
    trigger_targets: np.ndarray,
    final_target_predictions: np.ndarray,
    target_targets: np.ndarray,
    candidate_hit: np.ndarray,
    fallback_used: np.ndarray,
) -> dict[str, float]:
    trigger_pred = (trigger_logits >= 0.0).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        trigger_targets,
        trigger_pred,
        average="binary",
        zero_division=0,
    )

    positive_mask = target_targets >= 0
    metrics = {
        "trigger_precision": float(precision),
        "trigger_recall": float(recall),
        "trigger_f1": float(f1),
        "candidate_target_accuracy": 0.0,
        "candidate_macro_f1": 0.0,
        "candidate_topk_hit_rate": 0.0,
        "global_fallback_accuracy": 0.0,
        "global_fallback_rate": 0.0,
    }
    if positive_mask.any():
        true_targets = target_targets[positive_mask]
        pred_targets = final_target_predictions[positive_mask]
        metrics["candidate_target_accuracy"] = float(accuracy_score(true_targets, pred_targets))
        metrics["candidate_macro_f1"] = float(
            f1_score(true_targets, pred_targets, average="macro", zero_division=0)
        )
        metrics["candidate_topk_hit_rate"] = float(np.mean(candidate_hit[positive_mask]))
        fallback_positive = fallback_used[positive_mask]
        metrics["global_fallback_rate"] = float(np.mean(fallback_positive))
        if fallback_positive.any():
            metrics["global_fallback_accuracy"] = float(
                accuracy_score(true_targets[fallback_positive], pred_targets[fallback_positive])
            )
    return metrics


class SequenceWindowDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        frame: pd.DataFrame,
        seq_len: int,
        window_stride: int = 1,
        candidate_top_k: int | None = None,
        num_cells: int | None = None,
    ) -> None:
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if window_stride <= 0:
            raise ValueError("window_stride must be positive")
        if candidate_top_k is not None and candidate_top_k not in CANDIDATE_K_VALUES:
            raise ValueError(f"candidate_top_k must be one of {CANDIDATE_K_VALUES}")

        self.seq_len = seq_len
        self.window_stride = window_stride
        self.candidate_top_k = candidate_top_k
        ordered = frame.sort_values(["run_id", "imsi", "time"], ignore_index=True).copy()
        ordered["run_id"] = ordered["run_id"].astype("category")

        self.numeric = ordered[NUMERIC_FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True)
        self.serving_cell = ordered["serving_cell_index"].to_numpy(dtype=np.int64, copy=True)
        self.trigger = ordered["trigger_label"].to_numpy(dtype=np.float32, copy=True)
        self.target = ordered["target_cell_index"].to_numpy(dtype=np.int64, copy=True)
        self.time = ordered["time"].to_numpy(dtype=np.float32, copy=True)
        self.imsi = ordered["imsi"].to_numpy(dtype=np.int16, copy=True)
        self.run_code = ordered["run_id"].cat.codes.to_numpy(dtype=np.int16, copy=True)
        self.run_names = list(ordered["run_id"].cat.categories)
        self.candidate_padding_index = num_cells if num_cells is not None else -1
        self.candidate_cell = None
        self.candidate_mask = None
        self.candidate_features = None
        self.candidate_target_pos = None
        self.candidate_hit = None

        if candidate_top_k is not None:
            cell_columns = [candidate_cell_index_column(rank) for rank in range(1, candidate_top_k + 1)]
            mask_columns = [candidate_mask_column(rank) for rank in range(1, candidate_top_k + 1)]
            feature_columns = []
            for rank in range(1, candidate_top_k + 1):
                for basename in CANDIDATE_FEATURE_BASENAMES:
                    feature_columns.append(candidate_feature_column(rank, basename))

            candidate_cell = ordered[cell_columns].to_numpy(dtype=np.int64, copy=True)
            candidate_mask = ordered[mask_columns].to_numpy(dtype=np.int8, copy=True).astype(bool)
            if num_cells is None:
                raise ValueError("num_cells must be provided when candidate_top_k is enabled")
            candidate_cell = np.where(candidate_mask, candidate_cell, num_cells)
            raw_features = ordered[feature_columns].to_numpy(dtype=np.float32, copy=True)
            self.candidate_cell = candidate_cell
            self.candidate_mask = candidate_mask
            self.candidate_features = raw_features.reshape(
                len(ordered),
                candidate_top_k,
                len(CANDIDATE_FEATURE_BASENAMES),
            )
            self.candidate_target_pos = ordered[candidate_target_pos_column(candidate_top_k)].to_numpy(
                dtype=np.int64,
                copy=True,
            )
            self.candidate_hit = ordered[candidate_hit_column(candidate_top_k)].to_numpy(
                dtype=np.int8,
                copy=True,
            ).astype(bool)

        self.end_indices = self._build_end_indices(ordered)

    def _build_end_indices(self, frame: pd.DataFrame) -> np.ndarray:
        end_indices: list[np.ndarray] = []
        start = 0
        for _, group in frame.groupby(["run_id", "imsi"], sort=False, observed=True):
            length = len(group)
            if length >= self.seq_len:
                ends = np.arange(
                    start + self.seq_len - 1,
                    start + length,
                    self.window_stride,
                    dtype=np.int64,
                )
                end_indices.append(ends)
            start += length
        if not end_indices:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(end_indices)

    def __len__(self) -> int:
        return int(len(self.end_indices))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        end = int(self.end_indices[index])
        start = end - self.seq_len + 1
        sample = {
            "numeric": torch.from_numpy(self.numeric[start : end + 1]),
            "serving_cell": torch.from_numpy(self.serving_cell[start : end + 1]),
            "trigger": torch.tensor(self.trigger[end], dtype=torch.float32),
            "target": torch.tensor(self.target[end], dtype=torch.long),
            "time": torch.tensor(self.time[end], dtype=torch.float32),
            "imsi": torch.tensor(int(self.imsi[end]), dtype=torch.long),
            "run_code": torch.tensor(int(self.run_code[end]), dtype=torch.long),
        }
        if self.candidate_top_k is not None:
            sample.update(
                {
                    "candidate_cell": torch.from_numpy(self.candidate_cell[end]),
                    "candidate_mask": torch.from_numpy(self.candidate_mask[end]),
                    "candidate_features": torch.from_numpy(self.candidate_features[end]),
                    "candidate_target_pos": torch.tensor(
                        int(self.candidate_target_pos[end]), dtype=torch.long
                    ),
                    "candidate_hit": torch.tensor(bool(self.candidate_hit[end]), dtype=torch.bool),
                }
            )
        return sample
