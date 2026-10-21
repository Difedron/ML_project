"""Schema boundary for verified cached Parquet and official results_per_split.csv."""
from pathlib import Path
from collections.abc import Mapping

import numpy as np
import pandas as pd

from .model_mapping import exact_identifier, identify_model


class ResultsSchemaError(ValueError):
    pass


SPLIT_KEYS = ["dataset_source", "dataset_id", "split_id", "metric", "problem_type"]
OBSERVATION_KEYS = ["dataset_source", "dataset_id"]
ROW_KEYS = SPLIT_KEYS + ["exact_model_id"]
_REQUIRED = ["dataset", "fold", "method", "metric_error", "metric_error_val", "metric",
             "problem_type", "method_subtype", "config_type", "ta_name", "ta_suite"]


def standardize_results(frame: pd.DataFrame, *, artifact_id: str = "imported",
                        exported_csv: bool = False,
                        identity_map: Mapping[str, Mapping] | None = None) -> pd.DataFrame:
    """Preserve errors and identifiers; reject fabricated fills and ambiguous schemas."""
    if frame.empty or frame.columns.has_duplicates:
        raise ResultsSchemaError("Results must be nonempty and have unique column names")
    data = frame.copy()
    if "subtype" in data and "method_subtype" not in data:
        data = data.rename(columns={"subtype": "method_subtype"})
    if identity_map is not None and "method" in data:
        for column in ("ta_name", "ta_suite", "config_type", "method_subtype"):
            mapped = data["method"].map(lambda name: identity_map.get(name, {}).get(column))
            if column not in data:
                data[column] = mapped
    missing = sorted(set(_REQUIRED) - set(data.columns))
    if missing:
        raise ResultsSchemaError(f"Missing columns: {missing}. Supply an explicit identity_map for CSV versions; do not guess generations.")
    if exported_csv and "imputed" not in data:
        raise ResultsSchemaError("Official CSV import must carry imputed; re-export with fillna=None")
    if "imputed" in data:
        flag = data["imputed"].map(lambda x: str(x).lower())
        if not flag.isin(["false", "true", "0", "1"]).all():
            raise ResultsSchemaError("imputed must contain explicit booleans")
        if flag.isin(["true", "1"]).any():
            raise ResultsSchemaError("Imputed results are forbidden; export with fillna=None")
    for column in ("dataset", "method", "metric", "problem_type", "method_subtype", "config_type", "ta_name", "ta_suite"):
        if data[column].isna().any() or data[column].astype(str).str.strip().eq("").any():
            raise ResultsSchemaError(f"Missing identifier: {column}")
    for column in ("fold", "metric_error", "metric_error_val"):
        data[column] = pd.to_numeric(data[column], errors="raise")
        if not np.isfinite(data[column].to_numpy(dtype=float)).all():
            raise ResultsSchemaError(f"Non-finite {column}: missing results cannot be replaced by scores")
    if (data["fold"] < 0).any() or (data["fold"] % 1 != 0).any():
        raise ResultsSchemaError("fold must be a nonnegative integer")
    data["fold"] = data["fold"].astype(int)
    if not data["problem_type"].isin(["binary", "multiclass", "regression"]).all():
        raise ResultsSchemaError("Unknown problem_type")
    for dataset, group in data.groupby("dataset"):
        if group.metric.nunique() != 1 or group.problem_type.nunique() != 1:
            raise ResultsSchemaError(f"Dataset {dataset} mixes metrics/problem types; keep separate experiments")
    data = data.rename(columns={"dataset": "dataset_id", "fold": "split_id",
                                "metric_error": "test_error", "metric_error_val": "validation_error"})
    data["dataset_source"] = "tabarena"
    data["artifact_id"] = artifact_id
    data["imputed"] = False  # verified pre-comparison cached tables have no leaderboard filling
    identities = [identify_model(name, config) for name, config in zip(data.ta_name, data.config_type)]
    data["model_id"] = [identity.model_id for identity in identities]
    data["model_version"] = [identity.version for identity in identities]
    data["exact_model_id"] = [exact_identifier(suite, name, method)
                              for suite, name, method in zip(data.ta_suite, data.ta_name, data.method)]
    if data.duplicated(ROW_KEYS).any():
        raise ResultsSchemaError("Duplicate dataset/split/exact-model records")
    return data.sort_values(ROW_KEYS).reset_index(drop=True)


def load_results(path: str | Path, *, artifact_id: str = "imported", identity_map=None) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
        exported = False
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        exported = True
    else:
        raise ResultsSchemaError("Expected cached .parquet or exported .csv")
    return standardize_results(frame, artifact_id=artifact_id, exported_csv=exported, identity_map=identity_map)
