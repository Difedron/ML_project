"""Column roles and explicit input allowlists for the two pilot feature sets."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from meta_selector.datasets.shared_pilot import JOIN_KEYS


VARIANTS = {
    ("lightgbm", "baseline"): ("lightgbm_baseline", "lightgbm"),
    ("lightgbm", "priortune_t_k25"): ("lightgbm_priortune_t_k25", "lightgbm_priortune_t_k25"),
    ("tabpfn", "v2_6"): ("tabpfn_v2_6", "tabpfn_2_6"),
    ("tabicl", "v2"): ("tabicl_v2", "tabicl_v2"),
}
VARIANT_IDS = tuple(name for name, _ in VARIANTS.values())
OUTCOMES = {"balanced_accuracy": "ba", "roc_auc": "roc_auc", "log_loss": "log_loss"}
DIAGNOSTICS = ("validation_metric", "validation_score", "total_time_sec")
TARGETS = ("best_balanced_accuracy", "best_target", "best_target_is_tie", "best_target_count", "best_target_candidates")
SOURCES = ("dataset_source_rita", "dataset_source_diana", "dataset_source_normalized")
META_PREFIXES = ("feature::controls::", "feature::paper_features::")
LANDMARKER_MODELS = {"Logistic Regression": "logistic_regression", "Decision Tree": "decision_tree",
                    "kNN": "knn", "Naive Bayes": "gaussian_nb", "LDA": "lda"}
LANDMARKERS = {label: tuple(f"feature::custom_landmarking::{model}_balanced_accuracy_{stat}"
                            for stat in ("mean", "std")) for label, model in LANDMARKER_MODELS.items()}
LANDMARKER_NAMES = {column for columns in LANDMARKERS.values() for column in columns}
ROLE_NAMES = ("identity_columns", "meta_only_feature_columns", "landmarker_feature_columns",
              "candidate_outcome_columns", "target_columns", "diagnostic_columns")


@dataclass(frozen=True)
class ColumnRoles:
    identity_columns: tuple[str, ...]
    meta_only_feature_columns: tuple[str, ...]
    landmarker_feature_columns: tuple[str, ...]
    candidate_outcome_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    diagnostic_columns: tuple[str, ...]

    @classmethod
    def from_predictors(cls, predictors):
        roles = cls((*JOIN_KEYS, *SOURCES),
                    tuple(name for name in predictors if name.startswith(META_PREFIXES)),
                    tuple(name for name in predictors if name in LANDMARKER_NAMES),
                    tuple(f"{metric}_{variant}" for metric in OUTCOMES.values() for variant in VARIANT_IDS),
                    (*TARGETS, *(f"gap_{variant}" for variant in VARIANT_IDS)),
                    tuple(f"{metric}_{variant}" for metric in DIAGNOSTICS for variant in VARIANT_IDS))
        if len(predictors) != len(roles.meta_plus_landmarker_feature_columns):
            raise ValueError("Rita export has predictors without a recognized feature role")
        roles.validate(roles.columns)
        return roles

    @property
    def meta_plus_landmarker_feature_columns(self):
        return (*self.meta_only_feature_columns, *self.landmarker_feature_columns)

    @property
    def columns(self):
        return tuple(column for role in ROLE_NAMES for column in getattr(self, role))

    def validate(self, columns):
        declared = self.columns
        if len(declared) != len(set(declared)) or tuple(columns) != declared:
            raise ValueError("Column roles must partition the integrated schema exactly")
        if (not self.meta_only_feature_columns
                or any(not name.startswith(META_PREFIXES) for name in self.meta_only_feature_columns)
                or set(self.landmarker_feature_columns) != LANDMARKER_NAMES):
            raise ValueError("Leakage guard: X must use the exported meta-feature and landmarker allowlists")

    def as_dict(self):
        return {role: list(getattr(self, role)) for role in (*ROLE_NAMES, "meta_plus_landmarker_feature_columns")}

    def feature_frame(self, frame, *, include_landmarkers=False):
        """Select X from explicit feature names and leave all missing values intact."""
        self.validate(frame.columns)
        columns = self.meta_plus_landmarker_feature_columns if include_landmarkers else self.meta_only_feature_columns
        return frame.loc[:, list(columns)].copy()


def load_column_roles(manifest_path):
    declared = json.loads(Path(manifest_path).read_text(encoding="utf-8"))["column_roles"]
    roles = ColumnRoles(**{role: tuple(declared[role]) for role in ROLE_NAMES})
    roles.validate(roles.columns)
    if declared != roles.as_dict():
        raise ValueError("Manifest feature lists disagree with column roles")
    return roles


def feature_diagnostics(frame, roles):
    roles.validate(frame.columns)
    rows, representatives = [], []
    for name in roles.meta_plus_landmarker_feature_columns:
        values = frame[name]
        numeric = pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values)
        array = values.to_numpy(dtype=float, na_value=np.nan) if numeric else np.array([])
        positive, negative = int(np.isposinf(array).sum()), int(np.isneginf(array).sum())
        missing = int(values.isna().sum())
        duplicate = next((other for other in representatives if values.equals(frame[other])), None)
        if duplicate is None:
            representatives.append(name)
        rows.append(dict(column=name, role="landmarker" if name in LANDMARKER_NAMES else "meta_feature",
                         missing_count=missing, missing_fraction=missing / len(frame), inf_count=positive + negative,
                         positive_inf_count=positive, negative_inf_count=negative, constant=values.nunique() == 1,
                         all_nan=missing == len(frame), duplicate_of=duplicate, non_numeric=not numeric))
    return pd.DataFrame(rows)
