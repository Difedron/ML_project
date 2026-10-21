"""Explicit feature provenance and separate namespaces; original manifest is unknown."""
import importlib.metadata
import json
import warnings

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.impute import SimpleImputer

from meta_selector.artifacts import config_hash, environment
from meta_selector.landmarkers.preprocessing import clean_frame
from .controls import compute_controls


def feature_rows(values, *, identity, source, group, version, config, provenance, status="ok", error=""):
    return pd.DataFrame([dict(**identity, name=name, value=float(value), source=source,
                              group=group, extractor_version=version, config_id=config_hash(config),
                              status=status if np.isfinite(value) else "undefined", error_message=error,
                              provenance=json.dumps(provenance, sort_keys=True)) for name, value in values.items()])


def extract_controls(X_train, *, identity, provenance):
    config = {"log_base": "e", "missing": "cells", "categorical": "nonnumeric_including_bool", "version": "1"}
    return feature_rows(compute_controls(X_train), identity=identity, source="controls", group="controls",
                        version="meta-selector-controls-1", config=config, provenance=provenance)


def _pymfe_input(X_train):
    """Explicit reconstruction: imputation on training only, categories remain categories."""
    frame = clean_frame(X_train)
    categorical = []
    for index, column in enumerate(frame):
        numeric = is_numeric_dtype(X_train[column].dtype) and not is_bool_dtype(X_train[column].dtype)
        imputer = SimpleImputer(strategy="median" if numeric else "most_frequent", keep_empty_features=True)
        frame[column] = imputer.fit_transform(frame[[column]]).ravel()
        if not numeric:
            categorical.append(index)
            frame[column] = frame[column].astype(str)
    return frame.to_numpy(dtype=object), categorical


def extract_pymfe(X_train, y_train, *, identity, config, source, provenance):
    import os
    import tempfile
    from pathlib import Path
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "meta-selector-matplotlib"))
    from pymfe.mfe import MFE
    if source not in {"paper_features", "pymfe_standard_landmarking"}:
        raise ValueError("Unknown PyMFE namespace")
    groups = config["groups"]
    if not groups or isinstance(groups, str):
        raise ValueError("Supply an explicit PyMFE groups list")
    if source == "paper_features" and set(groups) & {"landmarking", "relative"}:
        raise ValueError("Keep standard landmarking/relative separate from paper-style features")
    if source == "pymfe_standard_landmarking" and groups != ["landmarking"]:
        raise ValueError("Standard landmarking namespace requires groups: [landmarking]")
    if y_train.isna().any():
        raise ValueError("PyMFE requires nonmissing training labels")
    options = {key: config[key] for key in ("num_cv_folds", "shuffle_cv_folds", "score") if key in config}
    if "landmarking" in groups and y_train.value_counts().min() < options.get("num_cv_folds", 10):
        raise ValueError("Insufficient training class support for PyMFE CV")
    values, cat_cols = _pymfe_input(X_train)
    version = importlib.metadata.version("pymfe")
    with warnings.catch_warnings(record=True) as found:
        warnings.simplefilter("always")
        extractor = MFE(groups=groups, summary=config["summaries"], random_state=config["random_seed"], **options)
        extractor.fit(values, y_train.to_numpy(), cat_cols=cat_cols, precomp_groups=None)
        names, scores = extractor.extract()
    group_lookup = {}
    for group in groups:
        for feature in MFE.valid_metafeatures(groups=[group]):
            group_lookup.setdefault(feature, []).append(group)
    context = {**provenance, "manifest": config, "environment": environment(),
               "label": config.get("label", "standard PyMFE landmarks"),
               "paper_exact_set_confirmed": config.get("confirmed", False),
               "warnings": sorted(set(str(item.message) for item in found))}
    tables = []
    for name, value in zip(names, scores):
        tables.append(feature_rows({name: value}, identity=identity, source=source,
                                   group="|".join(group_lookup.get(name.split('.')[0], groups)),
                                   version=version, config=config, provenance=context))
    if not tables:
        raise ValueError("PyMFE produced no features")
    return pd.concat(tables, ignore_index=True)
