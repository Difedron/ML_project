"""Five configurable diagnostic models; training-only CV, never candidate models."""
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from meta_selector.artifacts import config_hash, environment
from .preprocessing import make_preprocessor

MODEL_CLASSES = {"logistic_regression": LogisticRegression, "decision_tree": DecisionTreeClassifier,
                 "knn": KNeighborsClassifier, "gaussian_nb": GaussianNB, "lda": LinearDiscriminantAnalysis}
CUSTOM_LANDMARKER_MODELS = tuple(MODEL_CLASSES)
CUSTOM_FEATURE_NAMES = tuple(f"{name}_balanced_accuracy_{summary}" for name in MODEL_CLASSES
                             for summary in ("mean", "std"))


def classification_score(model, X, y, metric, classes):
    if metric == "balanced_accuracy":
        return float(balanced_accuracy_score(y, model.predict(X)))
    if metric == "macro_f1":
        return float(f1_score(y, model.predict(X), average="macro", zero_division=0))
    probabilities = model.predict_proba(X)
    if metric == "log_loss":
        return float(log_loss(y, probabilities, labels=classes))
    if metric == "roc_auc":
        return float(roc_auc_score(y, probabilities[:, 1] if len(classes) == 2 else probabilities,
                                   labels=classes, multi_class="ovr", average="macro"))
    raise ValueError(f"Unknown classification metric: {metric}")


def make_landmarker(name, X_train, config):
    if name not in MODEL_CLASSES:
        raise ValueError(f"Unknown custom landmarker: {name}")
    params = dict(config.get("parameters", {}).get(name, {}))
    if name in {"logistic_regression", "decision_tree"}:
        params.setdefault("random_state", config["random_seed"])
    model = MODEL_CLASSES[name](**params)
    return Pipeline([("preprocess", make_preprocessor(X_train, scale=name in {
        "logistic_regression", "knn", "lda"}, max_dense_cells=config["max_dense_cells"])), ("model", model)])


def run_custom_landmarkers(X_train, y_train, *, identity, config, split_provenance=None):
    """Only X_train/y_train enter this API. Return one status-bearing row per model."""
    if config["scoring"] not in {"balanced_accuracy", "macro_f1", "roc_auc", "log_loss"}:
        raise ValueError("Unknown custom scoring metric")
    names = config["models"]
    if not names or len(names) != len(set(names)) or not set(names) <= set(MODEL_CLASSES):
        raise ValueError("Custom models must be distinct registered identifiers")
    config_id = config_hash(config)
    provenance = dict(config=config, environment=environment(), split=split_provenance,
                      scoring_direction="minimize" if config["scoring"] == "log_loss" else "maximize",
                      input_role="outer_training_only", cv_std_ddof=0)
    rows = []
    for name in names:
        row = dict(**identity, model_id=name, metric=config["scoring"], config_id=config_id,
                   source="custom_landmarking", extractor_version=provenance["environment"]["scikit-learn"],
                   score_mean=np.nan, score_std=np.nan, fit_time_mean=np.nan, fit_time_std=np.nan,
                   n_folds=0, status="error", error_message="", provenance=json.dumps(provenance, sort_keys=True))
        try:
            if y_train.isna().any() or y_train.nunique() < 2:
                raise ValueError("Inner classification CV requires nonmissing labels and at least two classes")
            folds = int(config["n_splits"])
            if folds < 2 or y_train.value_counts().min() < folds:
                raise ValueError("Insufficient class support for configured StratifiedKFold")
            cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=config["random_seed"])
            scores, times, captured = [], [], []
            for train_indices, validation_indices in cv.split(X_train, y_train):
                model = make_landmarker(name, X_train, config)
                start = time.perf_counter()
                with warnings.catch_warnings(record=True) as found:
                    warnings.simplefilter("always")
                    model.fit(X_train.iloc[train_indices], y_train.iloc[train_indices])
                    times.append(time.perf_counter() - start)
                    scores.append(classification_score(model, X_train.iloc[validation_indices],
                                                       y_train.iloc[validation_indices], config["scoring"], model.classes_))
                    captured.extend(str(item.message) for item in found)
            if not np.isfinite(scores).all():
                raise ValueError("Non-finite inner CV scores")
            row.update(score_mean=float(np.mean(scores)), score_std=float(np.std(scores)),
                       fit_time_mean=float(np.mean(times)), fit_time_std=float(np.std(times)),
                       n_folds=folds, status="ok", provenance=json.dumps({**provenance, "warnings": sorted(set(captured))}, sort_keys=True))
        except Exception as error:
            row["error_message"] = f"{type(error).__name__}: {error}"
        rows.append(row)
    return pd.DataFrame(rows)
