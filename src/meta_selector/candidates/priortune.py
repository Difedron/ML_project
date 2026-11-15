"""Pinned PriorTune portfolio, validation-only trials and their research artifact."""
from dataclasses import dataclass
import hashlib
import json
from math import isfinite
from pathlib import Path

import pandas as pd

from meta_selector.artifacts import research_paths, upsert_table
from meta_selector.datasets.shared_pilot import JOIN_KEYS, contained_path
from meta_selector.metrics import aligned_class_probabilities, classification_probability_score


PORTFOLIO_ID = "lgbm/T/K25"
PORTFOLIO_PIN = {
    "repo_id": "mrsergazinov/PriorTune",
    "revision": "8b9cab4c0d6569ddda5cc2a18d78e0de75c526ee",
    "path": "portfolios/lgbm/T_K25.json",
    "sha256": "322a4e87966223ec268cdf390ba8d7a58e3b13d2a2a4a7f4dddd3989ab2688e5",
    "model": "lgbm", "rule": "T", "k": 25, "tabprep": False, "ensemble": False,
}
TRIAL_KEYS = (*JOIN_KEYS, "model_family", "variant", "portfolio_id", "portfolio_revision",
              "portfolio_sha256", "config_id")
PARAMETER_ALIASES = {"bagging_fraction": "subsample", "bagging_freq": "subsample_freq",
                     "feature_fraction": "colsample_bytree", "lambda_l1": "reg_alpha",
                     "lambda_l2": "reg_lambda", "min_data_in_leaf": "min_child_samples"}
INTEGER_PARAMETERS = {"bagging_freq": 1, "max_cat_to_onehot": 1, "min_data_in_leaf": 1,
                      "min_data_per_group": 1, "num_leaves": 2}
FRACTION_PARAMETERS = {"bagging_fraction", "feature_fraction"}
NONNEGATIVE_PARAMETERS = {"cat_l2", "cat_smooth", "lambda_l1", "lambda_l2"}
PUBLISHED_PARAMETERS = (INTEGER_PARAMETERS.keys() | FRACTION_PARAMETERS | NONNEGATIVE_PARAMETERS
                        | {"extra_trees", "learning_rate"})
TRAINING_POLICY = {"n_estimators": 1000, "stopping_rounds": 50, "first_metric_only": True,
                   "min_delta": 0.0, "verbose": False}


class PortfolioError(ValueError):
    pass


class IncompletePortfolioError(RuntimeError):
    pass


@dataclass(frozen=True)
class PortfolioConfiguration:
    config_id: int
    model_seed: int
    lightgbm: dict


def selection_rule(n_classes):
    return ("roc_auc", "maximize", "auc") if n_classes == 2 else ("log_loss", "minimize", "multi_logloss")


def _valid_parameters(parameters):
    if not isinstance(parameters, dict) or parameters.keys() != PUBLISHED_PARAMETERS:
        return False
    if not all(type(parameters[key]) is int and parameters[key] >= minimum
               for key, minimum in INTEGER_PARAMETERS.items()):
        return False
    numeric = FRACTION_PARAMETERS | NONNEGATIVE_PARAMETERS | {"learning_rate"}
    if not all(type(parameters[key]) in (int, float) and isfinite(parameters[key]) for key in numeric):
        return False
    return (type(parameters["extra_trees"]) is bool and parameters["learning_rate"] > 0
            and all(0 < parameters[key] <= 1 for key in FRACTION_PARAMETERS)
            and all(parameters[key] >= 0 for key in NONNEGATIVE_PARAMETERS))


def _unique_ids(values):
    return (isinstance(values, list) and len(values) == PORTFOLIO_PIN["k"]
            and all(type(value) is int and value >= 0 for value in values)
            and len(set(values)) == len(values))


def load_portfolio(path, specification):
    """Read the exact cached resource once. Downloads belong to the preflight command."""
    for key, expected in PORTFOLIO_PIN.items():
        actual = specification.get(key)
        if type(actual) is not type(expected) or actual != expected:
            raise PortfolioError(f"PriorTune portfolio identity mismatch for {key}: expected {expected!r}")
    path = Path(path)
    try:
        with path.open("rb") as stream:
            payload = stream.read(131073)
    except OSError as error:
        raise PortfolioError(f"Exact PriorTune portfolio unavailable at {path}. Run preflight --fetch-portfolio.") from error
    if len(payload) > 131072 or hashlib.sha256(payload).hexdigest() != PORTFOLIO_PIN["sha256"]:
        raise PortfolioError(f"PriorTune portfolio SHA256 mismatch at {path}")
    try:
        record = json.loads(payload)
    except (ValueError, UnicodeError) as error:
        raise PortfolioError("Invalid PriorTune portfolio JSON") from error
    if (not isinstance(record, dict) or record.get("model") != "lgbm" or record.get("objective") != "T"
            or type(record.get("K")) is not int or record["K"] != PORTFOLIO_PIN["k"]):
        raise PortfolioError("Expected a published lgbm/T/K=25 portfolio")
    ids, order, rows = (record.get(key) for key in ("config_ids", "add_order", "configs"))
    if not _unique_ids(ids) or not _unique_ids(order) or set(ids) != set(order):
        raise PortfolioError("Expected 25 unique config IDs and matching published add_order")
    if not isinstance(rows, list) or len(rows) != PORTFOLIO_PIN["k"]:
        raise PortfolioError("Expected exactly 25 published configurations")
    configurations = {}
    for row in rows:
        if not isinstance(row, dict) or row.keys() != {"config_id", "model_seed", "lightgbm", "tabprep", "tabprep_enabled"}:
            raise PortfolioError("Unexpected fields in a PriorTune configuration")
        config_id, model_seed = row["config_id"], row["model_seed"]
        if type(config_id) is not int or config_id not in ids or config_id in configurations:
            raise PortfolioError(f"Unexpected or duplicate PriorTune config ID: {config_id!r}")
        if (type(model_seed) is not int or not 0 <= model_seed <= 2**31 - 1 or row["tabprep"] is not None
                or row["tabprep_enabled"] is not False or not _valid_parameters(row["lightgbm"])):
            raise PortfolioError(f"Invalid plain LightGBM configuration: {config_id}")
        configurations[config_id] = PortfolioConfiguration(config_id, model_seed, row["lightgbm"])
    return tuple(configurations[config_id] for config_id in order)


def effective_parameters(config, seed, n_classes):
    """Map engine aliases to sklearn names so published values win over SDK defaults."""
    parameters = {PARAMETER_ALIASES.get(key, key): value for key, value in config.lightgbm.items()}
    parameters.update(
        boosting_type="gbdt", objective="binary" if n_classes == 2 else "multiclass",
        metric=selection_rule(n_classes)[2], n_estimators=TRAINING_POLICY["n_estimators"],
        random_state=config.model_seed, data_random_seed=seed, feature_fraction_seed=seed,
        bagging_seed=seed, extra_seed=seed, device_type="cpu", n_jobs=1, deterministic=True,
        force_col_wise=True, verbosity=-1,
    )
    if n_classes > 2:
        parameters["num_class"] = n_classes
    return parameters


def select_on_validation(configurations, train, y_train, validation, y_validation, *,
                         result, categorical_columns, estimator_factory, callback_factory, clock):
    """Fit and rank on train and validation. Retain only the best already-fitted model."""
    metric, direction, engine_metric = selection_rule(result.n_classes)
    callback_options = {key: value for key, value in TRAINING_POLICY.items() if key != "n_estimators"}
    trial_context = dict(dataset_id=result.dataset_id, split_id=result.split_id, seed=result.seed,
                         model_family=result.model_family, variant=result.variant,
                         portfolio_id=PORTFOLIO_ID, portfolio_revision=PORTFOLIO_PIN["revision"],
                         portfolio_sha256=PORTFOLIO_PIN["sha256"], selection_metric=metric,
                         selection_direction=direction)
    best = None
    for config in configurations:
        started = clock()
        params = effective_parameters(config, result.seed, result.n_classes)
        trial = dict(**trial_context, config_id=config.config_id, effective_parameters=params,
                     validation_score=None, model_init_time_sec=None, fit_time_sec=None,
                     validation_time_sec=None, status="ok", error_type="", error_message="")
        try:
            init_started = clock()
            try:
                model = estimator_factory("lightgbm", params.copy())
            finally:
                trial["model_init_time_sec"] = clock() - init_started
            fit_started = clock()
            try:
                model.fit(train.copy(deep=True), y_train.copy(),
                          eval_set=[(validation.copy(deep=True), y_validation.copy())], eval_metric=engine_metric,
                          categorical_feature=list(categorical_columns), callbacks=[callback_factory(**callback_options)])
            finally:
                trial["fit_time_sec"] = clock() - fit_started
            score_started = clock()
            try:
                probabilities = aligned_class_probabilities(model, validation.copy(deep=True), result.n_classes)
                score = classification_probability_score(y_validation, probabilities, metric)
                if not isfinite(score):
                    raise ValueError("Nonfinite validation score")
            finally:
                trial["validation_time_sec"] = clock() - score_started
            trial["validation_score"] = score
            rank = (-score if direction == "maximize" else score, config.config_id)
            if best is None or rank < best[0]:
                best = (rank, model, trial)
        except Exception as error:
            trial.update(status="error", error_type=type(error).__name__, error_message=str(error))
        finally:
            trial["total_trial_time_sec"] = clock() - started
            result.validation_trials.append(trial)
    return None if best is None else best[1:]


def write_validation_trials(result, project_root):
    """Upsert one complete batch, including failed trials, to the validation-only table."""
    derived = research_paths(project_root)["derived"]
    path = contained_path(derived, "priortune_validation/trials.parquet")
    rows = [{**trial, "effective_parameters": json.dumps(trial["effective_parameters"], sort_keys=True)}
            for trial in result.validation_trials]
    upsert_table(path, pd.DataFrame(rows), list(TRIAL_KEYS))
    return path
