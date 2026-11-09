"""Stage-3 candidate boundary. Models are imported only by an explicit run call.

The PriorTune wrapper reserves the API; portfolio execution belongs to stage 4.
No function here creates splits or writes experiment artifacts.
"""
from copy import deepcopy
import importlib.metadata
from pathlib import Path
import time

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import LabelEncoder

from meta_selector.artifacts import config_hash, environment
from meta_selector.datasets.shared_pilot import JOIN_KEYS
from meta_selector.metrics import classification_scores
from meta_selector.paths import DEFAULT_PROJECT_ROOT, read_yaml_mapping
from .devices import select_device
from .runner import CandidateResult, verified_weight


CANDIDATE_RESULT_KEYS = (*JOIN_KEYS, "model_family", "variant", "exact_model_id", "validation_metric")
CHECKPOINTS = {"tabpfn": "tabpfn-v2.6-classifier-v2.6_default.ckpt",
               "tabicl": "tabicl-classifier-v2-20260212.ckpt"}
METRICS = {"balanced_accuracy", "macro_f1", "roc_auc", "log_loss"}


class ExactModelUnavailableError(RuntimeError):
    pass


class Stage4RequiredError(NotImplementedError):
    pass


def official_estimator(family, params):
    """Exact installed official APIs; never select a latest/default checkpoint."""
    if family == "tabpfn":
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion
        return TabPFNClassifier.create_default_for_version(ModelVersion.V2_6, **params)
    if family == "tabicl":
        from tabicl import TabICLClassifier
        return TabICLClassifier(**params)
    if family == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**params)
    raise ValueError(f"Unknown model family: {family}")


def early_stopping_callback(**params):
    from lightgbm import early_stopping
    return early_stopping(**params)


class TrainOnlyFeatures:
    """Explicit Rita categories only. Raw inputs are copied, never modified.

    LightGBM category dictionaries are fitted on train. Neural SDKs receive category
    strings (including numeric codes such as mushroom's 0) and fit their own encoders.
    """
    def __init__(self, categorical_columns, family):
        self.categorical_columns, self.family = tuple(categorical_columns), family

    def fit_transform(self, X):
        self.columns_ = X.columns.copy()
        self.categories_ = {name: list(pd.unique(X[name].dropna())) for name in self.categorical_columns}
        return self.transform(X)

    def transform(self, X):
        if not X.columns.equals(self.columns_):
            raise ValueError("Feature names/order differ from training")
        result = X.copy(deep=True)
        for name in X.columns:
            if name in self.categories_:
                if self.family == "lightgbm":
                    known = X[name].where(X[name].isin(self.categories_[name]))
                    result[name] = pd.Categorical(known, categories=self.categories_[name])
                else:
                    result[name] = X[name].map(lambda value: str(value) if pd.notna(value) else np.nan).astype(object)
            elif not is_numeric_dtype(X[name].dtype):
                raise ValueError(f"Nonnumeric feature requires explicit categorical metadata: {name}")
            else:
                # Native missing-value handling; no imputation, scaling or row dropping.
                result[name] = X[name].to_numpy(dtype=float, na_value=np.nan)
        return result


def aligned_predictions(model, X, n_classes):
    classes = np.asarray(model.classes_)
    expected = np.arange(n_classes)
    if classes.shape != expected.shape or set(classes.tolist()) != set(expected.tolist()):
        raise ValueError("Estimator classes do not match training labels")
    probabilities = np.asarray(model.predict_proba(X), dtype=float)
    if probabilities.shape != (len(X), n_classes):
        raise ValueError("Estimator returned an invalid probability shape")
    order = [int(np.flatnonzero(classes == label)[0]) for label in expected]
    return np.asarray(model.predict(X)), probabilities[:, order]


class BaseCandidate:
    model_id = model_family = variant = model_version = ""

    def __init__(self, *, project_root=None, config=None, device=None, estimator_factory=None,
                 callback_factory=None, device_probe=None, version_probe=None, clock=None):
        self.root = Path(project_root or DEFAULT_PROJECT_ROOT).resolve()
        self.config = deepcopy(config or read_yaml_mapping(self.root / "config/candidates/pilot.yaml"))
        if self.config.get("schema_version") != 1:
            raise ValueError("Unsupported candidate config schema")
        self.resources = read_yaml_mapping(self.root / self.config["resources_config"])
        self.baseline = read_yaml_mapping(self.root / self.config["baseline_config"])
        self.requested_device = device if device is not None else self.config["device"]
        self.estimator_factory = estimator_factory or official_estimator
        self.callback_factory = callback_factory or early_stopping_callback
        self.device_probe = device_probe
        self.version_probe = version_probe or importlib.metadata.version
        self.clock = clock or time.perf_counter

    def available_models(self):
        """Implemented IDs only; resource/device readiness remains stage-1 preflight's job."""
        return [self.model_id]

    def _params(self, seed, X_train, categorical_columns, result):
        required = self.config["package_versions"][self.model_family]
        if self.model_family == "lightgbm":
            if required != self.baseline["package_version"]:
                raise ExactModelUnavailableError("Baseline/package version pins disagree")
            params = deepcopy(self.baseline["model_params"])
            # These four seeds belong to the baseline, not PriorTune's published configs.
            for name in ("random_state", "data_random_seed", "feature_fraction_seed", "bagging_seed"):
                params[name] = seed
            task = "binary" if result.n_classes == 2 else "multiclass"
            params["objective"] = self.baseline["objective_by_task"][task]
            if result.n_classes > 2:
                params["num_class"] = result.n_classes
            return params
        spec = self.resources["checkpoints"][self.model_family]
        if spec["filename"] != CHECKPOINTS[self.model_family]:
            raise ExactModelUnavailableError("unavailable_exact_version: checkpoint generation mismatch")
        configured = self.config["checkpoint_paths"][self.model_family]
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = self.root / path
        try:
            path, _ = verified_weight({"weights": {self.model_id: {
                "path": str(path), "sha256": spec["sha256"]}}}, self.model_id)
        except ValueError as error:
            raise ExactModelUnavailableError(str(error)) from error
        result.checkpoint_version = spec["filename"]
        result.provenance["checkpoint"] = dict(path=str(path), sha256=spec["sha256"],
                                                repo_id=spec["repo_id"], revision=spec["revision"])
        params = deepcopy(self.config[f"{self.model_family}_params"])
        reserved = {"model_path", "device", "random_state", "checkpoint_version", "allow_auto_download",
                    "categorical_features_indices"}
        if reserved.intersection(params):
            raise ValueError("Candidate overrides cannot replace identity, seed, device or category mapping")
        params.update(model_path=str(path), device=result.device, random_state=seed)
        if self.model_family == "tabpfn":
            params["categorical_features_indices"] = [X_train.columns.get_loc(name) for name in categorical_columns]
        else:
            params.update(checkpoint_version=spec["filename"], allow_auto_download=False)
        return params

    def _fit(self, model, features, X_train, y_train, X_validation, y_validation, n_classes):
        train = features.fit_transform(X_train)
        options = {}
        if self.model_family == "lightgbm":
            policy = self.baseline["fit_policy"]
            if policy["early_stopping_data"] != "validation_only" or policy["refit_train_plus_validation"]:
                raise ValueError("Baseline requires validation-only stopping and train-only fit")
            task = "binary" if n_classes == 2 else "multiclass"
            options = dict(eval_set=[(features.transform(X_validation), y_validation)],
                eval_metric=self.baseline["selection_metric_by_task"][task]["lightgbm_name"],
                categorical_feature=list(features.categorical_columns),
                callbacks=[self.callback_factory(stopping_rounds=policy["early_stopping_rounds"],
                    first_metric_only=policy["early_stopping_first_metric_only"],
                    min_delta=policy["early_stopping_min_delta"], verbose=False)])
        model.fit(train, y_train, **options)

    def run(self, model_id, X_train, y_train, X_validation, y_validation, X_test, y_test,
            *, dataset_id, split_id, metric=None, seed=None, dataset_source="pmlb", categorical_columns=()):
        """CandidateRunner-compatible call. Only an explicit call can instantiate/fit a model.

        `metric=None` uses ROC-AUC for binary and log-loss for multiclass. The three
        top-level quality fields are test metrics; all validation scores are in provenance.
        """
        started = self.clock()
        result = CandidateResult(dataset_id=dataset_id, split_id=split_id, model_id=self.model_id,
            model_version=self.model_version, exact_model_id="", validation_metric=metric or "",
            test_metric=metric or "", dataset_source=dataset_source, seed=seed,
            model_family=self.model_family, variant=self.variant,
            checkpoint_version=CHECKPOINTS.get(self.model_family))
        try:
            if model_id != self.model_id:
                raise ValueError(f"This wrapper only implements {self.model_id}")
            if type(seed) is not int:
                raise ValueError("An explicit shared split seed is required")
            if not dataset_id or not split_id or dataset_source != "pmlb":
                raise ValueError("Explicit PMLB dataset/split identity is required")
            portions = ((X_train, y_train), (X_validation, y_validation), (X_test, y_test))
            for X, y in portions:
                if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
                    raise ValueError("Candidates require pandas DataFrame/Series portions")
                if not len(X) or not X.shape[1] or len(X) != len(y) or y.isna().any():
                    raise ValueError("Need nonempty features and nonmissing, aligned labels")
                if not X.index.equals(y.index) or not X.columns.equals(X_train.columns) or X.columns.has_duplicates:
                    raise ValueError("Feature columns or feature/label row order mismatch")
                if y.name is not None and y.name in X:
                    raise ValueError("Target column must be excluded from features")
            if (len(categorical_columns) != len(set(categorical_columns))
                    or not set(categorical_columns) <= set(X_train)):
                raise ValueError("Categorical metadata must name distinct training features")
            labels = LabelEncoder().fit(y_train)
            n_classes = len(labels.classes_)
            if n_classes < 2:
                raise ValueError("Classification requires at least two training classes")
            y_parts = [labels.transform(y) for _, y in portions]
            if any(set(y) != set(range(n_classes)) for y in y_parts):
                raise ValueError("Every portion requires all training classes; review the shared split")
            result.n_train, result.n_validation, result.n_test = [len(y) for y in y_parts]
            result.n_features, result.n_classes = X_train.shape[1], n_classes
            result.validation_metric = result.test_metric = metric or ("roc_auc" if n_classes == 2 else "log_loss")
            if result.validation_metric not in METRICS:
                raise ValueError("Unknown classification metric")
            versions = environment()
            result.python_version = versions.pop("python")
            for package in ("torch", "tabpfn", "tabicl", "lightgbm"):
                try:
                    versions[package] = self.version_probe(package)
                except importlib.metadata.PackageNotFoundError:
                    versions[package] = None
            result.package_versions = versions
            result.provenance = dict(environment={"python": result.python_version, **versions},
                join_keys=list(JOIN_KEYS), join_key=[dataset_id, split_id, seed],
                problem_type="binary" if n_classes == 2 else "multiclass",
                label_classes=labels.classes_.tolist(), categorical_columns=list(categorical_columns),
                preprocessing_fit_on="train_only", requested_device=self.requested_device,
                metric_direction="minimize" if result.validation_metric == "log_loss" else "maximize",
                metric_scope="test", config_sha256=config_hash(self.config),
                source_revisions=self.resources["source_revisions"])
            if self.variant == "priortune_t_k25":
                result.device = "cpu"
                spec = self.resources["portfolio"]
                result.provenance["portfolio_contract"] = {
                    key: spec[key] for key in ("repo_id", "revision", "path", "sha256", "k", "rule")}
                result.provenance["portfolio_contract"]["selection_data"] = "validation_only"
                raise Stage4RequiredError("PriorTune K=25 execution/selection is reserved for stage 4")
            required = self.config["package_versions"][self.model_family]
            if versions[self.model_family] != required:
                raise ExactModelUnavailableError(f"Expected {self.model_family} package {required}")
            init_started = self.clock()
            result.device = select_device(self.requested_device, self.model_family, probe=self.device_probe)
            params = self._params(seed, X_train, categorical_columns, result)
            result.provenance["model_params"] = params
            identity_params = {key: value for key, value in params.items() if key != "model_path"}
            checkpoint = result.provenance.get("checkpoint")
            checkpoint_identity = {key: value for key, value in checkpoint.items() if key != "path"} if checkpoint else None
            result.exact_model_id = f"pilot::{self.model_id}::{config_hash(dict(
                model_family=self.model_family, variant=self.variant, params=identity_params,
                checkpoint=checkpoint_identity, package_version=versions[self.model_family]))}"
            try:
                model = self.estimator_factory(self.model_family, params)
            finally:
                result.model_init_time_sec = self.clock() - init_started
            features = TrainOnlyFeatures(categorical_columns, self.model_family)
            fit_started = self.clock()
            try:
                self._fit(model, features, X_train, y_parts[0], X_validation, y_parts[1], n_classes)
            finally:
                result.fit_time_sec = result.fit_time = self.clock() - fit_started
            result.tuning_time_sec = 0.0
            validation_started = self.clock()
            predicted, probabilities = aligned_predictions(model, features.transform(X_validation), n_classes)
            validation_scores = classification_scores(y_parts[1], predicted, probabilities, np.arange(n_classes))
            result.validation_time_sec = self.clock() - validation_started
            result.validation_score = validation_scores[result.validation_metric]
            result.provenance["validation_scores"] = validation_scores
            predict_started = self.clock()
            predicted, probabilities = aligned_predictions(model, features.transform(X_test), n_classes)
            scores = classification_scores(y_parts[2], predicted, probabilities, np.arange(n_classes))
            result.predict_time_sec = result.predict_time = self.clock() - predict_started
            result.test_score = scores[result.test_metric]
            result.balanced_accuracy, result.roc_auc, result.log_loss = (
                scores[name] for name in ("balanced_accuracy", "roc_auc", "log_loss"))
            result.provenance["test_scores"] = scores
            result.status = "ok"
        except Exception as error:
            result.error_type, result.error_message = type(error).__name__, str(error)
            result.status = ("not_implemented" if isinstance(error, Stage4RequiredError) else
                             "unavailable_exact_version" if isinstance(error, ExactModelUnavailableError) else "error")
        finally:
            result.total_time_sec = self.clock() - started
        return result


class TabPFN26Candidate(BaseCandidate):
    model_id, model_family, variant, model_version = "tabpfn_2_6", "tabpfn", "v2_6", "2.6"


class TabICLv2Candidate(BaseCandidate):
    model_id, model_family, variant, model_version = "tabicl_v2", "tabicl", "v2", "2"


class LightGBMBaselineCandidate(BaseCandidate):
    model_id, model_family, variant, model_version = "lightgbm", "lightgbm", "baseline", "4.7.0"


class LightGBMPriorTuneCandidate(BaseCandidate):
    model_id, model_family, variant, model_version = (
        "lightgbm_priortune_t_k25", "lightgbm", "priortune_t_k25", "4.7.0")

    def available_models(self):
        return []  # A common API exists; production portfolio execution is not implemented yet.


CANDIDATES = {cls.model_id: cls for cls in (
    TabPFN26Candidate, TabICLv2Candidate, LightGBMBaselineCandidate, LightGBMPriorTuneCandidate)}


class PilotCandidateRunner:
    """Small dispatcher over CandidateRunner-compatible wrappers; not a second artifact system."""
    def __init__(self, **options):
        self.candidates = {name: cls(**options) for name, cls in CANDIDATES.items()}

    def available_models(self):
        return [name for candidate in self.candidates.values() for name in candidate.available_models()]

    def run(self, model_id, *portions, **context):
        if model_id not in self.candidates:
            raise ValueError(f"Unknown pilot model: {model_id}")
        return self.candidates[model_id].run(model_id, *portions, **context)

    def run_shared(self, model_id, dataset, split, *, metric=None):
        """Consume stage-2 objects only; no make_split, row rewriting or artifact writes."""
        if dataset.dataset_source != "pmlb" or split.dataset_source != "pmlb":
            raise ValueError("Expected the canonical PMLB shared contract")
        portions = [split.portion(dataset, part) for part in ("train", "validation", "test")]
        result = self.run(model_id, *[value for pair in portions for value in pair],
            dataset_id=dataset.dataset_id, split_id=split.split_id, seed=split.seed,
            dataset_source=dataset.dataset_source, categorical_columns=dataset.categorical_columns, metric=metric)
        result.provenance["shared_split"] = dict(fingerprint=split.fingerprint,
            shared_manifest_sha256=split.metadata["shared_manifest_sha256"],
            rita_dataset_fingerprint=split.rita_dataset_fingerprint,
            row_order=split.metadata["row_order"])
        return result
