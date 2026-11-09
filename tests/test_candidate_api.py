"""Stage 3: fake estimators and tiny arrays only. Never import or fit real target models."""
from copy import deepcopy
from dataclasses import fields
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from meta_selector.artifacts import dataset_fingerprint, upsert_table
from meta_selector.candidates import (CandidateResult, LightGBMBaselineCandidate,
    LightGBMPriorTuneCandidate, PilotCandidateRunner, TabICLv2Candidate, TabPFN26Candidate)
from meta_selector.candidates.devices import DeviceUnavailableError, select_device
from meta_selector.candidates.pilot import (CANDIDATE_RESULT_KEYS, CHECKPOINTS, TrainOnlyFeatures,
                                          official_estimator)
from meta_selector.datasets.shared_pilot import PilotContractError, SharedSplit
from meta_selector.metrics import classification_score, classification_scores
from meta_selector.paths import DEFAULT_PROJECT_ROOT


WRAPPERS = [TabPFN26Candidate, TabICLv2Candidate, LightGBMBaselineCandidate]


@pytest.fixture
def setup(tmp_path):
    root = tmp_path
    (root / "config/candidates").mkdir(parents=True)
    for name in ("pilot", "preflight", "lightgbm_baseline"):
        data = yaml.safe_load((DEFAULT_PROJECT_ROOT / f"config/candidates/{name}.yaml").read_text())
        if name == "preflight":
            for family, filename in CHECKPOINTS.items():
                path = root / "checkpoints" / family / filename
                path.parent.mkdir(parents=True)
                payload = b"Unit-test fake checkpoint bytes, not model weights: " + family.encode()
                path.write_bytes(payload)
                data["checkpoints"][family]["sha256"] = hashlib.sha256(payload).hexdigest()
        (root / f"config/candidates/{name}.yaml").write_text(yaml.safe_dump(data))
    return root


def portions(n_classes=2):
    size = n_classes * 3
    X = pd.DataFrame({"signal": np.arange(size * 3) % n_classes,
                      "target_hint": np.arange(size * 3), "category": ["a", "b", None] * size})
    X.loc[size:size * 2 - 1, "category"] = "validation_only"
    X.loc[size * 2:, "category"] = "test_only"
    y = pd.Series([f"label_{i}" for i in X.signal], name="target")
    return [value for start in (0, size, size * 2) for value in (X.iloc[start:start + size], y.iloc[start:start + size])]


class FakeEstimator:
    def __init__(self, events, *, fail_fit=False):
        self.events, self.fail_fit = events, fail_fit

    def fit(self, X, y, **options):
        self.events.append(("fit", X.copy(deep=True), np.asarray(y).copy(), deepcopy(options)))
        if self.fail_fit:
            raise RuntimeError("Fake fit failure")
        self.classes_ = np.unique(y)[::-1]  # Deliberately reverse probability columns.
        return self

    def predict(self, X):
        self.events.append(("predict", X.index.tolist()))
        return X.signal.to_numpy(dtype=int)

    def predict_proba(self, X):
        self.events.append(("predict_proba", X.index.tolist()))
        values = np.full((len(X), len(self.classes_)), .2 / (len(self.classes_) - 1))
        for i, label in enumerate(X.signal):
            values[i, np.flatnonzero(self.classes_ == label)[0]] = .8
        return values


def dependencies(root, *, events=None, **options):
    events = events if events is not None else []
    def factory(family, params):
        events.append(("init", family, deepcopy(params)))
        return FakeEstimator(events)
    versions = {"tabpfn": "9.2.0", "tabicl": "2.2.0", "lightgbm": "4.7.0", "torch": "fake-unit-version"}
    ticks = itertools.count(0, .1)
    return dict(project_root=root, estimator_factory=factory, callback_factory=lambda **kwargs: ("stop", kwargs),
                version_probe=versions.__getitem__, device_probe=lambda device, family: device == "cpu",
                clock=lambda: next(ticks), **options)


def run(wrapper, data=None, **context):
    return wrapper.run(wrapper.model_id, *(data or portions()), dataset_id="pmlb_unit",
                       split_id="shared_rita", seed=77, categorical_columns=("category",), **context)


@pytest.mark.parametrize("wrapper_type", WRAPPERS)
@pytest.mark.parametrize("n_classes", [2, 3])
def test_common_api_scores_class_alignment_timings_and_train_isolation(setup, wrapper_type, n_classes):
    events, data = [], portions(n_classes)
    before = [value.copy(deep=True) for value in data]
    wrapper = wrapper_type(**dependencies(setup, events=events))
    result = run(wrapper, data)
    assert result.status == "ok", result.error_message
    assert result.seed == 77 and result.dataset_source == "pmlb"
    assert result.model_family == wrapper.model_family and result.variant == wrapper.variant
    assert result.validation_metric == ("roc_auc" if n_classes == 2 else "log_loss")
    assert result.balanced_accuracy == result.roc_auc == 1
    assert result.log_loss == pytest.approx(-np.log(.8))
    assert result.validation_score == pytest.approx(result.test_score)
    assert (result.n_train, result.n_validation, result.n_test) == (len(data[1]), len(data[3]), len(data[5]))
    assert result.n_features == 3 and result.n_classes == n_classes
    assert result.device == "cpu" and result.python_version
    assert result.package_versions[wrapper.model_family]
    assert result.provenance["join_key"] == ["pmlb_unit", "shared_rita", 77]
    fit = next(event for event in events if event[0] == "fit")
    assert fit[1].index.equals(data[0].index)
    assert "target_hint" in fit[1] and "target" not in fit[1]
    if wrapper.model_family == "lightgbm":
        assert list(fit[1].category.cat.categories) == ["a", "b"]
        assert fit[3]["eval_set"][0][0].index.equals(data[2].index)
        assert fit[3]["eval_set"][0][0].category.isna().all()
        assert fit[3]["eval_metric"] == ("auc" if n_classes == 2 else "multi_logloss")
        assert fit[3]["callbacks"][0][1]["stopping_rounds"] == 50
    else:
        assert fit[3] == {}  # Validation and test never enter neural fit.
    assert [event[1] for event in events if event[0] == "predict_proba"] == [data[2].index.tolist(), data[4].index.tolist()]
    for name in ("model_init_time_sec", "fit_time_sec", "validation_time_sec", "predict_time_sec", "total_time_sec"):
        assert np.isfinite(getattr(result, name)) and getattr(result, name) > 0
    assert result.tuning_time_sec == 0
    assert result.fit_time == result.fit_time_sec and result.predict_time == result.predict_time_sec
    for value, original in zip(data, before):
        assert value.equals(original)


@pytest.mark.parametrize("wrapper_type", WRAPPERS)
def test_exact_official_constructor_parameters_and_seed(setup, wrapper_type):
    events = []
    result = run(wrapper_type(**dependencies(setup, events=events)))
    assert result.status == "ok"
    params = events[0][2]
    assert params["random_state"] == 77
    if wrapper_type is TabPFN26Candidate:
        assert result.model_version == "2.6" and params["categorical_features_indices"] == [2]
        assert Path(params["model_path"]).name == CHECKPOINTS["tabpfn"]
    elif wrapper_type is TabICLv2Candidate:
        assert params["checkpoint_version"] == CHECKPOINTS["tabicl"]
        assert params["allow_auto_download"] is False
    else:
        assert params["n_estimators"] == 1000 and params["device_type"] == "cpu"
        for name in ("data_random_seed", "bagging_seed", "feature_fraction_seed"):
            assert params[name] == 77


def test_fourth_variant_is_api_only_without_any_portfolio_execution(setup):
    events = []
    wrapper = LightGBMPriorTuneCandidate(**dependencies(setup, events=events))
    result = run(wrapper)
    assert result.model_family == "lightgbm" and result.variant == "priortune_t_k25"
    assert result.status == "not_implemented" and result.error_type == "Stage4RequiredError"
    assert result.device == "cpu" and result.provenance["portfolio_contract"]["k"] == 25
    assert result.provenance["portfolio_contract"]["selection_data"] == "validation_only"
    assert events == [] and np.isnan(result.test_score)
    assert wrapper.available_models() == []


@pytest.mark.parametrize("family", ["tabpfn", "tabicl"])
@pytest.mark.parametrize("failure", ["missing", "hash", "generation", "package"])
def test_exact_version_failures_never_fallback_or_instantiate(setup, family, failure):
    events = []
    options = dependencies(setup, events=events)
    path = setup / "checkpoints" / family / CHECKPOINTS[family]
    if failure == "missing":
        path.unlink()
    elif failure == "hash":
        path.write_bytes(b"corrupted fake resource")
    elif failure == "generation":
        config_path = setup / "config/candidates/preflight.yaml"
        spec = yaml.safe_load(config_path.read_text())
        spec["checkpoints"][family]["filename"] = "wrong-latest.ckpt"
        config_path.write_text(yaml.safe_dump(spec))
    else:
        options["version_probe"] = lambda package: "999.0"
    result = run(({"tabpfn": TabPFN26Candidate, "tabicl": TabICLv2Candidate}[family])(**options))
    assert result.status == "unavailable_exact_version"
    assert result.error_type == "ExactModelUnavailableError" and events == []


@pytest.mark.parametrize("invalid", ["seed", "bool_seed", "source", "metric", "target", "row_order", "columns", "classes", "categories"])
def test_invalid_contract_stops_before_model_initialization(setup, invalid):
    events, data = [], portions()
    options = dict(dataset_id="pmlb_unit", split_id="rita", seed=77, categorical_columns=("category",))
    if invalid == "seed":
        options.pop("seed")
    elif invalid == "bool_seed":
        options["seed"] = True
    elif invalid == "source":
        options["dataset_source"] = "local"
    elif invalid == "metric":
        options["metric"] = "accuracy_typo"
    elif invalid == "target":
        for X, y in zip(data[::2], data[1::2]):
            X["target"] = y
    elif invalid == "row_order":
        data[3] = data[3].iloc[::-1]
    elif invalid == "columns":
        data[2] = data[2][list(reversed(data[2].columns))]
    elif invalid == "classes":
        data[3] = pd.Series("label_0", index=data[3].index, name="target")
    else:
        options["categorical_columns"] = ("nonexistent",)
    wrapper = TabPFN26Candidate(**dependencies(setup, events=events))
    result = wrapper.run(wrapper.model_id, *data, **options)
    assert result.status == "error" and result.error_type == "ValueError"
    assert events == [] and np.isnan(result.balanced_accuracy)


def test_error_status_does_not_fabricate_scores(setup):
    events = []
    options = dependencies(setup, events=events)
    options["estimator_factory"] = lambda family, params: FakeEstimator(events, fail_fit=True)
    result = run(TabPFN26Candidate(**options))
    assert result.status == "error" and result.error_type == "RuntimeError"
    assert result.error_message == "Fake fit failure" and result.total_time_sec > 0
    assert result.fit_time_sec > 0 and result.fit_time == result.fit_time_sec
    assert np.isnan(result.validation_score) and np.isnan(result.test_score)
    assert not any(event[0] == "predict_proba" for event in events)


def test_unavailable_package_stops_before_native_device_probe(setup):
    events = []
    options = dependencies(setup, events=events)
    options["version_probe"] = lambda package: None
    options["device_probe"] = lambda *args: pytest.fail("Native device probe must not run")
    result = run(TabPFN26Candidate(**options))
    assert result.status == "unavailable_exact_version" and events == []


def test_categories_nan_unknown_and_raw_immutability():
    train = pd.DataFrame({"number": [1.0, np.nan, 3], "ordinal": [0, 1, 0]})
    validation = pd.DataFrame({"number": [np.nan, 4], "ordinal": [9, 0]})
    before = train.copy(deep=True), validation.copy(deep=True)
    lightgbm = TrainOnlyFeatures(["ordinal"], "lightgbm")
    encoded = lightgbm.fit_transform(train)
    assert list(encoded.ordinal.cat.categories) == [0, 1]
    assert lightgbm.transform(validation).ordinal.isna().tolist() == [True, False]
    neural = TrainOnlyFeatures(["ordinal"], "tabicl")
    assert neural.fit_transform(train).ordinal.tolist() == ["0", "1", "0"]
    assert neural.transform(validation).ordinal.tolist() == ["9", "0"]
    assert train.equals(before[0]) and validation.equals(before[1])
    assert np.isnan(encoded.number.iloc[1]) and len(encoded) == len(train)


def test_device_preference_explicit_failures_and_cpu_lightgbm():
    probes = []
    def probe(device, family):
        probes.append((device, family))
        return device in {"mps", "cpu"}
    assert select_device("auto", "tabpfn", probe=probe) == "mps"
    assert probes == [("cuda", "tabpfn"), ("mps", "tabpfn")]
    probes.clear()
    with pytest.raises(DeviceUnavailableError):
        select_device("cuda", "tabpfn", probe=probe)
    assert probes == [("cuda", "tabpfn")]
    assert select_device("auto", "tabicl", probe=lambda device, family: device == "cpu") == "cpu"
    assert select_device("auto", "lightgbm", probe=lambda *args: pytest.fail("Must not import torch")) == "cpu"
    with pytest.raises(DeviceUnavailableError):
        select_device("mps", "lightgbm", probe=probe)
    with pytest.raises(ValueError):
        select_device("xpu", "tabicl", probe=probe)


@pytest.mark.parametrize("family", ["tabpfn", "tabicl", "lightgbm"])
def test_official_factory_dispatch_with_stub_modules(monkeypatch, family):
    calls = []
    class FakeClassifier:
        def __init__(self, **kwargs):
            calls.append(kwargs)
        @classmethod
        def create_default_for_version(cls, version, **kwargs):
            calls.append(version)
            return cls(**kwargs)
    module = ModuleType(family)
    setattr(module, {"tabpfn": "TabPFNClassifier", "tabicl": "TabICLClassifier", "lightgbm": "LGBMClassifier"}[family], FakeClassifier)
    monkeypatch.setitem(sys.modules, family, module)
    if family == "tabpfn":
        constants = ModuleType("tabpfn.constants")
        constants.ModelVersion = SimpleNamespace(V2_6="exact-v2.6-unit-sentinel")
        monkeypatch.setitem(sys.modules, "tabpfn.constants", constants)
    assert isinstance(official_estimator(family, {"random_state": 77}), FakeClassifier)
    assert calls[-1] == {"random_state": 77}
    if family == "tabpfn":
        assert calls[0] == "exact-v2.6-unit-sentinel"


def test_additive_result_schema_old_constructor_and_parquet(tmp_path):
    legacy = CandidateResult("unit", "s", "old", "1", "exact-old", "balanced_accuracy", "balanced_accuracy")
    legacy.fit_time, legacy.predict_time = 1.25, .5
    row = legacy.to_record()
    assert row["fit_time"] == row["fit_time_sec"] == 1.25
    assert row["predict_time"] == row["predict_time_sec"] == .5
    required = {"seed", "model_family", "variant", "checkpoint_version", "balanced_accuracy", "roc_auc", "log_loss",
        "model_init_time_sec", "fit_time_sec", "validation_time_sec", "tuning_time_sec", "predict_time_sec", "total_time_sec",
        "n_train", "n_validation", "n_test", "n_features", "n_classes", "device", "python_version", "package_versions", "error_type"}
    assert required <= {field.name for field in fields(CandidateResult)}
    assert row["schema_version"] == 2 and json.loads(row["package_versions"]) == {}
    path = tmp_path / "legacy.parquet"
    pd.DataFrame([row]).to_parquet(path)
    assert pd.read_parquet(path).test_metric.item() == "balanced_accuracy"


def test_new_records_keep_seed_in_keys_and_use_existing_artifact_writer(setup):
    result = run(LightGBMBaselineCandidate(**dependencies(setup)))
    row = result.to_record()
    assert json.loads(row["package_versions"])["lightgbm"] == "4.7.0"
    assert json.loads(row["provenance"])["join_keys"] == ["dataset_id", "split_id", "seed"]
    path = setup / "unit-results.parquet"
    upsert_table(path, pd.DataFrame([row]), list(CANDIDATE_RESULT_KEYS))
    second = {**row, "seed": 78}
    upsert_table(path, pd.DataFrame([second]), list(CANDIDATE_RESULT_KEYS))
    assert len(pd.read_parquet(path)) == 2


def test_exact_model_identity_does_not_depend_on_local_checkpoint_path(setup, tmp_path_factory):
    other = tmp_path_factory.mktemp("other-machine") / "project"
    shutil.copytree(setup, other)
    first = run(TabPFN26Candidate(**dependencies(setup)))
    second = run(TabPFN26Candidate(**dependencies(other)))
    assert first.status == second.status == "ok"
    assert first.provenance["checkpoint"]["path"] != second.provenance["checkpoint"]["path"]
    assert first.exact_model_id == second.exact_model_id


def test_shared_reader_boundary_uses_exact_positions_and_has_no_fallback(setup, monkeypatch):
    events, data = [], portions()
    X, y = pd.concat(data[::2]), pd.concat(data[1::2])
    dataset = SimpleNamespace(dataset_id="pmlb_unit", dataset_source="pmlb", split_id="rita", seed=77,
        fingerprint="a" * 64, categorical_columns=("category",), X=X, y=y)
    positions = [tuple(frame.index[::-1]) for frame in data[::2]]
    split = SharedSplit("pmlb_unit", "rita", 77, 88, *positions, "pmlb", "a" * 64, True, "b" * 64,
        dict(reader_dataframe_fingerprint=dataset_fingerprint(dataset), shared_manifest_sha256="c" * 64,
             row_order="raw_csv_data_rows_zero_based"))
    monkeypatch.setattr("meta_selector.datasets.splits.make_split", lambda *args, **kwargs: pytest.fail("No fallback"))
    runner = PilotCandidateRunner(**dependencies(setup, events=events))
    result = runner.run_shared("lightgbm", dataset, split)
    assert result.status == "ok", result.error_message
    fit = next(event for event in events if event[0] == "fit")
    assert tuple(fit[1].index) == positions[0]
    assert result.provenance["shared_split"]["fingerprint"] == dataset.fingerprint
    assert result.seed == 77 and result.split_id == "rita"
    assert len(runner.available_models()) == 3
    dataset.X.iloc[0, 0] = -99
    with pytest.raises(PilotContractError, match="modified"):
        runner.run_shared("lightgbm", dataset, split)
    with pytest.raises(ValueError, match="Unknown pilot"):
        runner.run("invented", *data)


@pytest.mark.parametrize("n_classes", [2, 3])
@pytest.mark.parametrize("metric", ["balanced_accuracy", "macro_f1", "roc_auc", "log_loss"])
def test_neutral_scoring_preserves_landmarker_import_and_values(n_classes, metric):
    from meta_selector.landmarkers.custom import classification_score as old_import
    assert old_import is classification_score
    events, data = [], portions(n_classes)
    model = FakeEstimator(events)
    # Scoring-only unit fixture; no estimator fit is necessary.
    model.classes_ = np.arange(n_classes)
    X, y = data[0], data[0].signal.to_numpy()
    probabilities = model.predict_proba(X)
    expected = classification_scores(y, model.predict(X), probabilities, model.classes_)[metric]
    assert classification_score(model, X, y, metric, model.classes_) == pytest.approx(expected)


@pytest.mark.parametrize("bad", ["shape", "nan", "negative", "sum", "missing_class", "unknown_prediction"])
def test_scoring_rejects_invalid_predictions(bad):
    y, predicted, probabilities = np.array([0, 1]), np.array([0, 1]), np.array([[.8, .2], [.2, .8]])
    if bad == "shape":
        probabilities = probabilities[:, 0]
    elif bad == "nan":
        probabilities[0, 0] = np.nan
    elif bad == "negative":
        probabilities[0] = [-.1, 1.1]
    elif bad == "sum":
        probabilities[0] = [.8, .8]
    elif bad == "missing_class":
        y[:] = 0
    else:
        predicted[0] = 9
    with pytest.raises(ValueError):
        classification_scores(y, predicted, probabilities, np.array([0, 1]))


def test_import_and_constructor_never_import_models_or_landmarkers():
    code = """
import sys
from meta_selector.candidates import PilotCandidateRunner
runner = PilotCandidateRunner()
assert len(runner.available_models()) == 3
for name in ('torch', 'tabpfn', 'tabicl', 'lightgbm', 'pymfe', 'meta_selector.landmarkers.custom'):
    assert name not in sys.modules, name
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=DEFAULT_PROJECT_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
