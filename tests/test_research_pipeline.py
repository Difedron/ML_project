"""Small offline integration tests; no candidate benchmark or foundation-model fitting."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import sys
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer

from meta_selector import load_dataset
from meta_selector.artifacts import (DEFAULT_PROJECT_ROOT, IDENTITY, research_config, research_paths,
                                     upsert_table)
from meta_selector.batch import local_ids, process_local
from meta_selector.candidates.runner import TabArenaCandidateRunner
from meta_selector.candidates.results import to_gap_records
from meta_selector.datasets.registry import read_registry, rebuild_registry, register_dataset
from meta_selector.datasets.splits import make_split, portion, check_manifest
from meta_selector.evaluation.experiments import EXPERIMENT_SPECS, prepare_experiment
from meta_selector.landmarkers.custom import MODEL_CLASSES, run_custom_landmarkers
from meta_selector.landmarkers.preprocessing import make_preprocessor
from meta_selector.meta_dataset import build_meta_dataset
from meta_selector.meta_features.extraction import extract_controls, extract_pymfe
from meta_selector.meta_model.lodo import evaluate_lodo
from meta_selector.meta_model.tabpfn_predictor import TabPFN26Predictor, ExactVersionUnavailable
from meta_selector.performance_gap.comparisons import COMPARISONS, restrict_coverage
from meta_selector.performance_gap.families import MLP, TREE, NON_TFM, TFM
from meta_selector.performance_gap.gap import calculate_dataset_gaps
from meta_selector.tabarena import TabArenaAdapter
from meta_selector.tabarena.model_mapping import MODELS, identify_model


@pytest.fixture
def project(tmp_path):
    (tmp_path / "config").mkdir()
    for name in ("project.yaml", "datasets.yaml", "research.yaml", "candidates.yaml"):
        shutil.copyfile(DEFAULT_PROJECT_ROOT / "config" / name, tmp_path / "config" / name)
    return tmp_path


def write_dataset(root, dataset_id="example", *, payload=True, frame=None):
    directory = root / "data/raw" / dataset_id
    directory.mkdir(parents=True)
    metadata = dict(dataset_id=dataset_id, name=dataset_id, task_type="classification", target="target",
                    source=dict(type="other", url="", external_id=None), file=dict(name="data.csv", format="csv"),
                    problem=dict(type="binary"), notes="Synthetic test fixture")
    (directory / "metadata.yaml").write_text(yaml.safe_dump(metadata))
    if payload:
        if frame is None:
            frame = pd.DataFrame({"numeric": np.arange(60, dtype=float), "category": ["a", "b", "c"] * 20,
                                  "target": [0, 1] * 30})
        frame.to_csv(directory / "data.csv", index=False)
    return directory


def test_registry_rebuild_180_metadata_entries_without_large_payloads(project):
    for i in range(180):
        write_dataset(project, f"dataset_{i:03d}", payload=i < 2)
    reports = rebuild_registry(project_root=project)
    rows = read_registry(project_root=project)
    assert len(reports) == len(rows) == 180
    assert sum(row["status"] == "valid" for row in rows) == 2
    assert sum(row["status"] == "missing_file" for row in rows) == 178
    assert len(local_ids(project)) == 2
    before = (project / "data/registry/datasets.csv").read_bytes()
    rebuild_registry(project_root=project)
    assert (project / "data/registry/datasets.csv").read_bytes() == before


def test_exact_family_partition():
    assert len(MODELS) == 21
    assert [len(f.members) for f in (MLP, TREE, NON_TFM, TFM)] == [5, 7, 14, 7]
    assert not set(MLP.members) & set(TREE.members)
    assert not set(NON_TFM.members) & set(TFM.members)
    assert set(NON_TFM.members + TFM.members) == set(MODELS)
    assert set(MLP.members) == {"fastai", "modernnca", "nn_torch", "realmlp", "tabm"}
    assert set(TREE.members) == {"catboost", "lightgbm", "xgboost", "ebm", "perpetualboost", "extra_trees", "random_forest"}
    assert set(TFM.members) == {"tabdpt", "tabstar", "tabicl_v2", "tabicl_v1", "tabpfn_v2", "tabpfn_2_6", "realtabpfn_v2_5"}


@pytest.mark.parametrize("model_id", list(MODELS))
def test_all_21_canonical_mappings_and_catalog(model_id):
    model = MODELS[model_id]
    identity = identify_model(model.ta_name, model.config_type)
    assert identity.model_id == model_id and identity.version == model.version
    catalog = TabArenaAdapter().artifacts.values()
    assert any(item["model_id"] == model_id and item["ta_name"] == model.ta_name for item in catalog)
    assert identify_model(model.ta_name, "unknown").version is None


@pytest.mark.parametrize("comparison", list(COMPARISONS.values()), ids=list(COMPARISONS))
def test_six_comparisons_on_available_real_cache(comparison):
    adapter = TabArenaAdapter()
    ids = adapter.config["normalization_reference"]["artifact_ids"]
    if any(not adapter.path_for(name).is_file() for name in ids):
        pytest.skip("Optional offline integration: aggregate result cache not downloaded")
    reference = adapter.load(ids)
    reference = reference[reference.method_subtype.isin(["default", "tuned"])].copy()
    cohort, excluded = restrict_coverage(reference, comparison)
    result = calculate_dataset_gaps(cohort, comparison.family_a, comparison.family_b,
                                    reference=reference, reference_label="test-historical-reconstruction")
    expected = {"C1": 51, "C2": 51, "C3": 51, "C4": 51, "C5": 33, "C6": 36}
    assert len(result.dataset_gaps) == expected[comparison.paper_id]
    assert not result.dataset_gaps.duplicated(["dataset_source", "dataset_id"]).any()
    assert len(result.dataset_gaps) + len(excluded) == 51


def test_reproducible_stratified_splits_do_not_modify_raw(project):
    directory = write_dataset(project)
    before = hashlib.sha256((directory / "data.csv").read_bytes()).hexdigest()
    dataset = load_dataset("example", project_root=project)
    first = make_split(dataset, root=project)
    assert first == make_split(dataset, root=project)
    positions = first["indices"]
    assert [len(positions[key]) for key in ("train", "validation", "test")] == [36, 12, 12]
    assert not set(positions["train"]) & set(positions["test"])
    assert all(dataset.y.iloc[value].nunique() == 2 for value in positions.values())
    assert before == hashlib.sha256((directory / "data.csv").read_bytes()).hexdigest()
    changed = copy.deepcopy(first)
    changed["indices"]["test"][0] = positions["train"][0]
    with pytest.raises(ValueError, match="overlap"):
        check_manifest(changed, dataset)
    dataset.X.iloc[0, 0] += 1
    with pytest.raises(ValueError, match="differs"):
        make_split(dataset, root=project)


def test_split_rejects_missing_labels_without_removing_rows(project):
    frame = pd.DataFrame({"x": range(60), "target": [0, 1] * 30})
    frame["target"] = frame.target.astype(float)
    frame.loc[0, "target"] = np.nan
    write_dataset(project, frame=frame)
    dataset = load_dataset("example", project_root=project)
    with pytest.raises(ValueError, match="missing target"):
        make_split(dataset, root=project)
    assert len(dataset.X) == 60


def training_data():
    X = pd.DataFrame({"number": np.arange(36, dtype=float), "category": ["a", "b", None] * 12})
    X.loc[0, "number"] = np.nan
    return X, pd.Series([0, 1, 2] * 12)


def test_five_custom_models_fit_only_inner_train(monkeypatch):
    X, y = training_data()
    config = research_config()["custom_landmarkers"]
    seen = []
    real_fit = SimpleImputer.fit

    def fit_spy(self, data, *args, **kwargs):
        if isinstance(data, pd.DataFrame) and data.columns.tolist() == ["number"]:
            seen.append(set(data.index))
        return real_fit(self, data, *args, **kwargs)

    monkeypatch.setattr(SimpleImputer, "fit", fit_spy)
    result = run_custom_landmarkers(X, y, identity=dict(dataset_source="local", dataset_id="cv", split_id="outer"), config=config)
    assert set(result.model_id) == set(MODEL_CLASSES)
    assert result.status.eq("ok").all(), result.error_message.tolist()
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    expected = [set(X.index[train]) for train, _ in cv.split(X, y)]
    assert len(seen) == 15
    assert all(indices == expected[i % 3] for i, indices in enumerate(seen))
    assert result.score_mean.between(0, 1).all()
    assert result.fit_time_mean.ge(0).all()


@pytest.mark.parametrize("metric", ["macro_f1", "roc_auc", "log_loss"])
def test_configurable_landmarker_metric(metric):
    X, y = training_data()
    config = copy.deepcopy(research_config()["custom_landmarkers"])
    config.update(models=["decision_tree"], scoring=metric)
    result = run_custom_landmarkers(X, y, identity=dict(dataset_source="local", dataset_id="cv", split_id="s"), config=config)
    assert result.status.eq("ok").all(), result.error_message.tolist()
    assert result.metric.item() == metric


def test_unknown_categories_and_dense_guard():
    X, _ = training_data()
    preprocessing = make_preprocessor(X, scale=True)
    preprocessing.fit(X.iloc[:20])
    held = X.iloc[20:].copy()
    held["category"] = "previously_unseen"
    assert np.isfinite(preprocessing.transform(held)).all()
    with pytest.raises(ValueError, match="max_dense_cells"):
        make_preprocessor(X, max_dense_cells=1).fit_transform(X)


def test_configurable_pymfe_namespaces_and_versions():
    X, y = training_data()
    identity = dict(dataset_source="local", dataset_id="features", split_id="s")
    config = research_config()
    for source in ("paper_features", "pymfe_standard_landmarking"):
        frame = extract_pymfe(X, y, identity=identity, config=config[source], source=source, provenance={})
        assert set(frame.source) == {source}
        assert frame.extractor_version.eq("0.4.4").all()
        assert set(IDENTITY + ["name", "value", "group", "extractor_version", "provenance"]) <= set(frame)
    bad = dict(config["paper_features"], groups=["landmarking"])
    with pytest.raises(ValueError, match="separate"):
        extract_pymfe(X, y, identity=identity, config=bad, source="paper_features", provenance={})


def test_batch_continues_and_upserts_derived(project):
    write_dataset(project)
    register_dataset("example", project_root=project)
    result = process_local("landmarkers", ["absent", "example"], root=project)
    assert result.set_index("dataset_id").status.to_dict() == {"absent": "error", "example": "ok"}
    path = research_paths(project)["derived"] / "landmarkers/custom_landmarkers.parquet"
    assert len(pd.read_parquet(path)) == 5
    process_local("landmarkers", ["example"], root=project)
    assert len(pd.read_parquet(path)) == 5
    features = process_local("meta_features", ["example"], root=project)
    assert features.status.eq("ok").all(), features.error_message.tolist()
    table = pd.read_parquet(research_paths(project)["derived"] / "meta_features/meta_features.parquet")
    assert set(table.source) == {"controls", "paper_features"}
    assert (research_paths(project)["derived"] / "landmarkers/pymfe_landmarkers.parquet").is_file()


def test_atomic_derived_keys(project):
    path = project / "derived.parquet"
    table = pd.DataFrame([dict(dataset_id="a", value=1), dict(dataset_id="b", value=2)])
    upsert_table(path, table, ["dataset_id"])
    upsert_table(path, pd.DataFrame([dict(dataset_id="a", value=5)]), ["dataset_id"])
    assert pd.read_parquet(path).set_index("dataset_id").value.to_dict() == {"a": 5, "b": 2}
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Duplicate"):
        upsert_table(path, pd.concat([table, table]), ["dataset_id"])
    assert path.read_bytes() == before


def test_candidate_availability_has_no_substitutions():
    config = yaml.safe_load((DEFAULT_PROJECT_ROOT / "config/candidates.yaml").read_text())
    missing = TabArenaCandidateRunner(config, dependency_probe=lambda name: False, gpu_probe=lambda: False)
    assert missing.available_models() == []
    assert len([missing.availability(name) for name in MODELS]) == 21
    assert "checkpoint" in missing.availability("tabpfn_2_6")["reason"]
    generator = SimpleNamespace(manual_configs=[{}], model_cls=object)
    ready = TabArenaCandidateRunner(config, dependency_probe=lambda name: True, gpu_probe=lambda: True,
                                    resolver=lambda model_id: generator)
    assert ready.availability("catboost")["available"]
    assert not ready.availability("tabpfn_2_6")["available"]
    X, y = training_data()
    result = missing.run("catboost", X, y, X, y, X, y, dataset_id="not_run", split_id="s")
    assert result.status == "unavailable" and np.isnan(result.test_score)
    assert "environment" in result.provenance


def test_candidate_dependency_generation_cannot_be_silently_upgraded():
    config = yaml.safe_load((DEFAULT_PROJECT_ROOT / "config/candidates.yaml").read_text())
    runner = TabArenaCandidateRunner(config, dependency_probe=lambda name: True, gpu_probe=lambda: True,
                                     version_probe=lambda name: "1.3.1", resolver=lambda name: SimpleNamespace(manual_configs=[{}], model_cls=object))
    availability = runner.availability("tabdpt")
    assert not availability["available"]
    assert "unavailable_exact_version: requires tabdpt<1.2" in availability["reason"]


def feature_fixture():
    gaps = pd.DataFrame([dict(dataset_source=source, dataset_id="same_id", gap=gap)
                         for source, gap in [("local", .2), ("tabarena", -.1)]])
    rows = []
    for source, value in [("local", 1), ("tabarena", 100)]:
        for split in ["s0", "s1"]:
            identity = dict(dataset_source=source, dataset_id="same_id", split_id=split)
            rows.append(extract_controls(pd.DataFrame({"x": range(value + 2)}), identity=identity, provenance={}))
    return gaps, pd.concat(rows, ignore_index=True)


def test_meta_dataset_joins_by_keys_and_aggregates_splits():
    gaps, features = feature_fixture()
    result = build_meta_dataset(gaps.iloc[::-1], [features.sample(frac=1, random_state=7)], namespaces=["controls"])
    assert len(result) == 2
    assert result.set_index("dataset_source")["feature::controls::log_n"].to_dict() == {
        "local": np.log(3), "tabarena": np.log(102)}
    with pytest.raises(ValueError, match="Duplicate"):
        build_meta_dataset(gaps, [pd.concat([features, features])], namespaces=["controls"])
    changed = features.copy()
    changed.loc[0, "config_id"] = "different"
    with pytest.raises(ValueError, match="configuration"):
        build_meta_dataset(gaps, [changed], namespaces=["controls"])
    with pytest.raises(ValueError, match="no matching features"):
        build_meta_dataset(gaps, [features[features.dataset_source.eq("local")]], namespaces=["controls"])


def meta_fixture():
    return pd.DataFrame({"dataset_source": ["tabarena"] * 4, "dataset_id": ["a", "b", "c", "d"],
                         "gap": [-.2, .1, .3, -.1], "feature::controls::x": [0., 1., 2., 1000000.]})


def test_lodo_isolation_and_training_only_preprocessing():
    meta = meta_fixture()
    seen = []
    class SpyMean:
        def fit(self, X, y):
            seen.append((set(X.index), X.mean().to_numpy()))
            self.value = np.mean(y)
        def predict(self, X):
            assert set(X.index).isdisjoint(seen[-1][0])
            return np.repeat(self.value, len(X))
    predictions, summary = evaluate_lodo(meta, predictor_factory=SpyMean)
    assert len(predictions) == 4 and len(seen) == 4
    assert all(len(indices) == 3 for indices, _ in seen)
    assert all(np.allclose(mean, 0, atol=1e-10) for _, mean in seen)
    assert summary["mae"] == pytest.approx(summary["baseline_mae"])
    for _, row in predictions.iterrows():
        assert row.dataset_id not in [item["dataset_id"] for item in json.loads(row.provenance)["train_datasets"]]
    with pytest.raises(ValueError, match="one meta-observation"):
        evaluate_lodo(pd.concat([meta, meta.iloc[:1]]), predictor_factory=SpyMean)


def test_lodo_related_origin_groups_are_held_out_together():
    meta = meta_fixture()
    meta["origin_group"] = ["related", "related", "other", "third"]
    class Mean:
        def fit(self, X, y):
            self.value = np.mean(y)
        def predict(self, X):
            return np.repeat(self.value, len(X))
    predictions, _ = evaluate_lodo(meta, predictor_factory=Mean)
    assert predictions.set_index("dataset_id").loc["a", "n_train_datasets"] == 2
    assert predictions.set_index("dataset_id").loc["b", "n_train_datasets"] == 2


def test_lodo_zero_forecasts_have_explicit_selection_tie_behavior():
    meta = meta_fixture()
    class Zero:
        def fit(self, X, y):
            pass
        def predict(self, X):
            return np.zeros(len(X))
    _, summary = evaluate_lodo(meta, predictor_factory=Zero)
    assert summary["sign_accuracy"] == 0
    assert summary["selection_accuracy"] == .5  # zero chooses A, which wins two datasets
    assert summary["by_source"]["tabarena"]["n_datasets"] == 4


def test_exact_predictor_requires_verified_26_weights():
    config = research_config()["meta_predictor"]
    predictor = TabPFN26Predictor(config)
    assert predictor.availability()["status"] == "unavailable_exact_version"
    with pytest.raises(ExactVersionUnavailable):
        predictor.fit(pd.DataFrame({"x": [1, 2]}), np.array([.1, .2]))
    newer = TabPFN26Predictor(dict(config, model_version="3.5"))
    assert newer.availability()["status"] == "unavailable_exact_version"


def test_all_six_experiment_definitions_and_no_automatic_local_admission():
    assert set(EXPERIMENT_SPECS) == {f"E{i}" for i in range(6)}
    assert not EXPERIMENT_SPECS["E0"].include_local
    assert EXPERIMENT_SPECS["E5"].include_local and EXPERIMENT_SPECS["E5"].include_custom
    gaps, features = feature_fixture()
    with pytest.raises(ValueError, match="Missing requested feature namespaces"):
        prepare_experiment(EXPERIMENT_SPECS["E0"], gaps, [features])
    with pytest.raises(ValueError, match="requires explicitly benchmarked local"):
        prepare_experiment(EXPERIMENT_SPECS["E1"], gaps[gaps.dataset_source.eq("tabarena")], [features])


def test_official_candidate_bridge_fits_train_and_passes_validation(monkeypatch):
    seen = {}
    class Cleaner:
        @classmethod
        def construct(cls, problem_type, y):
            seen["labels_fit"] = set(y.index)
            return cls()
        def transform(self, y):
            return y
    class Features:
        def fit_transform(self, X):
            seen["features_fit"] = set(X.index)
            return X
        def transform(self, X):
            return X
    class OfficialModel:
        def __init__(self, *, problem_type, hyperparameters):
            seen["hyperparameters"] = hyperparameters
        def fit(self, *, X, y, X_val, y_val, **kwargs):
            seen["model_fit"] = set(X.index)
            seen["validation"] = set(X_val.index)
        def predict(self, *, X):
            return np.zeros(len(X), dtype=int)
    for name in ("autogluon", "autogluon.core", "autogluon.core.data", "autogluon.features", "autogluon.features.generators"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules["autogluon.core.data"].LabelCleaner = Cleaner
    sys.modules["autogluon.features.generators"].AutoMLPipelineFeatureGenerator = Features
    config = yaml.safe_load((DEFAULT_PROJECT_ROOT / "config/candidates.yaml").read_text())
    official = SimpleNamespace(model_cls=OfficialModel, manual_configs=[{"iterations": 10}])
    runner = TabArenaCandidateRunner(config, resolver=lambda name: official,
                                     dependency_probe=lambda name: True, gpu_probe=lambda: True)
    X = pd.DataFrame({"x": range(30)})
    y = pd.Series([0, 1] * 15)
    result = runner.run("catboost", X.iloc[:18], y.iloc[:18], X.iloc[18:24], y.iloc[18:24], X.iloc[24:], y.iloc[24:],
                        dataset_id="synthetic-adapter-test", split_id="s")
    assert result.status == "ok", result.error_message
    assert seen["features_fit"] == seen["labels_fit"] == seen["model_fit"] == set(range(18))
    assert seen["validation"] == set(range(18, 24))
    assert seen["hyperparameters"] == {"iterations": 10}
    record = pd.DataFrame([result.to_record()])
    canonical = to_gap_records(record)
    assert canonical.dataset_source.item() == "local"
    assert canonical.test_error.item() == pytest.approx(1 - result.test_score)
    record["status"] = "unavailable"
    with pytest.raises(ValueError, match="successful"):
        to_gap_records(record)


def test_exact_version_factory_dispatch_without_real_model_loading(tmp_path, monkeypatch):
    path = tmp_path / "tabpfn-v2.6-regressor-v2.6_default.ckpt"
    path.write_bytes(b"explicit fake checkpoint for SDK dispatch unit test only")
    calls = []
    class Fake26Config:
        pass
    class FakeRegressor:
        @classmethod
        def create_default_for_version(cls, version, **kwargs):
            calls.append((version, kwargs))
            return cls()
        def fit(self, X, y):
            self.mean = np.mean(y)
            self.configs_ = [Fake26Config()]
        def predict(self, X):
            return np.repeat(self.mean, len(X))
    module = ModuleType("tabpfn")
    module.TabPFNRegressor = FakeRegressor
    module.__spec__ = __import__("importlib").machinery.ModuleSpec("tabpfn", loader=None)
    constants = ModuleType("tabpfn.constants")
    constants.ModelVersion = SimpleNamespace(V2_6="exact-dispatch-v2.6")
    monkeypatch.setitem(sys.modules, "tabpfn", module)
    monkeypatch.setitem(sys.modules, "tabpfn.constants", constants)
    architecture = ModuleType("tabpfn.architectures.tabpfn_v2_6")
    architecture.TabPFNV2p6Config = Fake26Config
    monkeypatch.setitem(sys.modules, "tabpfn.architectures.tabpfn_v2_6", architecture)
    config = dict(research_config()["meta_predictor"], checkpoint=str(path), checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    predictor = TabPFN26Predictor(config)
    X = pd.DataFrame({"x": [1., 2., 3.]})
    predictor.fit(X, [.1, .2, .3])
    assert predictor.predict(X).tolist() == pytest.approx([.2, .2, .2])
    assert calls[0][0] == "exact-dispatch-v2.6"
    assert calls[0][1]["model_path"] == str(path)
    def wrong_architecture(self, X, y):
        self.configs_ = [object()]
    monkeypatch.setattr(FakeRegressor, "fit", wrong_architecture)
    with pytest.raises(ExactVersionUnavailable, match="architecture"):
        TabPFN26Predictor(config).fit(X, [.1, .2, .3])
    path.write_bytes(b"changed")
    assert TabPFN26Predictor(config).availability()["status"] == "unavailable_exact_version"
