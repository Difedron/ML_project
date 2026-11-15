"""Stage 4 contracts. Every training call uses a fake estimator."""
import hashlib
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from meta_selector.artifacts import dataset_fingerprint
from meta_selector.candidates import CandidateResult, PilotCandidateRunner
from meta_selector.candidates import priortune
from meta_selector.datasets.shared_pilot import SharedSplit
from meta_selector.metrics import classification_scores
from meta_selector.paths import DEFAULT_PROJECT_ROOT


MODEL = "lightgbm_priortune_t_k25"
SPLIT_ID = "rita_holdout"
SEED = 77
PUBLISHED_PARAMS = dict(bagging_fraction=.8, bagging_freq=1, cat_l2=.1, cat_smooth=.2,
                        extra_trees=True, feature_fraction=.6, lambda_l1=.3, lambda_l2=.4,
                        learning_rate=.05, max_cat_to_onehot=5, min_data_in_leaf=4,
                        min_data_per_group=3, num_leaves=8)


@pytest.fixture
def portfolio(tmp_path, monkeypatch):
    # A test-only trust anchor, with the production identity fields and exact validation code.
    record = dict(model="lgbm", objective="T", K=25, config_ids=list(range(25)),
                  add_order=list(reversed(range(25))), configs=[
                      dict(config_id=i, model_seed=100 + i, lightgbm=PUBLISHED_PARAMS.copy(),
                           tabprep=None, tabprep_enabled=False) for i in range(25)])
    spec = {**priortune.PORTFOLIO_PIN, "cache_path": ".cache/test-portfolio.json"}
    path = tmp_path / spec["cache_path"]
    path.parent.mkdir()

    def write():
        payload = json.dumps(record).encode()
        path.write_bytes(payload)
        spec["sha256"] = hashlib.sha256(payload).hexdigest()
        monkeypatch.setattr(priortune, "PORTFOLIO_PIN", {key: spec[key] for key in priortune.PORTFOLIO_PIN})
        resources_path = tmp_path / "config/candidates/preflight.yaml"
        resources_path.parent.mkdir(parents=True, exist_ok=True)
        resources = yaml.safe_load((DEFAULT_PROJECT_ROOT / "config/candidates/preflight.yaml").read_text())
        resources["portfolio"] = spec
        resources_path.write_text(yaml.safe_dump(resources))

    for name in ("config/candidates/pilot.yaml", "config/candidates/lightgbm_baseline.yaml", "config/project.yaml"):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((DEFAULT_PROJECT_ROOT / name).read_bytes())
    write()
    return SimpleNamespace(root=tmp_path, path=path, spec=spec, record=record, write=write)


def test_actual_pinned_portfolio_is_valid_if_cached():
    spec = yaml.safe_load((DEFAULT_PROJECT_ROOT / "config/candidates/preflight.yaml").read_text())["portfolio"]
    assert priortune.PORTFOLIO_PIN["repo_id"] == "mrsergazinov/PriorTune"
    assert priortune.PORTFOLIO_PIN["revision"] == "8b9cab4c0d6569ddda5cc2a18d78e0de75c526ee"
    assert priortune.PORTFOLIO_PIN["path"] == "portfolios/lgbm/T_K25.json"
    assert priortune.PORTFOLIO_PIN["sha256"] == "322a4e87966223ec268cdf390ba8d7a58e3b13d2a2a4a7f4dddd3989ab2688e5"
    path = DEFAULT_PROJECT_ROOT / spec["cache_path"]
    if not path.is_file():
        pytest.skip("Local pinned resource is optional for unit tests")
    configurations = priortune.load_portfolio(path, spec)
    assert len(configurations) == len({config.config_id for config in configurations}) == 25
    assert configurations[0].config_id == 1452
    assert configurations[-1].config_id == 410


@pytest.mark.parametrize("field,replacement", [
    ("repo_id", "some/Other"), ("revision", "main"), ("path", "portfolios/lgbm/TE_K25.json"),
    ("sha256", "0" * 64), ("model", "lgbm-prep"), ("rule", "TE"), ("k", 2),
    ("tabprep", True), ("ensemble", True),
])
def test_loader_rejects_alternative_identity_before_reading(portfolio, field, replacement):
    with pytest.raises(priortune.PortfolioError, match=rf"identity mismatch for {field}"):
        priortune.load_portfolio(portfolio.path.with_suffix(".missing"), {**portfolio.spec, field: replacement})


def test_loader_rejects_missing_or_corrupt_bytes(portfolio):
    with pytest.raises(priortune.PortfolioError, match="unavailable"):
        priortune.load_portfolio(portfolio.path.with_suffix(".missing"), portfolio.spec)
    portfolio.path.write_bytes(portfolio.path.read_bytes() + b"\n")
    with pytest.raises(priortune.PortfolioError, match="SHA256 mismatch"):
        priortune.load_portfolio(portfolio.path, portfolio.spec)


@pytest.mark.parametrize("mutation,message", [
    (lambda d: d.update(K=24), "lgbm/T/K=25"),
    (lambda d: d.update(objective="TE"), "lgbm/T/K=25"),
    (lambda d: d["configs"].pop(), "exactly 25"),
    (lambda d: d["configs"].append(d["configs"][0]), "exactly 25"),
    (lambda d: d["config_ids"].__setitem__(1, 0), "unique config IDs"),
    (lambda d: d["add_order"].__setitem__(0, 99), "unique config IDs"),
    (lambda d: d["configs"][1].update(config_id=0), "duplicate"),
    (lambda d: d["configs"][1].update(config_id=99), "Unexpected or duplicate"),
    (lambda d: d["configs"][0].update(tabprep_enabled=True), "Invalid plain LightGBM"),
    (lambda d: d["configs"][0].update(tabprep={}), "Invalid plain LightGBM"),
    (lambda d: d["configs"][0].update(model_seed=True), "Invalid plain LightGBM"),
    (lambda d: d["configs"][0]["lightgbm"].update(device="gpu"), "Invalid plain LightGBM"),
    (lambda d: d["configs"][0]["lightgbm"].pop("lambda_l1"), "Invalid plain LightGBM"),
    (lambda d: d["configs"][0]["lightgbm"].update(feature_fraction=float("nan")), "Invalid plain LightGBM"),
    (lambda d: d["configs"][0].update(unexpected=True), "Unexpected fields"),
])
def test_damaged_portfolio_detected_even_with_matching_fixture_sha(portfolio, mutation, message):
    mutation(portfolio.record)
    portfolio.write()
    with pytest.raises(priortune.PortfolioError, match=message):
        priortune.load_portfolio(portfolio.path, portfolio.spec)


def data_parts(n_classes):
    label_order = [1, 0, *range(2, n_classes)] * 2
    size = len(label_order)
    frame = pd.DataFrame({"signal": label_order * 3,
                          "stalk-root": pd.array([0, 1] * n_classes + [2] * (size * 2), dtype="Int64"),
                          "safety": [0, 1] * (size * 3 // 2),
                          "numeric": [1, np.nan] * (size * 3 // 2)})
    targets = pd.Series([f"label_{label}" for label in frame.signal], name="target")
    return [value.iloc[start:start + size] for start in range(0, len(frame), size) for value in (frame, targets)]


def fake_options(portfolio, events, *, fail_ids=(), fail_validation_ids=(), ties=False, flip_test=False):
    class FakeLightGBM:
        def __init__(self, params):
            self.config_id = params["random_state"] - 100
            self.params = params

        def fit(self, X, y, **kwargs):
            self.classes_ = np.unique(y)[::-1]
            self.validation_indices = tuple(kwargs["eval_set"][0][0].index)
            events.append(("fit", self.config_id, X, y, kwargs))
            if self.config_id in fail_ids:
                raise RuntimeError("synthetic fit error")

        def predict(self, X):
            pytest.fail("Selection and final scoring should use one probability snapshot")

        def predict_proba(self, X):
            stage = "validation" if tuple(X.index) == self.validation_indices else "test"
            events.append((stage, self.config_id, X))
            n_classes = len(self.classes_)
            actual = X.signal.to_numpy(dtype=int)
            if stage == "validation" and self.config_id in fail_validation_ids:
                return np.full((len(X), n_classes), np.nan)
            if n_classes == 2 and stage == "validation":
                if ties or self.config_id == 2:
                    positive = np.where(actual == 1, .4, .1)  # AUC 1, BA .5.
                elif self.config_id == 10:
                    positive = np.array([.9, .3, .55, .6])  # AUC .75, BA .75.
                else:
                    positive = np.where(actual == 1, .1, .9)
                scores = np.column_stack([1 - positive, positive])
            else:
                if stage == "test":
                    favored_id = 10 if flip_test else 2
                    confidence = .9 if self.config_id == favored_id else .7
                else:
                    confidence = .8 if ties else (.55 if self.config_id == 2 else (.95 if self.config_id == 10 else .4))
                scores = np.full((len(X), n_classes), (1 - confidence) / (n_classes - 1))
                scores[np.arange(len(X)), actual] = confidence
            return scores[:, self.classes_]

    def factory(family, params):
        assert family == "lightgbm"
        return FakeLightGBM(params)

    ticks = itertools.count()
    versions = {"lightgbm": "4.7.0", "tabpfn": "9.2.0", "tabicl": "2.2.0", "torch": None}
    return dict(project_root=portfolio.root, estimator_factory=factory, version_probe=versions.__getitem__,
                callback_factory=lambda **options: options, clock=lambda: next(ticks) / 10)


def run(runner, n_classes=2, **context):
    return runner.run(MODEL, *data_parts(n_classes), dataset_id="pmlb_fake", split_id=SPLIT_ID,
                      seed=SEED, categorical_columns=("stalk-root", "safety"), **context)


@pytest.mark.parametrize("n_classes,metric,direction,winner", [(2, "roc_auc", "maximize", 2), (3, "log_loss", "minimize", 10)])
def test_full_candidate_selects_by_validation_and_predicts_test_only_once(portfolio, n_classes, metric, direction, winner):
    events = []
    runner = PilotCandidateRunner(**fake_options(portfolio, events))
    result = run(runner, n_classes, metric="balanced_accuracy")
    assert isinstance(result, CandidateResult) and result.schema_version == 2
    assert result.status == "ok", result.error_message
    assert (result.model_family, result.variant, result.device) == ("lightgbm", "priortune_t_k25", "cpu")
    assert (result.validation_metric, result.test_metric) == (metric, "balanced_accuracy")
    assert result.test_score == result.balanced_accuracy == result.roc_auc == 1
    assert result.log_loss == pytest.approx(-np.log(.9 if winner == 2 else .7))
    assert len(result.validation_trials) == len({trial["config_id"] for trial in result.validation_trials}) == 25
    provenance = result.provenance["priortune"]
    assert provenance["selected_config_id"] == winner
    assert provenance["selected_validation_metric"] == metric
    assert provenance["selected_validation_score"] == pytest.approx(1 if n_classes == 2 else -np.log(.95))
    assert provenance["selection_direction"] == direction
    assert provenance["tie_break_rule"] == "smallest_config_id"
    assert provenance["successful_trials"] == provenance["number_of_configs"] == 25
    assert provenance["failed_trials"] == 0 and provenance["portfolio"]["sha256"] == portfolio.spec["sha256"]
    assert [e[1] for e in events if e[0] == "fit"] == portfolio.record["add_order"]
    assert [e[1] for e in events if e[0] == "validation"] == portfolio.record["add_order"]
    assert [e[1] for e in events if e[0] == "test"] == [winner] and events[-1][:2] == ("test", winner)
    data = data_parts(n_classes)
    by_id = {trial["config_id"]: trial for trial in result.validation_trials}
    for _, config_id, X, y, options in (event for event in events if event[0] == "fit"):
        assert X.index.equals(data[0].index) and list(y) == data[0].signal.tolist()
        assert options["eval_set"][0][0].index.equals(data[2].index)
        assert list(options["eval_set"][0][1]) == data[2].signal.tolist()
        assert options["eval_set"][0][0]["stalk-root"].isna().all()
        assert list(X["stalk-root"].cat.categories) == ["0", "1"]
        assert list(X.safety.cat.categories) == ["0", "1"] and X.numeric.isna().any()
        assert options["categorical_feature"] == ["stalk-root", "safety"]
        assert options["eval_metric"] == ("auc" if n_classes == 2 else "multi_logloss")
        assert options["callbacks"][0]["stopping_rounds"] == 50
        trial = by_id[config_id]
        assert not any("test" in key for key in trial)
        params = trial["effective_parameters"]
        assert params["random_state"] == 100 + config_id
        assert all(params[name] == SEED for name in ("data_random_seed", "bagging_seed", "feature_fraction_seed", "extra_seed"))
        assert (params["subsample"], params["subsample_freq"], params["colsample_bytree"],
                params["reg_alpha"], params["reg_lambda"], params["min_child_samples"]) == (.8, 1, .6, .3, .4, 4)
        assert params["device_type"] == "cpu" and params["n_jobs"] == 1 and params["deterministic"]
        assert params.get("num_class") == (n_classes if n_classes > 2 else None)
        assert params["objective"] == ("binary" if n_classes == 2 else "multiclass")
        assert params["learning_rate"] == .05 and params["extra_trees"] is True
    assert events[-1][2]["stalk-root"].isna().all()
    winning_trial = by_id[winner]
    assert provenance["effective_parameters"] == winning_trial["effective_parameters"]
    assert result.fit_time_sec == result.fit_time == winning_trial["fit_time_sec"]
    assert result.validation_time_sec == winning_trial["validation_time_sec"]
    assert result.model_init_time_sec == winning_trial["model_init_time_sec"]
    assert result.predict_time_sec == result.predict_time > 0
    assert result.tuning_time_sec >= sum(trial["total_trial_time_sec"] for trial in by_id.values())
    assert result.total_time_sec >= result.tuning_time_sec + result.predict_time_sec
    assert "validation_trials" not in result.to_record()
    assert json.loads(result.to_record()["provenance"])["priortune"]["selected_config_id"] == winner


def test_balanced_accuracy_would_choose_differently(portfolio):
    truth = np.array([1, 0, 1, 0])
    winner_positive = np.array([.4, .1, .4, .1])
    other_positive = np.array([.9, .3, .55, .6])
    def scores(positive):
        return classification_scores(truth, positive >= .5, np.column_stack([1 - positive, positive]), np.arange(2))
    assert scores(winner_positive)["balanced_accuracy"] < scores(other_positive)["balanced_accuracy"]
    assert scores(winner_positive)["roc_auc"] > scores(other_positive)["roc_auc"]
    result = run(PilotCandidateRunner(**fake_options(portfolio, [])), metric="balanced_accuracy")
    assert result.provenance["priortune"]["selected_config_id"] == 2


@pytest.mark.parametrize("n_classes", [2, 3])
def test_tie_uses_smaller_id_even_when_smaller_id_is_last(portfolio, n_classes):
    result = run(PilotCandidateRunner(**fake_options(portfolio, [], ties=True)), n_classes)
    assert result.status == "ok", result.error_message
    assert result.provenance["priortune"]["selected_config_id"] == 0
    assert result.validation_trials[-1]["config_id"] == 0


def test_final_test_change_does_not_change_selection_score_or_identity(portfolio):
    normal = run(PilotCandidateRunner(**fake_options(portfolio, [])))
    changed = run(PilotCandidateRunner(**fake_options(portfolio, [], flip_test=True)))
    assert normal.status == changed.status == "ok"
    assert normal.provenance["priortune"] == changed.provenance["priortune"]
    assert normal.exact_model_id == changed.exact_model_id
    assert normal.log_loss != changed.log_loss


@pytest.mark.parametrize("fit_failures,score_failures", [({9}, {7}), (set(range(25)), set())])
def test_incomplete_portfolio_records_failures_and_does_not_touch_test(portfolio, fit_failures, score_failures):
    events = []
    runner = PilotCandidateRunner(**fake_options(portfolio, events, fail_ids=fit_failures,
                                                  fail_validation_ids=score_failures))
    result = run(runner, _load_test=lambda: pytest.fail("Incomplete portfolio must not read test"))
    assert result.status == "error" and result.error_type == "IncompletePortfolioError"
    provenance = result.provenance["priortune"]
    assert provenance["failed_trials"] == len(fit_failures | score_failures)
    assert provenance["successful_trials"] == 25 - provenance["failed_trials"]
    assert provenance["selected_config_id"] is None and np.isnan(result.test_score)
    assert len(result.validation_trials) == 25 and not any(e[0] == "test" for e in events)
    assert {trial["config_id"] for trial in result.validation_trials if trial["status"] == "error"} == fit_failures | score_failures
    assert all(trial["error_type"] and trial["error_message"] and trial["validation_score"] is None
               for trial in result.validation_trials if trial["status"] == "error")


def test_shared_pilot_defers_test_slice_and_upserts_validation_artifact(portfolio, monkeypatch):
    data = data_parts(2)
    dataset = SimpleNamespace(dataset_id="pmlb_fake", dataset_source="pmlb", split_id=SPLIT_ID, seed=SEED,
                              fingerprint="a" * 64, categorical_columns=("stalk-root", "safety"),
                              X=pd.concat(data[::2]), y=pd.concat(data[1::2]))
    positions = [tuple(frame.index[::-1]) for frame in data[::2]]
    split = SharedSplit("pmlb_fake", SPLIT_ID, SEED, 88, *positions, "pmlb", "a" * 64, True, "b" * 64,
        dict(reader_dataframe_fingerprint=dataset_fingerprint(dataset), shared_manifest_sha256="c" * 64,
             row_order="raw_csv_data_rows_zero_based"))
    events = []
    portion = SharedSplit.portion
    def checked_portion(self, dataset, part):
        events.append(("slice", part))
        if part == "test":
            assert sum(e[0] == "validation" for e in events) == 25
        return portion(self, dataset, part)
    monkeypatch.setattr(SharedSplit, "portion", checked_portion)
    monkeypatch.setattr("meta_selector.datasets.splits.make_split", lambda *args, **kwargs: pytest.fail("No split regeneration"))
    failed = PilotCandidateRunner(**fake_options(portfolio, events, fail_ids={9}))
    first = failed.run_shared(MODEL, dataset, split)
    assert first.status == "error" and ("slice", "test") not in events
    path = Path(first.provenance["validation_trials_artifact"])
    assert path == portfolio.root / "data/derived/priortune_validation/trials.parquet"
    table = pd.read_parquet(path)
    assert table.loc[table.config_id == 9, "status"].item() == "error"
    events.clear()
    ready = PilotCandidateRunner(**fake_options(portfolio, events))
    second = ready.run_shared(MODEL, dataset, split)
    assert second.status == "ok", second.error_message
    assert second.provenance["join_key"] == [dataset.dataset_id, SPLIT_ID, SEED]
    assert second.provenance["shared_split"]["fingerprint"] == dataset.fingerprint
    assert tuple(next(e for e in events if e[0] == "fit")[2].index) == positions[0]
    assert tuple(events[-1][2].index) == positions[2]
    table = pd.read_parquet(path)
    assert len(table) == 25 and not table.duplicated(list(priortune.TRIAL_KEYS)).any()
    assert table.status.eq("ok").all() and not any("test" in column for column in table)
    assert json.loads(table.loc[table.config_id == 9, "effective_parameters"].item())["random_state"] == 109


def test_baseline_does_not_use_portfolio_and_keeps_its_configuration(portfolio, monkeypatch):
    monkeypatch.setattr(priortune, "load_portfolio", lambda *args: pytest.fail("Baseline must not load a portfolio"))
    params_seen = []

    class FakeBaseline:
        classes_ = np.array([0, 1])

        def fit(self, X, y, **options):
            assert list(X["stalk-root"].cat.categories) == [0, 1]
            assert len(options["eval_set"]) == 1 and options["eval_metric"] == "auc"

        def predict(self, X):
            return X.signal.to_numpy(dtype=int)

        def predict_proba(self, X):
            return np.eye(2)[self.predict(X)]

    def factory(family, params):
        assert family == "lightgbm"
        params_seen.append(params)
        return FakeBaseline()

    options = fake_options(portfolio, [])
    options["estimator_factory"] = factory
    result = PilotCandidateRunner(**options).run("lightgbm", *data_parts(2), dataset_id="pmlb_fake",
        split_id=SPLIT_ID, seed=SEED, categorical_columns=("stalk-root", "safety"))
    assert result.status == "ok", result.error_message
    assert result.variant == "baseline" and result.validation_trials == [] and result.tuning_time_sec == 0
    assert params_seen[0]["random_state"] == SEED and params_seen[0]["num_leaves"] == 31
    assert params_seen[0]["learning_rate"] == .05 and params_seen[0]["n_estimators"] == 1000
    assert "priortune" not in result.provenance
