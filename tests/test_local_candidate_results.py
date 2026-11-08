"""Synthetic 10-task/40-result fixtures only; no classifiers or meta-models run."""
import copy
import importlib
import itertools
import json
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from meta_selector import load_dataset
from meta_selector.artifacts import config_hash, dataset_fingerprint, research_config
from meta_selector.candidates.results import (CandidateResultsValidationError, load_pmlb_candidate_results,
                                              to_gap_records, validate_pmlb_candidate_results)
from meta_selector.candidates.runner import CandidateResult
from meta_selector.datasets.pmlb import sha256_file
from meta_selector.meta_features.extraction import extract_controls
from meta_selector.paths import DEFAULT_PROJECT_ROOT as ROOT
from meta_selector.performance_gap.comparisons import COMPARISONS, PMLB_COMPARISONS
from meta_selector.performance_gap.families import PMLB_PILOT_MODELS
from meta_selector.performance_gap.gap import calculate_dataset_gaps


@pytest.fixture
def pilot(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    cli = importlib.import_module("build_performance_gap")
    monkeypatch.setattr(cli, "TabArenaAdapter", lambda *a, **k: pytest.fail("No TabArena backend for local input"))
    (tmp_path / "config").mkdir()
    for name in ("project.yaml", "research.yaml", "datasets.yaml"):
        shutil.copyfile(ROOT / "config" / name, tmp_path / "config" / name)
    config = research_config(tmp_path)
    specs, scores, features, custom_rows = [], [], [], []
    for i in range(10):
        dataset_id = f"pmlb_fixture_{i:02d}"
        k = 2 if i < 6 else 3
        folder = tmp_path / "data/raw" / dataset_id
        folder.mkdir(parents=True)
        frame = pd.DataFrame(dict(code=np.tile([0, 1, 2], 20), measurement=np.arange(60.) + i,
                                  other=np.linspace(1, 2, 60), count=np.tile([1, 2], 30),
                                  target=np.tile(np.arange(k), 60 // k)))
        csv_path = folder / "fixture.csv"
        frame.to_csv(csv_path, index=False)
        csv_hash = sha256_file(csv_path)
        metadata = dict(dataset_id=dataset_id, name="Synthetic fixture", target="target", task_type="classification",
                        source=dict(type="other", url="", external_id="synthetic-only"),
                        file=dict(name="fixture.csv", format="csv", sha256=csv_hash),
                        problem=dict(type="binary" if k == 2 else "multiclass"), categorical_columns=["code"],
                        notes="Synthetic test fixture, not PMLB measurements or real candidate results")
        (folder / "metadata.yaml").write_text(yaml.safe_dump(metadata), encoding="utf-8")
        dataset = load_dataset(dataset_id, project_root=tmp_path)
        manifest = dict(dataset_source="local", dataset_id=dataset_id, split_id="holdout_0",
                        dataset_fingerprint=dataset_fingerprint(dataset), protocol=config["splits"],
                        protocol_hash=config_hash(config["splits"]),
                        indices=dict(train=list(range(36)), validation=list(range(36, 48)), test=list(range(48, 60))))
        manifest_path = tmp_path / "data/splits" / dataset_id / "holdout_0.json"
        manifest_path.parent.mkdir(parents=True)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        specs.append(dict(dataset_id=dataset_id, sha256=csv_hash, n_classes=k,
                          categorical_columns=["code"], origin_group=dataset_id))
        context = dict(problem_type=metadata["problem"]["type"], dataset_fingerprint=manifest["dataset_fingerprint"],
                       protocol_hash=manifest["protocol_hash"], csv_sha256=csv_hash,
                       split_manifest_sha256=sha256_file(manifest_path), is_final=True,
                       fixture="synthetic-only")
        for j, model in enumerate(PMLB_PILOT_MODELS):
            provenance = dict(context)
            if model == "lightgbm_priortune_k25":
                provenance["priortune"] = dict(k=25, fixture="synthetic-only")
            scores.append(CandidateResult(
                dataset_id=dataset_id, split_id="holdout_0", model_id=model, model_version=None,
                exact_model_id=f"synthetic::{model}::final", validation_metric="balanced_accuracy",
                test_metric="balanced_accuracy", validation_score=[.88, .78, .68, .83][j] + .001 * i,
                test_score=[.90, .80, .70, .85][j] + .001 * i, status="ok", fit_time=.1,
                predict_time=.01, provenance=provenance).to_record())
        identity = dict(dataset_source="local", dataset_id=dataset_id, split_id="holdout_0")
        feature_context = dict(dataset_fingerprint=manifest["dataset_fingerprint"], protocol_hash=manifest["protocol_hash"],
                               train_indices=manifest["indices"]["train"], input_role="outer_training_only")
        features.extend(extract_controls(dataset.X.iloc[:36], identity=identity,
                                         provenance=feature_context).to_dict("records"))
        # Deliberately fake feature values to test joins, never run PyMFE for these fixtures.
        for feature in range(202):
            value = np.nan if i == 0 and feature == 0 else float(i + feature)
            features.append(dict(**identity, source="paper_features", group="synthetic-fixture",
                                 name=f"synthetic_paper_{feature:03d}", value=value,
                                 config_id=config_hash(config["paper_features"]),
                                 extractor_version="synthetic-fixture", status="undefined" if np.isnan(value) else "ok",
                                 provenance=json.dumps(feature_context)))
        for model in config["custom_landmarkers"]["models"]:
            custom_rows.append(dict(**identity, source="custom_landmarking", model_id=model,
                                    metric="balanced_accuracy", config_id=config_hash(config["custom_landmarkers"]),
                                    status="ok", n_folds=3, score_mean=.5 + i * .01, score_std=.01,
                                    fit_time_mean=.01, fit_time_std=.001,
                                    provenance=json.dumps(dict(split=feature_context))))
    (tmp_path / "config/pmlb_pilot.yaml").write_text(yaml.safe_dump(dict(schema_version=1, datasets=specs)), encoding="utf-8")
    feature_path = tmp_path / "data/derived/meta_features/meta_features.parquet"
    feature_path.parent.mkdir(parents=True)
    pd.DataFrame(features).to_parquet(feature_path, index=False)
    custom_path = tmp_path / "data/derived/landmarkers/custom_landmarkers.parquet"
    custom_path.parent.mkdir(parents=True)
    pd.DataFrame(custom_rows).to_parquet(custom_path, index=False)
    return tmp_path, pd.DataFrame(scores), cli


def write_scores(root, scores, suffix="csv"):
    path = root / f"synthetic_candidate_results.{suffix}"
    if suffix == "csv":
        scores.to_csv(path, index=False)
    else:
        scores.to_parquet(path, index=False)
    return path


def data_snapshot(root):
    return {p.relative_to(root).as_posix(): (sha256_file(p), p.stat().st_mtime_ns)
            for directory in ("data", "config") for p in (root / directory).rglob("*") if p.is_file()}


@pytest.mark.parametrize("suffix", ["csv", "parquet"])
def test_import_and_existing_bridge_roundtrip(pilot, suffix):
    root, scores, _ = pilot
    path = write_scores(root, scores, suffix)
    before = data_snapshot(root), sha256_file(path)
    loaded = load_pmlb_candidate_results(path, project_root=root)
    assert len(loaded) == 40 and set(loaded.model_id) == set(PMLB_PILOT_MODELS)
    reference = to_gap_records(loaded)
    np.testing.assert_allclose(reference.test_error, 1 - scores.test_score)
    np.testing.assert_allclose(reference.validation_error, 1 - scores.validation_score)
    assert not reference.imputed.any()
    assert before == (data_snapshot(root), sha256_file(path))


def test_all_six_pairs_use_identical_four_final_reference(pilot):
    root, scores, _ = pilot
    reference = to_gap_records(validate_pmlb_candidate_results(scores, project_root=root))
    pairs = {frozenset(c.family_a.members + c.family_b.members) for c in PMLB_COMPARISONS.values()}
    assert pairs == {frozenset(p) for p in itertools.combinations(PMLB_PILOT_MODELS, 2)}
    assert len(COMPARISONS) == 6  # Historical Table-3 registry is unchanged.
    normalized = dict(tabpfn_2_6=0., tabicl_v2=1., lightgbm=1., lightgbm_priortune_k25=2 / 3)
    raw_scores = dict(zip(PMLB_PILOT_MODELS, [.90, .80, .70, .85]))
    reference_hashes = set()
    for comparison in PMLB_COMPARISONS.values():
        result = calculate_dataset_gaps(reference, comparison.family_a, comparison.family_b,
                                        reference=reference, reference_label="synthetic-four-final-fixture")
        a, b = comparison.family_a.members[0], comparison.family_b.members[0]
        assert len(result.dataset_gaps) == len(result.split_gaps) == 10
        assert result.dataset_gaps.n_reference_models_min.eq(4).all()
        assert result.split_gaps.n_reference_variants_a.eq(4).all()
        np.testing.assert_allclose(result.dataset_gaps.gap, normalized[a] - normalized[b], atol=1e-12)
        np.testing.assert_allclose(result.dataset_gaps.raw_error_gap, raw_scores[b] - raw_scores[a], atol=1e-12)
        reference_hashes.add(result.provenance["reference_sha256"])
    assert len(reference_hashes) == 1


@pytest.mark.parametrize("case", [
    "missing_model", "missing_dataset", "unknown_dataset", "unknown_model", "duplicate", "extra_trial",
    "same_exact_id", "wrong_source", "wrong_split", "metric_mismatch", "wrong_metric", "imputed", "error", "unavailable",
    "unknown_status", "ok_with_message", "missing_error_message", "nan_score", "inf_score", "negative_score",
    "percent_score", "boolean_score", "nullable_nan_score", "negative_time", "infinite_time", "invalid_json", "dict_provenance",
    "dataset_fingerprint", "protocol_hash", "csv_sha256", "split_manifest_sha256", "problem_type",
    "not_final", "final_string", "wrong_k", "string_k", "missing_k", "missing_column", "empty",
])
def test_reject_invalid_or_incomplete_cohorts(pilot, case):
    root, original, _ = pilot
    scores = original.copy(deep=True)
    if case == "missing_model":
        scores = scores.iloc[1:]
    elif case == "missing_dataset":
        scores = scores[scores.dataset_id.ne(scores.dataset_id.iloc[0])]
    elif case in {"unknown_dataset", "unknown_model", "wrong_source", "wrong_split"}:
        column, value = {"unknown_dataset": ("dataset_id", "pmlb_unknown"), "unknown_model": ("model_id", "trial_25"),
                         "wrong_source": ("dataset_source", "tabarena"), "wrong_split": ("split_id", "different_split")}[case]
        scores.loc[0, column] = value
    elif case in {"duplicate", "extra_trial"}:
        extra = scores.iloc[[3]].copy()
        if case == "extra_trial":
            extra["exact_model_id"] = "synthetic::prior::trial_02"
        scores = pd.concat([scores, extra], ignore_index=True)
    elif case == "same_exact_id":
        scores.loc[1, "exact_model_id"] = scores.loc[0, "exact_model_id"]
    elif case in {"wrong_metric", "metric_mismatch"}:
        scores.loc[0, "validation_metric"] = "log_loss"
        if case == "wrong_metric":
            scores.loc[0, "test_metric"] = "log_loss"
    elif case == "imputed":
        scores["imputed"] = True
    elif case in {"error", "unavailable", "unknown_status", "missing_error_message"}:
        scores.loc[0, "status"] = "pending" if case == "unknown_status" else "error" if case == "missing_error_message" else case
        scores.loc[0, "error_message"] = "" if case == "missing_error_message" else "Synthetic failure, not a real model run"
        scores.loc[0, ["validation_score", "test_score"]] = np.nan
    elif case == "ok_with_message":
        scores.loc[0, "error_message"] = "Synthetic error"
    elif case in {"nan_score", "inf_score", "negative_score", "percent_score", "boolean_score"}:
        value = {"nan_score": np.nan, "inf_score": np.inf, "negative_score": -.1,
                 "percent_score": 90., "boolean_score": True}[case]
        scores["test_score"] = scores.test_score.astype(object)
        scores.loc[0, "test_score"] = value
    elif case in {"negative_time", "infinite_time"}:
        scores.loc[0, "fit_time"] = -1 if case == "negative_time" else np.inf
    elif case == "nullable_nan_score":
        scores["test_score"] = scores.test_score.astype("Float64")
        scores.loc[0, "test_score"] = pd.NA
    elif case in {"invalid_json", "dict_provenance"}:
        scores.at[0, "provenance"] = "not json" if case == "invalid_json" else {"is_final": True}
    elif case == "missing_column":
        scores = scores.drop(columns="validation_score")
    elif case == "empty":
        scores = scores.iloc[:0]
    else:
        number = 3 if case in {"wrong_k", "string_k", "missing_k"} else 0
        context = json.loads(scores.loc[number, "provenance"])
        if case in {"dataset_fingerprint", "protocol_hash", "csv_sha256", "split_manifest_sha256"}:
            context[case] = "0" * 64
        elif case == "problem_type":
            context[case] = "multiclass"
        elif case in {"not_final", "final_string"}:
            context["is_final"] = False if case == "not_final" else "true"
        else:
            context["priortune"] = {} if case == "missing_k" else {"k": "25" if case == "string_k" else 24}
        scores.loc[number, "provenance"] = json.dumps(context)
    with pytest.raises(CandidateResultsValidationError) as caught:
        validate_pmlb_candidate_results(scores, project_root=root)
    assert caught.value.report["status"] == "rejected"


@pytest.mark.parametrize("case", ["missing", "overlap", "protocol"])
def test_invalid_existing_manifests_are_not_regenerated(pilot, case):
    root, scores, _ = pilot
    path = root / "data/splits/pmlb_fixture_00/holdout_0.json"
    if case == "missing":
        path.unlink()
    else:
        manifest = json.loads(path.read_text())
        if case == "overlap":
            manifest["indices"]["test"][0] = manifest["indices"]["train"][0]
        else:
            manifest["protocol"]["random_seed"] = 99
            manifest["protocol_hash"] = config_hash(manifest["protocol"])
        path.write_text(json.dumps(manifest), encoding="utf-8")
    before = data_snapshot(root)
    with pytest.raises(CandidateResultsValidationError):
        validate_pmlb_candidate_results(scores, project_root=root)
    assert data_snapshot(root) == before


def test_cli_all_pairs_and_existing_builder_without_training(pilot):
    root, scores, cli = pilot
    input_path = write_scores(root, scores, "parquet")
    before = data_snapshot(root), sha256_file(input_path)
    assert cli.main(["--local-scores", str(input_path), "--project-root", str(root), "--all", "--build-meta-dataset"]) == 0
    base = root / "results/tables/pmlb_pilot"
    for comparison in PMLB_COMPARISONS:
        directory = base / comparison
        gaps = pd.read_csv(directory / "dataset_gaps.csv")
        meta = pd.read_parquet(directory / "meta_dataset.parquet")
        assert len(gaps) == len(meta) == 10
        assert len([c for c in meta if c.startswith("feature::")]) == 217
        assert not meta.duplicated(["dataset_source", "dataset_id"]).any()
        assert meta["feature::paper_features::synthetic_paper_000"].isna().sum() == 1
        assert meta.origin_group.notna().all()
        assert gaps.n_reference_models_min.eq(4).all() and gaps.n_reference_models_max.eq(4).all()
    assert len(pd.read_parquet(base / "candidate_results.parquet")) == 40
    assert before == (data_snapshot(root), sha256_file(input_path))
    assert not (root / "data/derived/candidate_scores").exists()


def test_cli_rejected_status_keeps_diagnostics_but_writes_no_gaps(pilot):
    root, scores, cli = pilot
    scores.loc[0, "status"] = "error"
    scores.loc[0, "error_message"] = "Synthetic timeout"
    scores.loc[0, ["validation_score", "test_score"]] = np.nan
    path = write_scores(root, scores)
    assert cli.main(["--local-scores", str(path), "--project-root", str(root), "--all"]) == 1
    base = root / "results/tables/pmlb_pilot"
    report = json.loads((base / "candidate_import.json").read_text())
    assert report["status"] == "rejected" and report["rows"] == 40
    assert report["rejected_runs"][0]["error_message"] == "Synthetic timeout"
    assert not list(base.glob("*/dataset_gaps.csv")) and not (base / "candidate_results.parquet").exists()


@pytest.mark.parametrize("case", ["split", "fingerprint", "train_indices", "config", "custom_metric", "custom_missing",
                                  "custom_nullable_score", "custom_nullable_folds"])
def test_meta_join_rejects_incompatible_features_without_recomputing_them(pilot, case):
    root, scores, cli = pilot
    path = root / "data/derived" / ("landmarkers/custom_landmarkers.parquet" if case.startswith("custom")
                                    else "meta_features/meta_features.parquet")
    table = pd.read_parquet(path)
    if case == "split":
        table.loc[0, "split_id"] = "other"
    elif case in {"fingerprint", "train_indices"}:
        context = json.loads(table.loc[0, "provenance"])
        context["dataset_fingerprint" if case == "fingerprint" else "train_indices"] = "wrong" if case == "fingerprint" else [0]
        table.loc[0, "provenance"] = json.dumps(context)
    elif case == "config":
        table.loc[table.source.eq("paper_features"), "config_id"] = "different"
    elif case == "custom_metric":
        table.loc[0, "metric"] = "accuracy"
    elif case == "custom_nullable_score":
        table["score_mean"] = table.score_mean.astype("Float64")
        table.loc[0, "score_mean"] = pd.NA
    elif case == "custom_nullable_folds":
        table["n_folds"] = table.n_folds.astype("Int64")
        table.loc[0, "n_folds"] = pd.NA
    else:
        table = table.iloc[1:]
    table.to_parquet(path, index=False)
    before = data_snapshot(root)
    input_path = write_scores(root, scores)
    assert cli.main(["--local-scores", str(input_path), "--project-root", str(root), "--all", "--build-meta-dataset"]) == 1
    base = root / "results/tables/pmlb_pilot"
    assert not list(base.glob("*/dataset_gaps.csv"))
    assert data_snapshot(root) == before


def test_gap_only_does_not_read_features(pilot, monkeypatch):
    root, scores, cli = pilot
    monkeypatch.setattr(cli, "_local_features", lambda *a: pytest.fail("No feature input requested"))
    path = write_scores(root, scores)
    assert cli.main(["--local-scores", str(path), "--project-root", str(root),
                     "--comparison", "lightgbm_vs_lightgbm_priortune_k25"]) == 0
    base = root / "results/tables/pmlb_pilot"
    assert len(list(base.glob("*/dataset_gaps.csv"))) == 1
    assert not list(base.glob("*/meta_dataset.parquet"))


@pytest.mark.parametrize("arguments", [["--fetch"], ["--config", "unused.yaml"], ["--comparison", "mlp_vs_tree"]])
def test_cli_refuses_mixing_local_and_tabarena_protocols(pilot, arguments):
    root, scores, cli = pilot
    path = write_scores(root, scores)
    with pytest.raises(SystemExit) as exc:
        cli.main(["--local-scores", str(path), "--project-root", str(root), *arguments])
    assert exc.value.code == 2


def test_legacy_bridge_still_accepts_nonpilot_metrics():
    row = CandidateResult(dataset_id="other", split_id="s0", model_id="a", model_version=None,
                          exact_model_id="legacy::a", validation_metric="roc_auc", test_metric="roc_auc",
                          validation_score=.8, test_score=.9, status="ok", provenance={"problem_type": "binary"})
    assert to_gap_records(pd.DataFrame([row.to_record()])).test_error.item() == pytest.approx(.1)


def test_nullable_timings_do_not_require_fabricated_times(pilot):
    root, scores, _ = pilot
    scores[["fit_time", "predict_time"]] = np.nan
    loaded = validate_pmlb_candidate_results(scores, project_root=root)
    assert loaded[["fit_time", "predict_time"]].isna().all().all()


@pytest.mark.parametrize("case", ["duplicate_headers", "unsupported_extension"])
def test_file_schema_errors_have_structured_diagnostics(pilot, case):
    root, _, _ = pilot
    path = root / ("bad.csv" if case == "duplicate_headers" else "bad.json")
    path.write_text("status,status\nok,ok\n" if case == "duplicate_headers" else "[]", encoding="utf-8")
    with pytest.raises(CandidateResultsValidationError, match="Cannot read"):
        load_pmlb_candidate_results(path, project_root=root)


def test_duplicate_dataframe_columns_rejected(pilot):
    root, scores, _ = pilot
    duplicate = pd.concat([scores, scores[["status"]]], axis=1)
    with pytest.raises(CandidateResultsValidationError, match="Duplicate CandidateResult columns"):
        validate_pmlb_candidate_results(duplicate, project_root=root)


def test_historical_cli_all_keeps_original_six_definitions(pilot, monkeypatch):
    _, _, cli = pilot
    class SyntheticAdapter:
        config = dict(normalization_reference=dict(artifact_ids=["fixture"], method_subtypes=["default"]))
        def __init__(self, *args):
            pass
        def load(self, ids):
            return pd.DataFrame([dict(artifact_id="fixture", method_subtype="default")])
    captured = []
    monkeypatch.setattr(cli, "TabArenaAdapter", SyntheticAdapter)
    monkeypatch.setattr(cli, "build_one", lambda adapter, ref, spec, excluded, comparison, output, multiple:
                        captured.append((comparison.id, multiple)))
    assert cli.main(["--all"]) == 0
    assert captured == [(name, True) for name in COMPARISONS]


@pytest.mark.parametrize("accepted", [True, False])
def test_public_cli_subprocess_with_synthetic_input_only(pilot, accepted):
    root, scores, _ = pilot
    if not accepted:
        context = json.loads(scores.loc[0, "provenance"])
        context["split_manifest_sha256"] = "0" * 64
        scores.loc[0, "provenance"] = json.dumps(context)
    path = write_scores(root, scores)
    before = data_snapshot(root)
    completed = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/build_performance_gap.py"),
                                "--local-scores", str(path), "--project-root", str(root), "--all", "--build-meta-dataset"],
                               cwd=root, capture_output=True, text=True)
    assert completed.returncode == (0 if accepted else 1), completed.stdout + completed.stderr
    report = json.loads((root / "results/tables/pmlb_pilot/candidate_import.json").read_text())
    assert report["status"] == ("accepted" if accepted else "rejected")
    assert data_snapshot(root) == before
