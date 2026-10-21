"""Offline boundary tests; synthetic scores below are test fixtures, never research results."""
import hashlib
import io
from pathlib import Path

import pandas as pd
import pytest
import yaml

from meta_selector.tabarena.adapter import TabArenaAdapter, export_official_results
from meta_selector.tabarena.model_mapping import identify_model
from meta_selector.tabarena.results_loader import ResultsSchemaError, load_results, standardize_results


@pytest.fixture
def native():
    return pd.DataFrame([
        dict(dataset="demo", fold=0, method="TABICLV2 (default)", metric_error=.2,
             metric_error_val=.3, metric="roc_auc", problem_type="binary", method_subtype="default",
             config_type="TABICLV2", ta_name="TabICLv2", ta_suite="tabarena-2026-02-16"),
        dict(dataset="demo", fold=0, method="TABPFN-V2.6 (default)", metric_error=.1,
             metric_error_val=.4, metric="roc_auc", problem_type="binary", method_subtype="default",
             config_type="TABPFN-V2.6", ta_name="TabPFN-v2.6", ta_suite="tabarena-2026-03-18"),
    ])


def test_preserves_real_schema_errors_and_exact_versions(native, tmp_path):
    path = tmp_path / "results.parquet"
    native.to_parquet(path, index=False)
    result = load_results(path)
    assert set(result.model_id) == {"tabicl_v2", "tabpfn_2_6"}
    assert set(result.model_version) == {"v2", "2.6"}
    assert set(result.dataset_source) == {"tabarena"}
    assert result.loc[result.model_id.eq("tabpfn_2_6"), "test_error"].item() == .1
    assert "tabarena-2026-03-18::TabPFN-v2.6::TABPFN-V2.6 (default)" in result.exact_model_id.tolist()


@pytest.mark.parametrize("name,config,expected", [
    ("TabPFNv2_GPU", "TABPFNV2_GPU", "tabpfn_v2"),
    ("TabICL_GPU", "TABICL_GPU", "tabicl_v1"),
    ("TabPFN-v3", "TABPFN-V3", "TabPFN-v3"),
    ("TabPFN-v2.6", "WRONG", "TabPFN-v2.6"),
])
def test_no_fuzzy_version_substitution(name, config, expected):
    assert identify_model(name, config).model_id == expected


@pytest.mark.parametrize("column,value", [
    ("metric_error", float("nan")), ("metric_error_val", float("inf")),
    ("fold", -.1), ("fold", .5), ("dataset", ""), ("ta_name", None),
    ("problem_type", "unknown"),
])
def test_rejects_missing_or_invalid_results(native, column, value):
    native[column] = native[column].astype(object)
    native.loc[0, column] = value
    with pytest.raises((ResultsSchemaError, ValueError)):
        standardize_results(native)


def test_rejects_duplicate_exact_split(native):
    with pytest.raises(ResultsSchemaError, match="Duplicate"):
        standardize_results(pd.concat([native, native.iloc[:1]]))


def test_rejects_mixed_dataset_metrics(native):
    native.loc[1, "metric"] = "log_loss"
    with pytest.raises(ResultsSchemaError, match="mixes metrics"):
        standardize_results(native)


def test_csv_requires_explicit_imputation_status_and_identity(native, tmp_path):
    path = tmp_path / "results_per_split.csv"
    native.to_csv(path, index=False)
    with pytest.raises(ResultsSchemaError, match="imputed"):
        load_results(path)
    native["imputed"] = False
    identity_fields = ["ta_name", "ta_suite", "config_type", "method_subtype"]
    mapping = native.set_index("method")[identity_fields].to_dict("index")
    native.drop(columns=identity_fields).to_csv(path, index=False)
    with pytest.raises(ResultsSchemaError, match="identity_map"):
        load_results(path)
    assert len(load_results(path, identity_map=mapping)) == 2


@pytest.mark.parametrize("flag", [True, "true", 1, "unknown", None])
def test_never_uses_imputed_scores(native, flag):
    native["imputed"] = flag
    with pytest.raises(ResultsSchemaError, match="[Ii]mputed"):
        standardize_results(native)


def make_catalog(tmp_path, payload, **overrides):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config = dict(schema_version=1, cache_dir=".cache/results", max_artifact_bytes=100000,
                  tabarena_revision="pinned-test", pair={"a": "tabicl_v2"}, artifacts=[dict(
                      artifact_id="tabicl_v2", model_id="tabicl_v2", ta_name="TabICLv2",
                      ta_suite="tabarena-2026-02-16", md5=hashlib.md5(payload).hexdigest(),
                      url="https://data.tabarena.ai/test-results.parquet")])
    config.update(overrides)
    path = config_dir / "results.yaml"
    path.write_text(yaml.safe_dump(config))
    return TabArenaAdapter(path)


def test_pinned_cache_verification_and_metadata(native, tmp_path):
    payload = native.iloc[:1].to_parquet(index=False)
    adapter = make_catalog(tmp_path, payload)
    with pytest.raises(FileNotFoundError, match="--fetch"):
        adapter.load()
    path = adapter.path_for("tabicl_v2")
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    assert len(adapter.load()) == 1
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        adapter.load()


def test_fetch_checks_hash_and_writes_provenance(native, tmp_path, monkeypatch):
    payload = native.iloc[:1].to_parquet(index=False)
    adapter = make_catalog(tmp_path, payload)
    calls = []

    def open_result(request, timeout):
        calls.append(request.full_url)
        return io.BytesIO(payload)

    monkeypatch.setattr("urllib.request.urlopen", open_result)
    result = adapter.fetch("tabicl_v2")
    assert result.read_bytes() == payload
    assert result.with_suffix(".provenance.json").is_file()
    adapter.fetch("tabicl_v2")
    assert len(calls) == 1  # cached verification never downloads


def test_explicit_exclusion_policy_reports_unknown_and_imputed(native, tmp_path):
    first = native.iloc[:1].copy()
    unknown = first.assign(fold=1, imputed=None)
    imputed = first.assign(fold=2, imputed=True)
    keep = first.assign(imputed=False)
    payload = pd.concat([keep, unknown, imputed]).to_parquet(index=False)
    adapter = make_catalog(tmp_path, payload, imputation_policy="exclude_imputed_or_unknown")
    path = adapter.path_for("tabicl_v2")
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    frame = adapter.load()
    assert frame.split_id.tolist() == [0]
    assert {row["reason"] for row in frame.attrs["excluded_results"]} == {"unknown_imputation", "imputed"}
    assert sum(row["rows"] for row in frame.attrs["excluded_results"]) == 2


def test_adapter_rejects_inconsistent_exact_identity(native, tmp_path):
    payload = native.iloc[1:].to_parquet(index=False)
    adapter = make_catalog(tmp_path, payload)
    path = adapter.path_for("tabicl_v2")
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    with pytest.raises(ResultsSchemaError, match="identity disagrees"):
        adapter.load()


@pytest.mark.parametrize("reason", ["size", "checksum"])
def test_failed_fetch_leaves_no_partial_artifact(native, tmp_path, monkeypatch, reason):
    payload = native.iloc[:1].to_parquet(index=False)
    adapter = make_catalog(tmp_path, payload, max_artifact_bytes=1 if reason == "size" else 100000)
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: io.BytesIO(payload if reason == "size" else b"wrong"))
    with pytest.raises(ValueError):
        adapter.fetch("tabicl_v2")
    assert not adapter.path_for("tabicl_v2").exists()
    assert not list(adapter.cache_dir.glob("*.part"))


def test_official_export_disables_imputation(tmp_path):
    class Context:
        def compare(self, output_dir, *, plot, fillna):
            assert plot is False and fillna is None
            Path(output_dir, "results_per_split.csv").write_text("imputed\nFalse\n")

    assert export_official_results(Context(), tmp_path).is_file()
