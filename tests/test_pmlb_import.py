"""Import/split tests only: no extractor, preprocessing fit or estimator execution."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import StratifiedKFold

from meta_selector import load_dataset, validate_dataset
from meta_selector.artifacts import dataset_fingerprint, research_config
from meta_selector.datasets.pmlb import import_pmlb, sha256_file
from meta_selector.datasets.registry import read_registry
from meta_selector.datasets.splits import check_manifest, make_split
from meta_selector.paths import DEFAULT_PROJECT_ROOT as ROOT


def prepare_project(root, specs=None):
    (root / "config").mkdir(parents=True)
    (root / "data/PMLB").mkdir(parents=True)
    for name in ("project.yaml", "datasets.yaml", "research.yaml"):
        shutil.copyfile(ROOT / "config" / name, root / "config" / name)
    manifest = dict(schema_version=1, selection_report="docs/PMLB_PILOT_SELECTION.md",
                    metadata_revision="local-test-fixture", datasets=specs or [])
    return manifest


def save_manifest(root, manifest):
    (root / "config/pmlb_pilot.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")


@pytest.fixture
def project(tmp_path):
    manifest = prepare_project(tmp_path)
    frame = pd.DataFrame(dict(code=np.tile([0, 1, 2], 40),
                              sexlike=np.tile([0., 1.], 60),
                              quantity=np.linspace(1, 100, 120),
                              integer_measurement=np.tile([1, 2, 3], 40),
                              target=np.tile([0, 1], 60)))
    source = tmp_path / "data/PMLB/example.csv"
    frame.to_csv(source, index=False)
    manifest["datasets"] = [dict(original_name="example", dataset_id="pmlb_example",
                                 sha256=sha256_file(source), n_rows=120, n_features=4, n_classes=2,
                                 categorical_columns=["code", "sexlike"], origin_group="example",
                                 description_url="https://example.org/fixture")]
    save_manifest(tmp_path, manifest)
    return tmp_path


def snapshot(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for directory in ("data", "results")
            for p in (root / directory).rglob("*") if p.is_file()}


def test_explicit_categories_preserve_values_and_numeric_measurements(project):
    source = project / "data/PMLB/example.csv"
    before = source.read_bytes()
    import_pmlb(project_root=project)
    dataset = load_dataset("pmlb_example", project_root=project)
    original = pd.read_csv(source)
    assert dataset.metadata.categorical_columns == ("code", "sexlike")
    assert isinstance(dataset.X.code.dtype, pd.CategoricalDtype)
    assert isinstance(dataset.X.sexlike.dtype, pd.CategoricalDtype)
    assert dataset.X.code.eq(0).sum() == 40  # 0 is a category, never an inferred missing value.
    for column in dataset.X:
        if column in dataset.metadata.categorical_columns:
            assert dataset.X[column].astype(original[column].dtype).equals(original[column])
        else:
            assert dataset.X[column].equals(original[column])
            assert dataset.X[column].dtype == original[column].dtype
    assert is_numeric_dtype(dataset.X.integer_measurement)  # Three values do not imply category.
    pd.testing.assert_series_equal(dataset.y, original.target)
    report = validate_dataset(dataset.dataset_id, project_root=project)
    assert report.is_valid and report.n_categorical_features == report.n_numeric_features == 2
    assert source.read_bytes() == before
    assert (project / "data/raw/pmlb_example/example.csv").read_bytes() == before
    assert dataset.metadata.file.sha256 == sha256_file(source)


@pytest.mark.parametrize("declaration", [None, "code", 3, ["code", "code"], ["target"], [""], [1], ["absent"]])
def test_bad_categorical_metadata_is_rejected(project, declaration):
    import_pmlb(project_root=project)
    path = project / "data/raw/pmlb_example/metadata.yaml"
    metadata = yaml.safe_load(path.read_text(encoding="utf-8"))
    metadata["categorical_columns"] = declaration
    path.write_text(yaml.safe_dump(metadata), encoding="utf-8")
    report = validate_dataset("pmlb_example", project_root=project)
    assert report.status == "invalid_metadata"
    assert "categorical_columns" in " ".join(report.errors)


def test_old_metadata_without_categories_or_checksum_still_loads(project):
    import_pmlb(project_root=project)
    path = project / "data/raw/pmlb_example/metadata.yaml"
    metadata = yaml.safe_load(path.read_text(encoding="utf-8"))
    metadata.pop("categorical_columns")
    metadata["file"].pop("sha256")
    path.write_text(yaml.safe_dump(metadata), encoding="utf-8")
    dataset = load_dataset("pmlb_example", project_root=project)
    assert dataset.metadata.categorical_columns == ()
    assert all(is_numeric_dtype(t) for t in dataset.X.dtypes)


def test_declared_category_keeps_actual_nan_without_inventing_missing_codes(project):
    import_pmlb(project_root=project)
    path = project / "data/raw/pmlb_example/example.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "code"] = np.nan
    frame.to_csv(path, index=False)
    metadata_path = path.parent / "metadata.yaml"
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    metadata["file"]["sha256"] = sha256_file(path)
    metadata_path.write_text(yaml.safe_dump(metadata), encoding="utf-8")
    dataset = load_dataset("pmlb_example", project_root=project)
    assert dataset.X.code.isna().sum() == 1
    assert dataset.X.code.eq(0).sum() == 39


def test_repeat_import_reuses_payload_metadata_manifest_and_registry(project):
    first = import_pmlb(project_root=project, create_splits=True)
    before = snapshot(project)
    protected = list((project / "data/raw").rglob("*")) + list((project / "data/splits").rglob("*.json"))
    mtimes = {p: p.stat().st_mtime_ns for p in protected if p.is_file()}
    second = import_pmlb(project_root=project, create_splits=True)
    assert first == second
    assert snapshot(project) == before
    assert all(p.stat().st_mtime_ns == value for p, value in mtimes.items())
    assert len(read_registry(project_root=project)) == 1


@pytest.mark.parametrize("conflict", ["source", "payload", "metadata", "unrelated"])
def test_conflicting_files_are_preserved(project, conflict):
    if conflict != "source":
        import_pmlb(project_root=project, create_splits=True)
    folder = project / "data/raw/pmlb_example"
    if conflict == "source":
        with (project / "data/PMLB/example.csv").open("ab") as stream:
            stream.write(b"\n")
    elif conflict == "payload":
        with (folder / "example.csv").open("ab") as stream:
            stream.write(b"\n")
        assert validate_dataset("pmlb_example", project_root=project).status == "invalid"
    elif conflict == "metadata":
        path = folder / "metadata.yaml"
        metadata = yaml.safe_load(path.read_text(encoding="utf-8"))
        metadata["categorical_columns"] = []
        path.write_text(yaml.safe_dump(metadata), encoding="utf-8")
    else:
        (folder / "user-notes.txt").write_text("Preserve user work", encoding="utf-8")
    before = snapshot(project)
    with pytest.raises(ValueError):
        import_pmlb(project_root=project, create_splits=True)
    assert snapshot(project) == before


def test_all_sources_are_checked_before_first_copy(project):
    path = project / "config/pmlb_pilot.yaml"
    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    second = dict(manifest["datasets"][0], original_name="missing", dataset_id="pmlb_missing")
    manifest["datasets"].append(second)
    save_manifest(project, manifest)
    with pytest.raises(FileNotFoundError):
        import_pmlb(project_root=project)
    assert not (project / "data/raw").exists()


def test_matching_payload_only_can_resume_an_interrupted_import(project):
    folder = project / "data/raw/pmlb_example"
    folder.mkdir(parents=True)
    shutil.copyfile(project / "data/PMLB/example.csv", folder / "example.csv")
    before = (folder / "example.csv").stat().st_mtime_ns
    import_pmlb(project_root=project, create_splits=True)
    assert (folder / "example.csv").stat().st_mtime_ns == before
    assert validate_dataset("pmlb_example", project_root=project).is_valid


def test_reused_and_fresh_splits_are_identical_and_reject_changes(project, tmp_path):
    import_pmlb(project_root=project, create_splits=True)
    dataset = load_dataset("pmlb_example", project_root=project)
    path = project / "data/splits/pmlb_example/holdout_0.json"
    original_bytes = path.read_bytes()
    manifest = make_split(dataset, root=project)
    assert path.read_bytes() == original_bytes
    # A fresh output root has the same values/dtypes/order and the same split indices.
    fresh = tmp_path / "fresh"
    prepare_project(fresh)
    fresh_manifest = make_split(dataset, root=fresh)
    assert manifest == fresh_manifest
    assert manifest["dataset_fingerprint"] == dataset_fingerprint(dataset)
    parts = [set(manifest["indices"][name]) for name in ("train", "validation", "test")]
    assert not (parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2])
    assert set.union(*parts) == set(range(len(dataset.y)))
    assert all(set(dataset.y.iloc[list(p)]) == set(dataset.y) for p in parts)
    damaged = copy.deepcopy(manifest)
    damaged["indices"]["test"][0] = damaged["indices"]["train"][0]
    with pytest.raises(ValueError, match="overlap"):
        check_manifest(damaged, dataset)
    dataset.X.loc[dataset.X.index[0], "quantity"] += 1
    with pytest.raises(ValueError, match="differs"):
        make_split(dataset, root=project)
    assert path.read_bytes() == original_bytes


def test_protocol_change_does_not_overwrite_existing_manifest(project):
    import_pmlb(project_root=project, create_splits=True)
    path = project / "data/splits/pmlb_example/holdout_0.json"
    before = path.read_bytes()
    config_path = project / "config/research.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["splits"]["random_seed"] = 99
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="new split_id"):
        import_pmlb(project_root=project, create_splits=True)
    assert path.read_bytes() == before
    import_pmlb(project_root=project, create_splits=True, split_id="holdout_99")
    assert (path.parent / "holdout_99.json").is_file()


def test_cli_import_and_repeat_from_another_working_directory(project):
    command = [sys.executable, "-B", str(ROOT / "scripts/import_pmlb.py"),
               "--project-root", str(project), "--create-splits"]
    for _ in range(2):
        completed = subprocess.run(command, cwd=project, capture_output=True, text=True)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert "pmlb_example valid 120 4 2 2 72 24 24" in completed.stdout


def test_typing_change_invalidates_existing_split_even_with_equal_values(project):
    import_pmlb(project_root=project, create_splits=True)
    dataset = load_dataset("pmlb_example", project_root=project)
    original = dataset_fingerprint(dataset)
    dataset.X["code"] = dataset.X.code.astype("int64")
    assert dataset_fingerprint(dataset) != original
    with pytest.raises(ValueError, match="differs"):
        make_split(dataset, root=project)


@pytest.mark.parametrize("file_format", ["csv", "xlsx", "parquet"])
def test_explicit_categories_are_applied_for_each_supported_format(project, file_format):
    import_pmlb(project_root=project)
    directory = project / "data/raw/pmlb_example"
    frame = pd.read_csv(directory / "example.csv")
    path = directory / f"typed.{file_format}"
    if file_format == "csv":
        frame.to_csv(path, index=False)
    elif file_format == "xlsx":
        frame.to_excel(path, index=False)
    else:
        frame.to_parquet(path, index=False)
    metadata_path = directory / "metadata.yaml"
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    metadata["file"].update(name=path.name, format=file_format, sha256=sha256_file(path))
    metadata_path.write_text(yaml.safe_dump(metadata), encoding="utf-8")
    dataset = load_dataset("pmlb_example", project_root=project)
    assert set(dataset.X.select_dtypes("category")) == {"code", "sexlike"}
    assert is_numeric_dtype(dataset.X.quantity) and is_numeric_dtype(dataset.X.integer_measurement)
    assert dataset.X.code.eq(0).sum() == 40
    assert dataset.y.tolist() == frame.target.tolist()


@pytest.mark.parametrize("checksum", ["invalid", "A" * 64, 123])
def test_malformed_file_checksum_is_invalid_metadata(project, checksum):
    import_pmlb(project_root=project)
    path = project / "data/raw/pmlb_example/metadata.yaml"
    metadata = yaml.safe_load(path.read_text(encoding="utf-8"))
    metadata["file"]["sha256"] = checksum
    path.write_text(yaml.safe_dump(metadata), encoding="utf-8")
    report = validate_dataset("pmlb_example", project_root=project)
    assert report.status == "invalid_metadata" and "sha256" in " ".join(report.errors)


def test_real_approved_cohort_matches_report_types_and_all_inner_folds(tmp_path):
    """Integration on ten real local CSVs, only index generation (never model fitting)."""
    manifest = yaml.safe_load((ROOT / "config/pmlb_pilot.yaml").read_text(encoding="utf-8"))
    expected = {"saheart": 1, "tic_tac_toe": 9, "kr_vs_kp": 36, "mushroom": 22,
                "Hill_Valley_without_noise": 0, "twonorm": 0, "penguins": 2,
                "car_evaluation": 6, "vehicle": 0, "waveform_21": 0}
    assert {s["original_name"] for s in manifest["datasets"]} == set(expected)
    if any(not (ROOT / "data/PMLB" / f"{n}.csv").is_file() for n in expected):
        pytest.skip("Local PMLB payloads are required for this integration test")
    prepare_project(tmp_path)
    save_manifest(tmp_path, manifest)
    for name in expected:
        shutil.copyfile(ROOT / "data/PMLB" / f"{name}.csv", tmp_path / "data/PMLB" / f"{name}.csv")
    source_hashes = {p: sha256_file(p) for p in (tmp_path / "data/PMLB").glob("*.csv")}
    first = import_pmlb(project_root=tmp_path, create_splits=True)
    before = snapshot(tmp_path)
    assert import_pmlb(project_root=tmp_path, create_splits=True) == first
    assert snapshot(tmp_path) == before
    config = research_config(tmp_path)
    fresh_root = tmp_path / "fresh_splits"
    prepare_project(fresh_root)
    for row in first:
        dataset = load_dataset(row["dataset_id"], project_root=tmp_path)
        assert row["status"] == "valid"
        assert row["n_categorical_features"] == expected[row["original_name"]]
        assert set(dataset.X.select_dtypes("category")) == set(dataset.metadata.categorical_columns)
        assert all(is_numeric_dtype(t) for c, t in dataset.X.dtypes.items()
                   if c not in dataset.metadata.categorical_columns)
        split = make_split(dataset, root=tmp_path)
        check_manifest(split, dataset)
        assert make_split(dataset, root=fresh_root) == split
        positions = split["indices"]["train"]
        X, y = dataset.X.iloc[positions], dataset.y.iloc[positions]
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=config["custom_landmarkers"]["random_seed"])
        for train, held in cv.split(X, y):
            assert len(train) >= 5 and len(train) > y.nunique()
            assert not set(train) & set(held)
            assert set(y.iloc[train]) == set(y.iloc[held]) == set(y)
        if row["original_name"] == "mushroom":
            assert isinstance(dataset.X["stalk-root"].dtype, pd.CategoricalDtype)
            assert dataset.X["stalk-root"].eq(0).sum() == 2480
            assert dataset.X["stalk-root"].isna().sum() == 0
        report_text = (ROOT / "docs/PMLB_PILOT_SELECTION.md").read_text(encoding="utf-8")
        assert row["sha256"] in report_text
    assert all(sha256_file(path) == digest for path, digest in source_hashes.items())
    assert len(read_registry(project_root=tmp_path)) == 10
    saved = json.loads((tmp_path / "results/logs/pmlb_import.json").read_text())
    assert len(saved["datasets"]) == 10
    assert not (tmp_path / "data/derived").exists()
