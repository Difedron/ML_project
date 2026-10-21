import csv
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd
import pytest
import yaml

from meta_selector import load_dataset, validate_dataset
from meta_selector.datasets import DatasetValidationError
from meta_selector.datasets.registry import (REGISTRY_FIELDS, read_registry, rebuild_registry,
                                              register_dataset)
from meta_selector.paths import DEFAULT_PROJECT_ROOT, project_paths


ROOT = DEFAULT_PROJECT_ROOT


@pytest.fixture
def catalog(tmp_path):
    (tmp_path / "config").mkdir()
    for file in ("project.yaml", "datasets.yaml"):
        shutil.copyfile(ROOT / "config" / file, tmp_path / "config" / file)
    (tmp_path / "data" / "raw").mkdir(parents=True)
    return tmp_path


def make_dataset(root, dataset_id="example", frame=None, file_format="csv", file_options=None):
    folder = root / "data" / "raw" / dataset_id
    folder.mkdir(parents=True, exist_ok=True)
    if frame is None:
        frame = pd.DataFrame({"age": [21, 30, 40, 50], "city": ["a", "b", None, "a"],
                              "target": [0, 1, 0, 1]})
    metadata = {"dataset_id": dataset_id, "name": "Example Dataset", "task_type": "classification",
                "target": "target", "source": {"type": "other", "url": "", "external_id": None},
                "file": {"name": f"original.{file_format}", "format": file_format},
                "problem": {"type": "binary"}, "notes": "Test fixture"}
    metadata["file"].update(file_options or {})
    path = folder / metadata["file"]["name"]
    if file_format == "csv":
        frame.to_csv(path, index=False, sep=metadata["file"].get("separator", ","),
                     encoding=metadata["file"].get("encoding", "utf-8"))
    elif file_format == "xlsx":
        frame.to_excel(path, index=False)
    else:
        frame.to_parquet(path, index=False)
    save_metadata(folder, metadata)
    return folder, metadata


def save_metadata(folder, metadata):
    (folder / "metadata.yaml").write_text(yaml.safe_dump(metadata, allow_unicode=True), encoding="utf-8")


def test_load_tracked_toy():
    dataset = load_dataset("_toy_binary")
    assert dataset.dataset_id == "_toy_binary"
    assert dataset.X.shape == (30, 3)
    assert dataset.y.name == "target"
    assert dataset.y.nunique() == 2
    assert dataset.X.city.iloc[0] == "kazan"
    assert dataset.X.isna().sum().sum() == 2
    assert dataset.metadata.file.name == "data.csv"


@pytest.mark.parametrize("file_format", ["csv", "xlsx", "parquet"])
def test_three_formats_preserve_data(catalog, file_format):
    original = pd.DataFrame({"age": [1.5, None, 3.0, 4.0], "city": ["a", "b", None, "a"],
                             "target": ["no", "yes", "no", "yes"]})
    folder, metadata = make_dataset(catalog, frame=original, file_format=file_format)
    payload = folder / metadata["file"]["name"]
    before = payload.read_bytes()
    dataset = load_dataset("example", project_root=catalog)
    assert dataset.X.columns.tolist() == ["age", "city"]
    assert dataset.X.age.isna().tolist() == original.age.isna().tolist()
    assert dataset.X.city.isna().tolist() == original.city.isna().tolist()
    assert dataset.X.city.dropna().tolist() == original.city.dropna().tolist()
    assert dataset.y.tolist() == original.target.tolist()
    assert payload.read_bytes() == before


@pytest.mark.parametrize("target", [None, "", "   "])
def test_target_is_required_and_never_inferred(catalog, target):
    folder, metadata = make_dataset(catalog, frame=pd.DataFrame({"class": [0, 1], "label": [0, 1]}))
    if target is None:
        del metadata["target"]
    else:
        metadata["target"] = target
    save_metadata(folder, metadata)
    report = validate_dataset("example", project_root=catalog)
    assert report.status == "invalid_metadata"
    assert "target" in " ".join(report.errors)
    with pytest.raises(DatasetValidationError):
        load_dataset("example", project_root=catalog)


def test_wrong_target(catalog):
    folder, metadata = make_dataset(catalog)
    metadata["target"] = "not_a_column"
    save_metadata(folder, metadata)
    report = validate_dataset("example", project_root=catalog)
    assert report.status == "invalid"
    assert "not_a_column" in report.errors[0]


@pytest.mark.parametrize("payload", ["", "age,target\n"])
def test_empty_dataset(catalog, payload):
    folder, metadata = make_dataset(catalog)
    (folder / metadata["file"]["name"]).write_text(payload)
    assert validate_dataset("example", project_root=catalog).status == "invalid"


def test_no_features(catalog):
    make_dataset(catalog, frame=pd.DataFrame({"target": [0, 1, 0, 1]}))
    report = validate_dataset("example", project_root=catalog)
    assert report.status == "invalid"
    assert report.n_features == 0


def test_one_class(catalog):
    make_dataset(catalog, frame=pd.DataFrame({"x": [1, 2, 3], "target": [1, 1, None]}))
    report = validate_dataset("example", project_root=catalog)
    assert report.status == "invalid"
    assert report.n_classes == 1


def test_multiclass_and_class_count_consistency(catalog):
    folder, metadata = make_dataset(catalog, frame=pd.DataFrame({"x": range(6), "target": [0, 1, 2, 0, 1, 2]}))
    assert validate_dataset("example", project_root=catalog).status == "invalid"
    metadata["problem"]["type"] = "multiclass"
    save_metadata(folder, metadata)
    report = validate_dataset("example", project_root=catalog)
    assert report.is_valid and report.n_classes == 3
    assert load_dataset("example", project_root=catalog).y.tolist() == [0, 1, 2, 0, 1, 2]


def test_missing_values_categories_and_imbalance_are_valid(catalog):
    original = pd.DataFrame({"empty": [None] * 12, "category": ["a", "b"] * 6,
                             "target": [0] * 9 + [1, None, None]})
    make_dataset(catalog, frame=original)
    report = validate_dataset("example", project_root=catalog)
    assert report.is_valid
    assert report.n_missing_values == 14
    assert report.n_missing_target == 2
    assert report.warnings
    dataset = load_dataset("example", project_root=catalog)
    assert len(dataset.y) == 12
    assert dataset.y.isna().sum() == 2
    assert dataset.X.category.tolist() == original.category.tolist()


@pytest.mark.parametrize("file_format", ["csv", "xlsx", "parquet"])
def test_duplicate_original_headers_are_rejected(catalog, file_format):
    folder, metadata = make_dataset(catalog, file_format=file_format)
    path = folder / metadata["file"]["name"]
    if file_format == "csv":
        path.write_text("age,age,target\n1,2,0\n3,4,1\n")
    elif file_format == "xlsx":
        from openpyxl import Workbook
        workbook = Workbook()
        workbook.active.append(["age", "age", "target"])
        workbook.active.append([1, 2, 0])
        workbook.active.append([3, 4, 1])
        workbook.save(path)
        workbook.close()
    else:
        import pyarrow as pa
        import pyarrow.parquet as pq
        pq.write_table(pa.table([pa.array([1, 2]), pa.array([3, 4]), pa.array([0, 1])],
                                names=["age", "age", "target"]), path)
    report = validate_dataset("example", project_root=catalog)
    assert report.status == "invalid"
    assert "Повторяющиеся исходные" in report.errors[0]


def test_csv_custom_separator_and_encoding(catalog):
    frame = pd.DataFrame({"город": ["Казань", "Пермь"], "target": [0, 1]})
    make_dataset(catalog, frame=frame, file_options={"separator": ";", "encoding": "cp1251"})
    dataset = load_dataset("example", project_root=catalog)
    assert dataset.X["город"].tolist() == ["Казань", "Пермь"]


@pytest.mark.parametrize("sheet", [None, "Data", 1])
def test_excel_sheet_selection(catalog, sheet):
    folder, metadata = make_dataset(catalog, file_format="xlsx")
    first = pd.DataFrame({"x": [1, 2], "target": [0, 1]})
    second = pd.DataFrame({"x": [10, 20], "target": [0, 1]})
    with pd.ExcelWriter(folder / metadata["file"]["name"], engine="openpyxl") as writer:
        first.to_excel(writer, index=False, sheet_name="First")
        second.to_excel(writer, index=False, sheet_name="Data")
    metadata["file"]["sheet_name"] = sheet
    # CSV-only settings are ignored for XLSX.
    metadata["file"]["separator"] = None
    metadata["file"]["encoding"] = None
    save_metadata(folder, metadata)
    dataset = load_dataset("example", project_root=catalog)
    assert dataset.X.x.tolist() == ([1, 2] if sheet is None else [10, 20])


@pytest.mark.parametrize("dataset_id", ["../outside", "has space", "Adult", "bad-id"])
def test_bad_dataset_id(catalog, dataset_id):
    assert validate_dataset(dataset_id, project_root=catalog).status == "invalid_metadata"


def test_missing_metadata_and_file_have_distinct_statuses(catalog):
    folder, metadata = make_dataset(catalog)
    (folder / metadata["file"]["name"]).unlink()
    report = validate_dataset("example", project_root=catalog)
    assert report.status == "missing_file" and report.metadata.target == "target"
    (folder / "metadata.yaml").unlink()
    assert validate_dataset("example", project_root=catalog).status == "invalid_metadata"


@pytest.mark.parametrize("change", ["id", "task", "format", "filename", "separator", "encoding", "yaml_duplicate", "yaml_broken"])
def test_invalid_metadata(catalog, change):
    folder, metadata = make_dataset(catalog)
    if change == "id":
        metadata["dataset_id"] = "different"
    elif change == "task":
        metadata["task_type"] = "regression"
    elif change == "format":
        metadata["file"]["format"] = "zip"
    elif change == "filename":
        metadata["file"]["name"] = "../outside.csv"
    elif change == "separator":
        metadata["file"]["separator"] = "::"
    elif change == "encoding":
        metadata["file"]["encoding"] = "not_a_codec"
    save_metadata(folder, metadata)
    if change == "yaml_duplicate":
        with (folder / "metadata.yaml").open("a") as stream:
            stream.write("target: overwritten\n")
    elif change == "yaml_broken":
        (folder / "metadata.yaml").write_text("target: [broken")
    assert validate_dataset("example", project_root=catalog).status == "invalid_metadata"


def test_global_defaults_are_used_without_cwd_dependency(catalog, monkeypatch, tmp_path):
    make_dataset(catalog, file_options={"separator": ";"})
    folder = catalog / "data/raw/example"
    metadata = yaml.safe_load((folder / "metadata.yaml").read_text())
    del metadata["file"]["separator"]
    save_metadata(folder, metadata)
    (catalog / "config/datasets.yaml").write_text('csv:\n  separator: ";"\n')
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert validate_dataset("example", project_root=catalog).is_valid
    assert project_paths().root == ROOT
    assert load_dataset("_toy_binary").X.shape == (30, 3)


def test_registry_upsert_rebuild_and_invalid_entries(catalog):
    make_dataset(catalog, dataset_id="z_valid")
    folder, metadata = make_dataset(catalog, dataset_id="a_missing")
    (folder / metadata["file"]["name"]).unlink()
    broken = catalog / "data/raw/b_no_metadata"
    broken.mkdir()
    register_dataset("z_valid", project_root=catalog)
    register_dataset("z_valid", project_root=catalog)
    assert len(read_registry(project_root=catalog)) == 1
    reports = rebuild_registry(project_root=catalog)
    assert len(reports) == 3
    rows = read_registry(project_root=catalog)
    assert [row["dataset_id"] for row in rows] == ["a_missing", "b_no_metadata", "z_valid"]
    assert [row["status"] for row in rows] == ["missing_file", "invalid_metadata", "valid"]
    assert rows[0]["error_message"] and rows[1]["error_message"]
    assert rows[0]["target"] == "target"
    path = project_paths(catalog).registry
    before = path.read_bytes()
    path.unlink()
    rebuild_registry(project_root=catalog)
    assert path.read_bytes() == before
    with path.open(newline="") as stream:
        assert next(csv.reader(stream)) == REGISTRY_FIELDS


def test_registry_atomic_replace_failure_preserves_previous_file(catalog, monkeypatch):
    import meta_selector.datasets.registry as registry

    make_dataset(catalog)
    rebuild_registry(project_root=catalog)
    path = project_paths(catalog).registry
    before = path.read_bytes()

    def fail_replace(source, target):
        assert Path(source).parent == path.parent
        assert Path(source).read_bytes().startswith(b"dataset_id,")
        raise OSError("simulated interruption before replace")

    monkeypatch.setattr(registry.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        rebuild_registry(project_root=catalog)
    assert path.read_bytes() == before
    assert not list(path.parent.glob("*.tmp"))


def test_all_cli_commands_preserve_raw_bytes_and_work_outside_project(catalog, tmp_path):
    folder, _ = make_dataset(catalog)
    before = {file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in folder.iterdir()}
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    commands = [("validate_dataset.py", "example"), ("register_dataset.py", "example"),
                ("register_dataset.py", "--rebuild"), ("list_datasets.py",)]
    for script, *args in commands:
        result = subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args,
                                 "--project-root", str(catalog)], cwd=tmp_path, env=environment,
                                 capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "example" in result.stdout
    after = {file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in folder.iterdir()}
    assert before == after
    result = subprocess.run([sys.executable, str(ROOT / "scripts/validate_dataset.py"), "_toy_binary"],
                            cwd=tmp_path, env=environment, capture_output=True, text=True)
    assert result.returncode == 0 and "Status: VALID" in result.stdout


def test_cli_invalid_dataset_is_indexed_with_nonzero_exit(catalog, tmp_path):
    folder, metadata = make_dataset(catalog)
    del metadata["target"]
    save_metadata(folder, metadata)
    result = subprocess.run([sys.executable, str(ROOT / "scripts/register_dataset.py"), "example",
                             "--project-root", str(catalog)], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 1
    assert "target" in result.stdout
    assert read_registry(project_root=catalog)[0]["status"] == "invalid_metadata"


def test_gitignore_tracks_metadata_and_toy_but_not_payloads():
    ignored = ["data/raw/adult/data.csv", "data/raw/adult/original_name.csv",
               "data/raw/credit/data.xlsx", "data/raw/credit/other.parquet",
               "data/raw/credit/archive.zip", "data/processed/adult/result.csv", "results/logs/run.log"]
    visible = ["data/raw/adult/metadata.yaml", "data/raw/_toy_binary/data.csv",
               "data/raw/_toy_binary/metadata.yaml", "data/registry/datasets.csv",
               "data/processed/.gitkeep", "results/tables/.gitkeep", "results/figures/.gitkeep",
               "results/logs/.gitkeep"]
    for filename, expected_code in [(name, 0) for name in ignored] + [(name, 1) for name in visible]:
        result = subprocess.run(["git", "check-ignore", "--no-index", "-q", filename], cwd=ROOT,
                                capture_output=True, text=True)
        assert result.returncode == expected_code, f"{filename}: {result.stderr}"
