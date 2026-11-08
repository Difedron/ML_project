"""Transfer integrity tests: copied fixtures only, no models and no split generation."""
import csv
import importlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest
import yaml

from meta_selector.paths import DEFAULT_PROJECT_ROOT as ROOT


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    exporter = importlib.import_module("export_pmlb_pilot")
    receiver = importlib.import_module("pmlb_package_verify")
    specs = yaml.safe_load((ROOT / "config/pmlb_pilot.yaml").read_text(encoding="utf-8"))["datasets"]
    if any(not (ROOT / "data/PMLB" / f"{s['original_name']}.csv").is_file() for s in specs):
        pytest.skip("Approved local CSVs required")
    files = ["config/project.yaml", "config/datasets.yaml", "config/research.yaml", "config/pmlb_pilot.yaml",
             "data/registry/datasets.csv", "docs/PMLB_DIANA_PACKAGE_README.md", "scripts/pmlb_package_verify.py"]
    for spec in specs:
        name, dataset_id = spec["original_name"], spec["dataset_id"]
        files += [f"data/PMLB/{name}.csv", f"data/raw/{dataset_id}/{name}.csv",
                  f"data/raw/{dataset_id}/metadata.yaml", f"data/splits/{dataset_id}/holdout_0.json"]
    for name in files:
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    monkeypatch.setattr(exporter, "ROOT", tmp_path)
    splits = importlib.import_module("meta_selector.datasets.splits")
    monkeypatch.setattr(splits, "make_split", lambda *a, **k: pytest.fail("Export must never generate splits"))
    archive = tmp_path / "exports/first.zip"
    report, rows = exporter.export(archive)
    unpacked = tmp_path / "unpacked"
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(unpacked)
    return exporter, receiver, tmp_path, archive, unpacked, report, rows


def update_checksum(receiver, root, name):
    path = root / "SHA256SUMS.json"
    values = json.loads(path.read_text(encoding="utf-8"))
    values["files"][name] = receiver.sha256(root / name)
    path.write_text(json.dumps(values), encoding="utf-8")


def test_export_is_exact_allowlisted_and_deterministic(bundle):
    exporter, receiver, root, archive, unpacked, report, rows = bundle
    assert len(rows) == 10 and report["sources_unchanged"] and report["zip_crc_ok"]
    result = receiver.verify(unpacked, report["checksums_sha256"])
    assert result["verified_files"] == 64 and result["index_files"] == 30
    with zipfile.ZipFile(archive) as zipped:
        assert len(zipped.namelist()) == 64
        assert not any("meta_features" in n or "landmarkers" in n or n.endswith(".parquet") for n in zipped.namelist())
        for row in rows:
            dataset_id = row["dataset_id"]
            assert zipped.read(row["csv_path"]) == (root / "data/PMLB" / row["original_csv"]).read_bytes()
            assert zipped.read(row["metadata_path"]) == (root / "data/raw" / dataset_id / "metadata.yaml").read_bytes()
            assert zipped.read(row["manifest_path"]) == (root / "data/splits" / dataset_id / "holdout_0.json").read_bytes()
    second = root / "exports/second.zip"
    second_report, _ = exporter.export(second)
    assert second.read_bytes() == archive.read_bytes()
    assert second_report["zip_sha256"] == report["zip_sha256"]


def test_standalone_verifier_works_without_project_imports_from_other_cwd(bundle):
    _, _, root, _, unpacked, report, _ = bundle
    result = subprocess.run([sys.executable, "-I", "-B", str(unpacked / "verify_package.py"),
                             "--checksums-sha256", report["checksums_sha256"]],
                            cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["verified_files"] == 64


@pytest.mark.parametrize("field", ["csv_path", "metadata_path", "manifest_path", "train_indices_path", "verifier"])
def test_changed_file_is_detected(bundle, field):
    _, receiver, _, _, unpacked, _, rows = bundle
    name = "verify_package.py" if field == "verifier" else rows[0][field]
    with (unpacked / name).open("ab") as stream:
        stream.write(b"\nchanged\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        receiver.verify(unpacked)


def test_reordered_indices_rejected_even_after_updating_file_checksum(bundle):
    _, receiver, _, _, unpacked, _, rows = bundle
    name = rows[0]["train_indices_path"]
    path = unpacked / name
    lines = path.read_text().splitlines()
    path.write_text("\n".join(reversed(lines)) + "\n", encoding="utf-8")
    update_checksum(receiver, unpacked, name)
    with pytest.raises(ValueError, match="differs from manifest"):
        receiver.verify(unpacked)


@pytest.mark.parametrize("parts,n_rows", [
    ({"train": [0], "validation": [0], "test": [1]}, 2),
    ({"train": [0], "validation": [1], "test": [2]}, 4),
    ({"train": [True], "validation": [0], "test": [1]}, 3),
    ({"train": [0], "validation": [1], "test": [3]}, 3),
])
def test_overlap_incomplete_or_nonpositional_indices_rejected(monkeypatch, parts, n_rows):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    receiver = importlib.import_module("pmlb_package_verify")
    with pytest.raises(ValueError):
        receiver.check_parts(parts, n_rows)


def test_unexpected_file_rejected(bundle):
    _, receiver, _, _, unpacked, _, _ = bundle
    (unpacked / "unrelated.txt").write_text("not in transfer contract", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected"):
        receiver.verify(unpacked)


def test_wrong_second_seed_rejected_even_after_updating_checksum(bundle):
    _, receiver, _, _, unpacked, _, _ = bundle
    path = unpacked / "datasets.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    rows[0]["second_stage_seed"] = "44"
    memory = io.StringIO(newline="")
    writer = csv.DictWriter(memory, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(memory.getvalue(), encoding="utf-8")
    update_checksum(receiver, unpacked, "datasets.csv")
    with pytest.raises(ValueError, match="seeds 42 / 43"):
        receiver.verify(unpacked)
