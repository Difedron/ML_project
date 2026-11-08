"""Stage-2 unit tests: tiny synthetic contracts only; no candidate/landmarker fits."""
import csv
import hashlib
import io
import json
from pathlib import Path
import stat
from zipfile import ZipFile, ZipInfo

import pandas as pd
import pytest
import yaml

from meta_selector.artifacts import config_hash
from meta_selector.datasets.pilot_import import import_pilot_bundle, safe_extract_bundle
from meta_selector.datasets.shared_pilot import (JOIN_KEYS, PilotContractError, contained_path,
    file_sha256, load_pmlb_dataset, load_shared_split, read_pilot_manifest, validate_indices, validate_pilot)


@pytest.fixture
def pilot(tmp_path):
    """Authoritative-format fake bundle, with seed 77/88 to catch guessed 42/43."""
    raw = tmp_path / "data/PMLB"
    raw.mkdir(parents=True)
    contents = {"README.md": "Индексы **позиционные, от нуля**, после заголовка CSV.".encode(),
                "verify_package.py": b"raise RuntimeError('This file must never be executed')\n"}
    rows = []
    for i in range(10):
        dataset_id, filename = f"pmlb_test_{i}", f"Original_{i}.csv"
        frame = pd.DataFrame({"number": range(12), "target_hint": [0, 1, 2] * 4, "class": [0, 1] * 6})
        payload = frame.to_csv(index=False).encode()
        (raw / filename).write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        meta = dict(dataset_id=dataset_id, name=filename, task_type="classification", target="class",
                    source=dict(type="pmlb", url="https://example.invalid/unit-test", external_id=filename),
                    file=dict(name=filename, format="csv", encoding="utf-8", separator=",", sha256=digest),
                    problem=dict(type="binary"), categorical_columns=["target_hint"], notes="Synthetic unit fixture",
                    provenance=dict(origin_group=dataset_id))
        protocol = dict(random_seed=77, validation_fraction=.25, test_fraction=.25)
        parts = dict(train=[5, 4, 3, 2, 1, 0], validation=[8, 6, 7], test=[11, 10, 9])
        manifest = dict(schema_version=1, fingerprint_version=2, dataset_id=dataset_id, dataset_source="local",
                        split_id="rita_split", dataset_fingerprint="a" * 64, protocol=protocol,
                        protocol_hash=config_hash(protocol), indices=parts)
        row = dict(dataset_id=dataset_id, original_csv=filename, target_column="class", split_id="rita_split",
                   seed=77, second_stage_seed=88, train_size=6, validation_size=3, test_size=3,
                   csv_sha256=digest, categorical_columns=json.dumps(["target_hint"]),
                   n_rows=12, n_features=2, n_classes=2, dataset_fingerprint="a" * 64,
                   protocol_hash=manifest["protocol_hash"], csv_path=f"datasets/{dataset_id}/{filename}",
                   metadata_path=f"datasets/{dataset_id}/metadata.yaml", manifest_path=f"splits/{dataset_id}/rita_split.json",
                   **{f"{part}_indices_path": f"indices/{dataset_id}/{part}.txt" for part in parts})
        contents[row["csv_path"]] = payload
        contents[row["metadata_path"]] = yaml.safe_dump(meta).encode()
        contents[row["manifest_path"]] = json.dumps(manifest, indent=2).encode()
        for part, indices in parts.items():
            contents[row[f"{part}_indices_path"]] = "".join(f"{value}\n" for value in indices).encode()
        rows.append(row)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    contents["datasets.csv"] = stream.getvalue().encode()
    contents["SHA256SUMS.json"] = json.dumps(dict(schema_version=1, files={
        name: hashlib.sha256(value).hexdigest() for name, value in contents.items()})).encode()
    archive = tmp_path / "authoritative.zip"
    with ZipFile(archive, "w") as package:
        for name, value in contents.items():
            package.writestr(name, value)
    report = import_pilot_bundle(archive, project_root=tmp_path)
    return dict(root=tmp_path, archive=archive, report=report, contents=contents)


def catalog(pilot):
    return yaml.safe_load((pilot["root"] / "config/pilot_datasets.yaml").read_text())


def save_catalog(pilot, data):
    (pilot["root"] / "config/pilot_datasets.yaml").write_text(yaml.safe_dump(data, sort_keys=False))


def mutate_split(pilot, change):
    data = catalog(pilot)
    item = data["datasets"][0]
    path = pilot["root"] / "config/pilot_splits" / item["shared_manifest"]
    manifest = json.loads(path.read_text())
    change(manifest)
    path.write_text(json.dumps(manifest))
    item["shared_manifest_sha256"] = file_sha256(path)  # exercise semantic checks after integrity
    save_catalog(pilot, data)


def test_exact_id_loading_target_exclusion_and_no_global_categorical_fit(pilot):
    dataset = load_pmlb_dataset("pmlb_test_0", project_root=pilot["root"])
    assert dataset.dataset_source == "pmlb" and dataset.rita_dataset_source == "local"
    assert dataset.metadata.target == "class" and dataset.y.name == "class"
    assert dataset.X.columns.tolist() == ["number", "target_hint"]
    assert dataset.categorical_columns == ("target_hint",)
    assert dataset.X.target_hint.dtype.kind in "iu"  # annotation only, no learned vocabulary
    assert dataset.seed == 77 and dataset.rita_fingerprint_present
    with pytest.raises(PilotContractError, match="not in Rita"):
        load_pmlb_dataset("Original_0", project_root=pilot["root"])


def test_manifest_reader_preserves_positions_seed_source_and_all_classes(pilot):
    root = pilot["root"]
    dataset = load_pmlb_dataset("pmlb_test_0", project_root=root)
    split = load_shared_split(dataset.dataset_id, "rita_split", project_root=root, dataset=dataset)
    assert JOIN_KEYS == ("dataset_id", "split_id", "seed")
    assert split.join_key == ("pmlb_test_0", "rita_split", 77)
    assert split.second_stage_seed == 88
    assert split.train_indices == (5, 4, 3, 2, 1, 0)
    assert split.metadata["original_manifest"]["dataset_source"] == "local"
    X, y = split.portion(dataset, "train")
    assert X.index.tolist() == [5, 4, 3, 2, 1, 0] and y.tolist() == [1, 0, 1, 0, 1, 0]
    assert split.metadata["class_counts"]["validation"] == {"0": 2, "1": 1}
    assert not (root / "data/splits").exists()


@pytest.mark.parametrize("indices,match", [
    ({"train": [0, 1], "validation": [2], "test": [4]}, "Out-of-bounds"),
    ({"train": [-1, 1], "validation": [2], "test": [3]}, "Out-of-bounds"),
    ({"train": [0, True], "validation": [2], "test": [3]}, "noninteger"),
    ({"train": [0, 1.0], "validation": [2], "test": [3]}, "noninteger"),
    ({"train": [0, 0], "validation": [2], "test": [3]}, "Duplicate"),
    ({"train": [0, 1], "validation": [1, 2], "test": [3]}, "overlap"),
    ({"train": [0], "validation": [2], "test": [3]}, "cover all"),
    ({"train": [], "validation": [0, 1], "test": [2, 3]}, "Empty"),
    ({"train": [0, 1], "test": [2, 3]}, "must contain"),
])
def test_bad_indices_are_rejected(indices, match):
    with pytest.raises(PilotContractError, match=match):
        validate_indices(indices, 4)


def test_missing_manifest_does_not_generate_fallback(pilot, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Automatic split generation must never be called")
    monkeypatch.setattr("meta_selector.datasets.splits.make_split", forbidden)
    item = catalog(pilot)["datasets"][0]
    path = pilot["root"] / "config/pilot_splits" / item["shared_manifest"]
    path.unlink()
    with pytest.raises(PilotContractError, match="shared pilot manifests are required"):
        load_shared_split("pmlb_test_0", "rita_split", project_root=pilot["root"])
    assert not path.exists() and not (pilot["root"] / "data/splits").exists()


def test_missing_pilot_catalog_is_an_error(tmp_path):
    with pytest.raises(PilotContractError, match="shared pilot manifests are required"):
        read_pilot_manifest(project_root=tmp_path)


@pytest.mark.parametrize("split_id,seed", [("other", 77), ("rita_split", 42), ("rita_split", True)])
def test_wrong_join_identity_fails(pilot, split_id, seed):
    with pytest.raises(PilotContractError, match="split_id/seed"):
        load_shared_split("pmlb_test_0", split_id, seed=seed, project_root=pilot["root"])


def test_raw_fingerprint_rejects_content_and_row_order_change(pilot):
    path = pilot["root"] / "data/PMLB/Original_0.csv"
    frame = pd.read_csv(path)
    frame.iloc[::-1].to_csv(path, index=False)
    with pytest.raises(PilotContractError, match="fingerprint mismatch"):
        load_pmlb_dataset("pmlb_test_0", project_root=pilot["root"])


def test_explicit_target_is_never_guessed(pilot):
    data = catalog(pilot)
    data["datasets"][0]["target"] = "nonexistent_target"
    data["datasets"][0]["rita_metadata"]["target"] = "nonexistent_target"
    save_catalog(pilot, data)
    with pytest.raises(PilotContractError, match="Missing explicit target"):
        load_pmlb_dataset("pmlb_test_0", project_root=pilot["root"])


def test_missing_raw_does_not_fallback_to_bundle_copy(pilot):
    (pilot["root"] / "data/PMLB/Original_0.csv").unlink()
    with pytest.raises(PilotContractError, match="CSV not found"):
        load_pmlb_dataset("pmlb_test_0", project_root=pilot["root"])


def test_unknown_row_basis_is_not_inferred(pilot):
    data = catalog(pilot)
    data["row_positions"]["base"] = 1
    save_catalog(pilot, data)
    with pytest.raises(PilotContractError, match="index basis"):
        load_shared_split("pmlb_test_0", "rita_split", project_root=pilot["root"])


def test_single_class_is_not_classification(pilot):
    path = pilot["root"] / "data/PMLB/Original_0.csv"
    frame = pd.read_csv(path)
    frame["class"] = 0
    frame.to_csv(path, index=False)
    data = catalog(pilot)
    data["datasets"][0]["fingerprint"] = file_sha256(path)
    data["datasets"][0]["rita_metadata"]["file"]["sha256"] = file_sha256(path)
    save_catalog(pilot, data)
    with pytest.raises(PilotContractError, match="at least two classes"):
        load_pmlb_dataset("pmlb_test_0", project_root=pilot["root"])


def test_shared_json_byte_integrity(pilot):
    path = pilot["root"] / "config/pilot_splits/pmlb_test_0/rita_split.json"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(PilotContractError, match="manifest SHA256"):
        load_shared_split("pmlb_test_0", "rita_split", project_root=pilot["root"])


@pytest.mark.parametrize("change,match", [
    (lambda m: m["indices"]["train"].__setitem__(0, -1), "Out-of-bounds"),
    (lambda m: m["indices"]["train"].__setitem__(0, 4), "Duplicate"),
    (lambda m: m["indices"]["train"].__setitem__(0, 8), "overlap"),
    (lambda m: m["protocol"].update(random_seed=42), "seed mismatch"),
    (lambda m: m.update(protocol_hash="0" * 64), "protocol hash"),
    (lambda m: m.update(dataset_fingerprint="0" * 64), "fingerprint provenance"),
])
def test_reader_checks_semantics_even_after_updated_manifest_hash(pilot, change, match):
    mutate_split(pilot, change)
    with pytest.raises(PilotContractError, match=match):
        load_shared_split("pmlb_test_0", "rita_split", project_root=pilot["root"])


def test_missing_class_requires_review_and_is_not_repaired(pilot):
    def missing_class(manifest):
        manifest["indices"] = dict(train=list(range(6)), validation=[6, 8, 10], test=[7, 9, 11])
    mutate_split(pilot, missing_class)
    path = pilot["root"] / "config/pilot_splits/pmlb_test_0/rita_split.json"
    before = path.read_bytes()
    with pytest.raises(PilotContractError, match="requires review"):
        load_shared_split("pmlb_test_0", "rita_split", project_root=pilot["root"])
    assert path.read_bytes() == before


def test_no_preprocessing_or_filtering_before_positions(pilot):
    dataset = load_pmlb_dataset("pmlb_test_0", project_root=pilot["root"])
    split = load_shared_split(dataset.dataset_id, dataset.split_id, project_root=pilot["root"], dataset=dataset)
    dataset.X.iloc[0, 0] += 100
    with pytest.raises(PilotContractError, match="modified"):
        split.portion(dataset, "train")
    with pytest.raises(PilotContractError, match="filtered/reordered/modified"):
        load_shared_split(dataset.dataset_id, dataset.split_id, project_root=pilot["root"], dataset=dataset)


def test_authoritative_import_is_idempotent_and_raw_unchanged(pilot):
    root = pilot["root"]
    before = {path.name: file_sha256(path) for path in (root / "data/PMLB").glob("*.csv")}
    source_hash = file_sha256(pilot["archive"])
    report = import_pilot_bundle(pilot["archive"], project_root=root)
    assert len(report["datasets"]) == 10 and report["verified_bundle_files"] == 64
    assert all(row["status"] == "PASS" for row in validate_pilot(project_root=root))
    assert source_hash == file_sha256(pilot["archive"])
    assert before == {path.name: file_sha256(path) for path in (root / "data/PMLB").glob("*.csv")}
    assert not (root / "data/raw").exists()
    for item in catalog(pilot)["datasets"]:
        raw_manifest = root / "config/pilot_splits" / item["shared_manifest"]
        assert raw_manifest.read_bytes() == pilot["contents"][item["rita_manifest_path"]]


def test_import_does_not_overwrite_modified_shared_contract(pilot):
    path = pilot["root"] / "config/pilot_datasets.yaml"
    path.write_text(path.read_text() + "\n# existing collaborator edit\n")
    before = path.read_bytes()
    with pytest.raises(PilotContractError, match="rather than overwrite"):
        import_pilot_bundle(pilot["archive"], project_root=pilot["root"])
    assert path.read_bytes() == before


@pytest.mark.parametrize("name", ["../outside", "/tmp/outside", "C:/outside", "a\\outside"])
def test_unsafe_zip_paths_are_rejected(tmp_path, name):
    archive = tmp_path / "unsafe.zip"
    with ZipFile(archive, "w") as package:
        from zipfile import ZipInfo
        info = ZipInfo("placeholder")
        info.filename = name
        package.writestr(info, b"unsafe")
    with pytest.raises(PilotContractError, match="Unsafe archive"):
        safe_extract_bundle(archive, tmp_path / "cache")


def test_zip_symlinks_are_rejected(tmp_path):
    archive = tmp_path / "symlink.zip"
    info = ZipInfo("link")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with ZipFile(archive, "w") as package:
        package.writestr(info, b"../outside")
    with pytest.raises(PilotContractError, match="symlinks"):
        safe_extract_bundle(archive, tmp_path / "cache")


def test_corrupt_archive_contents_fail_before_publication(tmp_path):
    archive = tmp_path / "corrupt.zip"
    with ZipFile(archive, "w") as package:
        package.writestr("README.md", b"modified after hashing")
        package.writestr("SHA256SUMS.json", json.dumps(dict(schema_version=1, files={"README.md": "0" * 64})))
    with pytest.raises(PilotContractError, match="Bundle checksum mismatch"):
        safe_extract_bundle(archive, tmp_path / "cache")
    assert not (tmp_path / "config").exists()


def test_external_artifact_symlink_is_rejected(tmp_path):
    inside, outside = tmp_path / "inside", tmp_path / "outside"
    inside.mkdir()
    outside.mkdir()
    (inside / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(PilotContractError, match="escapes"):
        contained_path(inside, "link/file")
