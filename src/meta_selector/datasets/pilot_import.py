"""Validate the authoritative Rita bundle in quarantine; publish only text manifests."""
import csv
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
from zipfile import ZipFile

import yaml

from meta_selector.artifacts import config_hash
from meta_selector.paths import DEFAULT_PROJECT_ROOT
from .shared_pilot import (PARTS, JOIN_KEYS, PilotContractError, contained_path, file_sha256,
                            validate_indices, validate_pilot)


def _require(condition, message):
    if not condition:
        raise PilotContractError(message)


def safe_extract_bundle(archive, cache_root):
    """No extractall(), execution, symlinks, traversal, duplicate names or unbounded expansion."""
    archive, cache_root = Path(archive), Path(cache_root)
    digest = file_sha256(archive)
    cache_root.mkdir(parents=True, exist_ok=True)
    destination = cache_root / digest
    with ZipFile(archive) as package:
        entries = package.infolist()
        _require(len(entries) <= 256 and sum(item.file_size for item in entries) <= 32 * 1024 * 1024,
                 "Bundle exceeds safe extraction limits")
        names = set()
        for entry in entries:
            name = entry.filename
            parts = PurePosixPath(name)
            _require(name and not parts.is_absolute() and ".." not in parts.parts and "\\" not in name
                     and all(":" not in part for part in parts.parts), "Unsafe archive path")
            _require(parts.as_posix().casefold() not in names, "Duplicate/case-colliding archive path")
            names.add(parts.as_posix().casefold())
            _require(not stat.S_ISLNK(entry.external_attr >> 16), "Archive symlinks are forbidden")
            _require(not entry.flag_bits & 1 and entry.file_size <= 8 * 1024 * 1024,
                     "Encrypted or oversized archive entry")
        if destination.exists():
            verify_bundle_files(destination)
            expected = {entry.filename for entry in entries if not entry.is_dir()}
            _require(expected == {p.relative_to(destination).as_posix() for p in destination.rglob('*') if p.is_file()},
                     "Existing quarantine differs from bundle")
            for entry in entries:
                if not entry.is_dir():
                    _require(contained_path(destination, entry.filename).read_bytes() == package.read(entry),
                             "Quarantine bytes differ from source archive")
            return destination
        with tempfile.TemporaryDirectory(prefix=".extract-", dir=cache_root) as temporary:
            staging = Path(temporary) / "contents"
            staging.mkdir()
            for entry in entries:
                target = contained_path(staging, entry.filename)
                if entry.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as stream:
                    stream.write(package.read(entry))
            verify_bundle_files(staging)
            staging.rename(destination)
    _require(file_sha256(archive) == digest, "Archive changed during extraction")
    return destination


def verify_bundle_files(root):
    root = Path(root).resolve()
    checksums = json.loads((root / "SHA256SUMS.json").read_text(encoding="utf-8"))
    _require(checksums.get("schema_version") == 1 and isinstance(checksums.get("files"), dict),
             "Invalid bundle checksum manifest")
    files = checksums["files"]
    _require("SHA256SUMS.json" not in files, "Checksum manifest cannot hash itself")
    actual = set()
    for path in root.rglob("*"):
        _require(not path.is_symlink(), "Symlink in extracted bundle")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    _require(actual == set(files) | {"SHA256SUMS.json"}, "Bundle has missing/unexpected files")
    for name, expected in files.items():
        _require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected), "Invalid source SHA256")
        _require(file_sha256(contained_path(root, name)) == expected, f"Bundle checksum mismatch: {name}")
    return files


def _canonical_catalog(bundle, raw_root, archive_digest):
    hashes = verify_bundle_files(bundle)
    readme = (bundle / "README.md").read_text(encoding="utf-8")
    _require("Индексы **позиционные, от нуля**" in readme and "после заголовка CSV" in readme,
             "Rita row-order/index basis is not explicit in this bundle schema")
    with (bundle / "datasets.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    _require(len(rows) == 10 and len({row["dataset_id"] for row in rows}) == 10,
             "Rita bundle must name exactly ten distinct datasets")
    entries = []
    from meta_selector.paths import read_yaml_mapping
    for row in rows:
        dataset_id, split_id = row["dataset_id"], row["split_id"]
        _require(re.fullmatch(r"pmlb_[a-z0-9_]+", dataset_id) and re.fullmatch(r"[a-zA-Z0-9_-]+", split_id),
                 "Unsafe dataset/split identity")
        filename = row["original_csv"]
        _require(PurePosixPath(filename).name == filename and filename.endswith(".csv") and "\\" not in filename,
                 "Invalid original PMLB filename")
        meta = read_yaml_mapping(contained_path(bundle, row["metadata_path"]))
        original_path = contained_path(bundle, row["manifest_path"])
        manifest = json.loads(original_path.read_text(encoding="utf-8"))
        seed, second_seed = int(row["seed"]), int(row["second_stage_seed"])
        _require(meta["dataset_id"] == manifest["dataset_id"] == dataset_id and manifest["split_id"] == split_id,
                 "Dataset/split identity mismatch in Rita bundle")
        _require(meta["task_type"] == "classification" and meta["target"] == row["target_column"],
                 "Explicit classification target missing or inconsistent")
        _require(meta["source"]["type"] == "pmlb" and manifest["dataset_source"] == "local",
                 "Unexpected Rita source label; explicit local→pmlb normalization requires review")
        _require(meta["file"]["name"] == filename and meta["file"]["format"] == "csv", "Metadata CSV mapping mismatch")
        _require(type(manifest["protocol"]["random_seed"]) is int and manifest["protocol"]["random_seed"] == seed,
                 "Seed mismatch between Rita table and manifest")
        _require(config_hash(manifest["protocol"]) == manifest["protocol_hash"] == row["protocol_hash"],
                 "Protocol hash mismatch")
        csv_digest = file_sha256(contained_path(bundle, row["csv_path"]))
        _require(csv_digest == row["csv_sha256"] == meta["file"]["sha256"], "Rita CSV fingerprint mismatch")
        raw = contained_path(raw_root, filename)
        _require(raw.is_file(), f"Required PMLB CSV does not exist: {filename}")
        _require(file_sha256(raw) == csv_digest, f"Current PMLB CSV differs from Rita: {filename}")
        fingerprint = manifest.get("dataset_fingerprint")
        _require((fingerprint or "") == row.get("dataset_fingerprint", ""), "Rita typed fingerprint fields disagree")
        cats = json.loads(row["categorical_columns"])
        _require(cats == meta["categorical_columns"], "Categorical semantics disagree")
        n_rows = int(row["n_rows"])
        validate_indices(manifest["indices"], n_rows)
        sizes = {}
        for part in PARTS:
            text = contained_path(bundle, row[f"{part}_indices_path"]).read_text(encoding="utf-8").splitlines()
            _require(all(re.fullmatch(r"0|[1-9][0-9]*", line) for line in text), "Invalid index TXT")
            positions = [int(line) for line in text]
            _require(positions == manifest["indices"][part], "TXT indices differ from Rita JSON sequence")
            _require(len(positions) == int(row[f"{part}_size"]), "Split size mismatch in Rita table")
            sizes[part] = len(positions)
        entries.append(dict(dataset_id=dataset_id, dataset_source="pmlb", rita_dataset_source=manifest["dataset_source"],
            file=filename, target=row["target_column"], split_id=split_id, seed=seed, second_stage_seed=second_seed,
            n_rows=n_rows, n_features=int(row["n_features"]), n_classes=int(row["n_classes"]),
            fingerprint=csv_digest, fingerprint_method="sha256_raw_csv_bytes", rita_fingerprint_present=bool(fingerprint),
            rita_dataset_fingerprint=fingerprint, rita_fingerprint_version=manifest.get("fingerprint_version"),
            categorical_columns=cats, split_sizes=sizes, protocol_hash=row["protocol_hash"],
            shared_manifest=f"{dataset_id}/{split_id}.json", shared_manifest_sha256=hashes[row["manifest_path"]],
            rita_manifest_path=row["manifest_path"], rita_metadata=meta))
    _require(len({entry["seed"] for entry in entries}) == 1 and len({entry["second_stage_seed"] for entry in entries}) == 1,
             "Pilot does not declare a common seed protocol; review with Rita")
    return dict(schema_version=1, pilot_id="rita_pmlb_pilot_10", join_keys=list(JOIN_KEYS),
        source_bundle=dict(archive_sha256=archive_digest, checksums_sha256=file_sha256(bundle / "SHA256SUMS.json"),
            table_sha256=hashes["datasets.csv"], row_basis_source="README.md", readme_sha256=hashes["README.md"]),
        row_positions=dict(kind="positional", base=0, order="raw_csv_data_rows_after_header_no_filtering"),
        source_label_mapping={"rita": "local", "canonical": "pmlb"}, datasets=entries)


def import_pilot_bundle(archive, *, project_root=None):
    """Quarantine and validate every dataset before publishing any new canonical files."""
    root = Path(project_root or DEFAULT_PROJECT_ROOT).resolve()
    archive = Path(archive).resolve()
    digest = file_sha256(archive)
    bundle = safe_extract_bundle(archive, root / ".cache/rita_pilot_import")
    try:
        catalog = _canonical_catalog(bundle, root / "data/PMLB", digest)
    except PilotContractError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise PilotContractError(f"Incomplete/invalid authoritative bundle contract: {error}") from error
    # Stage the complete contract outside production; check adapter/reader on real raw bytes.
    with tempfile.TemporaryDirectory(prefix=".canonical-", dir=bundle.parent) as temporary:
        staging = Path(temporary)
        config = staging / "config"
        config.mkdir()
        raw_link = staging / "data/PMLB"
        raw_link.parent.mkdir()
        raw_link.symlink_to(root / "data/PMLB", target_is_directory=True)
        payloads = {"config/pilot_datasets.yaml": yaml.safe_dump(catalog, sort_keys=False, allow_unicode=True).encode()}
        for item in catalog["datasets"]:
            payloads["config/pilot_splits/" + item["shared_manifest"]] = contained_path(bundle, item["rita_manifest_path"]).read_bytes()
        for relative, data in payloads.items():
            path = contained_path(staging, relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        results = validate_pilot(project_root=staging)
        # Preflight every existing destination before writing. Never overwrite changed shared science.
        for relative, data in payloads.items():
            destination = contained_path(root, relative)
            _require(not destination.exists() or destination.read_bytes() == data,
                     f"Existing shared contract differs; review rather than overwrite: {relative}")
        for relative, data in payloads.items():
            destination = contained_path(root, relative)
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary_file = destination.with_suffix(destination.suffix + ".importing")
                try:
                    with temporary_file.open("xb") as stream:
                        stream.write(data)
                    os.replace(temporary_file, destination)
                finally:
                    temporary_file.unlink(missing_ok=True)
    _require(file_sha256(archive) == digest, "Source archive changed during import")
    return dict(status="PASS", archive_sha256=digest, extracted_to=str(bundle),
                verified_bundle_files=len(verify_bundle_files(bundle)) + 1,
                canonical_manifest="config/pilot_datasets.yaml", datasets=results)
