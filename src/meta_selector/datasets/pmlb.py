"""Import the explicitly approved local PMLB cohort; no downloads or estimators."""
import hashlib
from pathlib import Path
import re
import shutil

import yaml

from meta_selector.artifacts import _atomic, atomic_json, environment, research_config, research_paths
from meta_selector.paths import project_paths, read_yaml_mapping
from .loader import check_dataset_id, load_dataset
from .registry import register_dataset
from .splits import make_split, portion


def sha256_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _inside(root, path):
    path = path.resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Import paths must remain inside the project: {path}")
    return path


def _metadata(spec, config):
    original = spec["original_name"]
    return dict(
        dataset_id=spec["dataset_id"], name=original, task_type="classification", target="target",
        source=dict(type="pmlb", external_id=original,
                    url=f"https://github.com/EpistasisLab/pmlb/tree/{config['metadata_revision']}/datasets/{original}"),
        file=dict(name=f"{original}.csv", format="csv", separator=",", encoding="utf-8",
                  sha256=spec["sha256"]),
        problem=dict(type="binary" if spec["n_classes"] == 2 else "multiclass"),
        categorical_columns=spec["categorical_columns"],
        notes="Explicit pilot typing; ordinal factors use categorical representation. "
              "Original CSV bytes and unknown-category codes are preserved.",
        provenance=dict(original_name=original, original_path=f"data/PMLB/{original}.csv",
                        selection_report=config["selection_report"],
                        metadata_revision=config["metadata_revision"],
                        description_url=spec["description_url"], origin_group=spec["origin_group"],
                        unknown_categories=spec.get("unknown_categories", {})),
    )


def _preflight(root, paths):
    """Check the entire cohort and existing destinations before copying any payload."""
    config = read_yaml_mapping(root / "config/pmlb_pilot.yaml")
    if config.get("schema_version") != 1 or not isinstance(config.get("datasets"), list) or not config["datasets"]:
        raise ValueError("Invalid PMLB pilot manifest")
    planned, ids, names = [], set(), set()
    for spec in config["datasets"]:
        original, dataset_id = spec["original_name"], spec["dataset_id"]
        check_dataset_id(dataset_id)
        if (not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", original)
                or dataset_id != "pmlb_" + original.lower()):
            raise ValueError("PMLB names must use the stable pmlb_<original_name.lower()> mapping")
        if dataset_id in ids or original in names:
            raise ValueError("Duplicate PMLB pilot identity")
        ids.add(dataset_id)
        names.add(original)
        if not isinstance(spec["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", spec["sha256"]):
            raise ValueError(f"Invalid approved SHA-256: {dataset_id}")
        source = _inside(root, root / "data/PMLB" / f"{original}.csv")
        destination = _inside(root, paths.raw_data / dataset_id / source.name)
        metadata_path = _inside(root, destination.parent / "metadata.yaml")
        metadata = _metadata(spec, config)
        if sha256_file(source) != spec["sha256"]:
            raise ValueError(f"Source SHA-256 differs from approved selection: {source}")
        if destination.exists() and sha256_file(destination) != spec["sha256"]:
            raise ValueError(f"Preserved conflicting imported CSV: {destination}")
        if metadata_path.exists() and read_yaml_mapping(metadata_path) != metadata:
            raise ValueError(f"Preserved conflicting metadata: {metadata_path}")
        if destination.parent.exists():
            unexpected = set(destination.parent.iterdir()) - {destination, metadata_path}
            if unexpected:
                raise ValueError(f"Preserved unrelated files in {destination.parent}: {unexpected}")
        planned.append((spec, source, destination, metadata_path, metadata))
    return planned


def import_pmlb(*, project_root=None, create_splits=False, split_id="holdout_0"):
    """Copy verified bytes, use the existing validator/registry and optional split API.

    One writer per checkout, as with the existing registry and artifact utilities.
    Matching existing CSV/metadata/manifests are reused; conflicting files are never replaced.
    A partially completed import can be resumed with the same approved manifest.
    """
    paths = project_paths(project_root)
    root = paths.root
    outputs = research_paths(root)
    for path in (paths.raw_data, paths.registry, paths.results,
                 *outputs.values()):
        _inside(root, path)
    if not split_id or not all(c.isalnum() or c in "_-" for c in split_id):
        raise ValueError("Unsafe split ID")
    config = research_config(root)
    planned = _preflight(root, paths)
    records = []
    for spec, source, destination, metadata_path, metadata in planned:
        if not destination.exists():
            def copy_verified(temporary):
                shutil.copyfile(source, temporary)
                if sha256_file(temporary) != spec["sha256"]:
                    raise ValueError(f"CSV changed during copying: {source}")
            _atomic(destination, copy_verified)
        if not metadata_path.exists():
            _atomic(metadata_path, lambda temporary: temporary.write_text(
                yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True), encoding="utf-8"))
        report = register_dataset(spec["dataset_id"], project_root=root)
        if not report.is_valid:
            raise ValueError(report.to_text())
        for field in ("n_rows", "n_features", "n_classes"):
            if getattr(report, field) != spec[field]:
                raise ValueError(f"Imported {spec['dataset_id']} differs from approved {field}")
        dataset = load_dataset(spec["dataset_id"], project_root=root)
        record = dict(dataset_id=dataset.dataset_id, original_name=spec["original_name"],
                      status=report.status.value, sha256=spec["sha256"],
                      source_path=source.relative_to(root).as_posix(),
                      imported_path=destination.relative_to(root).as_posix(),
                      n_rows=report.n_rows, n_features=report.n_features, n_classes=report.n_classes,
                      n_numeric_features=report.n_numeric_features,
                      n_categorical_features=report.n_categorical_features,
                      n_missing_values=report.n_missing_values,
                      categorical_columns=list(dataset.metadata.categorical_columns),
                      class_counts={str(k): int(v) for k, v in dataset.y.value_counts().sort_index().items()})
        if create_splits:
            manifest = make_split(dataset, root=root, split_id=split_id, protocol=config["splits"])
            # make_split and portion both call the existing check_manifest.
            parts = {name: portion(dataset, manifest, name)[1] for name in ("train", "validation", "test")}
            if parts["train"].value_counts().min() < config["custom_landmarkers"]["n_splits"]:
                raise ValueError(f"Insufficient training class support: {dataset.dataset_id}")
            record.update(split_id=split_id, dataset_fingerprint=manifest["dataset_fingerprint"],
                          protocol_hash=manifest["protocol_hash"],
                          split_manifest=(outputs["splits"] / dataset.dataset_id / f"{split_id}.json").relative_to(root).as_posix(),
                          split_sizes={name: len(y) for name, y in parts.items()},
                          split_class_counts={name: {str(k): int(v) for k, v in y.value_counts().sort_index().items()}
                                              for name, y in parts.items()})
        records.append(record)
    for spec, source, destination, _, _ in planned:
        if sha256_file(source) != spec["sha256"] or sha256_file(destination) != spec["sha256"]:
            raise ValueError(f"Source/import checksum changed: {source}")
    atomic_json(paths.results / "logs/pmlb_import.json",
                dict(schema_version=1, environment=environment(), protocol=config["splits"], datasets=records))
    return records
