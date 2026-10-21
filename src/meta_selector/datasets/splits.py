"""Persisted positional stratified splits, validated against the unchanged dataset."""
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from meta_selector.artifacts import (atomic_json, config_hash, dataset_fingerprint, environment,
                                     research_config, research_paths)


def make_split(dataset, *, root=None, split_id="holdout_0", protocol=None):
    if not split_id or not all(c.isalnum() or c in "_-" for c in split_id):
        raise ValueError("Unsafe split ID")
    protocol = protocol or research_config(root)["splits"]
    test, validation = float(protocol["test_fraction"]), float(protocol["validation_fraction"])
    if test <= 0 or validation <= 0 or test + validation >= 1:
        raise ValueError("Split fractions must be positive and sum to less than 1")
    if dataset.y.isna().any():
        raise ValueError("Cannot stratify missing target labels; raw rows were not removed")
    fingerprint = dataset_fingerprint(dataset)
    path = research_paths(root)["splits"] / dataset.dataset_id / f"{split_id}.json"
    if path.exists():
        import json
        manifest = json.loads(path.read_text())
        if manifest["dataset_fingerprint"] != fingerprint or manifest["protocol"] != protocol:
            raise ValueError("Existing split differs from dataset/protocol; choose a new split_id")
    else:
        indices = np.arange(len(dataset.y))
        train_val, held_test = train_test_split(indices, test_size=test, stratify=dataset.y,
                                                random_state=protocol["random_seed"])
        train, val = train_test_split(train_val, test_size=validation / (1 - test),
                                     stratify=dataset.y.iloc[train_val], random_state=protocol["random_seed"] + 1)
        manifest = dict(schema_version=1, fingerprint_version=2, dataset_source="local", dataset_id=dataset.dataset_id,
                        split_id=split_id, dataset_fingerprint=fingerprint, protocol=protocol,
                        protocol_hash=config_hash(protocol), environment=environment(),
                        indices={"train": sorted(train.tolist()), "validation": sorted(val.tolist()),
                                 "test": sorted(held_test.tolist())})
        check_manifest(manifest, dataset)
        atomic_json(path, manifest)
    check_manifest(manifest, dataset)
    return manifest


def check_manifest(manifest, dataset):
    if manifest["dataset_source"] != "local" or manifest["dataset_id"] != dataset.dataset_id:
        raise ValueError("Split dataset identity mismatch")
    if manifest["dataset_fingerprint"] != dataset_fingerprint(dataset):
        raise ValueError("Dataset changed since splitting")
    if manifest["protocol_hash"] != config_hash(manifest["protocol"]):
        raise ValueError("Split protocol hash mismatch")
    parts = manifest["indices"]
    if set(parts) != {"train", "validation", "test"}:
        raise ValueError("Split needs train, validation and test indices")
    flat = [value for name in parts for value in parts[name]]
    if any(not isinstance(value, int) or isinstance(value, bool) for value in flat):
        raise ValueError("Indices must be positional integers")
    if len(flat) != len(set(flat)) or sorted(flat) != list(range(len(dataset.y))):
        raise ValueError("Split indices overlap or do not partition the dataset")
    classes = set(dataset.y.unique())
    if any(not positions or set(dataset.y.iloc[positions].unique()) != classes for positions in parts.values()):
        raise ValueError("Each split portion must contain every classification label; increase dataset size")


def portion(dataset, manifest, name):
    check_manifest(manifest, dataset)
    positions = manifest["indices"][name]
    return dataset.X.iloc[positions].copy(), dataset.y.iloc[positions].copy()
