"""Atomic derived tables with explicit keys; raw datasets are never written here."""
import hashlib
import importlib.metadata
import json
import os
import platform
import tempfile
from pathlib import Path

import pandas as pd

from .paths import DEFAULT_PROJECT_ROOT, read_yaml_mapping

IDENTITY = ["dataset_source", "dataset_id", "split_id"]


def research_config(root=None):
    return read_yaml_mapping(Path(root or DEFAULT_PROJECT_ROOT) / "config/research.yaml")


def research_paths(root=None):
    root = Path(root or DEFAULT_PROJECT_ROOT).resolve()
    configured = read_yaml_mapping(root / "config/project.yaml")["paths"]
    return {key: (root / configured.get(key, f"data/{key}")).resolve() for key in ("splits", "derived")}


def config_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def environment():
    result = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "scikit-learn", "pymfe", "tabpfn", "tabarena", "autogluon.tabular"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def atomic_json(path, value):
    _atomic(Path(path), lambda temporary: temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"))


def _atomic(path, writer):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(handle)
    temporary = Path(name)
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def upsert_table(path, frame, keys):
    """One writer per checkout; distinct config_id values coexist rather than overwrite."""
    path = Path(path)
    if frame.empty or not set(keys) <= set(frame) or frame[keys].isna().any().any():
        raise ValueError("Derived records need nonempty, explicit keys")
    if frame.duplicated(keys).any():
        raise ValueError("Duplicate derived keys")
    if path.exists():
        existing = pd.read_parquet(path)
        if existing.duplicated(keys).any():
            raise ValueError("Existing derived table has duplicate keys")
        existing = existing.merge(frame[keys].assign(_replace=True), on=keys, how="left")
        existing = existing[existing._replace.isna()].drop(columns="_replace")
        frame = pd.concat([existing, frame], ignore_index=True)
    frame = frame.sort_values(keys).reset_index(drop=True)
    _atomic(path, lambda temporary: frame.to_parquet(temporary, index=False))


def dataset_fingerprint(dataset):
    """Fingerprint content, order, column names and dtypes; pandas version is in provenance."""
    table = dataset.X
    digest = hashlib.sha256(pd.util.hash_pandas_object(table, index=False).to_numpy().tobytes())
    digest.update(pd.util.hash_pandas_object(dataset.y, index=False).to_numpy().tobytes())
    digest.update(str(dataset.y.dtype).encode())
    digest.update(json.dumps([(str(c), str(t)) for c, t in zip(table.columns, table.dtypes)]).encode())
    return digest.hexdigest()
