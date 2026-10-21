"""Explicit local batch operations. Dataset registration never calls this module."""
import json
from pathlib import Path

import pandas as pd

from . import load_dataset
from .artifacts import IDENTITY, config_hash, environment, research_config, research_paths, upsert_table
from .datasets.registry import read_registry
from .datasets.splits import make_split, portion
from .landmarkers.custom import run_custom_landmarkers
from .meta_features.extraction import extract_controls, extract_pymfe


def local_ids(root=None):
    """Registry is local by design; preserve its existing fixed 14-column schema."""
    return [row["dataset_id"] for row in read_registry(project_root=root)
            if row.get("dataset_source", "local") == "local" and row["status"] == "valid"
            and row["task_type"] == "classification"]


def process_local(kind, dataset_ids, *, root=None, config=None, split_id="holdout_0", include_standard=True):
    if kind not in {"meta_features", "landmarkers"}:
        raise ValueError("Unknown batch operation")
    config = config or research_config(root)
    base = research_paths(root)["derived"]
    statuses = []
    for dataset_id in dataset_ids:
        identity = dict(dataset_source="local", dataset_id=dataset_id, split_id=split_id)
        try:
            dataset = load_dataset(dataset_id, project_root=root)
            manifest = make_split(dataset, root=root, split_id=split_id, protocol=config["splits"])
            X, y = portion(dataset, manifest, "train")
            provenance = dict(dataset_fingerprint=manifest["dataset_fingerprint"], protocol_hash=manifest["protocol_hash"],
                              train_indices=manifest["indices"]["train"], environment=environment(), input_role="outer_training_only")
            operations = ["custom"] if kind == "landmarkers" else ["controls", "paper_features"]
            if kind == "meta_features" and include_standard:
                operations.append("pymfe_standard_landmarking")
            for namespace in operations:
                try:
                    if namespace == "custom":
                        table = run_custom_landmarkers(X, y, identity=identity, config=config["custom_landmarkers"],
                                                       split_provenance=provenance)
                        path, keys = base / "landmarkers/custom_landmarkers.parquet", IDENTITY + ["model_id", "metric", "config_id"]
                    else:
                        table = (extract_controls(X, identity=identity, provenance=provenance) if namespace == "controls"
                                 else extract_pymfe(X, y, identity=identity, config=config[namespace], source=namespace, provenance=provenance))
                        path = base / ("landmarkers/pymfe_landmarkers.parquet" if namespace == "pymfe_standard_landmarking"
                                       else "meta_features/meta_features.parquet")
                        keys = IDENTITY + ["source", "name", "config_id"]
                    upsert_table(path, table, keys)
                    failed = table.status.eq("error").any()
                    statuses.append(dict(**identity, namespace=namespace, status="error" if failed else "ok",
                                         error_message="; ".join(table.error_message[table.status.eq("error")].tolist()), rows=len(table)))
                except Exception as error:
                    statuses.append(dict(**identity, namespace=namespace, status="error", rows=0,
                                         error_message=f"{type(error).__name__}: {error}"))
        except Exception as error:
            statuses.append(dict(**identity, namespace="dataset", status="error", rows=0,
                                 error_message=f"{type(error).__name__}: {error}"))
    if not statuses:
        return pd.DataFrame(columns=IDENTITY + ["namespace", "status", "rows", "error_message"])
    report = pd.DataFrame(statuses)
    report["config_id"] = config_hash(config)
    report["provenance"] = json.dumps(dict(config=config, environment=environment()), sort_keys=True)
    upsert_table(base / kind / "run_status.parquet", report, IDENTITY + ["namespace", "config_id"])
    return report
