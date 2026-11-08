"""Read-only PMLB mapping and Rita split consumption. No split generation or preprocessing."""
from dataclasses import dataclass
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath

import pandas as pd

from meta_selector.artifacts import config_hash, dataset_fingerprint
from meta_selector.paths import DEFAULT_PROJECT_ROOT, read_yaml_mapping
from .loader import _check_headers
from .models import Dataset, DatasetMetadata, FileMetadata, ProblemMetadata, SourceMetadata


JOIN_KEYS = ("dataset_id", "split_id", "seed")
PARTS = ("train", "validation", "test")


class PilotContractError(ValueError):
    pass


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def contained_path(root, relative):
    """Resolve manifest paths without absolute paths, traversal or external symlinks."""
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise PilotContractError("Invalid shared artifact path")
    parts = PurePosixPath(relative)
    if parts.is_absolute() or ".." in parts.parts or any(":" in value for value in parts.parts):
        raise PilotContractError("Unsafe shared artifact path")
    root = Path(root).resolve()
    result = root.joinpath(*parts.parts).resolve()
    if not result.is_relative_to(root):
        raise PilotContractError("Shared artifact path escapes its root")
    return result


def read_pilot_manifest(*, project_root=None, manifest_path=None):
    root = Path(project_root or DEFAULT_PROJECT_ROOT).resolve()
    path = Path(manifest_path) if manifest_path else root / "config/pilot_datasets.yaml"
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        raise PilotContractError(f"shared pilot manifests are required: {path}")
    data = read_yaml_mapping(path)
    if data.get("schema_version") != 1 or data.get("join_keys") != list(JOIN_KEYS):
        raise PilotContractError("Unsupported pilot schema or join keys")
    if data.get("row_positions") != dict(kind="positional", base=0, order="raw_csv_data_rows_after_header_no_filtering"):
        raise PilotContractError("Unknown raw row ordering or positional index basis")
    entries = data.get("datasets", [])
    if len(entries) != 10 or len({item["dataset_id"] for item in entries}) != 10:
        raise PilotContractError("Authoritative pilot must contain exactly ten distinct datasets")
    identities = [(item["dataset_id"], item["split_id"], item["seed"]) for item in entries]
    if len(set(identities)) != len(identities) or any(type(item["seed"]) is not int for item in entries):
        raise PilotContractError("Invalid dataset/split/seed join identities")
    return data


def pilot_entry(dataset_id, *, project_root=None, manifest_path=None):
    data = read_pilot_manifest(project_root=project_root, manifest_path=manifest_path)
    for item in data["datasets"]:
        if item["dataset_id"] == dataset_id:
            return item
    raise PilotContractError(f"Dataset is not in Rita's authoritative pilot: {dataset_id}")


@dataclass
class PMLBDataset(Dataset):
    dataset_source: str
    split_id: str
    seed: int
    categorical_columns: tuple[str, ...]
    fingerprint: str
    rita_fingerprint_present: bool
    rita_dataset_fingerprint: str | None
    rita_dataset_source: str
    origin_group: str


def load_pmlb_dataset(dataset_id, *, project_root=None, manifest_path=None):
    root = Path(project_root or DEFAULT_PROJECT_ROOT).resolve()
    item = pilot_entry(dataset_id, project_root=root, manifest_path=manifest_path)
    if item["dataset_source"] != "pmlb" or item["fingerprint_method"] != "sha256_raw_csv_bytes":
        raise PilotContractError("Expected canonical PMLB source and byte fingerprint")
    path = contained_path(root / "data/PMLB", item["file"])
    if path.name != item["file"] or path.suffix != ".csv" or not path.is_file():
        raise PilotContractError(f"Required PMLB CSV not found: {item['file']}")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != item["fingerprint"]:
        raise PilotContractError(f"Dataset fingerprint mismatch: {dataset_id}; raw CSV bytes/order changed")
    meta = item["rita_metadata"]
    if meta.get("dataset_id") != dataset_id or meta.get("task_type") != "classification" or meta.get("target") != item["target"]:
        raise PilotContractError("Authoritative target/task metadata mismatch")
    options = meta["file"]
    if options["name"] != item["file"] or options["sha256"] != item["fingerprint"]:
        raise PilotContractError("Rita file identity differs from canonical mapping")
    header = next(csv.reader(io.StringIO(payload.decode(options["encoding"]), newline=""), delimiter=options["separator"]), [])
    _check_headers(header)
    if item["target"] not in header:
        raise PilotContractError(f"Missing explicit target: {item['target']}")
    frame = pd.read_csv(io.BytesIO(payload), sep=options["separator"], encoding=options["encoding"])
    if frame.empty or len(frame.columns) < 2:
        raise PilotContractError("Classification requires rows and at least one feature")
    X, y = frame.drop(columns=[item["target"]]), frame[item["target"]]
    if y.isna().any() or y.nunique() < 2:
        raise PilotContractError("Classification requires nonmissing labels and at least two classes")
    if (len(frame), X.shape[1], y.nunique()) != (item["n_rows"], item["n_features"], item["n_classes"]):
        raise PilotContractError("PMLB dimensions/classes disagree with authoritative mapping")
    if meta["problem"]["type"] != ("binary" if y.nunique() == 2 else "multiclass"):
        raise PilotContractError("Classification problem type disagrees with labels")
    cats = item["categorical_columns"]
    if len(cats) != len(set(cats)) or not set(cats) <= set(X):
        raise PilotContractError("Categorical columns must be distinct features, never the target")
    if cats != meta["categorical_columns"]:
        raise PilotContractError("Categorical semantics differ from authoritative metadata")
    metadata = DatasetMetadata(dataset_id, meta["name"], "classification", item["target"],
        SourceMetadata("pmlb", meta["source"]["url"], meta["source"].get("external_id")),
        FileMetadata(item["file"], "csv", options["separator"], options["encoding"]),
        ProblemMetadata(meta["problem"]["type"]), meta.get("notes", ""))
    return PMLBDataset(dataset_id, X, y, metadata, "pmlb", item["split_id"], item["seed"], tuple(cats),
                       item["fingerprint"], item["rita_fingerprint_present"], item.get("rita_dataset_fingerprint"),
                       item["rita_dataset_source"], meta["provenance"]["origin_group"])


def validate_indices(indices, n_rows):
    if not isinstance(indices, dict) or set(indices) != set(PARTS):
        raise PilotContractError("Shared split must contain train/validation/test")
    all_indices = []
    for part in PARTS:
        positions = indices[part]
        if not isinstance(positions, list) or not positions:
            raise PilotContractError(f"Empty or invalid {part} indices")
        if any(type(value) is not int or not 0 <= value < n_rows for value in positions):
            raise PilotContractError(f"Out-of-bounds/noninteger positional indices: {part}")
        if len(positions) != len(set(positions)):
            raise PilotContractError(f"Duplicate indices within {part}")
        all_indices.extend(positions)
    if len(all_indices) != len(set(all_indices)):
        raise PilotContractError("Train/validation/test indices overlap")
    if len(all_indices) != n_rows or set(all_indices) != set(range(n_rows)):
        raise PilotContractError("Shared split does not cover all raw CSV rows")


def class_counts(values):
    return {str(label): int(count) for label, count in values.value_counts(dropna=False).sort_index().items()}


@dataclass(frozen=True)
class SharedSplit:
    dataset_id: str
    split_id: str
    seed: int
    second_stage_seed: int
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    dataset_source: str
    fingerprint: str
    rita_fingerprint_present: bool
    rita_dataset_fingerprint: str | None
    metadata: dict

    @property
    def join_key(self):
        return (self.dataset_id, self.split_id, self.seed)

    def portion(self, dataset, part):
        if (dataset.dataset_id, dataset.split_id, dataset.seed) != self.join_key or dataset.fingerprint != self.fingerprint:
            raise PilotContractError("Dataset/split identity mismatch before positional slicing")
        if part not in PARTS:
            raise PilotContractError("Unknown split portion")
        if dataset_fingerprint(dataset) != self.metadata["reader_dataframe_fingerprint"]:
            raise PilotContractError("Dataset was modified before positional slicing")
        positions = list(getattr(self, part + "_indices"))
        # Preserve the original positional index/order. No filtering, category fitting or sorting.
        return dataset.X.iloc[positions].copy(), dataset.y.iloc[positions].copy()


def load_shared_split(dataset_id, split_id, *, seed=None, project_root=None, manifest_path=None, dataset=None):
    root = Path(project_root or DEFAULT_PROJECT_ROOT).resolve()
    item = pilot_entry(dataset_id, project_root=root, manifest_path=manifest_path)
    if item["split_id"] != split_id or (seed is not None and (type(seed) is not int or seed != item["seed"])):
        raise PilotContractError("Shared split_id/seed does not match the authoritative mapping")
    path = contained_path(root / "config/pilot_splits", item["shared_manifest"])
    if not path.is_file():
        raise PilotContractError(f"shared pilot manifests are required: {path}")
    if file_sha256(path) != item["shared_manifest_sha256"]:
        raise PilotContractError("Shared manifest SHA256 mismatch; Rita's indices must not be rewritten")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (manifest["dataset_id"], manifest["split_id"], manifest["dataset_source"]) != (
            dataset_id, split_id, item["rita_dataset_source"]):
        raise PilotContractError("Rita manifest identity mismatch")
    if type(manifest["protocol"]["random_seed"]) is not int or manifest["protocol"]["random_seed"] != item["seed"]:
        raise PilotContractError("Rita protocol seed mismatch")
    if config_hash(manifest["protocol"]) != manifest["protocol_hash"] or manifest["protocol_hash"] != item["protocol_hash"]:
        raise PilotContractError("Rita protocol hash mismatch")
    if bool(manifest.get("dataset_fingerprint")) != item["rita_fingerprint_present"] or manifest.get("dataset_fingerprint") != item.get("rita_dataset_fingerprint"):
        raise PilotContractError("Rita fingerprint provenance mismatch")
    current = load_pmlb_dataset(dataset_id, project_root=root, manifest_path=manifest_path)
    if dataset is not None:
        if dataset.dataset_id != dataset_id or not dataset.X.equals(current.X) or not dataset.y.equals(current.y):
            raise PilotContractError("Dataset was filtered/reordered/modified before applying Rita's indices")
    indices = manifest["indices"]
    validate_indices(indices, len(current.y))
    classes = set(current.y.unique())
    distributions = {}
    for part in PARTS:
        if len(indices[part]) != item["split_sizes"][part]:
            raise PilotContractError("Split portion size differs from authoritative table")
        labels = current.y.iloc[indices[part]]
        if set(labels.unique()) != classes:
            raise PilotContractError(f"Dataset requires review: missing classification labels in {part}")
        distributions[part] = class_counts(labels)
    return SharedSplit(dataset_id, split_id, item["seed"], item["second_stage_seed"],
        *(tuple(indices[part]) for part in PARTS), "pmlb", item["fingerprint"], item["rita_fingerprint_present"],
        item.get("rita_dataset_fingerprint"), dict(original_manifest=manifest, class_counts=distributions,
            raw_class_counts=class_counts(current.y), shared_manifest_sha256=item["shared_manifest_sha256"],
            reader_dataframe_fingerprint=dataset_fingerprint(current),
            rita_dataset_source=item["rita_dataset_source"], row_order="raw_csv_data_rows_zero_based"))


def validate_pilot(*, project_root=None, manifest_path=None):
    catalog = read_pilot_manifest(project_root=project_root, manifest_path=manifest_path)
    rows = []
    for item in catalog["datasets"]:
        dataset = load_pmlb_dataset(item["dataset_id"], project_root=project_root, manifest_path=manifest_path)
        split = load_shared_split(item["dataset_id"], item["split_id"], seed=item["seed"], project_root=project_root,
                                  manifest_path=manifest_path, dataset=dataset)
        rows.append(dict(dataset_id=dataset.dataset_id, dataset_source="pmlb", csv=item["file"],
                         rows=len(dataset.y), features=dataset.X.shape[1], classes=dataset.y.nunique(),
                         train=len(split.train_indices), validation=len(split.validation_indices), test=len(split.test_indices),
                         seed=split.seed, second_stage_seed=split.second_stage_seed, split_id=split.split_id,
                         fingerprint=dataset.fingerprint, rita_fingerprint_present=dataset.rita_fingerprint_present,
                         class_counts=split.metadata["class_counts"], status="PASS"))
    return rows
