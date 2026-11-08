"""Standalone Diana package verifier: Python standard library only, no split generation."""
import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys


PARTS = ("train", "validation", "test")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def package_path(root, name):
    path = PurePosixPath(name)
    require(not path.is_absolute() and ".." not in path.parts and "\\" not in name, f"Unsafe path: {name}")
    result = root.joinpath(*path.parts)
    require(result.resolve().is_relative_to(root), f"External path: {name}")
    return result


def check_parts(parts, n_rows):
    require(set(parts) == set(PARTS), "Expected train, validation, test indices")
    flattened = []
    for name in PARTS:
        values = parts[name]
        require(isinstance(values, list) and values, f"Empty/invalid {name} indices")
        require(all(type(v) is int and 0 <= v < n_rows for v in values), f"Invalid zero-based {name} position")
        flattened.extend(values)
    require(len(flattened) == len(set(flattened)), "Duplicate or overlapping positions")
    require(sorted(flattened) == list(range(n_rows)), "Indices do not cover every CSV data row")


def verify(root, expected_checksums_sha256=None):
    root = Path(root).resolve()
    checksum_path = root / "SHA256SUMS.json"
    checksum_digest = sha256(checksum_path)
    if expected_checksums_sha256 is not None:
        require(checksum_digest == expected_checksums_sha256, "SHA256SUMS.json SHA-256 mismatch")
    checksums = json.loads(checksum_path.read_text(encoding="utf-8"))
    require(checksums.get("schema_version") == 1 and isinstance(checksums.get("files"), dict), "Invalid checksums manifest")
    expected = checksums["files"]
    require("SHA256SUMS.json" not in expected, "Checksums manifest cannot hash itself")
    actual = set()
    for path in root.rglob("*"):
        require(not path.is_symlink(), f"Symlink in package: {path}")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    require(actual == set(expected) | {"SHA256SUMS.json"}, "Missing or unexpected package files")
    verified_hashes = {"SHA256SUMS.json": checksum_digest}
    for name, digest in expected.items():
        require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest), f"Invalid SHA-256: {name}")
        observed = sha256(package_path(root, name))
        require(observed == digest, f"SHA-256 mismatch: {name}")
        verified_hashes[name] = observed
    with (root / "datasets.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 10, "Expected exactly ten datasets")
    require(len({r["dataset_id"] for r in rows}) == 10, "Duplicate dataset IDs")
    allowed = {"README.md", "verify_package.py", "datasets.csv", "SHA256SUMS.json"}
    results = []
    binary = 0
    for row in rows:
        dataset_id = row["dataset_id"]
        require(re.fullmatch(r"pmlb_[a-z0-9_]+", dataset_id), f"Invalid dataset ID: {dataset_id}")
        require(row["split_id"] == "holdout_0", f"Unexpected split: {dataset_id}")
        require(int(row["seed"]) == 42 and int(row["second_stage_seed"]) == 43, "Expected seeds 42 / 43")
        original = row["original_csv"]
        require(PurePosixPath(original).name == original and original.endswith(".csv"), "Invalid original CSV name")
        canonical = dict(csv_path=f"datasets/{dataset_id}/{original}",
                         metadata_path=f"datasets/{dataset_id}/metadata.yaml",
                         manifest_path=f"splits/{dataset_id}/holdout_0.json",
                         **{f"{part}_indices_path": f"indices/{dataset_id}/{part}.txt" for part in PARTS})
        for field, name in canonical.items():
            require(row[field] == name, f"Unexpected {field}: {dataset_id}")
            allowed.add(name)
        csv_path = package_path(root, row["csv_path"])
        require(verified_hashes[row["csv_path"]] == row["csv_sha256"], f"CSV/table SHA mismatch: {dataset_id}")
        with csv_path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            header = reader.fieldnames
            records = list(reader)
        require(header and len(header) == len(set(header)), f"Invalid CSV headers: {dataset_id}")
        require(all(None not in r and None not in r.values() for r in records), f"Malformed CSV row: {dataset_id}")
        n_rows = len(records)
        require(n_rows == int(row["n_rows"]), f"Wrong row count: {dataset_id}")
        require(len(header) - 1 == int(row["n_features"]), f"Wrong feature count: {dataset_id}")
        target = row["target_column"]
        require(target in header, f"Target absent: {dataset_id}")
        cats = json.loads(row["categorical_columns"])
        require(isinstance(cats, list) and all(isinstance(c, str) and c in header and c != target for c in cats),
                f"Invalid categorical columns: {dataset_id}")
        require(len(cats) == len(set(cats)), f"Duplicate categorical columns: {dataset_id}")
        labels = [r[target] for r in records]
        require(all(value != "" for value in labels), f"Missing target: {dataset_id}")
        classes = set(labels)
        require(len(classes) == int(row["n_classes"]), f"Wrong class count: {dataset_id}")
        binary += len(classes) == 2
        manifest = json.loads(package_path(root, row["manifest_path"]).read_text(encoding="utf-8"))
        require(manifest["dataset_id"] == dataset_id and manifest["dataset_source"] == "local", "Manifest identity mismatch")
        require(manifest["split_id"] == row["split_id"], "Manifest split mismatch")
        require(manifest["dataset_fingerprint"] == row["dataset_fingerprint"], "Fingerprint field mismatch")
        protocol = manifest["protocol"]
        require(protocol == {"random_seed": 42, "validation_fraction": 0.2, "test_fraction": 0.2}, "Protocol mismatch")
        protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True, allow_nan=False).encode()).hexdigest()
        require(protocol_hash == manifest["protocol_hash"] == row["protocol_hash"], "Protocol hash mismatch")
        parts = manifest["indices"]
        check_parts(parts, n_rows)
        for part in PARTS:
            lines = package_path(root, row[f"{part}_indices_path"]).read_text(encoding="utf-8").splitlines()
            require(all(re.fullmatch(r"0|[1-9][0-9]*", line) for line in lines), f"Invalid {part}.txt")
            positions = [int(line) for line in lines]
            require(positions == parts[part], f"{part}.txt differs from manifest: {dataset_id}")
            require(len(positions) == int(row[f"{part}_size"]), f"Wrong {part} size: {dataset_id}")
            require({labels[i] for i in positions} == classes, f"Class lost in {part}: {dataset_id}")
        results.append(dict(dataset_id=dataset_id, split_id=row["split_id"], seed=42, second_stage_seed=43,
                            sizes={p: len(parts[p]) for p in PARTS}, csv_sha256=row["csv_sha256"],
                            status="ok"))
    require(binary == 6, "Expected six binary and four multiclass datasets")
    require(actual == allowed, "Package contains files outside the data/split contract")
    return dict(status="ok", datasets=results, dataset_count=10, index_files=30,
                verified_files=len(actual), checksums_sha256=checksum_digest,
                all_indices_match_manifests=True, disjoint_and_complete=True,
                all_classes_preserved=True, file_sha256=verified_hashes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--checksums-sha256", help="Optional SHA-256 from the sender's external verification report")
    args = parser.parse_args()
    try:
        result = verify(args.root, args.checksums_sha256)
    except (OSError, ValueError, KeyError, TypeError, csv.Error) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
