"""Export approved CSV/metadata/existing holdout indices, then verify the unpacked ZIP."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

import _bootstrap  # noqa: F401
from meta_selector import load_dataset
from meta_selector.artifacts import _atomic, atomic_json, research_config, research_paths
from meta_selector.datasets.pmlb import sha256_file
from meta_selector.datasets.splits import check_manifest
from meta_selector.paths import DEFAULT_PROJECT_ROOT as ROOT, project_paths, read_yaml_mapping
from pmlb_package_verify import PARTS, require


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded_json(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def input_snapshot(specs):
    paths = list((ROOT / "data/PMLB").glob("*.csv"))
    paths += [ROOT / p for p in ("config/pmlb_pilot.yaml", "config/research.yaml", "data/registry/datasets.csv")]
    for spec in specs:
        folder = project_paths(ROOT).raw_data / spec["dataset_id"]
        paths += [folder / f"{spec['original_name']}.csv", folder / "metadata.yaml",
                  research_paths(ROOT)["splits"] / spec["dataset_id"] / "holdout_0.json"]
    return {p.relative_to(ROOT).as_posix(): dict(sha256=sha256_file(p), mtime_ns=p.stat().st_mtime_ns)
            for p in paths}


def collect(specs):
    protocol = research_config(ROOT)["splits"]
    require(protocol == dict(random_seed=42, validation_fraction=0.2, test_fraction=0.2), "Unexpected holdout protocol")
    require(len(specs) == 10 and len({s["dataset_id"] for s in specs}) == 10, "Expected the ten approved tasks")
    payload, rows = {}, []
    for spec in specs:
        dataset_id, original = spec["dataset_id"], f"{spec['original_name']}.csv"
        dataset = load_dataset(dataset_id, project_root=ROOT)
        require(dataset.metadata.target == "target", f"Unexpected target: {dataset_id}")
        require(list(dataset.metadata.categorical_columns) == spec["categorical_columns"], "Categorical typing mismatch")
        require((len(dataset.y), dataset.X.shape[1], dataset.y.nunique()) ==
                (spec["n_rows"], spec["n_features"], spec["n_classes"]), "Dataset shape mismatch")
        manifest_path = research_paths(ROOT)["splits"] / dataset_id / "holdout_0.json"
        manifest_bytes = manifest_path.read_bytes()  # Missing manifest is an error, never generated.
        manifest = json.loads(manifest_bytes)
        check_manifest(manifest, dataset)
        require(manifest["protocol"] == protocol and manifest["split_id"] == "holdout_0", "Split protocol mismatch")
        folder = project_paths(ROOT).raw_data / dataset_id
        csv_bytes = (ROOT / "data/PMLB" / original).read_bytes()
        require(digest(csv_bytes) == spec["sha256"] == dataset.metadata.file.sha256, "Approved CSV SHA-256 mismatch")
        require((folder / dataset.metadata.file.name).read_bytes() == csv_bytes, "Raw/imported CSV mismatch")
        csv_name = f"datasets/{dataset_id}/{original}"
        metadata_name = f"datasets/{dataset_id}/metadata.yaml"
        manifest_name = f"splits/{dataset_id}/holdout_0.json"
        payload[csv_name] = csv_bytes
        payload[metadata_name] = (folder / "metadata.yaml").read_bytes()
        payload[manifest_name] = manifest_bytes
        row = dict(dataset_id=dataset_id, original_csv=original, target_column=dataset.metadata.target,
                   split_id="holdout_0", seed=protocol["random_seed"], second_stage_seed=protocol["random_seed"] + 1,
                   **{f"{part}_size": len(manifest["indices"][part]) for part in PARTS},
                   csv_sha256=spec["sha256"], categorical_columns=json.dumps(spec["categorical_columns"]),
                   n_rows=len(dataset.y), n_features=dataset.X.shape[1], n_classes=int(dataset.y.nunique()),
                   dataset_fingerprint=manifest["dataset_fingerprint"], protocol_hash=manifest["protocol_hash"],
                   csv_path=csv_name, metadata_path=metadata_name, manifest_path=manifest_name)
        for part in PARTS:
            name = f"indices/{dataset_id}/{part}.txt"
            payload[name] = ("\n".join(str(i) for i in manifest["indices"][part]) + "\n").encode("utf-8")
            row[f"{part}_indices_path"] = name
        rows.append(row)
    table = io.StringIO(newline="")
    writer = csv.DictWriter(table, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    payload["datasets.csv"] = table.getvalue().encode("utf-8")
    payload["README.md"] = (ROOT / "docs/PMLB_DIANA_PACKAGE_README.md").read_bytes()
    payload["verify_package.py"] = (ROOT / "scripts/pmlb_package_verify.py").read_bytes()
    payload["SHA256SUMS.json"] = encoded_json(dict(schema_version=1, files={name: digest(data) for name, data in payload.items()}))
    require(len(payload) == 64, "Unexpected package content count")
    return payload, rows


def zip_bytes(payload):
    memory = io.BytesIO()
    with zipfile.ZipFile(memory, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(payload):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, payload[name], compresslevel=9)
    return memory.getvalue()


def export(output=None):
    output = Path(output or ROOT / "results/exports/pmlb_pilot_for_diana.zip").resolve()
    require(output.is_relative_to(ROOT) and output.suffix == ".zip", "ZIP must be inside this project")
    specs = read_yaml_mapping(ROOT / "config/pmlb_pilot.yaml")["datasets"]
    before = input_snapshot(specs)
    payload, rows = collect(specs)
    packed = zip_bytes(payload)
    if output.exists():
        require(output.read_bytes() == packed, "Existing export differs; preserving it")
    else:
        _atomic(output, lambda temporary: temporary.write_bytes(packed))
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)) == 64 and set(names) == set(payload), "Unexpected/duplicate ZIP entries")
        require(archive.testzip() is None, "ZIP CRC check failed")
        for name in names:
            require(archive.read(name) == payload[name], f"ZIP bytes differ: {name}")
        # Extract only this exact allowlisted package, entirely inside the project.
        with tempfile.TemporaryDirectory(prefix=".verify_pmlb_", dir=output.parent) as directory:
            archive.extractall(directory)
            completed = subprocess.run(
                [sys.executable, "-B", str(Path(directory) / "verify_package.py"),
                 "--checksums-sha256", digest(payload["SHA256SUMS.json"])],
                cwd=directory, capture_output=True, text=True, encoding="utf-8")
            require(completed.returncode == 0, f"Unpacked standalone verifier failed: {completed.stdout}\n{completed.stderr}")
            checked = json.loads(completed.stdout)
    require(input_snapshot(specs) == before, "Source CSV/metadata/config/manifests changed")
    checksum = sha256_file(output)
    report = dict(archive=output.relative_to(ROOT).as_posix(), bytes=output.stat().st_size,
                  zip_sha256=checksum, zip_crc_ok=True, sources_unchanged=True, protected_files=len(before),
                  checksums_sha256=digest(payload["SHA256SUMS.json"]),
                  standalone_verification=checked)
    atomic_json(output.with_suffix(".verification.json"), report)
    _atomic(output.with_suffix(".zip.sha256"), lambda temporary: temporary.write_text(
        f"{checksum}  {output.name}\n", encoding="utf-8"))
    return report, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report, rows = export(args.output)
    print(f"ZIP: {report['archive']} ({report['bytes']} bytes)")
    print(f"SHA-256: {report['zip_sha256']}")
    for row in rows:
        print(row["dataset_id"], row["split_id"], row["seed"], row["second_stage_seed"],
              row["train_size"], row["validation_size"], row["test_size"])
    print(f"PASS: {report['standalone_verification']['verified_files']} files; 10 datasets; 30 index files; "
          f"ZIP CRC; hashes; partition/classes; protected sources unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
