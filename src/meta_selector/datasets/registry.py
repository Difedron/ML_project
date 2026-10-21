"""A reproducible, derived index. Metadata and raw data remain the source of truth."""
import csv
import os
import tempfile
from pathlib import Path

from meta_selector.paths import project_paths
from .models import ValidationReport
from .validator import validate_dataset


REGISTRY_FIELDS = ["dataset_id", "name", "target", "task_type", "source_type", "source_url",
                   "external_id", "file_name", "file_format", "status", "n_rows", "n_features",
                   "n_classes", "error_message"]


def report_row(report: ValidationReport) -> dict[str, str]:
    metadata = report.metadata
    row = dict.fromkeys(REGISTRY_FIELDS, "")
    row.update(dataset_id=report.dataset_id, status=report.status.value,
               error_message="; ".join(report.errors))
    for name in ("n_rows", "n_features", "n_classes"):
        value = getattr(report, name)
        row[name] = "" if value is None else str(value)
    if metadata:
        row.update(name=metadata.name, target=metadata.target, task_type=metadata.task_type,
                   source_type=metadata.source.type, source_url=metadata.source.url,
                   external_id=metadata.source.external_id or "", file_name=metadata.file.name,
                   file_format=metadata.file.format)
    return row


def read_registry(*, project_root: str | Path | None = None) -> list[dict[str, str]]:
    path = project_paths(project_root).registry
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != REGISTRY_FIELDS:
            raise ValueError("Registry имеет другую структуру. Выполните register_dataset.py --rebuild")
        rows = list(reader)
    if any(set(row) != set(REGISTRY_FIELDS) or None in row.values() for row in rows):
        raise ValueError("Registry повреждён. Выполните register_dataset.py --rebuild")
    return sorted(rows, key=lambda row: row["dataset_id"])


def _write_registry(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=REGISTRY_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda row: row["dataset_id"]))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def register_dataset(dataset_id: str, *, project_root: str | Path | None = None) -> ValidationReport:
    paths = project_paths(project_root)
    report = validate_dataset(dataset_id, project_root=paths.root)
    rows = {row["dataset_id"]: row for row in read_registry(project_root=paths.root)}
    rows[dataset_id] = report_row(report)
    _write_registry(paths.registry, list(rows.values()))
    return report


def rebuild_registry(*, project_root: str | Path | None = None) -> list[ValidationReport]:
    paths = project_paths(project_root)
    reports = []
    if paths.raw_data.exists():
        for folder in sorted(paths.raw_data.iterdir()):
            if folder.is_dir() and not folder.name.startswith("."):
                reports.append(validate_dataset(folder.name, project_root=paths.root))
    _write_registry(paths.registry, [report_row(report) for report in reports])
    return reports
