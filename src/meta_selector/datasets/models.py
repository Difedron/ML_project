from dataclasses import dataclass, field
from enum import StrEnum

import pandas as pd


class DatasetStatus(StrEnum):
    VALID = "valid"
    INVALID = "invalid"
    MISSING_FILE = "missing_file"
    INVALID_METADATA = "invalid_metadata"


@dataclass(frozen=True)
class SourceMetadata:
    type: str
    url: str
    external_id: str | None = None


@dataclass(frozen=True)
class FileMetadata:
    name: str
    format: str
    separator: str = ","
    encoding: str = "utf-8"
    sheet_name: str | int | None = None


@dataclass(frozen=True)
class ProblemMetadata:
    type: str


@dataclass(frozen=True)
class DatasetMetadata:
    dataset_id: str
    name: str
    task_type: str
    target: str
    source: SourceMetadata
    file: FileMetadata
    problem: ProblemMetadata
    notes: str = ""


@dataclass
class Dataset:
    dataset_id: str
    X: pd.DataFrame
    y: pd.Series
    metadata: DatasetMetadata


@dataclass
class ValidationReport:
    dataset_id: str
    status: DatasetStatus
    metadata: DatasetMetadata | None = None
    n_rows: int | None = None
    n_features: int | None = None
    n_classes: int | None = None
    n_numeric_features: int | None = None
    n_categorical_features: int | None = None
    n_missing_values: int | None = None
    n_missing_target: int | None = None
    column_types: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.status == DatasetStatus.VALID

    def to_text(self) -> str:
        def number(value):
            return "—" if value is None else f"{value:,}".replace(",", " ")

        target = self.metadata.target if self.metadata else "—"
        missing = "—" if self.n_missing_values is None else ("yes" if self.n_missing_values else "no")
        lines = [f"Dataset: {self.dataset_id}", "", f"Status: {self.status.upper()}", "",
                 f"Rows: {number(self.n_rows)}", f"Features: {number(self.n_features)}",
                 f"Target: {target}", f"Classes: {number(self.n_classes)}",
                 f"Numeric features: {number(self.n_numeric_features)}",
                 f"Categorical features: {number(self.n_categorical_features)}", f"Missing values: {missing}"]
        if self.column_types:
            lines += ["", "Column types:"] + [f"  {name}: {dtype}" for name, dtype in self.column_types.items()]
        for label, messages in [("Warnings", self.warnings), ("Errors", self.errors)]:
            if messages:
                lines += ["", label + ":"] + [f"  - {message}" for message in messages]
        lines += ["", "Dataset is ready for further processing." if self.is_valid else
                  "Dataset needs attention. Исправьте ошибки и повторите проверку."]
        return "\n".join(lines)


class DatasetReadError(ValueError):
    def __init__(self, status: DatasetStatus, message: str):
        self.status = status
        super().__init__(message)


class DatasetValidationError(ValueError):
    def __init__(self, report: ValidationReport):
        self.report = report
        super().__init__(report.to_text())
