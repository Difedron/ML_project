from pathlib import Path

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from meta_selector.paths import ProjectPaths, project_paths
from .loader import read_frame, read_metadata
from .models import DatasetMetadata, DatasetReadError, DatasetStatus, ValidationReport


def validate_frame(metadata: DatasetMetadata, frame: pd.DataFrame) -> ValidationReport:
    report = ValidationReport(metadata.dataset_id, DatasetStatus.VALID, metadata=metadata,
                              n_rows=len(frame), column_types={str(c): str(t) for c, t in frame.dtypes.items()})
    if frame.columns.has_duplicates:
        report.errors.append("Названия колонок повторяются")
    if frame.empty:
        report.errors.append("Таблица пустая: необходимы строки и столбцы")
    if metadata.target not in frame.columns:
        report.errors.append(f"Target {metadata.target!r} отсутствует в таблице. Доступные колонки: {list(frame.columns)!r}")
    if report.errors:
        report.status = DatasetStatus.INVALID
        return report
    X, y = frame.drop(columns=[metadata.target]), frame[metadata.target]
    report.n_features = X.shape[1]
    report.n_classes = int(y.nunique(dropna=True))
    report.n_numeric_features = sum(is_numeric_dtype(t) and not is_bool_dtype(t) for t in X.dtypes)
    report.n_categorical_features = report.n_features - report.n_numeric_features
    report.n_missing_values = int(frame.isna().sum().sum())
    report.n_missing_target = int(y.isna().sum())
    if report.n_features < 1:
        report.errors.append("Нужен хотя бы один feature помимо target")
    if report.n_classes < 2:
        report.errors.append("Target должен содержать минимум два непустых класса")
    if metadata.problem.type == "binary" and report.n_classes > 2:
        report.errors.append("problem.type=binary, но target содержит более двух классов")
    if metadata.problem.type == "multiclass" and report.n_classes == 2:
        report.errors.append("problem.type=multiclass, но target содержит только два класса; укажите binary")
    if report.n_missing_values:
        report.warnings.append(f"Пропущенных значений: {report.n_missing_values}. Строки и значения сохранены без изменений.")
    if report.n_missing_target:
        report.warnings.append(f"Пропусков в target: {report.n_missing_target}; перед будущим обучением потребуется отдельное решение.")
    if report.errors:
        report.status = DatasetStatus.INVALID
    return report


def _inspect_dataset(dataset_id: str, paths: ProjectPaths) -> tuple[ValidationReport, pd.DataFrame | None]:
    report = ValidationReport(dataset_id, DatasetStatus.INVALID_METADATA)
    try:
        metadata = read_metadata(dataset_id, paths)
        report.metadata = metadata
        frame = read_frame(metadata, paths)
        return validate_frame(metadata, frame), frame
    except DatasetReadError as exc:
        report.status = exc.status
        report.errors.append(str(exc))
    except Exception as exc:
        report.status = DatasetStatus.INVALID
        report.errors.append(f"Ошибка проверки: {type(exc).__name__}: {exc}")
    return report, None


def validate_dataset(dataset_id: str, *, project_root: str | Path | None = None) -> ValidationReport:
    report, _ = _inspect_dataset(dataset_id, project_paths(project_root))
    return report
