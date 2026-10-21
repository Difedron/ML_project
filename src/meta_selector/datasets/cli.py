import argparse
import sys
from pathlib import Path

from .registry import read_registry, rebuild_registry, register_dataset
from .validator import validate_dataset


def _parser(description: str):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--project-root", type=Path, help="Другой корень проекта (обычно не требуется)")
    return parser


def _guard(function):
    try:
        return function()
    except (OSError, ValueError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2


def validate_main(argv=None) -> int:
    parser = _parser("Проверить локальный датасет без изменения исходных файлов")
    parser.add_argument("dataset_id")
    args = parser.parse_args(argv)

    def execute():
        report = validate_dataset(args.dataset_id, project_root=args.project_root)
        print(report.to_text())
        return 0 if report.is_valid else 1

    return _guard(execute)


def register_main(argv=None) -> int:
    parser = _parser("Проверить датасет и обновить производный CSV-реестр")
    parser.add_argument("dataset_id", nargs="?")
    parser.add_argument("--rebuild", action="store_true", help="Пересобрать весь реестр из data/raw")
    args = parser.parse_args(argv)
    if bool(args.dataset_id) == args.rebuild:
        parser.error("Укажите либо dataset_id, либо --rebuild")

    def execute():
        if args.rebuild:
            reports = rebuild_registry(project_root=args.project_root)
            valid = sum(report.is_valid for report in reports)
            for report in reports:
                print(f"{report.dataset_id}: {report.status}")
                for error in report.errors:
                    print(f"  {error}")
            print(f"Registry rebuilt: {len(reports)} dataset(s), {valid} valid, {len(reports) - valid} need attention.")
            return 0 if valid == len(reports) else 1
        report = register_dataset(args.dataset_id, project_root=args.project_root)
        print(report.to_text())
        print("\nRegistry updated.")
        return 0 if report.is_valid else 1

    return _guard(execute)


def list_main(argv=None) -> int:
    parser = _parser("Посмотреть локальный реестр без чтения или изменения raw-файлов")
    args = parser.parse_args(argv)

    def execute():
        rows = read_registry(project_root=args.project_root)
        if not rows:
            print("Registry is empty or missing. Выполните python scripts/register_dataset.py --rebuild")
            return 0
        fields = ["dataset_id", "status", "n_rows", "n_features", "target"]
        headers = ["ID", "Status", "Rows", "Features", "Target"]
        values = [headers] + [[row[field] or "—" for field in fields] for row in rows]
        widths = [max(len(row[index]) for row in values) for index in range(len(fields))]
        for row in values:
            print("  ".join(value.ljust(width) for value, width in zip(row, widths)).rstrip())
        print("\nЭто сохранённый индекс. После изменения данных обновите регистрацию или выполните --rebuild.")
        return 0

    return _guard(execute)
