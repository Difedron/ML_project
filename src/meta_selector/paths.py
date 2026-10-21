"""Resolve paths against the source project, never the current working directory."""
from dataclasses import dataclass
from pathlib import Path

import yaml


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ProjectConfigError(ValueError):
    """The project configuration cannot be used."""


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    raw_data: Path
    processed_data: Path
    registry: Path
    results: Path


def read_yaml_mapping(path: Path) -> dict:
    class UniqueKeyLoader(yaml.SafeLoader):
        pass

    def mapping(loader, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in result:
                raise ValueError(f"Повторяющийся ключ YAML: {key!r}")
            result[key] = loader.construct_object(value_node, deep=deep)
        return result

    UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    with path.open(encoding="utf-8") as stream:
        value = yaml.load(stream, Loader=UniqueKeyLoader)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} должен содержать YAML mapping (поля key: value)")
    return value


def project_paths(project_root: str | Path | None = None) -> ProjectPaths:
    root = Path(project_root).resolve() if project_root is not None else DEFAULT_PROJECT_ROOT
    try:
        configured = read_yaml_mapping(root / "config" / "project.yaml")["paths"]
        resolved = {}
        for key in ("raw_data", "processed_data", "registry", "results"):
            value = configured[key]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"paths.{key} должен быть непустой строкой")
            resolved[key] = (root / value).resolve()
        return ProjectPaths(root=root, **resolved)
    except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError) as exc:
        raise ProjectConfigError(f"Ошибка config/project.yaml: {exc}") from exc


def dataset_defaults(paths: ProjectPaths) -> dict:
    try:
        defaults = read_yaml_mapping(paths.root / "config" / "datasets.yaml")
        for section in ("csv", "xlsx"):
            if section in defaults and not isinstance(defaults[section], dict):
                raise ValueError(f"{section} должен содержать mapping")
        return defaults
    except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError) as exc:
        raise ProjectConfigError(f"Ошибка config/datasets.yaml: {exc}") from exc
