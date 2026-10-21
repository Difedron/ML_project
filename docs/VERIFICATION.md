# Проверка реализации

Ниже фактический вывод команд текущего этапа. Модели и эксперименты не запускались.

## .venv/bin/python -m pytest -q

```text
........................................                                 [100%]
40 passed in 2.63s
```

Код завершения: 0.

## .venv/bin/python scripts/validate_dataset.py _toy_binary

```text
Dataset: _toy_binary

Status: VALID

Rows: 30
Features: 3
Target: target
Classes: 2
Numeric features: 2
Categorical features: 1
Missing values: yes

Column types:
  age: int64
  income: float64
  city: str
  target: int64

Warnings:
  - Пропущенных значений: 2. Строки и значения сохранены без изменений.

Dataset is ready for further processing.
```

Код завершения: 0.

## .venv/bin/python scripts/register_dataset.py _toy_binary

```text
Dataset: _toy_binary

Status: VALID

Rows: 30
Features: 3
Target: target
Classes: 2
Numeric features: 2
Categorical features: 1
Missing values: yes

Column types:
  age: int64
  income: float64
  city: str
  target: int64

Warnings:
  - Пропущенных значений: 2. Строки и значения сохранены без изменений.

Dataset is ready for further processing.

Registry updated.
```

Код завершения: 0.

## .venv/bin/python scripts/register_dataset.py --rebuild

```text
_toy_binary: valid
Registry rebuilt: 1 dataset(s), 1 valid, 0 need attention.
```

Код завершения: 0.

## .venv/bin/python scripts/list_datasets.py

```text
ID           Status  Rows  Features  Target
_toy_binary  valid   30    3         target

Это сохранённый индекс. После изменения данных обновите регистрацию или выполните --rebuild.
```

Код завершения: 0.

## Неизменность raw

SHA-256 исходного CSV и metadata.yaml до и после всех команд совпадают.

## Фактическая проверка .gitignore

```text
data/raw/adult/original.csv: IGNORED
data/raw/adult/data.xlsx: IGNORED
data/raw/adult/anything.parquet: IGNORED
data/raw/adult/metadata.yaml: NOT IGNORED
data/raw/_toy_binary/data.csv: NOT IGNORED
data/raw/_toy_binary/metadata.yaml: NOT IGNORED
data/registry/datasets.csv: NOT IGNORED
```

## Итоговое дерево

Служебные Git/IDE-файлы, виртуальное окружение и кеши не показаны.

```text
VAK_ML/
├── .gitignore
├── README.md
├── config/
│   ├── datasets.yaml
│   └── project.yaml
├── data/
│   ├── processed/
│   │   └── .gitkeep
│   ├── raw/
│   │   └── _toy_binary/
│   │       ├── data.csv
│   │       └── metadata.yaml
│   └── registry/
│       └── datasets.csv
├── docs/
│   ├── ARCHITECTURE.md
│   ├── COLLABORATION.md
│   ├── DATASETS.md
│   └── VERIFICATION.md
├── pyproject.toml
├── results/
│   ├── figures/
│   │   └── .gitkeep
│   ├── logs/
│   │   └── .gitkeep
│   └── tables/
│       └── .gitkeep
├── scripts/
│   ├── _bootstrap.py
│   ├── list_datasets.py
│   ├── register_dataset.py
│   └── validate_dataset.py
├── src/
│   └── meta_selector/
│       ├── __init__.py
│       ├── candidates/
│       │   └── __init__.py
│       ├── datasets/
│       │   ├── __init__.py
│       │   ├── cli.py
│       │   ├── loader.py
│       │   ├── models.py
│       │   ├── registry.py
│       │   └── validator.py
│       ├── landmarkers/
│       │   └── __init__.py
│       ├── meta_features/
│       │   └── __init__.py
│       ├── meta_model/
│       │   └── __init__.py
│       └── paths.py
└── tests/
    └── test_dataset_loading.py
```

## Созданные файлы

```text
config/datasets.yaml
config/project.yaml
data/processed/.gitkeep
data/raw/_toy_binary/data.csv
data/raw/_toy_binary/metadata.yaml
data/registry/datasets.csv
docs/ARCHITECTURE.md
docs/COLLABORATION.md
docs/DATASETS.md
docs/VERIFICATION.md
results/figures/.gitkeep
results/logs/.gitkeep
results/tables/.gitkeep
scripts/_bootstrap.py
scripts/list_datasets.py
scripts/register_dataset.py
scripts/validate_dataset.py
src/meta_selector/__init__.py
src/meta_selector/candidates/__init__.py
src/meta_selector/datasets/__init__.py
src/meta_selector/datasets/cli.py
src/meta_selector/datasets/loader.py
src/meta_selector/datasets/models.py
src/meta_selector/datasets/registry.py
src/meta_selector/datasets/validator.py
src/meta_selector/landmarkers/__init__.py
src/meta_selector/meta_features/__init__.py
src/meta_selector/meta_model/__init__.py
src/meta_selector/paths.py
tests/test_dataset_loading.py
```

## Изменённые файлы

```text
.gitignore
README.md
pyproject.toml
```

## Удалённые файлы предыдущего проекта

```text
configs/local.example.json
configs/pilot.json
configs/smoke.json
configs/tabarena.json
docs/protocol.md
requirements-lock.txt
results/model_access.json
results/model_weights.json
scripts/analyze_completed.py
scripts/check_models.py
src/dataset_router/__init__.py
src/dataset_router/__main__.py
src/dataset_router/cli.py
src/dataset_router/data.py
src/dataset_router/experiment.py
src/dataset_router/features.py
src/dataset_router/meta.py
src/dataset_router/models.py
tests/test_protocol.py
```

Дополнительно удалены игнорируемые артефакты предыдущих экспериментов из results/ и старый .cache/.

Git-история, .idea и локальное .venv сохранены. Старый editable-пакет dataset-router удалён из окружения; установлен meta-selector.
