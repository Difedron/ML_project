# Архитектура текущего этапа

Локальный каталог читает, проверяет и индексирует вручную найденные датасеты.
Новый независимый слой импортирует агрегированные результаты TabArena и вычисляет
dataset-level performance gap для всех шести сравнений. Configurable feature extraction,
пять custom landmarkers, keyed meta-dataset builder и LODO реализованы; реальные
candidate/meta-predictor запуски требуют внешнего backend/checkpoints.
Методологические источники и предположения разделены в
[аудите](PAPER_REPLICATION_AUDIT.md).

## Основные папки

| Папка | Назначение |
|---|---|
| `config/` | Общие пути, зарезервированный seed и настройки чтения |
| `data/raw/<dataset_id>/` | Неизменяемый исходный файл и его `metadata.yaml` |
| `data/processed/` | Пустое место для будущих нормализованных данных |
| `data/registry/` | Производный восстанавливаемый CSV-индекс |
| `src/meta_selector/` | Устанавливаемый Python-пакет |
| `scripts/` | Тонкие оболочки команд пользователя |
| `tests/` | Проверки поведения на локальных тестовых файлах |
| `docs/` | Инструкции и архитектурные решения |
| `results/tables/`, `figures/`, `logs/` | Производные таблицы и будущие артефакты |
| `.cache/tabarena_results/` | Неизменённые опубликованные aggregate tables и checksums |

## Источник истины и поток данных

```text
data/raw/<id>/metadata.yaml + исходный файл
                ↓
    проверка схемы и настроек чтения
                ↓
    проверка исходных заголовков
                ↓
          pandas DataFrame
                ↓
    проверка target и структуры
        ↙                    ↘
  Dataset(X, y)        ValidationReport
                             ↓
                 производный datasets.csv
```

Loader никогда не использует CSV-реестр для поиска target или файла. Отсутствие
или устаревание реестра не мешает загрузить корректный локальный датасет.

## Интерфейсы

```python
from meta_selector import load_dataset, validate_dataset
from meta_selector.datasets.registry import register_dataset, rebuild_registry, read_registry

dataset = load_dataset("_toy_binary")
report = validate_dataset("_toy_binary")
report = register_dataset("_toy_binary")
reports = rebuild_registry()
rows = read_registry()
```

Везде поддерживается именованный аргумент `project_root` для другого каталога
(в первую очередь изолированных тестов). По умолчанию корень определяется
от расположения исходного пакета; `Path.cwd()` для этого не используется.
Поддерживаемый способ установки — editable (`pip install -e .`) из checkout.
Это локальный репозиторий, не wheel с встроенными пользовательскими данными.

`Dataset`: `dataset_id`, `X: DataFrame`, `y: Series`, `metadata: DatasetMetadata`.
Типизированные метаданные содержат вложенные `source`, `file`, `problem`.
`load_dataset` выполняет те же проверки, что validator, и при ошибке выдаёт
`DatasetValidationError`; подробности доступны как `exception.report`.

`ValidationReport`: статус, метаданные при их доступности, размеры, типы,
число пропусков, списки ошибок и предупреждений. `report.is_valid` даёт bool,
`report.to_text()` — человеческий отчёт. Булевы и остальные нечисловые признаки
в этой диагностике относятся к категориальным; значения и dtype не меняются.

## Распределение ответственности

- `paths.py` читает общую конфигурацию и безопасный YAML с проверкой повторяющихся ключей.
- `datasets/models.py` описывает структуры данных, статусы и исключения.
- `datasets/loader.py` проверяет метаданные и читает исходные форматы.
- `datasets/validator.py` проверяет содержимое и формирует отчёт.
- `datasets/registry.py` формирует строки индекса и выполняет атомарную запись.
- `datasets/cli.py` отвечает за аргументы, вывод и коды завершения.
- `scripts/_bootstrap.py` позволяет вызывать скрипты напрямую и из другой директории.

Форматы соответствуют расширениям `.csv`, `.xlsx`, `.parquet`. `file.name`
содержит только имя файла в папке датасета. Выход через `../`, абсолютный путь
или ссылка на payload вне этой папки отклоняются. ID не может содержать путь.
CSV/XLSX проверяются по исходной первой строке до pandas-нормализации,
Parquet — по именам в схеме. Заголовки не переименовываются.

## Схема реестра

Порядок полей фиксирован:

```text
dataset_id,name,target,task_type,source_type,source_url,external_id,file_name,file_format,status,n_rows,n_features,n_classes,error_message
```

Статусы ограничены: `valid`, `invalid`, `missing_file`, `invalid_metadata`.
Неизвестные значения при ошибке оставляются пустыми. Если метаданные не удалось
проверить, запись сохраняет ID папки, статус и ошибку; поля из непроверенных
метаданных не выдаются за достоверные. Timestamps не записываются.

`register_dataset` обновляет одну строку и сохраняет остальные. `rebuild_registry`
заново обходит все нескрытые папки raw и заменяет весь индекс. Одна проблемная
папка не исключает остальные; в итоговом CSV она представлена своим статусом.
Строки сортируются по ID. Атомарная запись: временный файл рядом с целевым,
полная запись и flush/fsync, затем `os.replace`; при ошибке временный файл удаляется.

Детерминированность означает одинаковый результат при неизменных файлах,
метаданных, конфигурации и версиях reader-библиотек. Текст системных ошибок
может зависеть от ОС; локальные статусы разных участников могут отличаться,
если у них разный набор скачанных payload-файлов.

## Ограничения

Таблица читается целиком в память; большие файлы сначала оценивайте по размеру.
Не реализованы удалённые источники, архивы, многотабличные данные, специальные
строки заголовка, объединение файлов, выбор target и поиск дубликатов датасетов.
Не реализована блокировка параллельных обновлений в одном checkout: запускайте
одну команду регистрации за раз. Для совместной работы через Git реестр
пересобирается после объединения метаданных.

Размеры и пропуски в локальном отчёте — диагностика загрузки. Их нельзя автоматически
использовать вместо признаков benchmark training portions.

## Слой репликации

```text
config/tabarena_results.yaml (revision + URLs + MD5 + reference roster)
             ↓ explicit --fetch
.cache/tabarena_results/*.parquet (original bytes + provenance)
             ↓ schema / identity / imputation checks
canonical split-level result table
             ↓ validation normalization using its own anchors
family representative per dataset/split (validation only)
             ↓ separate test normalization using all declared reference methods
split gap = normalized_test_A − normalized_test_B
             ↓ mean over splits of each dataset
dataset_gaps.csv + split_gaps.csv + provenance.json
```

`tabarena/adapter.py` загружает только маленькие aggregate tables, проверяет MD5,
лимит размера и metadata. Внешний TabArena не вендорится и не импортируется при
обычной работе. `export_official_results` — мост к переданному извне `TabArenaContext`:
`compare(output_dir=..., plot=False, fillna=None)`; создание контекста остаётся у
внешнего окружения TabArena. Полный `compare()` может загрузить широкий roster,
поэтому для текущей пары используется выборочное чтение cached Parquet.

`tabarena/results_loader.py` приводит schema к `dataset_source`, `dataset_id`,
`split_id`, `model_id`, `model_version`, `exact_model_id`, `validation_error`,
`test_error`, `metric`, `problem_type`. Exact ID = `ta_suite::ta_name::method`.
В `model_mapping.py` поколения сопоставляются только явно. Неизвестная версия
сохраняет исходное имя и неизвестную version, а не превращается в TabPFN-2.6.
CSV с отсутствующими полями identity требует явный YAML `--identity-map`:

```yaml
"TABPFN-V2.6 (default)":
  ta_name: TabPFN-v2.6
  ta_suite: tabarena-2026-03-18
  config_type: TABPFN-V2.6
  method_subtype: default
```

CSV обязан содержать `imputed=False`; NaN/Inf ошибок и дубли ключей отклоняются.
Для исторического roster manifest явно задаёт `exclude_imputed_or_unknown`:
адаптер исключает подстановки и неизвестные флаги **до** strict loader и сохраняет
сводку исключений. У двух основных артефактов колонки imputed нет; это исходные
результаты default-модели до leaderboard filling. Исходные файлы не меняются.

`performance_gap/families.py` задаёт точные множества моделей. `normalization.py`
использует все объявленные reference variants, а не только представителей семейства;
проверяет присутствие тех же candidate records и запрещает pair-only anchors.
`gap.py` проверяет одинаковое покрытие A/B, без молчаливого пересечения splits.
Ничьи после validation clipping разрешаются по exact ID; test errors в выборе
не участвуют. Сохраняются обе validation/test ошибки и якоря, reference counts,
SHA256 reference tables и полный manifest. 9 и 30 splits усредняются **внутри**
датасета; каждый датасет остаётся одной meta-observation.

## Граница реализованного и будущего

| Модуль | Сейчас | Позднее |
|---|---|---|
| `meta_features/controls.py` | Пять контролей от X_train | Уточнение исходных определений |
| `meta_features/pymfe_features.py` | Configurable PyMFE extractor | Подтверждение exact original set |
| `feature_sets.py`, `meta_dataset/` | Namespaced features, keyed joins, split aggregation | Научные ablations |
| `landmarkers/` | Пять моделей, train-only CV, separate standard PyMFE | Проверка дополнительного сигнала |
| `meta_model/tabpfn_predictor.py` | Real exact-version wrapper + checkpoints guard | Интеграция реальных локальных весов |
| `meta_model/baselines.py` | Mean / majority внутри LODO | Согласованный эксперимент |
| `meta_model/lodo.py` | Dataset/group isolation и fold-local preprocessing | Оценка на реальном meta-dataset |
| `evaluation/` | MAE/sign/regret, исполняемые E0–E5 definitions | Полный benchmark |
| `candidates/` | Availability и официальный registry executor | Установка backend и реальная проверка |

PyMFE и sklearn входят в зависимости. TabArena/AutoGluon остаются внешним backend;
их отсутствие возвращает unavailable, а не подменяет модель. TabPFN — отдельная
optional dependency; exact regressor checkpoint пока не настроен.

Для будущего local-pipeline составной ключ всегда `(dataset_source='local', dataset_id)`;
для внешнего benchmark — `('tabarena', dataset_id)`. Регистрация local-данных только
проверяет/индексирует файл. Допуск к E1/E4/E5 потребует отдельной команды, фиксированных
splits, metric и candidate results. Candidate scores для научного local benchmark
ещё не получены; автоматически включать local
в текущую таблицу TabArena нельзя. Feature extraction использует только training
portion; meta-preprocessing fit — только training datasets соответствующего LODO fold.

Данные нового слоя: `data/splits/` — JSON manifests; `data/derived/meta_features/`,
`landmarkers/`, `candidate_scores/` — атомарно обновляемые Parquet-таблицы с ключами
source/id/split/config. Подробности и команды — [RESEARCH_PIPELINE.md](RESEARCH_PIPELINE.md).

## Замена предыдущего прототипа

Удалены `src/dataset_router/`, `configs/`, два старых экспериментальных скрипта,
`tests/test_protocol.py`, `docs/protocol.md`, старый `requirements-lock.txt` и
содержимое `results/` предыдущего этапа. `README.md`, `pyproject.toml`, `.gitignore`
заменены описанием нового проекта. Git-история и настройки IDE сохранены.
Существующее локальное `.venv` используется для проверки; тяжёлые ML-библиотеки
не входят в зависимости нового проекта и не импортируются его кодом.
