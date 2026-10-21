# Meta Selector — данные и pairwise performance gap

Фундамент совместного исследовательского проекта Дианы и Риты. Сейчас проект
умеет читать, проверять и регистрировать **вручную скачанные локальные датасеты**,
а также импортировать опубликованные результаты TabArena и вычислять gap между
всех шести сравнений Table 3. Добавлены reproducible local splits, PyMFE,
пять custom landmarkers, builder meta-dataset и LODO с обёрткой TabPFNRegressor 2.6.
Реальные candidate/TabPFN-запуски требуют внешнего backend и проверенных весов.

Аудит [статьи arXiv:2605.28418](https://arxiv.org/html/2605.28418v3), официального
TabArena и PyMFE: [PAPER_REPLICATION_AUDIT.md](docs/PAPER_REPLICATION_AUDIT.md).
Это подготовка репликации: точный feature set авторов не подтверждён, а binary
метрика cached results отличается от описания Appendix A. Полной численной
репликацией результаты текущего этапа не являются.

## Быстрый старт

Требуется Python 3.11 или новее. Из корня проекта:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'

python scripts/validate_dataset.py _toy_binary
python scripts/register_dataset.py _toy_binary
python scripts/register_dataset.py --rebuild
python scripts/list_datasets.py
pytest
```

Если окружение уже существует, достаточно активировать его и установить проект.
Небольшой `_toy_binary` хранится в Git; скачивать данные для этих команд не нужно.

## Диана и Рита нашли новый датасет. Что делать?

1. Создать папку `data/raw/my_dataset/`.
2. Положить туда исходный CSV, XLSX или Parquet. Имя файла можно оставить оригинальным.
3. Создать рядом `metadata.yaml`; в `file.name` записать имя файла, а в `target` — **точное имя целевой колонки**.
4. Запустить `python scripts/validate_dataset.py my_dataset`.
5. Запустить `python scripts/register_dataset.py my_dataset`.
6. Проверить список: `python scripts/list_datasets.py`.

Полный пошаговый пример, шаблон метаданных и разбор ошибок:
[docs/DATASETS.md](docs/DATASETS.md).

## Структура

```text
config/                 Общие настройки путей и чтения файлов
data/raw/<dataset_id>/  Исходный файл + metadata.yaml
data/processed/         Зарезервировано для будущих обработанных данных
data/registry/          Восстанавливаемый CSV-индекс
src/meta_selector/      Python-пакет
scripts/                Команды для работы без написания Python-кода
tests/                  Автоматические проверки инфраструктуры
docs/                   Инструкции и описание архитектуры
results/                Производные таблицы gap, provenance и будущие результаты
.cache/tabarena_results/ Проверенные маленькие внешние таблицы (не Git)
```

## Implemented

- Структура локального каталога: один датасет — одна папка.
- Единая схема `metadata.yaml` без автоматического определения target.
- Загрузка CSV, Parquet и XLSX в pandas; результат содержит `X`, `y`, `metadata`, `dataset_id`.
- Проверка метаданных, исходных заголовков, структуры таблицы, классов и пропусков.
- Восстанавливаемый реестр с атомарной записью и четырьмя статусами.
- CLI для проверки, регистрации, пересборки и просмотра.
- Toy-пример, инструкции для совместной работы и тесты.
- Внешний адаптер TabArena: CSV/Parquet, exact identifiers, pinned checksums.
- Выбор варианта по нормализованному validation error; отдельная нормализация
  test errors и одна gap-запись на `(dataset_source, dataset_id)`.
- Canonical mapping всех 21 метода, явные семейства и шесть сравнений.
- Неизменяемые стратифицированные local split manifests с seed и fingerprint.
- Training-only controls и configurable PyMFE paper-style extraction.
- Отдельные standard PyMFE landmarks и пять custom CV-моделей: LR, Tree, kNN, NB, LDA.
- Batch-команды, keyed derived tables и явный builder meta-dataset.
- LODO с training-fold preprocessing, mean/majority baselines, MAE/sign/regret.
- Таблица доступности кандидатов и adapter официального TabArena model registry.
- Обёртка TabPFNRegressor 2.6 с проверкой checkpoint/SHA256/architecture и отказом без весов.

## Planned

- Подтверждение точного исходного набора PyMFE-признаков авторов.
- Загрузка training portions по фиксированным TabArena splits.
- Установка/интеграционная проверка официального candidate backend и точных checkpoints.
- Реальные согласованные E0–E5 и проверка научных гипотез.

Регистрация local-датасета не запускает их и не загружает benchmark или веса.

Подробные команды, конфигурация и границы проверенного backend:
[RESEARCH_PIPELINE.md](docs/RESEARCH_PIPELINE.md).
[Отчёт нового этапа и все 21 модели](docs/STAGE3_REPORT.md).

## Проверка опубликованных результатов TabArena

```bash
# Только две маленькие таблицы результатов; загрузка — явно через --fetch.
python scripts/inspect_tabarena_results.py --fetch

# 21 историческая таблица reference cohort (~1.36 MB суммарно).
python scripts/inspect_tabarena_results.py --reference --fetch

# Работает офлайн после загрузки таблиц.
python scripts/build_performance_gap.py
python scripts/build_performance_gap.py --all
```

Можно совместить последние два шага: `python scripts/build_performance_gap.py --fetch`.
Скрипты не скачивают датасеты, predictions или веса и не запускают модели.
Версии, адреса и MD5 закреплены в `config/tabarena_results.yaml`.

Нормализация требует общего reference cohort, а не только сравниваемой пары.
Наш исторический roster явно помечен как реконструкция; строки с imputed или
неизвестным imputation-статусом исключены с отчётом. Набор авторов не подтверждён.
Результат: `results/tables/tabicl_v2_vs_tabpfn_2_6/{split_gaps.csv,dataset_gaps.csv,provenance.json}`.
Положительный gap означает преимущество TabPFN-2.6, отрицательный — TabICLv2.

Проверенные строки, ограничения и воспроизводимые команды:
[REPLICATION_STAGE_REPORT.md](docs/REPLICATION_STAGE_REPORT.md).
Эти команды не запускают E0. Полный benchmark на 150 датасетах не запускался.

## Python-интерфейс

```python
from meta_selector import load_dataset, validate_dataset

report = validate_dataset("_toy_binary")
print(report.to_text())

dataset = load_dataset("_toy_binary")
print(dataset.dataset_id)
print(dataset.X.shape)
print(dataset.y.name)
print(dataset.metadata.target)
```

При невалидных данных `load_dataset` выдаёт `DatasetValidationError` с отчётом.
Для отдельного каталога можно передать `project_root="/absolute/path/to/project"`.
Raw-файлы не изменяются; категории, NaN и значения target сохраняются.

## Пути и статусы

Пути берутся из `config/project.yaml` относительно расположения проекта, а не
текущей директории терминала. Поэтому можно запускать скрипт по абсолютному пути:

```bash
/path/to/project/.venv/bin/python /path/to/project/scripts/validate_dataset.py _toy_binary
```

Статусы: `valid`, `invalid`, `missing_file`, `invalid_metadata`. Подробности
ошибок находятся в отчёте и `error_message` реестра. Пропуски, категории и
дисбаланс сами по себе не являются ошибкой.

Коды завершения: `0` — успешно; `1` — есть невалидный датасет; `2` — проблема
команды, конфигурации, чтения или записи реестра. При пересборке невалидные
датасеты тоже записываются в индекс; итоговый код при их наличии — `1`.

## Документация

- [Как добавить датасет](docs/DATASETS.md)
- [Архитектура и интерфейсы](docs/ARCHITECTURE.md)
- [Как работать вдвоём](docs/COLLABORATION.md)
- [Аудит статьи, TabArena и PyMFE](docs/PAPER_REPLICATION_AUDIT.md)
- [Результат текущего этапа](docs/REPLICATION_STAGE_REPORT.md)

Старый экспериментальный код удалён по новому заданию. Git-история и настройки
IDE сохранены. Результаты предыдущего прототипа не являются частью этого этапа.
