# Отчёт: аудит и слой performance gap

Исторический отчёт предыдущего milestone (86 тестов). Новый слой описан в
[STAGE3_REPORT.md](STAGE3_REPORT.md); ограничения исходного аудита сохранены.

## Что выполнено

Сохранён ручной каталог `data/raw/<dataset_id>/metadata.yaml`, loader, validator,
реестр и CLI. Изучен полный текст arXiv:2605.28418 v3, включая методологию,
Appendix A/B и predictive evaluation. До реализации исследовательского слоя
создан [PAPER_REPLICATION_AUDIT.md](PAPER_REPLICATION_AUDIT.md) с разделением
подтверждённых фактов, неизвестного и наших предположений.

Из внешнего TabArena получены 21 небольшая опубликованная таблица результатов:
1 361 478 байт суммарно. Все MD5 проверены по официальному pinned manifest.
Код TabArena не вендорится; модели, weights, raw datasets и prediction repositories
не загружались. Архив выборочно прочитанных исходников для аудита лежит в
игнорируемом `.cache/replication_audit/`.

Точные идентификаторы первых кандидатов подтверждены:

| model_id | ta_name | method | suite | строк |
|---|---|---|---|---:|
| tabicl_v2 | TabICLv2 | TABICLV2 (default) | tabarena-2026-02-16 | 816 |
| tabpfn_2_6 | TabPFN-v2.6 | TABPFN-V2.6 (default) | tabarena-2026-03-18 | 816 |

Наличие результатов не подтверждает доступ к локальным весам для будущего
TabPFNRegressor 2.6. Модель другой версии автоматически не подставляется.

## Несколько реальных строк

APSFailure, metric=`roc_auc`; числа ниже — **error**, меньше лучше (не сама AUC):

| model_id | split | validation_error | test_error |
|---|---:|---:|---:|
| tabicl_v2 | 0 | 0.007687740 | 0.004506892 |
| tabicl_v2 | 1 | 0.005961454 | 0.006703254 |
| tabicl_v2 | 2 | 0.007498792 | 0.005393167 |
| tabpfn_2_6 | 0 | 0.009957023 | 0.004894319 |
| tabpfn_2_6 | 1 | 0.007447346 | 0.008343157 |
| tabpfn_2_6 | 2 | 0.008292575 | 0.007283272 |

[CSV с исходной точностью и exact identifiers](TABARENA_REAL_RESULTS_SAMPLE.csv).
[Полный индекс датасетов](TABARENA_DATASET_INVENTORY.csv): 30 binary, 8 multiclass,
13 regression. Датасеты имеют 9 или 30 splits. В каждой таблице пары 816 строк,
но независимых dataset-level observations **51**.

## Что получилось вычислить

Нормализация validation/test выполнена отдельно, выбор представителя использует
только normalized validation error. Нормализованный test gap усредняется по splits.
Проверка на реальных данных дала 816 парных split-записей и 51 dataset-запись.

Результаты в `results/tables/tabicl_v2_vs_tabpfn_2_6/`:

- `dataset_gaps.csv` — одна строка на `(dataset_source, dataset_id)`;
- `split_gaps.csv` — выбранные exact IDs, ошибки, якоря и gap каждого split;
- `provenance.json` — revision, артефакты, checksums, reference roster, исключения,
  SHA256 тестовой и validation reference-таблиц, знак gap и ограничения.

Примеры dataset-level gap: APSFailure = −0.567689; Bioresponse = +0.800742.
Положительное значение означает преимущество TabPFN-2.6, отрицательное — TabICLv2.
Это проверка вычисления целевой величины, не оценка качества метапредиктора.

Reference — явная **реконструкция** исторического roster Table 3: 21 модель,
default/tuned subtypes. Нормализация по одной паре запрещена. Из всех загруженных
строк исключены 7 254 imputed/unknown-imputation; после фильтра subtype осталось
24 732 reference-строки. Покрытие reference — 19–21 модель на split. Подробная
политика исключений находится в аудите; она не заявляется правилом авторов статьи.

## Почему полная репликация пока не заявляется

1. Appendix A описывает classification log-loss; binary cached results используют
   ROC-AUC error. По агрегированной AUC нельзя восстановить log-loss.
2. Полный original PyMFE feature/group/summary manifest и код анализа не найдены.
   Текст подтверждает statistical, information-theoretic и model-based summaries,
   но не позволяет восстановить точный `groups=`.
3. Наличие landmarking в default PyMFE **не доказывает** его использование авторами.
   Ни включение, ни исключение этой группы в original baseline не подтверждено.
4. Не подтверждены точный normalization roster/snapshot, near-constant threshold,
   некоторые определения контролей, агрегация training-split meta-features и
   checkpoint/config метапредиктора. Эти решения нельзя молча угадать.

Следовательно, MAE/sign accuracy из статьи ещё не воспроизведены. Реализованы
формулы, импорт, агрегация, controls и функции baseline/metrics; полный PyMFE
extractor, landmarker fitting, LODO runner и TabPFN fitting не реализованы.

## PyMFE и будущее расширение

В проверенном PyMFE уже есть stumps, LDA, GaussianNB и 1-NN/elite-NN; defaults
включают landmarking. Наши LR, дерево иной глубины и kNN с k>1 могут давать другое
поведенческое описание, но прирост качества и научная новизна ещё не проверены.
Подготовлены имена шести custom-признаков mean/std и описание CV только внутри
benchmark training portion. Для категорий/NaN preprocessing будет fit внутри CV.

A/B/C/D и E0–E5 описаны как будущие серии. Пять original predictive variants
(controls/all/controls+all/robust/controls+robust) выделены отдельно. Global robust
screening авторов отмечен как ограничение; строгий nested вариант требует fit
внутри каждого LODO fold.

## Дополнительные датасеты позднее

Диана и Рита создают `data/raw/my_dataset/`, кладут исходный CSV/XLSX/Parquet и
`metadata.yaml` с обязательным target, затем выполняют:

```bash
python scripts/validate_dataset.py my_dataset
python scripts/register_dataset.py my_dataset
python scripts/list_datasets.py
```

Полный шаблон — [DATASETS.md](DATASETS.md). Регистрация не запускает обучение
и не добавляет dataset в текущую таблицу benchmark. В будущем отдельная команда
зафиксирует splits, metric, preprocessing и candidate results. Ключ local-данных
будет `('local', dataset_id)`, TabArena — `('tabarena', dataset_id)`.

## Проверка и следующие команды

Окружение проверки: Python 3.13.7; numpy 2.5.4; pandas 3.0.6; pyarrow 26.0.0;
PyYAML 6.0.3; openpyxl 3.1.5; pytest 9.1.1. Проверки не требуют сети или ML-пакетов.

```bash
source .venv/bin/activate
python -m pytest -q
python scripts/validate_dataset.py _toy_binary
python scripts/list_datasets.py

# Только точная пара; --fetch нужен при отсутствии локального кэша.
python scripts/inspect_tabarena_results.py --fetch

# Полный объявленный reference roster; только маленькие aggregate tables.
python scripts/inspect_tabarena_results.py --reference --fetch
python scripts/build_performance_gap.py
```

86 tests passed, включая прежние 40 проверок каталога. Новые проверки покрывают
точные версии, schema/CSV, отсутствие имитации scores, checksum, явные exclusions,
validation selection/clipping ties, отдельные anchors, coverage, sign, averaging,
разделение local/tabarena, controls, baselines и metrics. Toy успешно валидируется
и остаётся единственной записью local registry. `git diff --check` проходит.

Следующий milestone: уточнить неизвестные настройки и metric discrepancy,
получить training portions по фиксированным splits, реализовать original features
и проверенный TabPFNRegressor 2.6, затем согласованно выполнить E0 LODO. На текущем
этапе E0, custom landmarkers, массовое обучение и научные сравнения не запускались.
