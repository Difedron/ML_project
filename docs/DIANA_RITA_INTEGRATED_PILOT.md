# Этап 8. Интеграция пилота Дианы и Риты

Интегрированная таблица содержит **10 датасетов и 256 колонок**. Строка соответствует одному shared split одного датасета. Из неё можно явно выбрать два набора входов для будущего эксперимента. Интеграция сама не запускает обучение, модельный inference, извлечение признаков или LODO.

## Источники

- Экспорт Риты из `results/exports/pmlb_pilot_meta_features/` в `main` на коммите `7ff999d`. Экспорт добавлен в `7d5943c`. Основной вход — `pmlb_pilot_meta_features_217.parquet`, CSV — эквивалентная копия для сверки. README и `manifest.json` фиксируют имена, порядок и контрольные суммы.
- Канонический результат Дианы — `data/derived/candidate_full_pilot/candidate_results.parquet`, этап 7, код из `6e8038b`. Подтверждены 40 записей CandidateResult v2 и `status=ok` для всех. Проверены SHA самого артефакта, 40 точных variant identities из `protocol.json`, `run_summary.json` и provenance каждой записи.
- Candidate protocol ID — `5c33c6db32a0e2832f1851d6c7c7077c4f69848d75deb30060b783202d305495`. SHA256 таблицы Дианы — `66031c5ce8e46fea1a81bb3e15e38f766423d8f61607ef434b5ba938a5b0dba5`.

## Экспорт Риты

**Export artifact integrity — PASS.** Оба формата содержат ровно **10 × 221**: 4 идентификатора и 217 числовых predictors типа `float64`. Сверены полный список и порядок колонок из README и manifest, контрольные суммы обоих файлов и точное равенство значений CSV и Parquet с `float_precision="round_trip"`. Позиции всех 41 пропусков и число пропусков по датасету совпали. `NaN` в CSV соответствует null в Parquet. Экспорт содержит только идентификаторы и признаки, включая оценки landmarkers по train.

| Артефакт | SHA256 |
|---|---|
| `pmlb_pilot_meta_features_217.parquet` | `17372bd6d7b78b77645ee53dc428371dac269766dab391c5c9e9fde53db1842c` |
| `pmlb_pilot_meta_features_217.csv` | `87796426e670d4a89cf7da4a1a4612e79e12e0422db669cd854cf8b3ec039acb` |
| `README.md` | `877ff4f97e8bef5b7b624ca368e4e0038f5e5e6fb35ca051004effdb44c0c709` |

Predictors Риты относятся только к outer train. Группа PyMFE — paper-style reconstruction. Landmark scores — Balanced Accuracy по трём внутренним StratifiedKFold, стандартное отклонение с `ddof=0`.

### CRLF и raw provenance

**Rita raw SHA provenance — KNOWN STALE METADATA due to CRLF restoration. Impact on integration — NONE.** Текущие `data/raw`-копии отличаются от записанных export-time SHA на **10 из 10** датасетов. Каждая текущая копия по байтам совпадает с canonical CSV в `data/PMLB`. Для каждой копии хеш после замены CRLF на LF **в памяти** совпал со старым `raw_copy_sha256`. Это проверка истории файлов, отдельная от integrity экспортированных predictors. Входные raw-файлы, shared manifests и оба экспорта сохранены в исходном виде.

## Датасеты и ключи

`pmlb_saheart`, `pmlb_tic_tac_toe`, `pmlb_kr_vs_kp`, `pmlb_mushroom`, `pmlb_hill_valley_without_noise`, `pmlb_twonorm`, `pmlb_penguins`, `pmlb_car_evaluation`, `pmlb_vehicle`, `pmlb_waveform_21`.

Scientific key — **`dataset_id + split_id + seed`**, у всех `holdout_0` и `42`. До join сверены SHA canonical CSV, dataset fingerprint версии Риты, split protocol hash, SHA самого shared manifest, размеры train и полных таблиц. Для каждой записи Дианы эти данные сверены с исторической идентичностью этапа 7. Актуальные исходники кандидатов в `main` могут отличаться от сохранённого снимка исходников этапа 7.

После pivot для каждого ключа есть ровно одна feature row и четыре валидных candidate variants. Join `one_to_one` дал 10 совпадений, 0 unmatched Rita, 0 unmatched Diana и 0 дублей. В таблице сохранены `dataset_source_rita=local` и `dataset_source_diana=pmlb`. Отдельная `dataset_source_normalized=pmlb` использует документированное соответствие `local → pmlb`.

## Landmarkers и роли колонок

Каждой модели соответствуют два фактических имени из экспорта. Все 100 значений landmarkers присутствуют и конечны.

| Модель | Mean | Std |
|---|---|---|
| Logistic Regression | `feature::custom_landmarking::logistic_regression_balanced_accuracy_mean` | `feature::custom_landmarking::logistic_regression_balanced_accuracy_std` |
| Decision Tree | `feature::custom_landmarking::decision_tree_balanced_accuracy_mean` | `feature::custom_landmarking::decision_tree_balanced_accuracy_std` |
| kNN | `feature::custom_landmarking::knn_balanced_accuracy_mean` | `feature::custom_landmarking::knn_balanced_accuracy_std` |
| Naive Bayes | `feature::custom_landmarking::gaussian_nb_balanced_accuracy_mean` | `feature::custom_landmarking::gaussian_nb_balanced_accuracy_std` |
| LDA | `feature::custom_landmarking::lda_balanced_accuracy_mean` | `feature::custom_landmarking::lda_balanced_accuracy_std` |

| Роль | Колонок | Содержание |
|---|---:|---|
| Identity | 6 | 3 shared keys, 2 исходных source labels и normalized source |
| Meta-only features | 207 | 202 `paper_features` и 5 `controls` |
| Landmarker features | 10 | 5 моделей × mean и std |
| Candidate outcomes | 12 | `ba_*`, `roc_auc_*` и `log_loss_*` для каждого варианта |
| Targets | 9 | 5 полей best и tie, 4 `gap_*` |
| Diagnostics | 12 | `validation_metric_*`, `validation_score_*` и `total_time_sec_*` |

**Meta-only — 207, Meta + Landmarkers — 217.** Списки с исходными именами записаны в JSON. `load_column_roles()` и `ColumnRoles.feature_frame()` разрешают получить X из явных наборов и проверяют полное разбиение схемы. Тесты и независимая сверка подтверждают отсутствие identity, outcome, runtime, target и tie колонок в обоих X. Meta-only исключает все 10 колонок landmarkers.

Candidate variants имеют суффиксы `lightgbm_baseline`, `lightgbm_priortune_t_k25`, `tabpfn_v2_6`, `tabicl_v2`. Для каждого суффикса сохранены результаты из test. Candidate validation metrics хранятся отдельно для диагностики.

## Target, ничьи и gaps

`best_balanced_accuracy` — максимум четырёх test BA. `best_target_candidates` — список всех точных максимумов. `best_target_count` и `best_target_is_tie` отражают размер этого списка. При одном максимуме `best_target` содержит его variant, при ничьей — null. `gap_<variant> = best_balanced_accuracy − ba_<variant>`.

Из восьми датасетов с одним победителем у `tabpfn_v2_6` четыре и у `tabicl_v2` четыре. Две точные ничьи, обе с BA `1.0`:

- `pmlb_mushroom` — `tabpfn_v2_6` и `tabicl_v2`.
- `pmlb_penguins` — `lightgbm_baseline` и `tabicl_v2`.

На Mushroom сохранённый результат PriorTune с `config_id=95`, test BA `0.5` и gap `0.5` сохранён. Аудит этапа 7 подтвердил ожидаемое следствие фиксированного `argmax` при идеальном ранжировании. Это не причина менять outcome или правило выбора.

## Диагностика признаков

В 14 колонках 41 пропуск. `inf` и `-inf` отсутствуют. Есть 3 колонки с одним уникальным непустым значением, 0 all-NaN, 3 колонки с точным дублированием значений другой колонки, 0 нечисловых predictors. Duplicate-by-value учитывает совпадение позиций пропусков. Все 217 колонок и все значения сохранены. Поколоночный Parquet содержит роль, число и долю пропусков, оба знака бесконечности, constant/all-NaN, тип и имя первого точного дубликата.

## Артефакты и воспроизведение

```bash
.venv/bin/python scripts/build_pilot_meta_integration.py
```

Путь — `data/derived/meta_integration/`. В нём `pmlb_pilot_integrated.parquet` с 10 × 256, `manifest.json` со сверенными SHA, provenance, schema и диагностикой, `feature_diagnostics.parquet`, а также `meta_only_features.json` и `meta_plus_landmarkers_features.json`. Parquet — primary truth. Интегрированная таблица перечитана независимо и сравнена с обоими источниками по ключам. Проверены точные maximum и ties, неотрицательность gaps, наличие нуля у каждого максимума и сохранение исходных feature-значений.

## Тесты

```bash
.venv/bin/python -m pytest -q tests/test_pilot_meta_integration.py \
  tests/test_pmlb_meta_features_export.py tests/test_shared_pilot.py \
  tests/test_full_pilot_cli.py tests/test_candidate_api.py tests/test_local_candidate_results.py
```

**201 passed.** Исправлены две существующие тестовые фикстуры. Тесты экспорта теперь берут значения из отслеживаемого Parquet вместо локального кэша с тремя smoke-датасетами. Негативный кейс со словарём в provenance явно использует pandas dtype `object`, чтобы проверка достигала валидатора и на pandas 3. Изменения относятся только к тестам. Полный набор тестов включает этапы исследования за пределами интеграции, поэтому использована целевая команда выше.

Следующий этап — проектирование meta-model experiment и решение tie policy.
