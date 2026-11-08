# Приём результатов Дианы: PMLB, четыре финальных варианта

Реализован локальный вход в существующий `scripts/build_performance_gap.py`.
Используются прежние `to_gap_records()`, `calculate_dataset_gaps()` и
`build_meta_dataset()`. Нормализация и LODO не изменены. Классификаторы,
извлечение признаков и meta-learning при подготовке/приёме таблицы не запускаются.

Программный вход в существующем `candidates/results.py`:
`load_pmlb_candidate_results(path, project_root=...)` и
`validate_pmlb_candidate_results(dataframe, project_root=...)` возвращают проверенную
CandidateResult-таблицу для прежнего `to_gap_records()`. Общий bridge сохраняет
совместимость с другими локальными задачами и ранее поддержанными метриками.

Реальных результатов Дианы пока нет. Числовые проверки выполнялись исключительно
на синтетических test fixtures внутри `.cache/`, без подстановки их в данные пилота.

## 1. Точный формат входного файла

Один UTF-8 CSV с разделителем `,` или один Parquet. Схема соответствует
`CandidateResult.to_record()`. В файле **ровно 40 записей**: десять ID из
`config/pmlb_pilot.yaml` × четыре model_id. Порядок строк/столбцов таблицы результатов
не важен; идентичность определяется ключами. Данные исходных датасетов не меняются.

Заголовок CSV (все 15 полей обязательны; nullable-поля также присутствуют):

```csv
dataset_id,split_id,model_id,model_version,exact_model_id,validation_metric,test_metric,dataset_source,validation_score,test_score,fit_time,predict_time,status,error_message,provenance
```

| Поле | Формат и значение |
|---|---|
| `dataset_id` | Точный `pmlb_*` ID из утверждённого пакета; без переименования |
| `split_id` | Только `holdout_0` |
| `model_id` | `tabpfn_2_6`, `tabicl_v2`, `lightgbm`, `lightgbm_priortune_k25` |
| `model_version` | Версия модели, строка или null; SDK/версии библиотек рекомендуется отдельно записать в provenance |
| `exact_model_id` | Непустой стабильный ID конкретного метода/конфигурации. Разным model_id нельзя давать один exact_model_id; одинаковая конфигурация одного метода может повторяться на разных датасетах |
| `validation_metric`, `test_metric` | Обе строки строго `balanced_accuracy`, для всех 40 записей |
| `dataset_source` | Строго `local`, не `pmlb` и не `tabarena` |
| `validation_score`, `test_score` | Исходная Balanced Accuracy финального варианта на соответствующих outer-частях, конечное число в [0,1], без перевода в проценты/ошибку и без округления для отчёта |
| `fit_time`, `predict_time` | Неотрицательные секунды либо null, если время неизвестно. Не подставлять вымышленные нули |
| `status` | Для принятого cohort все строки `ok`. При ошибке передавать явный статус, например `error`/`unavailable`; весь импорт тогда отклоняется с диагностикой |
| `error_message` | Пустая строка/null при `ok`; непустое описание при неуспехе |
| `provenance` | Сериализованная JSON-строка, в том числе в Parquet; обязательные поля ниже |

Если дополнительно передан столбец `imputed`, допустим только явный false/0.
Подставленные или неизвестные результаты не допускаются в reference.
Дополнительные метрики следует хранить отдельно: локальный режим этого пилота
фиксирует одну основную метрику — Balanced Accuracy.

### Обязательный provenance

Ниже показана структура, а не пример измеренных результатов. Все угловые
placeholder-строки нужно заменить значениями из переданного пакета:

```json
{
  "problem_type": "binary",
  "dataset_fingerprint": "<dataset_fingerprint из holdout_0.json>",
  "protocol_hash": "<protocol_hash из holdout_0.json>",
  "csv_sha256": "<csv_sha256 из datasets.csv>",
  "split_manifest_sha256": "<SHA-256 байтов переданного holdout_0.json>",
  "is_final": true,
  "priortune": {"k": 25}
}
```

- `problem_type` — `binary` либо `multiclass` по конкретному датасету.
- `is_final` — именно JSON boolean `true`, не строка `"true"` и не номер trial.
- Объект `priortune` с целым `k: 25` обязателен только для
  `model_id=lightgbm_priortune_k25`. Для остальных методов его можно не передавать.
- `split_manifest_sha256` фиксирует **сами индексы**. Fingerprint датасета и hash
  протокола отдельно недостаточны для проверки идентичности списков строк.
  Нужный SHA-256 уже есть в ZIP-файле `SHA256SUMS.json` для ключа
  `splits/<dataset_id>/holdout_0.json`. Manifest не нужно пересохранять или форматировать.
- Дополнительные поля provenance сохраняются: версии SDK/весов и их hashes,
  гиперпараметры, seed модели, смысл K, prior и правило выбора конфигурации,
  preprocessing, границы измерения fit/tuning time.

Для CSV JSON должен быть корректно экранирован. Проще сформировать строки через
`CandidateResult(..., provenance=...).to_record()` либо вызвать `json.dumps()`
для словаря provenance перед `DataFrame.to_csv(index=False)` / `to_parquet(index=False)`.

## 2. Что считается финальным вариантом PriorTune

Одна запись на датасет для `lightgbm_priortune_k25`, с фактическими validation/test
scores выбранной конфигурации и её exact_model_id. История отдельных испытаний
передаётся отдельным журналом и не подаётся в этот импортёр.

Дубликаты `(dataset_id, split_id, model_id)`, дополнительные trial model_id,
`is_final=false`, другой K и отсутствующий вариант приводят к отказу. Таким
образом, 25 конфигураций не могут незаметно изменить reference median.
Параметры и правило финального выбора нужно записать в provenance; test не должен
использоваться для выбора конфигурации. Импортёр не реализует и не запускает PriorTune.

## 3. Команды после получения реальных результатов

Из корня checkout в проектной `.venv`, с TEMP/TMP и кэшами внутри проекта:

```powershell
# Все шесть попарных gap, без обращения к TabArena:
& ".venv/Scripts/python.exe" -B scripts/build_performance_gap.py --local-scores "data/incoming/diana_candidate_results.csv" --all

# То же с подготовкой meta-dataset из уже рассчитанных PyMFE/controls/custom:
& ".venv/Scripts/python.exe" -B scripts/build_performance_gap.py --local-scores "data/incoming/diana_candidate_results.parquet" --all --build-meta-dataset

# Только эффект PriorTune относительно обычного LightGBM, reference всё равно из четырёх вариантов:
& ".venv/Scripts/python.exe" -B scripts/build_performance_gap.py --local-scores "data/incoming/diana_candidate_results.csv" --comparison lightgbm_vs_lightgbm_priortune_k25
```

Это инструкции будущего запуска: файлы с реальными scores сейчас не создавались.
`--project-root` задаёт другой checkout; `--output-dir` — каталог внутри его
`results/` (по умолчанию `results/tables/pmlb_pilot`). Для сохранения разных поставок
результатов используйте отдельные output dirs. Входной файл не перезаписывается.

Без `--all/--comparison` выбирается `tabicl_v2_vs_tabpfn_2_6`.
Даже для одной пары обязателен полный input 10 × 4: reference одинаков для всех пар.
`--fetch` и TabArena `--config` в локальном режиме запрещены.
Без `--local-scores` прежние TabArena-команды и исторический `--all` сохранены.

## 4. Шесть сравнений и знак gap

| comparison_id | A | B (преимущество при gap > 0) |
|---|---|---|
| `tabicl_v2_vs_tabpfn_2_6` | TabICLv2 | TabPFN-2.6 |
| `lightgbm_vs_tabpfn_2_6` | LightGBM | TabPFN-2.6 |
| `lightgbm_priortune_k25_vs_tabpfn_2_6` | LightGBM + PriorTune K=25 | TabPFN-2.6 |
| `lightgbm_vs_tabicl_v2` | LightGBM | TabICLv2 |
| `lightgbm_priortune_k25_vs_tabicl_v2` | LightGBM + PriorTune K=25 | TabICLv2 |
| `lightgbm_vs_lightgbm_priortune_k25` | LightGBM | LightGBM + PriorTune K=25 |

Существующая формула не меняется: `error=1-score`,
`gap=normalized_error_A-normalized_error_B`.
Reference состоит строго из четырёх финальных записей на dataset/split, включая
обе стороны сравнения. Validation/test anchors считаются отдельно существующим
нормализатором. `raw_error_gap` также сохраняется; для Balanced Accuracy это
`test_score_B-test_score_A`. Clipping нормализованной ошибки в [0,1] может дать
нулевой normalized gap при ненулевом raw gap — это прежний протокол.

Исторические Table-3 families/comparisons и registry исполнителей не расширялись
новыми алгоритмами; четыре варианта поддержаны именно на приёме локальных scores.

## 5. Валидация и поведение при отказе

Проверяются schema, полное покрытие 10 × 4, уникальность final/exact identities,
статусы, оба metric, диапазон и конечность scores. Для всех десяти задач loader
проверяет approved CSV SHA-256/категории; `check_manifest()` — fingerprint данных,
protocol hash, непересечение и покрытие строк, сохранение классов.
Manifest только читается, `make_split()` не вызывается.
Затем проверяется совпадение каждого provenance с этими существующими данными
и точным SHA-256 manifest.

При ошибке записывается `candidate_import.json` со статусом `rejected`, причинами,
числом строк и ошибочными runs. CLI возвращает 1; новые candidate snapshot/gap/meta
не публикуются. Строки с ошибками не удаляются из cohort для продолжения расчёта,
scores не заполняются. Старые результаты другого запуска не удаляются.

Валидная поставка фиксируется целиком в `candidate_results.parquet` (40 строк),
без смешивания с прошлыми exact configs. На нормализацию передаётся только
текущий проверенный input, не общий пул старых результатов.

## 6. Подготовка объединения

`--build-meta-dataset` по умолчанию читает, но не изменяет:

- `data/derived/meta_features/meta_features.parquet`;
- `data/derived/landmarkers/custom_landmarkers.parquet`.

Флаги `--features` и `--custom` позволяют явно выбрать другие существующие Parquet.
Проверяются dataset/split coverage, config IDs, пять успешных custom-моделей,
3-fold Balanced Accuracy, dataset fingerprint, protocol hash и точные outer train
indices в provenance признаков. Builder вызывается с namespaces
`controls/paper_features` и custom Balanced Accuracy; стандартный PyMFE landmarking
не требуется. Его функция и формула агрегации не менялись.

На каждое сравнение создаются отдельные 10 dataset-level строк. Для текущих
реальных features это 217 predictor columns: 207 PyMFE/controls + 10 custom mean/std.
NaN/undefined сохраняются; fit times не становятся predictors. `origin_group`
переносится из pilot config. LODO/обучение не запускаются, fake target gaps не создаются.

## 7. Выходы и проверки реализации

```text
results/tables/pmlb_pilot/
  candidate_import.json
  candidate_results.parquet
  <comparison_id>/
    split_gaps.csv
    dataset_gaps.csv
    provenance.json
    meta_dataset.parquet       # только с --build-meta-dataset
```

В provenance gap сохраняются reference hashes, входной SHA-256, hash pilot config,
comparison ID и правило четырёх финальных методов. Математические функции,
builder и LODO переиспользуются без новых реализаций.

Автотесты `tests/test_local_candidate_results.py` используют десять небольших
**синтетических** датасетов/manifest fixtures и 40 искусственных test scores только
в `.cache/pytest-local-candidates`. Проверяются CSV/Parquet, все шесть пар, численный
результат существующей формулы, четыре reference variants, сборка 217 колонок,
сохранение NaN, ошибки/дубликаты/trials/неверные fingerprints и manifests,
несовместимые features, настоящий CLI subprocess и совместимость старого bridge/CLI.
Никакие classifiers/predictors не fit-ятся.

Команда проверки в `.venv`:

```powershell
& ".venv/Scripts/python.exe" -B -m pytest tests/test_local_candidate_results.py tests/test_performance_gap.py -q --basetemp ".cache/pytest-local-candidates" -o cache_dir=".cache/pytest-cache"
```

Результат проверки: **80 passed in 21.83s**, без предупреждений.
