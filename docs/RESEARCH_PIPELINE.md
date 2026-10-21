# Работа с исследовательским pipeline

## Границы текущего этапа

Инфраструктура реализована и проверена на `_toy_binary`, `_smoke_iris`,
`_smoke_mixed`. Исходный каталог, 14-полевая schema реестра и TabArena cached
artifacts сохранены. Группы/сравнения Table 3 теперь заданы полностью, но exact
численная репликация статьи по-прежнему не заявляется: см. [аудит](PAPER_REPLICATION_AUDIT.md).

Новые зависимости: `scikit-learn>=1.5,<2`, `pymfe==0.4.4`. Установка:

```bash
source .venv/bin/activate
python -m pip install -e '.[test]'
```

## Куда Диана и Рита складывают 150 датасетов

**Сюда же: `data/raw/<dataset_id>/`.** Рядом с исходным CSV/XLSX/Parquet лежит
`metadata.yaml`, target задан явно. Нового raw-каталога нет. Правила добавления
не изменились: [DATASETS.md](DATASETS.md).

```text
data/
├── raw/<dataset_id>/             original file + metadata.yaml
├── registry/datasets.csv         производный local индекс
├── processed/                   резерв обработанных данных
├── splits/<dataset_id>/
│   └── holdout_0.json            seed, protocol, fingerprint, train/validation/test indices
└── derived/
    ├── meta_features/
    │   ├── meta_features.parquet controls + paper-style namespaces
    │   └── run_status.parquet    статусы, включая ошибки
    ├── landmarkers/
    │   ├── custom_landmarkers.parquet
    │   ├── pymfe_landmarkers.parquet
    │   └── run_status.parquet
    └── candidate_scores/
        └── candidate_scores.parquet
```

Raw никогда не перезаписывается. `data/splits/`, `processed/`, `derived/`, `.cache/`
и вычисленные `results/` игнорируются Git. Сохраняются code/config/docs, metadata
и `.gitkeep`. Каждый участник пересоздаёт результаты локально. Версии sklearn/pandas
входят в provenance; для побитового повторения нужно совпадающее окружение.
Один checkout использует один writer за раз: блокировки конкурентных Parquet-записей
не реализованы, атомарная замена защищает от частичного файла.

## Splits

`config/research.yaml`: seed=42, train=60%, validation=20%, test=20%; два вызова
stratified train_test_split с seed 42 и 43. Индексы позиционные, части непересекаются
и покрывают все строки. На каждой части должны присутствовать все классы.
Редкие классы и missing target приводят к понятной ошибке; строки не удаляются.
Недостаточно большой dataset не получает нестратифицированный fallback.

Manifest сохраняется при первой feature/landmarker/candidate-команде и затем
переиспользуется. Проверяются fingerprint значений/порядка/dtype, protocol hash,
размеры и отсутствие пересечений. Если данные или protocol изменились, нужен новый
`--split-id`; старый manifest не перезаписывается.

Это **local holdout protocol**, а не воспроизведение многократной TabArena CV.
Официальные исходные 51 datasets используют свои cached split identifiers.

## Meta-features и пять landmarkers

```bash
python scripts/create_smoke_datasets.py
python scripts/extract_meta_features.py _toy_binary _smoke_iris _smoke_mixed
python scripts/run_landmarkers.py _toy_binary _smoke_iris _smoke_mixed
```

Для будущего явного запуска всех зарегистрированных valid local classification datasets:

```bash
python scripts/extract_meta_features.py --all-local
python scripts/run_landmarkers.py --all-local
```

Команды обходят registry. Его schema остаётся local-only, поэтому `dataset_source=local`
добавляется на границе pipeline; `source_type=kaggle/openml/...` означает источник
скачивания и не меняет origin. Ошибка dataset/namespace/model записывается в status,
остальные продолжают обработку; CLI возвращает 1, если есть ошибки. Повторный запуск
upsert-ит те же ключи, другая конфигурация имеет другой `config_id` и хранится отдельно.

Четыре раздельных namespace:

| Namespace | Конфигурация / значения |
|---|---|
| controls | ln(n), ln(d), d/n, categorical fraction, missing-cell fraction |
| paper_features | general/statistical/info-theory/model-based; mean/sd/min/max/median |
| pymfe_standard_landmarking | Только landmarking; mean/sd; accuracy; 3 shuffled folds |
| custom_landmarking | Пять configurable sklearn pipelines; balanced_accuracy; 3 shuffled folds |

`paper_features.confirmed=false`; label всегда **paper-style reconstruction**.
Точный исходный set авторов неизвестен. Infinite values переводятся в NaN в
рабочей training-копии; PyMFE получает median/mode-imputed training data и явные
categorical indices. Это задокументированная реконструкция, не авторский preprocessing.
PyMFE может вернуть mathematically undefined features: они остаются NaN со статусом
`undefined`, а не превращаются в искусственные числа. Импутация meta-features для
предиктора выполняется уже внутри каждого LODO fold.

Стандартные PyMFE landmarks используют собственный алгоритм библиотеки после
preprocessing всего outer training portion; это отличается от fold-local custom
preprocessing. Поэтому их результаты не считаются идентичными даже для NB/LDA.
Оба набора не видят outer validation/test. Никакая научная новизна не заявляется.

| Custom ID | Конфигурация | Scaling |
|---|---|---|
| logistic_regression | C=1, max_iter=1000, random_state=42 | Да |
| decision_tree | max_depth=3, random_state=42 | Нет |
| knn | n_neighbors=5 | Да |
| gaussian_nb | var_smoothing=1e-9 | Нет |
| lda | solver=lsqr, shrinkage=auto | Да |

Список и параметры находятся в `custom_landmarkers` в `config/research.yaml`;
изменение списка не требует переписывать pipeline. Новый тип estimator добавляется
явной регистрацией в `MODEL_CLASSES`.
Внутренняя StratifiedKFold работает только с X_train/y_train. В каждой inner
training части заново fit-ятся median/mode imputation, OHE(ignore unknown) и scaling.
Dense OHE имеет ограничение max_dense_cells. CV не уменьшает число folds/k молча.
Маленькие/вырожденные задачи возвращают error для соответствующей модели.

Custom table сохраняет score mean/std и fit_time mean/std (std ddof=0), metric,
model ID, sklearn version, config, warnings и split provenance. Метрики поддержаны:
balanced_accuracy, macro_f1, roc_auc (binary/OVR multiclass), log_loss. Log-loss
сохраняется положительным и минимизируется; направление записано в provenance.

Meta-feature rows содержат name/value/source/group/extractor_version/config_id,
dataset_source/dataset_id/split_id/status/provenance. Имена в meta-dataset включают
namespace и, для custom, metric; разные scores не смешиваются.

## Все шесть сравнений

```bash
python scripts/build_performance_gap.py --comparison mlp_vs_tree
python scripts/build_performance_gap.py --comparison non_tfm_vs_tfm
python scripts/build_performance_gap.py --comparison tabicl_v2_vs_tabpfn_2_6
python scripts/build_performance_gap.py --comparison realtabpfn_v2_5_vs_tabpfn_2_6
python scripts/build_performance_gap.py --comparison tabpfn_v2_vs_tabpfn_2_6
python scripts/build_performance_gap.py --comparison tabicl_v1_vs_tabicl_v2
python scripts/build_performance_gap.py --all
```

Без flags сохранено прежнее сравнение C3. `--fetch` явно загружает отсутствующие
маленькие aggregate tables. Результат каждого сравнения хранится в `results/tables/<comparison_id>/`.
C1–C3 требуют одинакового покрытия A/B. Для version checks выбран явный
`complete_datasets`: оставляются только целые datasets с результатом обеих сторон
на всех ожидаемых splits, список исключений сохраняется. Это консервативная
реконструкция применимого подмножества, не подтверждённый список авторов.

Artifact IDs/URLs/MD5 не изменены; model_id всех методов приведены к canonical schema
из `model_mapping.py`. Старые производные gap-таблицы пересоздаются командой выше.

## Кандидаты — официальный backend

```bash
python scripts/check_candidate_models.py
# Позднее, явно одна модель / один dataset:
python scripts/run_candidate.py my_dataset --model catboost --metric roc_auc
```

`CandidateRunner` и `CandidateResult` определены в `candidates/runner.py`.
Используется официальный `get_configs_generator_from_name` и default manual config,
затем AutoGluon feature generator/LabelCleaner и model.fit(X, y, X_val, y_val).
Пример API проверен в [официальном файле TabArena](https://github.com/autogluon/tabarena/blob/4b055decdd272314cf44078d98eafc37b105d785/examples/advanced/run_tabarena_model_on_your_data.py).
Feature/label generators fit только на train. Test scores вычисляются после fit;
они не используются для выбора конфигурации. Алгоритмы не реализуются заново.

Backend устанавливается отдельно из закреплённого внешнего TabArena checkout
с совместимым AutoGluon и model extras. Не ставьте произвольный новый registry,
выдавая его за исторический: live class/config/env сохраняются в provenance.
Этот executor реализует single-default local holdout, **не** исторические 200 HPO
configs/8-fold bagging/refit. Для их точной репликации нужно отдельное согласование
официального experiment runner. GPU-варианты не заменяются CPU-версиями.

`config/candidates.yaml`: compute budget, time limit, per-model overrides и
local weights `{path, sha256, parameter}`. Для 2.6/Real2.5 проверена передача
`zip_model_path`; для других checkpoint-моделей parameter должен быть задан явно
по API конкретной версии. Веса должны заранее находиться локально; HF offline
включён при запуске. Availability выводит зависимости, GPU и checkpoint/license
причины. Отсутствие backend — unavailable, ошибка fit — error; scores остаются NaN.
Фактическое обучение с AutoGluon здесь ещё не проверено, так как он не установлен;
контракт интеграции проверен тестовым backend без обучения реальных кандидатов.

`candidates/results.py:to_gap_records` преобразует успешные local scores в ошибки
с учётом направления metric. Source сохраняется local, пропуски/неуспех отклоняются.
Для gap необходим отдельный явно заданный multi-model reference cohort и реальные
scores кандидатов; добавление raw-файла само по себе target gap не создаёт.

## Meta-dataset, TabPFN-2.6 и LODO

`build_meta_dataset` объединяет long feature tables и **одно выбранное сравнение**
по `(dataset_source, dataset_id)`. Первичные features имеют также split_id:
сначала проверяются configuration, duplicates, полнота namespace/features и
покрытие splits, затем training-split values усредняются в одну dataset-запись.
Это предположение реконструкции; по порядку строк ничего не соединяется.
Custom в predictor по умолчанию даёт score mean/std; fit times хранятся для анализа
стоимости, но не включаются как признаки. При нескольких config_id нужно явно
выбрать нужные строки до builder, смешивать их автоматически запрещено.
CLI поддерживает повторяемый `--feature-config source=config_id` и
`--custom-config-id`; custom metric берётся из research config.

`config/research.yaml:meta_predictor` задаёт local checkpoint **регрессора**:
`tabpfn-v2.6-regressor-v2.6_default.ckpt`, SHA256 из доверенного источника,
device и seed. Установленный SDK 9.2.0 поддерживает официальный V2_6 factory;
версия SDK не равна поколению модели. Wrapper вызывает
`create_default_for_version(ModelVersion.V2_6, model_path=...)`, после fit проверяет
тип загруженной architecture config. Без exact checkpoint возвращается
`unavailable_exact_version`, код команды 2; весов нет — обучения нет.
Токены в чат/конфиги Git не записывать; загружать веса через этот pipeline не требуется.

```bash
# Команда для будущего реального E0 после получения benchmark training-features и весов:
python scripts/run_meta_experiment.py --experiment E0 \
  --gaps results/tables/tabicl_v2_vs_tabpfn_2_6/dataset_gaps.csv \
  --features /path/to/tabarena_training_meta_features.parquet \
  --output-dir results/experiments/E0
```

Для E2/E5 добавьте второй `--features .../pymfe_landmarkers.parquet`;
для E3/E4/E5 нужен `--custom .../custom_landmarkers.parquet`. E1/E4/E5 требуют
local candidate gaps в объединённой таблице, а не автоматического подключения raw.
E0/E2/E3 оценивают TabArena; E1/E4/E5 расширяют dataset cohort. При сравнении
качества нужно отдельно отчитать общий исходный cohort: его нельзя напрямую
сравнить со средним на другом наборе задач. Feature manifest и набор datasets
сохраняются в meta_dataset/prediction provenance.
Summary содержит также `by_source`, чтобы отдельно оценивать исходные TabArena задачи.

LODO отвергает несколько строк одного dataset/source. Optional origin_group
удерживает связанные/дублированные datasets вместе. На каждом fold новый predictor,
median imputer и scaler fit только на оставшихся datasets. ID/gap/performance колонки
никогда не становятся features. Mean/majority обучаются на том же training fold.
Метрики: MAE, sign/selection accuracy, pairwise normalized regret; нулевой forecast
для выбора модели/regret выбирает A, а при реальной ничьей любой выбор корректен.
Sign accuracy отдельно сравнивает трёхзначные signs. Это не заменяет regret в исходной metric, для него есть
отдельная direction-aware функция. Robust selection глобально не запускается.

E0–E5 используют один predictor/protocol; они реализованы как исполняемые definitions,
но научные эксперименты с реальным TabPFN ещё не выполнены. Smoke LODO тестирует
только изоляцию с простым тестовым predictor, его MAE не выдаётся за результат статьи.
