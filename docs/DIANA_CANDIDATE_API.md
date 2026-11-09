# Диана: этап 3 — единый Candidate API

Дата: 2026-11-09. Результат: **PASS для API и unit tests**.
Этапы 1 и 2 приняты. Реальные модели на этом этапе не запускались; это не отчёт
об их inference readiness или качестве. Этап 4 не начат.

## Реализованный контракт

Расширен существующий boundary `meta_selector.candidates`, сохранены
`CandidateRunner`, `CandidateResult`, исторический TabArena runner и
`artifacts.py`. Нет второго registry, генератора splits или отдельного формата
хранения результатов. Новые wrappers используют официальный classifier API напрямую;
старый runner продолжает обслуживать исторический TabArena pipeline.

| Wrapper | model_id | model_family | variant | Состояние на этапе 3 |
|---|---|---|---|---|
| TabPFN26Candidate | tabpfn_2_6 | tabpfn | v2_6 | API реализован; проверен fake estimator |
| TabICLv2Candidate | tabicl_v2 | tabicl | v2 | API реализован; проверен fake estimator |
| LightGBMBaselineCandidate | lightgbm | lightgbm | baseline | API реализован; проверен fake estimator |
| LightGBMPriorTuneCandidate | lightgbm_priortune_t_k25 | lightgbm | priortune_t_k25 | Контракт зарезервирован; исполнение — этап 4 |

Все wrappers имеют `run(model_id, X_train, y_train, X_validation, y_validation,
X_test, y_test, *, dataset_id, split_id, seed, ...)` в стиле существующего
`CandidateRunner`. Дополнительно передаются `dataset_source="pmlb"`,
`categorical_columns` и необязательный `metric`.
Отсутствующий seed — ошибка; значение по умолчанию не угадывается.

`PilotCandidateRunner` — небольшой dispatcher над этими wrappers.
Его `run_shared(model_id, dataset, split)` принимает объекты этапа 2 и вызывает
`SharedSplit.portion`, сохраняя исходный порядок positional indices и проверку
fingerprint. Он не загружает/генерирует альтернативное разбиение и не вызывает
`make_split`. Ошибка shared identity выходит как ошибка контракта до обучения.

`available_models()` у новых wrappers/dispatcher перечисляет реализованные IDs,
**не выполняет native import или проверку аппаратной готовности**. Сейчас это
три IDs; PriorTune не объявляется исполняемым. Readiness проверяет preflight этапа 1.

Пример будущего авторизованного запуска; на этапе 3 он **не исполнялся**:

```python
from meta_selector.candidates import PilotCandidateRunner
from meta_selector.datasets.shared_pilot import load_pmlb_dataset, load_shared_split

dataset = load_pmlb_dataset("pmlb_saheart")
split = load_shared_split(dataset.dataset_id, dataset.split_id, seed=dataset.seed, dataset=dataset)
result = PilotCandidateRunner().run_shared("tabpfn_2_6", dataset, split)
record = result.to_record()
```

Вызов `run`/`run_shared` для реализованных variants действительно инициирует fit;
это не dry-run. Обычный import и создание wrappers/dispatcher модели не загружают.
Автоматических записей результатов и нового CLI обучения на этапе 3 нет.

## Точные модели и preprocessing

Конфигурация API: `config/candidates/pilot.yaml`. Она ссылается на уже принятые
`preflight.yaml` и `lightgbm_baseline.yaml`; исходные настройки этапа 1 не изменены.
Версии packages зафиксированы: tabpfn 9.2.0, tabicl 2.2.0, lightgbm 4.7.0.
Несовпадение версии возвращает `unavailable_exact_version` до native import.

Пути checkpoints взяты из ручного PASS этапа 1. Перед конструктором проверяются
локальный файл и SHA256 через существующий `verified_weight`. Отсутствующий файл,
неверный SHA или другое поколение не заменяются иной моделью.

- TabPFN вызывает `TabPFNClassifier.create_default_for_version(ModelVersion.V2_6, ...)`
  с явными local `model_path`, device, seed и categorical indices. Default/latest
  classifier не выбирается. Использованы
  [официальный source](https://github.com/PriorLabs/TabPFN/blob/4992bca68cc6cb4db4551b4668d1b95a201b9006/src/tabpfn/classifier.py)
  и установленный source package 9.2.0.
- TabICL получает exact `tabicl-classifier-v2-20260212.ckpt`, local model_path,
  `allow_auto_download=False`, device и seed. Проверено по
  [официальному source](https://github.com/soda-inria/tabicl/blob/d83091b845a8c5f7c087bd54964f755a9e560f45/src/tabicl/_sklearn/classifier.py)
  и установленному package 2.2.0.
- LightGBM baseline использует явные параметры прежнего YAML; четыре baseline
  random seeds берутся из shared seed. Objective и num_class зависят от train labels.
  Early stopping использует только validation, train+validation refit отключён.
  API `eval_set`, `eval_metric`, categorical features и callbacks сверены с
  [официальной документацией 4.7.0](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html).

LabelEncoder обучается только на train. Вероятности переставляются по
`estimator.classes_` в canonical training class order перед scoring;
все части должны содержать все train classes.

Категориальные признаки берутся только из контракта Риты. LightGBM получает pandas
categories со словарём, построенным по train; новые категории validation/test
становятся missing. Числовые пропуски сохраняются. Для neural SDK объявленные
категории передаются как строки, а их native encoders обучаются на train.
Числовой код категории `0` остаётся категорией `"0"`, в частности у mushroom.
Ordinal факторы car_evaluation также получают categorical semantics.
Глобального fit, масштабирования, imputation и удаления строк нет. Входные X/y не меняются.

`auto` для neural models проверяет CUDA → MPS → CPU через arithmetic probe;
TabICL учитывает его официальный guard для MPS на виртуализированном Mac.
Явно недоступный accelerator вызывает ошибку, без незаметного повторного обучения
на CPU. LightGBM — CPU; neural device probe ему не нужен.
Фактический device записывается в результат.

## CandidateResult: совместимая схема 2

Все прежние позиционные аргументы и поля сохранены. Новые поля добавлены после
старых с defaults; старый конструктор продолжает работать. `to_record()` сохраняет
JSON-string provenance, дополнительно сериализует package_versions для Parquet.
Для legacy runner значения fit_time/predict_time переносятся в новые `_sec` поля
при сериализации. Старый score-to-gap consumer проверен без изменения его кода.

| Группа | Поля / смысл |
|---|---|
| Identity | dataset_id, dataset_source, split_id, seed |
| Модель | model_id, model_family, variant, model_version, checkpoint_version, exact_model_id |
| Validation | validation_metric, validation_score; полный набор в provenance.validation_scores |
| Test | test_metric, test_score; balanced_accuracy, roc_auc, log_loss — именно test scores |
| Размеры | n_train, n_validation, n_test, n_features, n_classes |
| Окружение | device, python_version, package_versions |
| Результат | schema_version=2, status, error_type, error_message, provenance |

При `metric=None` validation/test metric — ROC-AUC для binary, log-loss для
multiclass; можно явно выбрать поддерживаемый metric. Это не реализует portfolio
selection. Время измеряется в секундах через perf_counter:

- model_init_time_sec — device/resource preparation и estimator constructor;
- fit_time_sec — train-only preprocessing и model.fit; для baseline также подготовка eval_set;
- validation_time_sec — validation transform, predict/predict_proba и scoring;
- predict_time_sec — аналогичный этап для test;
- tuning_time_sec — 0 для трёх single-model wrappers, NaN у неисполненного PriorTune;
- total_time_sec — весь вызов, включая проверки/versions/прочие накладные расходы.

Неначатые/неуспешно завершённые измерения могут оставаться NaN; fake clock проверяет
структуру timing, а не скорость реальных моделей. На ошибке fit сохраняется время
попытки и не создаются успешные scores. Статусы: ok, error,
unavailable_exact_version, not_implemented. Mocks не являются research results.

Canonical join с Ритой: **dataset_id + split_id + seed**.
Для будущего upsert через существующий `artifacts.upsert_table` объявлен
`CANDIDATE_RESULT_KEYS`: join + model_family + variant + exact_model_id + validation_metric.
Общий `artifacts.IDENTITY` не изменён. Model exact identity зависит от параметров,
версии backend и checkpoint repo/revision/SHA, но не от локального пути весов.
Сам путь хранится отдельно в provenance. Старый local-only gap bridge не используется
для автоматического подключения PMLB результатов к исследованию.

## Scoring и граница с Ритой

`classification_score` перенесён без изменения вычислений в `meta_selector.metrics`.
`landmarkers.custom` импортирует и тем самым реэкспортирует тот же helper:
старый import сохраняется. Его CV, preprocessing и scientific logic не изменены.
Candidate code больше не импортирует landmarkers. Новый `classification_scores`
считает несколько метрик из одного prediction snapshot с проверкой вероятностей.

## PriorTune: только API до этапа 4

LightGBMPriorTuneCandidate имеет model_family=lightgbm, variant=priortune_t_k25.
В provenance записываются опубликованный portfolio reference и правило
validation-only selection. Вызов возвращает **not_implemented / Stage4RequiredError**
до создания estimator, чтения portfolio configs или fitting.

Загрузка portfolio, 25 trials, выбор winner, test только для winner, trial artifacts
и debug execution в этой задаче не реализованы. Они относятся к этапу 4.
Нет TabPrep, ансамбля, synthetic meta-training или генерации новых configs.

## Проверки

Выполнены только unit tests, без реальных target models и скачивания checkpoint:

```bash
.venv/bin/python -m pytest -q \
  tests/test_candidate_api.py tests/test_shared_pilot.py tests/test_dataset_loading.py \
  tests/test_research_pipeline.py::test_candidate_availability_has_no_substitutions \
  tests/test_research_pipeline.py::test_candidate_dependency_generation_cannot_be_silently_upgraded \
  tests/test_research_pipeline.py::test_official_candidate_bridge_fits_train_and_passes_validation
```

53 новых unit cases; 79 regression cases datasets/shared contract; 3 legacy candidate
cases с fake backend. Итог: **135 passed**, без warnings.
Покрыты binary/multiclass, порядок probability columns, identities, классы и target,
доступность версии/ресурса, отсутствие fallback/download, timers, обработка ошибки,
train-only categories, NaN, shared row ordering/mutation guard, seed-aware Parquet
upsert и стабильный exact ID при разных локальных путях.
Отдельный subprocess доказывает, что import и создание dispatcher не импортируют
torch/tabpfn/tabicl/lightgbm/PyMFE/landmarkers. Все estimators в проверках — fake.

Полный pytest не выполнялся: он включает обучение landmarkers/meta-model из других
этапов. Unit PASS не заменяет будущие реальные smoke/device/native-runtime проверки.
Процессная изоляция реальных запусков (важная для macOS OpenMP из preflight) и
benchmark orchestration относятся к последующим этапам.

## Файлы и границы выполненной работы

Созданы:

- src/meta_selector/metrics.py;
- src/meta_selector/candidates/pilot.py;
- src/meta_selector/candidates/devices.py;
- config/candidates/pilot.yaml;
- tests/test_candidate_api.py;
- docs/DIANA_CANDIDATE_API.md.

Изменены:

- src/meta_selector/candidates/runner.py — additive schema и neutral helper import;
- src/meta_selector/candidates/__init__.py — exports API;
- src/meta_selector/landmarkers/custom.py — только перенос/re-export scoring helper.

Данные, shared manifests, registry, stage-1 resources config и baseline config
сохранены. Checkpoints/токены не копировались и не коммитились. Commit/push не выполнялись.
Блокирующих проблем для этапа 3 нет. От Дианы и Риты для завершения этапа 3 ничего не требуется.
**Этап 4 не начат; переход только по отдельной команде пользователя.**
