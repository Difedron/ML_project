# Этап 4. LightGBM + PriorTune K=25

Вариант `lightgbm / priortune_t_k25` выбирает одну из 25 опубликованных
конфигураций LightGBM. Для всего портфеля используется один shared split Риты.
По завершении настройки на test оценивается уже обученный победитель.

| Поле | Значение |
|---|---|
| HF repository | `mrsergazinov/PriorTune` |
| Revision | `8b9cab4c0d6569ddda5cc2a18d78e0de75c526ee` |
| Artifact | `portfolios/lgbm/T_K25.json` |
| SHA256 | `322a4e87966223ec268cdf390ba8d7a58e3b13d2a2a4a7f4dddd3989ab2688e5` |
| Portfolio ID | `lgbm/T/K25` |
| Local cache | `.cache/diana_preflight/hf/portfolios/lgbm/T_K25.json` |

Загрузчик сверяет repository, revision, artifact и SHA256 с зафиксированным
ресурсом. Он проверяет тип `lgbm`, правило `T`, размер `K=25`, уникальность и
согласованность `config_id` и `add_order`, схему полей и значения настроек.
У всех записей должны быть `tabprep: null` и `tabprep_enabled: false`.
Отсутствующий файл и несовпадение SHA завершают запуск со статусом `error`
и диагностикой `PortfolioError`.

Раннер читает файл локально. Загрузка ресурса вынесена в команду preflight
`python scripts/candidates/check_models.py --fetch-portfolio`. Если локальный
файл повреждён, переместите его из cache перед повторной командой.
[Опубликованный портфель](https://huggingface.co/datasets/mrsergazinov/PriorTune/blob/8b9cab4c0d6569ddda5cc2a18d78e0de75c526ee/portfolios/lgbm/T_K25.json).

## Train, validation и test

1. `PilotCandidateRunner.run_shared` применяет позиции train и validation из
   `SharedSplit`. Кодирование меток и словари категорий строятся по train.
2. Каждая конфигурация в опубликованном `add_order` обучается на train. В
   `eval_set` для early stopping передаётся только validation.
3. Для binary побеждает максимальный validation ROC-AUC. Положительный класс
   имеет код `1` в `LabelEncoder`, обученном на train. Для multiclass побеждает
   минимальный validation log-loss. Колонки вероятностей выравниваются по
   `estimator.classes_`. При точном равенстве выигрывает меньший `config_id`.
4. После 25 успешных попыток раннер запрашивает test у `SharedSplit`. Только
   сохранённый победитель вычисляет вероятности на test, одним вызовом
   `predict_proba`. Из них вычисляются Balanced Accuracy, ROC-AUC и log-loss.

Произвольный `metric=` выбирает только `test_score` для этого варианта.
`validation_metric` всегда соответствует правилу PriorTune. Balanced Accuracy
остаётся метрикой отчёта. Выбор использует одну validation-метрику и сохраняет
уже обученный экземпляр модели для test без refit.

## Преобразование конфигурации в LightGBM

Используется `LGBMClassifier` версии 4.7.0 на CPU, один поток, `deterministic=True`
и `force_col_wise=True`. Опубликованные значения сохраняются. Алиасы
`bagging_fraction`, `bagging_freq`, `feature_fraction`, `lambda_l1`, `lambda_l2`,
`min_data_in_leaf` передаются как `subsample`, `subsample_freq`,
`colsample_bytree`, `reg_alpha`, `reg_lambda`, `min_child_samples`, чтобы значения
не конкурировали с параметрами конструктора. `objective` и `num_class` выводятся
из train labels. Опубликованный `model_seed` идёт в `random_state`. Shared `seed`
идёт в `data_random_seed`, `feature_fraction_seed`, `bagging_seed` и `extra_seed`.
[Параметры LightGBM 4.7.0](https://lightgbm.readthedocs.io/en/v4.7.0/Parameters.html).

Для нашего native runner зафиксированы до 1 000 итераций и остановка после 50
итераций без улучшения первой validation-метрики (`min_delta=0`). В исходном
коде PriorTune используется AutoGluon `LGBModel` с его callbacks. Поэтому
воспроизведение здесь относится к опубликованным конфигурациям и правилу
выбора. Численное совпадение с AutoGluon не проверялось.
[Официальная реализация обучения](https://github.com/yandex-research/PriorTune/blob/104a255b4ad86755638c183e7749d60a64a6d903/bin/models/_runtime/cpu.py).

Список категориальных признаков берётся из метаданных Риты. Объявленные категории
приводятся к строкам и передаются в LightGBM как pandas categorical. Код `0`
у mushroom — категория `"0"`, а ordinal-признаки car_evaluation остаются
категориальными. Новые validation/test категории попадают в missing по
train-словарю. Числовые пропуски передаются LightGBM. Baseline использует свою
существующую конфигурацию и способ подготовки данных.

## Ошибки и артефакты

Политика `require_all_25` требует успешной оценки каждой конфигурации. При
сбое одной попытки остальные продолжаются, чтобы собрать полный журнал.
Любая ошибка fit, включая ошибку конструктора, или невалидная validation-оценка
делает итог `error / IncompletePortfolioError`. В таком случае победителя нет,
test-порция не запрашивается, в provenance остаются счётчики успешных и упавших
попыток. Если упали все 25, применяется та же политика.

`CandidateResult.validation_trials` содержит по строке на каждую начатую
конфигурацию. `run_shared` сохраняет их в
`data/derived/priortune_validation/trials.parquet` через `artifacts.upsert_table`,
включая ошибки. Таблица хранит dataset/split/seed, variant, portfolio identity,
`config_id`, параметры в JSON, validation-метрику и её направление, score,
времена и диагностику. В таблице отсутствуют test-поля. Повтор этой комбинации
обновляет те же строки. Ключ включает `dataset_id + split_id + seed`,
`model_family + variant`, portfolio ID/revision/SHA и `config_id`.
Для записи таблицы действует существующий режим одного писателя на checkout.

`CandidateResult.to_record()` сохраняет схему v2. В `provenance.priortune`
лежат `selected_config_id`, метрика и score победителя, направление, число
конфигураций, число успешных и failed trials, portfolio identity, правило
tie-break, параметры победителя и training policy. При `run_shared` записываются
fingerprint split и путь к таблице trials. `exact_model_id` определяется
идентификатором варианта, портфелем, training policy и версией LightGBM. ID
победителя хранится отдельно, поэтому повторный run с тем же протоколом
совместим с существующим `CANDIDATE_RESULT_KEYS`. Низкоуровневый `run` возвращает
trial rows в памяти. `run_shared` — вход для запуска с записью таблицы.

## Время

- `tuning_time_sec` измеряет подготовку train/validation и все 25 попыток с
  выбором победителя.
- `model_init_time_sec`, `fit_time_sec` и `validation_time_sec` в итоговом
  результате относятся к победителю. Они входят в `tuning_time_sec`. Для
  отдельных конфигураций эти времена и `total_trial_time_sec` лежат в таблице.
- `predict_time_sec` измеряет получение test-порции, её transform, вычисление
  вероятностей и расчёт test metrics.
- `total_time_sec` — время вызова `Candidate.run` целиком, без суммирования
  вложенных интервалов. Оно включает чтение портфеля. Получение train/validation
  у `run_shared` и запись Parquet происходят за пределами этого интервала.

## Проверки этапа

Проверен локальный опубликованный JSON. Код обучения протестирован с fake
LightGBM estimators. Покрыты обе метрики, `config_id` tie-break, перестановка
классов, train-only categories, все 25 попыток, ошибки, задержка чтения test,
повторный upsert и совместимость с baseline и shared splits.

```bash
.venv/bin/python -m pytest -q \
  tests/test_priortune.py tests/test_candidate_api.py \
  tests/test_shared_pilot.py tests/test_dataset_loading.py
```

Первый запуск с реальным LightGBM, проверка native early stopping и измерение
качества на одном shared PMLB dataset относятся к этапу 5.
