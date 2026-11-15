# Этап 5. Реальный smoke test на pmlb_saheart

**PASS · 15 ноября 2026 · `main`**

Четыре кандидата завершили реальный запуск с `status=ok`. Таблица показывает
результаты проверки runtime на одном датасете. Для сравнения алгоритмов в статье
требуется отдельный эксперимент.

## Shared split

`pmlb_saheart / holdout_0 / seed=42`, 462 строки, 9 признаков, 2 класса.
Train — 276, validation — 93, test — 93. Во всех процессах использованы
`load_pmlb_dataset` и `load_shared_split`. `SharedSplit.portion` применил позиции
Риты к текущему CSV в исходном порядке.

- CSV SHA256 `4dec0d1a9a62519922240607c51d5edf6f74371d3725186eb87502f7a0f8c6b7`.
- Shared manifest SHA256 `087e8d781b318621d406a8ddb31e0f117c008e7514df1aded1ab01a88a0eff76`.
- Единственный категориальный признак — `Famhist`. У TabPFN он передан как индекс `[4]`.
  Для TabICL признак подготовлен как строковая категория. У обоих LightGBM
  словарь категорий обучен на train.
- Метки `0` и `1` сверены с `estimator.classes_` у всех четырёх моделей.
  Для binary ROC-AUC используется колонка вероятностей положительного класса `1`.

## Environment и preflight

MacBook Pro M3 Pro, macOS 26.6.2 arm64, Python 3.13.7, PyTorch 2.14.1,
TabPFN SDK 9.2.0, TabICL 2.2.0, LightGBM 4.7.0.
Нативный preflight подтвердил MPS и локальные ресурсы. В песочнице PyTorch
определял только CPU. Для итогового запуска использовано то же нативное
окружение, в котором preflight подтвердил MPS.

| Ресурс | Exact local artifact | SHA256 | Preflight |
|---|---|---|---|
| TabPFN-2.6 | `checkpoints/tabpfn/tabpfn-v2.6-classifier-v2.6_default.ckpt` | `0578fa56f97e11024e31735aaec2c4e7332584b7730242fbaf6c0bbd0299206a` | READY |
| TabICLv2 | `checkpoints/tabicl/tabicl-classifier-v2-20260212.ckpt` | `bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0` | READY |
| PriorTune | `portfolios/lgbm/T_K25.json`, `mrsergazinov/PriorTune`, revision `8b9cab4c0d6569ddda5cc2a18d78e0de75c526ee` | `322a4e87966223ec268cdf390ba8d7a58e3b13d2a2a4a7f4dddd3989ab2688e5` | READY |
| LightGBM | Локальная библиотека `4.7.0` | — | READY |

`auto` выбрал MPS для TabPFN и TabICL. Обе обученные модели подтвердили `mps`
через атрибуты SDK. LightGBM baseline и PriorTune выполнены на CPU.
TabPFN использовал фабрику `ModelVersion.V2_6` с явным путём к checkpoint.
TabICL получил явный путь и `allow_auto_download=False`. Все файлы уже имелись
локально. При запуске использовался offline-режим Hugging Face.

## Smoke results

| Family / variant | Status | Device | Validation ROC-AUC | Test balanced accuracy | Test ROC-AUC | Test log-loss | Total, с |
|---|---|---|---:|---:|---:|---:|---:|
| `lightgbm / baseline` | ok | CPU | 0.755123 | 0.500000 | 0.745389 | 0.598410 | 0.080428 |
| `lightgbm / priortune_t_k25` | ok | CPU | 0.804816 | 0.569928 | 0.777664 | 0.574477 | 0.307759 |
| `tabpfn / v2_6` | ok | MPS | 0.783811 | 0.705943 | 0.778689 | 0.526697 | 5.321470 |
| `tabicl / v2` | ok | MPS | 0.798156 | 0.761014 | 0.804816 | 0.513790 | 2.868916 |

| Вариант | Init, с | Fit, с | Validation, с | Tuning, с | Final test, с |
|---|---:|---:|---:|---:|---:|
| LightGBM baseline | 0.031390 | 0.019511 | 0.005738 | 0 | 0.003940 |
| PriorTune T/K=25 | 0.000019 | 0.006329 | 0.001439 | 0.271853 | 0.022619 |
| TabPFN-2.6 | 1.790626 | 0.153116 | 1.767038 | 0 | 1.598572 |
| TabICLv2 | 0.682512 | 0.233870 | 1.114793 | 0 | 0.825132 |

Init, fit и validation у PriorTune относятся к победителю и входят во время
tuning. Total измерен внутри вызова кандидата. Запуск subprocess и запись
Parquet находятся за пределами этого интервала.

### PriorTune и native early stopping

Выполнено 25 успешных конфигураций, ошибок 0. Конфигурация `2428` заняла первое
место при независимой сортировке журнала по validation ROC-AUC по убыванию и
`config_id` по возрастанию. Её validation ROC-AUC — `0.8048155737704918` с
округлением до `0.804816` в таблице. `best_iteration_` у baseline — 7, у
победителя PriorTune — 40, по всем 25 конфигурациям — от 3 до 110. Это подтверждает
работу early stopping при лимите 1 000. Effective parameters всех строк
сверены с закреплённым портфелем после преобразования в параметры sklearn API.
LightGBM 4.7.0 сообщает о депрекации `eval_set`. Этот API успешно выполнил
validation-only early stopping в обоих вариантах LightGBM.

## Test leakage audit

- У всех вариантов `fit` получает train. У LightGBM `eval_set` содержит только
  validation. Категориальные словари и кодирование меток строятся по train.
- У baseline и обеих foundation models логи фиксируют `fit_train`, затем
  `score_validation`, затем `final_test`. Test используется моделью только в
  финальном scoring. Shared loader проверяет целостность и наличие классов во
  всех частях до запуска моделей.
- PriorTune получает только train и validation в `select_on_validation`.
  `run_shared` откладывает извлечение test-порции до вызова финальной оценки.
  В логе ровно 25 успешных validation trials, затем запись о выборе `2428`,
  затем один `final_test`. Победитель уже обучен, его `predict_proba`
  вызывается один раз. В trial table ровно 25 уникальных `config_id` для
  `(pmlb_saheart, holdout_0, 42)`. Test-колонок в ней нет.

## Изменения и исправления

`scripts/candidates/run_saheart_smoke.py` запускает baseline, PriorTune, TabPFN,
затем TabICL. Каждый кандидат работает в отдельном процессе, чтобы разделить
нативные библиотеки и OpenMP runtime. Родитель сохраняет `CandidateResult` в
Parquet перед запуском следующего процесса. stdout и stderr пишутся в отдельный
лог каждого кандидата. Запись с ошибкой процесса хранится в той же схеме.

При первом прогоне `log_loss` предупреждал о сумме вероятностей. Softmax обеих
foundation models возвращал float32, а преобразование во float64 делало заметным
отклонение суммы от единицы в пределах погрешности округления. После проверки
диапазона и исходной суммы строки нормализуются для `log_loss`. ROC-AUC
по-прежнему использует исходные probabilities. Во втором прогоне предупреждение
исчезло. Добавлены регрессионные проверки округления и диагностики несовпадения
устройства SDK. В provenance записываются classes, устройство обученной модели и
стадия ошибки, если она возникла. Для LightGBM записывается `best_iteration`.

## Generated artifacts

- `data/derived/candidate_smoke/candidate_results.parquet` — 4 записи `CandidateResult.to_record()`.
- `data/derived/candidate_smoke/preflight.json` — нативный preflight.
- `data/derived/candidate_smoke/*.log` — 4 итоговых лога с порядком выполнения.
- `data/derived/candidate_smoke/initial_logs/*.log` — 4 лога первого прогона.
- `data/derived/candidate_smoke/cache/` — локальные runtime cache.
- `data/derived/priortune_validation/trials.parquet` — 25 validation trials.

## Regression tests

```bash
.venv/bin/python -m pytest -q \
  tests/test_priortune.py tests/test_candidate_api.py \
  tests/test_shared_pilot.py tests/test_dataset_loading.py
```

**170 passed.** Итоговый нативный прогон завершился с кодом `0`. Независимая
проверка Parquet и логов подтвердила размеры, SHA, версии, device, class order,
отбор по validation, состав таблицы trials и порядок стадий.

## Git

На `main` команда `git pull --rebase origin main` вернула `Already up to date`.
`git diff --stat` показал пять изменённых отслеживаемых файлов, 91 добавление и
4 удаления. Этот отчёт и smoke CLI перечислены в `git status --short` как новые
файлы. `checkpoints/` и `incoming/` были локальными неотслеживаемыми папками
до этапа 5. Результаты и логи из `data/derived/` игнорируются Git. Изменения
остались в рабочей копии без коммита и push.

От Дианы и Риты ничего не требуется. Этап 6, mini-pilot на 2–3 shared datasets,
ожидает отдельной команды.
