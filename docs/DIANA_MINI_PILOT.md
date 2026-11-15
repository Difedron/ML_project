# Этап 6. Mini-pilot на трёх shared datasets

**PASS · 15 ноября 2026 · `main`**

Завершились 12 из 12 реальных запусков и 75 из 75 validation trials PriorTune.
Данные ниже предназначены для проверки runtime и протокола на трёх датасетах.

## Environment и preflight

MacBook Pro M3 Pro, macOS 26.6.2 arm64, Python 3.13.7, PyTorch 2.14.1,
TabPFN SDK 9.2.0, TabICL 2.2.0, LightGBM 4.7.0. Один нативный preflight
подтвердил MPS, два exact checkpoints и опубликованный портфель. Кандидаты
выполнялись с локальными ресурсами и offline-настройками Hugging Face.

| Ресурс | Закреплённый файл | SHA256 | Preflight |
|---|---|---|---|
| TabPFN-2.6 | `tabpfn-v2.6-classifier-v2.6_default.ckpt` | `0578fa56f97e11024e31735aaec2c4e7332584b7730242fbaf6c0bbd0299206a` | READY |
| TabICLv2 | `tabicl-classifier-v2-20260212.ckpt` | `bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0` | READY |
| PriorTune T/K=25 | `mrsergazinov/PriorTune` / `8b9cab4c0d6569ddda5cc2a18d78e0de75c526ee` / `portfolios/lgbm/T_K25.json` | `322a4e87966223ec268cdf390ba8d7a58e3b13d2a2a4a7f4dddd3989ab2688e5` | READY |
| LightGBM | Локальный `LGBMClassifier` 4.7.0 | — | READY |

## Datasets

Все три используют shared `holdout_0`, seed `42`. CLI проверил размеры всех
трёх через `load_pmlb_dataset` и `load_shared_split` до запуска первой модели.
Каждый worker повторил проверку. Во всех частях строки совпали с исходной
таблицей при выборке по позициям манифеста в исходном порядке.

| Dataset | Задача | Строки | Признаки | Классы | Train | Validation | Test | Категории Риты |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `pmlb_tic_tac_toe` | Binary | 958 | 9 | 2 | 574 | 192 | 192 | `t1`–`t9` |
| `pmlb_penguins` | Multiclass | 333 | 7 | 3 | 199 | 67 | 67 | `island`, `sex` |
| `pmlb_hill_valley_without_noise` | Binary | 1212 | 100 | 2 | 726 | 243 | 243 | — |

На tic-tac-toe все 9 признаков переданы как категориальные по metadata. На
Hill Valley артефакт подтвердил 100 входных признаков у каждого кандидата.
Обёртки передают все столбцы после shared slicing, сохраняя их порядок и
числовые пропуски. Оба нейросетевых кандидата закончили inference на MPS.
Ошибок памяти или аварий процессов не наблюдалось. Пиковая память отдельно
не измерялась.

## Candidate results

| Dataset | Family / variant | Status | Device | Validation metric | Val score | Test balanced accuracy | Test ROC-AUC | Test log-loss | Total, с |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| `pmlb_tic_tac_toe` | `lightgbm / baseline` | ok | CPU | ROC-AUC | 0.999278 | 0.988537 | 0.999045 | 0.065471 | 0.100050 |
| `pmlb_tic_tac_toe` | `lightgbm / priortune_t_k25` | ok | CPU | ROC-AUC | 0.998797 | 0.969612 | 0.998090 | 0.105825 | 1.398159 |
| `pmlb_tic_tac_toe` | `tabpfn / v2_6` | ok | MPS | ROC-AUC | 0.999639 | 0.992537 | 1.000000 | 0.022029 | 18.469738 |
| `pmlb_tic_tac_toe` | `tabicl / v2` | ok | MPS | ROC-AUC | 0.999278 | 0.985075 | 0.999881 | 0.020989 | 4.761455 |
| `pmlb_penguins` | `lightgbm / baseline` | ok | CPU | Log-loss | 0.097323 | 1.000000 | 1.000000 | 0.010212 | 0.077156 |
| `pmlb_penguins` | `lightgbm / priortune_t_k25` | ok | CPU | Log-loss | 0.020079 | 0.988506 | 1.000000 | 0.019954 | 2.216093 |
| `pmlb_penguins` | `tabpfn / v2_6` | ok | MPS | Log-loss | 0.016210 | 0.988506 | 1.000000 | 0.037632 | 16.135097 |
| `pmlb_penguins` | `tabicl / v2` | ok | MPS | Log-loss | 0.014003 | 1.000000 | 1.000000 | 0.010013 | 2.853425 |
| `pmlb_hill_valley_without_noise` | `lightgbm / baseline` | ok | CPU | ROC-AUC | 0.568225 | 0.546850 | 0.570257 | 0.744713 | 0.327790 |
| `pmlb_hill_valley_without_noise` | `lightgbm / priortune_t_k25` | ok | CPU | ROC-AUC | 0.610467 | 0.525203 | 0.517141 | 0.699643 | 5.904743 |
| `pmlb_hill_valley_without_noise` | `tabpfn / v2_6` | ok | MPS | ROC-AUC | 0.998306 | 0.979268 | 0.997696 | 0.072693 | 35.903428 |
| `pmlb_hill_valley_without_noise` | `tabicl / v2` | ok | MPS | ROC-AUC | 0.986518 | 0.942785 | 0.989533 | 0.130920 | 22.151628 |

Шесть LightGBM runs выполнились на CPU. Во всех шести нейросетевых запусках
обученный estimator подтвердил устройство `mps`, зафиксированное в provenance
и логах. Общий wall-clock — **126.582 с**, включая загрузку shared данных,
старт subprocess, расчёты и сохранение Parquet. Preflight измерялся отдельно.
`total_time_sec` каждой строки берётся из `CandidateResult`.

## PriorTune

| Dataset | Selection metric | Направление | Победитель | Validation score | Успешно | Ошибки | Best iteration | Tuning, с |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `pmlb_tic_tac_toe` | ROC-AUC | max | 926 | 0.998797499 | 25 | 0 | 504 | 1.374360 |
| `pmlb_penguins` | Log-loss | min | 3921 | 0.020078943 | 25 | 0 | 999 | 2.193550 |
| `pmlb_hill_valley_without_noise` | ROC-AUC | max | 1487 | 0.610467480 | 25 | 0 | 172 | 5.881897 |

Независимая сортировка сохранённых validation score подтвердила все выбранные
ID. При равенстве используется меньший `config_id`. Все 75 наборов effective
parameters совпали с параметрами из закреплённого портфеля после преобразования
в LightGBM API. Для Penguins некоторые конфигурации достигли установленного
предела 1 000 итераций.

### Multiclass check

На `pmlb_penguins` все три train-класса `[0, 1, 2]` присутствуют во всех
частях, а канонический порядок совпал с `estimator.classes_` четырёх моделей.
Общий код выравнивает колонки `predict_proba` перед расчётом. PriorTune выбрал
`3921` по минимальному validation log-loss `0.020078942531799936` среди 25
попыток. В итоговой таблице сохранены test Balanced Accuracy и multiclass
ROC-AUC macro one-vs-rest. Для multiclass ROC-AUC используются вероятности всех
трёх классов.

## Leakage audit

- В 12 результатах сверены dataset fingerprint, SHA shared manifest, split,
  seed, train/validation/test размеры, classes и список категорий. Словарь
  категорий и `LabelEncoder` обучались по train. Fit у LightGBM использовал
  train и передавал только validation в `eval_set`.
- В девяти логах без PriorTune порядок стадий — `fit_train`, `score_validation`,
  `final_test`. Test участвовал только в финальном scoring.
- В каждом из трёх логов PriorTune последовательно записаны 25 validation
  trials в опубликованном `add_order`, затем winner, затем один `final_test`.
  `run_shared` запрашивает test-порцию после selection. На test вызывается
  только обученный winner, без refit.
- В общей trial table проверены ровно 75 уникальных строк по трём датасетам
  mini-pilot. Ранее сохранённые строки SAheart исключены из этого подсчёта.
  Trial table содержит только validation и времена попыток.

## Execution, ошибки и исправления

Общий helper запускает каждый кандидат в отдельном процессе последовательно.
Он сохраняет `CandidateResult.to_record()` перед следующим запуском. Сбой
subprocess отмечается ошибкой для его canonical identity и `error_stage` в
provenance. Полный запуск вернул код `0`. Багов нативного выполнения не обнаружено.
В логах LightGBM 4.7.0 сохранилось предупреждение о депрекации `eval_set` из
этапа 5. Ошибок, требовавших правки model pipeline, не было.

```bash
.venv/bin/python scripts/candidates/run_mini_pilot.py --all
```

Для отдельного повторного запуска CLI принимает `--dataset` и `--model`.
Существующая запись обновляется по `dataset_id + split_id + seed + model_family +
variant`. Это проверено тестом на 12 временных строках с упавшим worker и
повтором одной записи. Старый `run_saheart_smoke.py` продолжает использовать
SAheart и свой отдельный output path. Совместимость проверена через `--help`
и тест с fake subprocess.

## Artifacts

- `data/derived/candidate_mini_pilot/candidate_results.parquet` — 12 строк v2.
- `data/derived/candidate_mini_pilot/preflight.json` — нативный preflight.
- `data/derived/candidate_mini_pilot/run_summary.json` — 12 статусов и wall-clock.
- `data/derived/candidate_mini_pilot/<dataset_id>/*.log` — 12 отдельных логов.
- `data/derived/candidate_mini_pilot/<dataset_id>/cache/` — runtime cache процессов.
- `data/derived/priortune_validation/trials.parquet` — общая таблица с 75 новыми
  строками mini-pilot.

## Regression tests и Git

```bash
.venv/bin/python -m pytest -q \
  tests/test_mini_pilot_cli.py tests/test_priortune.py tests/test_candidate_api.py \
  tests/test_shared_pilot.py tests/test_dataset_loading.py
```

**177 passed**. Семь новых проверок покрывают ограничение scope, проверку
размеров до первого worker, порядок запуска, отказ процесса, повтор записи
без дублей и совместимость CLI этапа 5.

`main` синхронизирована с `origin/main`, базовый commit `4b0b31b`. Добавлены
`_isolated_candidates.py`, `run_mini_pilot.py`, `test_mini_pilot_cli.py` и этот
отчёт. `run_saheart_smoke.py` использует общий helper. Generated artifacts
игнорируются Git. Изменения остались в рабочей копии без коммита и push.

От Дианы и Риты ничего не требуется. Этап 7, полный pilot на 10 shared
PMLB datasets, ожидает отдельной команды.
