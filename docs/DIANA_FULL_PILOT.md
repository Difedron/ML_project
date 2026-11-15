# Этап 7. Полный real pilot на 10 shared PMLB datasets

**PASS · 15 ноября 2026 · `main`**

Из 40 ожидаемых candidate runs успешны **40**, ошибок **0**. Из 250 validation trials PriorTune успешны **250**, ошибок **0**, отсутствуют **0**. Wall-clock расчёта — **700.962 с** (11.68 мин), включая проверку данных, subprocess и запись candidate rows. Нативный preflight и финальный аудит измерялись отдельно.

## Environment и exact resources

MacBook Pro M3 Pro, macOS 26.6.2 arm64, Python 3.13.7, PyTorch 2.14.1, TabPFN SDK 9.2.0, TabICL SDK 2.2.0. Нативный preflight прошёл один раз. Все ресурсы были доступны локально. Foundation models запускались с `auto`, остальные на CPU. Фактические устройства по CandidateResult: CPU — 20, MPS — 20.

| Ресурс | Зафиксированный файл / версия | SHA256 |
| --- | --- | --- |
| TabPFN-2.6 | `tabpfn-v2.6-classifier-v2.6_default.ckpt` | `0578fa56f97e11024e31735aaec2c4e7332584b7730242fbaf6c0bbd0299206a` |
| TabICLv2 | `tabicl-classifier-v2-20260212.ckpt` | `bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0` |
| PriorTune T/K=25 | `mrsergazinov/PriorTune` / `8b9cab4c0d6569ddda5cc2a18d78e0de75c526ee` / `portfolios/lgbm/T_K25.json` | `322a4e87966223ec268cdf390ba8d7a58e3b13d2a2a4a7f4dddd3989ab2688e5` |
| LightGBM | `4.7.0` | — |

## Datasets

Все размеры сверены до первого запуска через `load_pmlb_dataset` и `load_shared_split`. Использованы `holdout_0`, seed `42` и исходные позиции Риты. CSV и shared manifest SHA сохранены в provenance.

| Dataset | Строки | Признаки | Классы | Train | Validation | Test |
| --- | --- | --- | --- | --- | --- | --- |
| `pmlb_saheart` | 462 | 9 | 2 | 276 | 93 | 93 |
| `pmlb_tic_tac_toe` | 958 | 9 | 2 | 574 | 192 | 192 |
| `pmlb_kr_vs_kp` | 3196 | 36 | 2 | 1917 | 639 | 640 |
| `pmlb_mushroom` | 8124 | 22 | 2 | 4874 | 1625 | 1625 |
| `pmlb_hill_valley_without_noise` | 1212 | 100 | 2 | 726 | 243 | 243 |
| `pmlb_twonorm` | 7400 | 20 | 2 | 4440 | 1480 | 1480 |
| `pmlb_penguins` | 333 | 7 | 3 | 199 | 67 | 67 |
| `pmlb_car_evaluation` | 1728 | 6 | 4 | 1036 | 346 | 346 |
| `pmlb_vehicle` | 846 | 18 | 4 | 507 | 169 | 170 |
| `pmlb_waveform_21` | 5000 | 21 | 3 | 3000 | 1000 | 1000 |

На mushroom код `stalk-root=0` сохранён у 1 497 train-строк во всех четырёх ветках подготовки. Baseline использует числовую pandas-категорию `0`, остальные преобразуют её в строку `"0"`. Все шесть ordinal-факторов car_evaluation переданы как категории. Проверка использовала train-порции после shared slicing.

## Candidate execution

Все 10 датасетов заново запущены последовательно. На каждом использован порядок LightGBM baseline → PriorTune → TabPFN → TabICL. Каждый worker изолирован в отдельном subprocess, а CandidateResult сохраняется атомарно перед запуском следующего. Исторические candidate tables этапов 5–6 остались отдельно. В архиве `stage5_stage6_trials.parquet` сохранены ранее существовавшие строки trials до обновления канонической таблицы под теми же ключами. `reused_candidate_rows=0` для этого запуска.

| Dataset | Кандидат | Статус | Device | Validation metric | Validation score | Test BA | Test ROC-AUC | Test log-loss | Total, с |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `pmlb_saheart` | LightGBM baseline | ok | CPU | roc_auc | 0.755123 | 0.500000 | 0.745389 | 0.598410 | 0.056406 |
| `pmlb_saheart` | PriorTune T/K=25 | ok | CPU | roc_auc | 0.804816 | 0.569928 | 0.777664 | 0.574477 | 0.295442 |
| `pmlb_saheart` | TabPFN-2.6 | ok | MPS | roc_auc | 0.783811 | 0.705943 | 0.778689 | 0.526697 | 16.902537 |
| `pmlb_saheart` | TabICLv2 | ok | MPS | roc_auc | 0.798156 | 0.761014 | 0.804816 | 0.513790 | 2.802094 |
| `pmlb_tic_tac_toe` | LightGBM baseline | ok | CPU | roc_auc | 0.999278 | 0.988537 | 0.999045 | 0.065471 | 0.091606 |
| `pmlb_tic_tac_toe` | PriorTune T/K=25 | ok | CPU | roc_auc | 0.998797 | 0.969612 | 0.998090 | 0.105825 | 1.412134 |
| `pmlb_tic_tac_toe` | TabPFN-2.6 | ok | MPS | roc_auc | 0.999639 | 0.992537 | 1.000000 | 0.022029 | 18.163391 |
| `pmlb_tic_tac_toe` | TabICLv2 | ok | MPS | roc_auc | 0.999278 | 0.985075 | 0.999881 | 0.020989 | 3.977536 |
| `pmlb_kr_vs_kp` | LightGBM baseline | ok | CPU | roc_auc | 0.999411 | 0.993875 | 0.999912 | 0.017272 | 0.189425 |
| `pmlb_kr_vs_kp` | PriorTune T/K=25 | ok | CPU | roc_auc | 0.999146 | 0.982848 | 0.998581 | 0.071374 | 2.066904 |
| `pmlb_kr_vs_kp` | TabPFN-2.6 | ok | MPS | roc_auc | 0.999666 | 0.993875 | 0.999892 | 0.015911 | 46.777388 |
| `pmlb_kr_vs_kp` | TabICLv2 | ok | MPS | roc_auc | 0.999843 | 0.996869 | 0.999951 | 0.009635 | 21.940007 |
| `pmlb_mushroom` | LightGBM baseline | ok | CPU | roc_auc | 1.000000 | 0.998723 | 1.000000 | 0.021512 | 0.182499 |
| `pmlb_mushroom` | PriorTune T/K=25 | ok | CPU | roc_auc | 1.000000 | 0.500000 | 1.000000 | 0.667438 | 1.732072 |
| `pmlb_mushroom` | TabPFN-2.6 | ok | MPS | roc_auc | 1.000000 | 1.000000 | 1.000000 | 0.000020 | 95.549723 |
| `pmlb_mushroom` | TabICLv2 | ok | MPS | roc_auc | 1.000000 | 1.000000 | 1.000000 | 0.000002 | 41.698456 |
| `pmlb_hill_valley_without_noise` | LightGBM baseline | ok | CPU | roc_auc | 0.568225 | 0.546850 | 0.570257 | 0.744713 | 0.367849 |
| `pmlb_hill_valley_without_noise` | PriorTune T/K=25 | ok | CPU | roc_auc | 0.610467 | 0.525203 | 0.517141 | 0.699643 | 5.792125 |
| `pmlb_hill_valley_without_noise` | TabPFN-2.6 | ok | MPS | roc_auc | 0.998306 | 0.979268 | 0.997696 | 0.072681 | 39.025162 |
| `pmlb_hill_valley_without_noise` | TabICLv2 | ok | MPS | roc_auc | 0.986518 | 0.942785 | 0.989533 | 0.130920 | 19.327816 |
| `pmlb_twonorm` | LightGBM baseline | ok | CPU | roc_auc | 0.994795 | 0.975685 | 0.997604 | 0.066333 | 0.679620 |
| `pmlb_twonorm` | PriorTune T/K=25 | ok | CPU | roc_auc | 0.995650 | 0.975683 | 0.997916 | 0.062927 | 20.755055 |
| `pmlb_twonorm` | TabPFN-2.6 | ok | MPS | roc_auc | 0.996266 | 0.977709 | 0.998262 | 0.057471 | 85.006004 |
| `pmlb_twonorm` | TabICLv2 | ok | MPS | roc_auc | 0.996220 | 0.981761 | 0.998433 | 0.052485 | 29.796377 |
| `pmlb_penguins` | LightGBM baseline | ok | CPU | log_loss | 0.097323 | 1.000000 | 1.000000 | 0.010212 | 0.114322 |
| `pmlb_penguins` | PriorTune T/K=25 | ok | CPU | log_loss | 0.020079 | 0.988506 | 1.000000 | 0.019954 | 2.219623 |
| `pmlb_penguins` | TabPFN-2.6 | ok | MPS | log_loss | 0.016210 | 0.988506 | 1.000000 | 0.037632 | 16.618668 |
| `pmlb_penguins` | TabICLv2 | ok | MPS | log_loss | 0.014003 | 1.000000 | 1.000000 | 0.010013 | 2.046457 |
| `pmlb_car_evaluation` | LightGBM baseline | ok | CPU | log_loss | 0.008595 | 0.957792 | 0.998877 | 0.047681 | 0.750007 |
| `pmlb_car_evaluation` | PriorTune T/K=25 | ok | CPU | log_loss | 0.038905 | 0.939935 | 0.999667 | 0.047495 | 9.465884 |
| `pmlb_car_evaluation` | TabPFN-2.6 | ok | MPS | log_loss | 0.032375 | 0.960006 | 0.999869 | 0.040423 | 19.052292 |
| `pmlb_car_evaluation` | TabICLv2 | ok | MPS | log_loss | 0.020529 | 0.956759 | 0.999918 | 0.028642 | 4.557928 |
| `pmlb_vehicle` | LightGBM baseline | ok | CPU | log_loss | 0.501853 | 0.790016 | 0.932881 | 0.483336 | 0.150659 |
| `pmlb_vehicle` | PriorTune T/K=25 | ok | CPU | log_loss | 0.466003 | 0.813014 | 0.942239 | 0.438013 | 9.420667 |
| `pmlb_vehicle` | TabPFN-2.6 | ok | MPS | log_loss | 0.281330 | 0.889881 | 0.968346 | 0.276453 | 18.604298 |
| `pmlb_vehicle` | TabICLv2 | ok | MPS | log_loss | 0.207524 | 0.890422 | 0.976186 | 0.234036 | 4.200191 |
| `pmlb_waveform_21` | LightGBM baseline | ok | CPU | log_loss | 0.340515 | 0.836505 | 0.960707 | 0.368235 | 0.486901 |
| `pmlb_waveform_21` | PriorTune T/K=25 | ok | CPU | log_loss | 0.303975 | 0.846517 | 0.965518 | 0.334799 | 39.294295 |
| `pmlb_waveform_21` | TabPFN-2.6 | ok | MPS | log_loss | 0.281084 | 0.860477 | 0.969968 | 0.308158 | 47.535204 |
| `pmlb_waveform_21` | TabICLv2 | ok | MPS | log_loss | 0.276269 | 0.855501 | 0.971794 | 0.300322 | 19.622742 |

## PriorTune winners

Для шести binary datasets отбор использует максимальный validation ROC-AUC. Для четырёх multiclass datasets используется минимальный validation log-loss. Независимый расчёт сортирует таблицу validation trials по нужному направлению, затем по возрастанию `config_id`. Balanced Accuracy остаётся test reporting metric. В multiclass reporting ROC-AUC вычисляется как macro one-vs-rest по вероятностям всех классов после выравнивания колонок с `estimator.classes_`.

| Dataset | Метрика | Config ID | Validation score | Успешно | Ошибки | Best iteration | Tuning, с |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `pmlb_saheart` | roc_auc | 2428 | 0.804815574 | 25 | 0 | 40 | 0.260294 |
| `pmlb_tic_tac_toe` | roc_auc | 926 | 0.998797499 | 25 | 0 | 504 | 1.386680 |
| `pmlb_kr_vs_kp` | roc_auc | 2005 | 0.999145970 | 25 | 0 | 240 | 2.028978 |
| `pmlb_mushroom` | roc_auc | 95 | 1.000000000 | 25 | 0 | 5 | 1.700991 |
| `pmlb_hill_valley_without_noise` | roc_auc | 1487 | 0.610467480 | 25 | 0 | 172 | 5.769719 |
| `pmlb_twonorm` | roc_auc | 926 | 0.995650102 | 25 | 0 | 741 | 20.638492 |
| `pmlb_penguins` | log_loss | 3921 | 0.020078943 | 25 | 0 | 999 | 2.198201 |
| `pmlb_car_evaluation` | log_loss | 926 | 0.038904859 | 25 | 0 | 997 | 9.396347 |
| `pmlb_vehicle` | log_loss | 2428 | 0.466002634 | 25 | 0 | 773 | 9.390847 |
| `pmlb_waveform_21` | log_loss | 3545 | 0.303975265 | 25 | 0 | 648 | 39.188293 |

## Best-by-Balanced-Accuracy и gaps

Диагностика вычислена из 40 канонических CandidateResult. Для полного набора из четырёх успешных вариантов `gap_to_best` равен максимальной test Balanced Accuracy минус значение варианта. Точные равенства записаны с `tie=true` и списком вариантов. Для неполного набора поля best и gap отсутствуют.

| Dataset | Максимум BA | BA | Ничья | Baseline BA | PriorTune BA | TabPFN BA | TabICL BA |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `pmlb_saheart` | tabicl/v2 | 0.761014 | нет | 0.500000 | 0.569928 | 0.705943 | 0.761014 |
| `pmlb_tic_tac_toe` | tabpfn/v2_6 | 0.992537 | нет | 0.988537 | 0.969612 | 0.992537 | 0.985075 |
| `pmlb_kr_vs_kp` | tabicl/v2 | 0.996869 | нет | 0.993875 | 0.982848 | 0.993875 | 0.996869 |
| `pmlb_mushroom` | tabpfn/v2_6, tabicl/v2 | 1.000000 | да | 0.998723 | 0.500000 | 1.000000 | 1.000000 |
| `pmlb_hill_valley_without_noise` | tabpfn/v2_6 | 0.979268 | нет | 0.546850 | 0.525203 | 0.979268 | 0.942785 |
| `pmlb_twonorm` | tabicl/v2 | 0.981761 | нет | 0.975685 | 0.975683 | 0.977709 | 0.981761 |
| `pmlb_penguins` | lightgbm/baseline, tabicl/v2 | 1.000000 | да | 1.000000 | 0.988506 | 0.988506 | 1.000000 |
| `pmlb_car_evaluation` | tabpfn/v2_6 | 0.960006 | нет | 0.957792 | 0.939935 | 0.960006 | 0.956759 |
| `pmlb_vehicle` | tabicl/v2 | 0.890422 | нет | 0.790016 | 0.813014 | 0.889881 | 0.890422 |
| `pmlb_waveform_21` | tabpfn/v2_6 | 0.860477 | нет | 0.836505 | 0.846517 | 0.860477 | 0.855501 |

## Leakage audit и integrity

- Для каждой строки проверены `dataset_id`, `split_id`, `seed`, family/variant, `exact_model_id`, размеры, совпадение shared fingerprints, версии пакетов и устройство. Для foundation models сверены SHA checkpoint и фактическое устройство обученного estimator.
- Fit использует train, включая fit словаря категорий и `LabelEncoder`. LightGBM получает только validation в `eval_set`. Сверка логов проверяет последовательность fit → validation → final test.
- Для PriorTune тестовая порция запрашивается после отбора. В каждом успешном логе сначала идут 25 validation trials в опубликованном `add_order`, затем выбранный ID, затем ровно один `final_test`. Test получает уже обученный победитель. Canonical trial table содержит только validation и времена.
- Ключ строк результатов включает `dataset_id + split_id + seed + model_family + variant + exact_model_id`. Повтор единственного слота заменяет предыдущую строку для того же dataset и variant. Resume сверяет status, exact ID, shared fingerprints, digest лога, метрики, protocol digest и комплект trials для PriorTune.

## Artifacts

- `data/derived/candidate_full_pilot/candidate_results.parquet` — primary truth, 40 строк v2.
- `data/derived/priortune_validation/trials.parquet` — канонические 250 validation trials текущего `full_pilot_protocol_id`.
- `data/derived/candidate_full_pilot/performance_matrix.parquet` и `best_by_balanced_accuracy.json` — диагностические представления.
- `data/derived/candidate_full_pilot/run_summary.json`, `protocol.json`, `preflight.json` и 40 отдельных логов workers.
- `data/derived/candidate_full_pilot/stage5_stage6_trials.parquet` — снимок ранее сохранённых trials.

## Errors и fixes

Ошибок выполнения и нарушений протокола не выявлено. Модельный pipeline не требовал исправлений.

LightGBM 4.7.0 выдаёт известное предупреждение о `eval_set`. Этот интерфейс использован для сохранения закреплённого validation-only протокола.

## Regression tests и воспроизведение

```bash
.venv/bin/python scripts/candidates/run_full_pilot.py --all
.venv/bin/python scripts/candidates/run_full_pilot.py --all --audit-only
.venv/bin/python -m pytest -q \
  tests/test_full_pilot_cli.py tests/test_mini_pilot_cli.py tests/test_priortune.py \
  tests/test_candidate_api.py tests/test_shared_pilot.py tests/test_dataset_loading.py
```

**183 passed**. Тесты используют fake workers для оркестрации и проверяют фиксированный scope, порядок сохранения, отказы, fail-fast проверку размеров, повтор без дублей, strict resume и точное равенство BA. Совместимость CLI этапов 5–6 проверена регрессионными тестами.

## Scientific limitation

Эти 10 datasets являются pilot-набором. Результаты не являются окончательным сравнением алгоритмов статьи.

`main` синхронизирована с `origin/main` на базовом коммите `3268bb2`. Код и отчёт находятся в рабочей копии. Данные запуска игнорируются Git. От Дианы и Риты ничего не требуется для этого этапа. Этап 8 ожидает отдельной команды.
