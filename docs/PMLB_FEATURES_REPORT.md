# PMLB-пилот: метапризнаки и пять custom landmarkers

Дата: 2026-11-09. Статус: **вычисления и повторная проверка завершены**.
Состав — десять задач из `config/pmlb_pilot.yaml`; split ID — `holdout_0`.

## 1. Итог

- На каждом датасете получено **202 PyMFE-признака + 5 controls = 207 признаков**.
  Всего сохранено **2070 long-format строк**, из них 2029 конечных и 41 NaN.
- Стандартный PyMFE landmarking не запускался. Файл
  `data/derived/landmarkers/pymfe_landmarkers.parquet` не создавался.
- Получены **50 успешных custom-записей**: десять датасетов × пять моделей.
  У каждой записи три folds, Balanced Accuracy mean/std и fit time mean/std.
  Итоговых ошибок, пропущенных моделей и нечисловых custom scores нет.
- Все десять датасетов обработаны. Для extraction и CV использован исключительно
  outer train из существующих manifests; outer validation/test не передавались
  extractor или landmarker API.
- Повторный проход дал точное совпадение метапризнаков, статусов, предупреждений,
  provenance и оценок моделей. Время fit измеряется заново и не обязано совпадать.
- Исходные CSV, импортированные копии, metadata, утверждённый состав, research
  config, реестр и split manifests сохранены без изменений.

TabPFN, TabICLv2, LightGBM, PriorTune и мета-обучение не запускались.
Внутренние model-based вычисления самого PyMFE соответствуют существующей
конфигурации и не являются запуском целевых классификаторов.

## 2. Окружение

В проекте отсутствовала `.venv`. Она создана внутри checkout с **Python 3.12.0**;
все последующие Python/pip/pytest-команды выполнялись через
`.venv/Scripts/python.exe`. Системный Python не изменялся, глобальной установки
пакетов не выполнялось. Администраторские права и учётные данные не потребовались.

Установка: `pip install -e ".[test]"` из `pyproject.toml`, с constraints для
numpy/pandas/sklearn, совпадающими с окружением подготовки manifests.

| Компонент | Версия |
|---|---|
| Python | 3.12.0 |
| numpy | 2.1.2 |
| pandas | 2.3.3 |
| scikit-learn | 1.9.1 |
| PyMFE | 0.4.4 |
| pyarrow | 26.0.1 |
| scipy | 1.18.1 |
| statsmodels | 0.15.0 |
| PyYAML | 6.0.3 |
| openpyxl | 3.1.5 |
| pytest | 9.2.0 |

`pip check`: **No broken requirements found**. Импорт PyMFE и остальных основных
зависимостей проверен до запуска. Полный состав пакетов сохранён в
`results/logs/pmlb_features/environment.json` и `requirements.freeze.txt`.

Все TEMP/TMP, pip cache, matplotlib/XDG/joblib/numba caches направлены в `.cache/`
внутри проекта. BLAS/OpenMP/NumExpr ограничены одним потоком для устойчивого
повторения численных результатов. Локальные настройки PowerShell сохранены
в `.cache/pilot-env.ps1`; constraints — в `.cache/pilot-constraints.txt`.
Эти служебные файлы не отслеживаются Git.

Старое environment в manifests отражает момент их создания (например, PyMFE тогда
не был установлен); оно не переписывалось. В provenance результатов записано
новое фактическое окружение вычислений.

## 3. Протокол и команды

Использована неизменённая `config/research.yaml`:

- PyMFE groups: `general`, `statistical`, `info-theory`, `model-based`;
  summaries: `mean`, `sd`, `min`, `max`, `median`; seed 42.
- Пять custom pipelines: Logistic Regression (`C=1`, `max_iter=1000`),
  Decision Tree (`max_depth=3`), kNN (`k=5`), GaussianNB (`var_smoothing=1e-9`),
  LDA (`solver=lsqr`, `shrinkage=auto`).
- Custom scoring: `balanced_accuracy`; shuffled StratifiedKFold(3), seed 42.
- Custom std вычисляется с `ddof=0` по трём folds.
- Imputation/OHE/scaling fit выполняется заново внутри inner train каждого fold.
  Scaling числовых признаков включён у LR, kNN и LDA.

Обёртка `scripts/run_pmlb_features_pilot.py` вызывает существующие CLI-файлы
через `runpy`, передавая им явный список десяти ID, и добавляет runtime assertions,
таймеры, журналы и snapshots. Модели, scoring, preprocessing и правила сохранения
результатов не заменяются. Для extraction всегда передаётся существующий флаг:

```text
scripts/extract_meta_features.py <10 утверждённых ID> --without-standard-landmarking
scripts/run_landmarkers.py <те же 10 ID>
```

Фактические команды успешных проходов после настройки окружения:

```powershell
. ".cache/pilot-env.ps1"
& ".venv/Scripts/python.exe" -B scripts/run_pmlb_features_pilot.py --phase features --run-id baseline
& ".venv/Scripts/python.exe" -B scripts/run_pmlb_features_pilot.py --phase landmarkers --run-id corrected
& ".venv/Scripts/python.exe" -B scripts/run_pmlb_features_pilot.py --phase features --run-id repeat
& ".venv/Scripts/python.exe" -B scripts/run_pmlb_features_pilot.py --phase landmarkers --run-id repeat
& ".venv/Scripts/python.exe" -B scripts/verify_pmlb_features.py
```

При новом запуске нужен новый `--run-id`: существующие журналы не перезаписываются.
Если отсутствует хотя бы один approved manifest, обёртка останавливается до CLI.
Существующий `make_split` вызывался только для проверки/чтения имеющихся manifests;
новых разбиений не создавалось.

## 4. Метапризнаки, NaN и предупреждения

202 уникальных PyMFE-столбца распределены так: general — 15, statistical — 105,
info-theory — 28, model-based — 54. Ещё пять controls образуют отдельный namespace.
Названия, группы, версии и config IDs сохранены в `feature_manifest.parquet`.
Это **paper-style reconstruction**, а не подтверждённый exact set исходной статьи.

| Датасет (без `pmlb_`) | PyMFE | Controls | Конечных значений | NaN / undefined | Inf | Warning-пары |
|---|---:|---:|---:|---:|---:|---:|
| saheart | 202 | 5 | 205 | 2 | 0 | 3 |
| tic_tac_toe | 202 | 5 | 204 | 3 | 0 | 2 |
| kr_vs_kp | 202 | 5 | 204 | 3 | 0 | 2 |
| mushroom | 202 | 5 | 204 | 3 | 0 | 3 |
| Hill_Valley_without_noise | 202 | 5 | 205 | 2 | 0 | 1 |
| twonorm | 202 | 5 | 195 | 12 | 0 | 12 |
| penguins | 202 | 5 | 206 | 1 | 0 | 2 |
| car_evaluation | 202 | 5 | 205 | 2 | 0 | 2 |
| vehicle | 202 | 5 | 205 | 2 | 0 | 2 |
| waveform_21 | 202 | 5 | 196 | 11 | 0 | 11 |
| **Всего** | **2020** | **50** | **2029** | **41** | **0** | **40** |

Warning-пара — уникальное сообщение в provenance dataset/namespace. Всего
**16 разных текстов**, 40 таких пар; это не число всех повторных испусканий
одного warning внутри библиотеки. Существующий extractor сохраняет сообщения
с дедупликацией. Дополнительные предупреждения вне его внутреннего capture
сохраняет audit wrapper; в успешных проходах таких дополнительных предупреждений нет.

Все 41 NaN находятся в `paper_features` и имеют статус `undefined`; controls
конечны. Ни одного Inf не сохранено. Значения не удалялись, не заменялись нулями
и не импутировались на уровне meta-dataset.

| Неопределённые признаки | Число значений | Диагностика |
|---|---:|---|
| `can_cor.sd` | 6 | Во всех binary-задачах только одна каноническая корреляция; summary sd не определён |
| `cat_to_num` | 4 | Нет исходных количественных признаков в полностью категориальных задачах |
| `num_to_cat` | 4 | Нет категориальных признаков в полностью числовых задачах |
| `sd_ratio` | 7 | Не конечен расчёт через лог-детерминанты ковариационных матриц в PyMFE 0.4.4; возможны вырожденность, численная неустойчивость и особенности реализации |
| `g_mean.{mean,sd,min,max,median}` | 10 | twonorm и waveform содержат отрицательные значения; геометрическое среднее не даёт допустимого результата |
| `h_mean.{mean,sd,min,max,median}` | 10 | Те же две задачи: harmonic mean в используемой реализации не допускает отрицательные значения |

Основные сообщения: невозможность summary для `can_cor`/`g_mean`/`h_mean`,
`divide by zero encountered in log`, `invalid value encountered in log`,
`invalid value encountered in scalar subtract`,
`It is not possible make equal discretization`.
Для `sd_ratio` не применялась произвольная регуляризация и не менялась библиотечная
формула. Операционный статус `undefined` означает non-finite output; он сам по себе
не доказывает, что причина каждого NaN является только свойством исходных данных.
Полные списки по dataset/name и warning/message доступны в CSV и provenance.

## 5. Пять landmarkers: Balanced Accuracy

Таблица содержит итоговый проход `repeat`, **mean ± std по трём inner folds**,
в единицах 0–1. Все 50 записей имеют `status=ok`, `n_folds=3`, пустой error_message.
Ни NaN/Inf в scores или fit times, ни warnings у итоговых custom-моделей нет.

| Датасет (без `pmlb_`) | Logistic Regression | Decision Tree | kNN | Gaussian NB | LDA |
|---|---:|---:|---:|---:|---:|
| saheart | 0.6500 ± 0.0238 | 0.5795 ± 0.0733 | 0.5778 ± 0.0444 | 0.6687 ± 0.0103 | 0.6545 ± 0.0268 |
| tic_tac_toe | 0.9104 ± 0.0221 | 0.6933 ± 0.0218 | 0.8807 ± 0.0085 | 0.6512 ± 0.0279 | 0.7821 ± 0.0335 |
| kr_vs_kp | 0.9576 ± 0.0039 | 0.8959 ± 0.0024 | 0.8876 ± 0.0049 | 0.5957 ± 0.0079 | 0.9258 ± 0.0095 |
| mushroom | 0.9989 ± 0.0011 | 0.9842 ± 0.0020 | 0.9996 ± 0.0006 | 0.9518 ± 0.0168 | 0.9960 ± 0.0022 |
| Hill_Valley_without_noise | 0.7071 ± 0.0486 | 0.4864 ± 0.0104 | 0.5411 ± 0.0168 | 0.5046 ± 0.0136 | 0.6422 ± 0.0468 |
| twonorm | 0.9779 ± 0.0025 | 0.7403 ± 0.0094 | 0.9714 ± 0.0011 | 0.9800 ± 0.0012 | 0.9791 ± 0.0015 |
| penguins | 0.9829 ± 0.0242 | 0.9877 ± 0.0104 | 0.9915 ± 0.0121 | 0.8821 ± 0.0284 | 1.0000 ± 0.0000 |
| car_evaluation | 0.6790 ± 0.0565 | 0.4195 ± 0.0119 | 0.4947 ± 0.0199 | 0.8570 ± 0.0113 | 0.6196 ± 0.0634 |
| vehicle | 0.7582 ± 0.0078 | 0.6407 ± 0.0513 | 0.7034 ± 0.0265 | 0.5144 ± 0.0092 | 0.7526 ± 0.0070 |
| waveform_21 | 0.8705 ± 0.0114 | 0.7134 ± 0.0167 | 0.8217 ± 0.0081 | 0.8132 ± 0.0105 | 0.8590 ± 0.0061 |

Эти диагностические оценки не являются результатами целевых классификаторов,
не используются здесь для смены состава пилота и не означают оценки на outer test.

## 6. Время вычислений

Wall time существующих CLI под аудитом (не включает установку окружения/pytest):

| Попытка | Этап | Секунды | Результат |
|---|---|---:|---|
| first | features | 0.002 | Ошибка argparse до обработки данных |
| baseline | features | 15.007 | Успешно, 10 задач |
| baseline | landmarkers | 19.371 | 40 ok, 10 LDA error из-за неверной проверки аудитора |
| corrected | landmarkers | 19.373 | Успешно, 50 записей |
| repeat | features | 16.515 | Успешно, воспроизводимость подтверждена |
| repeat | landmarkers | 23.694 | Успешно, 50 записей, воспроизводимость подтверждена |

Первый полный успешный комплект: **34.380 с**; повторный: **40.209 с**.
Суммарно по всем шести попыткам, включая диагностику: **93.962 с**.
Это wall time с проверками и записью промежуточных результатов, не чистое fit time.

Ниже **fit time mean ± std, миллисекунды**, по трём folds последнего прохода.
Измеряется fit всего pipeline (preprocessing + estimator), без predict/score.
Небольшие runtime assertions также входят в инструментированный вызов fit.
В исходном Parquet эти величины хранятся в секундах.

| Датасет (без `pmlb_`) | Logistic Regression | Decision Tree | kNN | Gaussian NB | LDA |
|---|---:|---:|---:|---:|---:|
| saheart | 15.32 ± 0.34 | 11.04 ± 0.60 | 12.27 ± 0.54 | 11.27 ± 0.47 | 14.28 ± 0.96 |
| tic_tac_toe | 21.78 ± 1.77 | 19.71 ± 1.21 | 18.87 ± 1.67 | 19.67 ± 1.71 | 22.37 ± 1.86 |
| kr_vs_kp | 141.40 ± 9.23 | 156.27 ± 3.70 | 168.80 ± 7.03 | 164.21 ± 0.05 | 183.78 ± 27.63 |
| mushroom | 303.11 ± 18.86 | 296.12 ± 20.15 | 242.19 ± 21.57 | 222.32 ± 1.10 | 248.94 ± 9.46 |
| Hill_Valley_without_noise | 55.24 ± 1.01 | 56.69 ± 1.49 | 48.69 ± 1.08 | 47.03 ± 0.55 | 56.63 ± 0.34 |
| twonorm | 22.68 ± 0.56 | 33.75 ± 0.08 | 18.80 ± 1.16 | 19.26 ± 0.10 | 23.73 ± 0.50 |
| penguins | 15.54 ± 1.28 | 13.88 ± 0.37 | 14.29 ± 0.31 | 13.39 ± 0.48 | 16.92 ± 0.52 |
| car_evaluation | 29.71 ± 1.60 | 22.38 ± 0.19 | 21.61 ± 0.45 | 21.85 ± 0.46 | 26.74 ± 0.21 |
| vehicle | 24.95 ± 1.68 | 14.25 ± 0.34 | 14.02 ± 0.51 | 13.66 ± 0.45 | 18.45 ± 0.16 |
| waveform_21 | 28.33 ± 0.40 | 26.55 ± 1.44 | 19.07 ± 0.92 | 17.43 ± 1.40 | 23.00 ± 0.23 |

Сумма fit time 150 завершённых custom fits последнего прохода — **9.217 с**;
в corrected-проходе — 7.715 с. Различие wall/fit включает prediction, загрузку,
аудит и сохранение. Время по датасетам/namespace вынесено в `dataset_compute_times.csv`.

## 7. Проверка отсутствия утечки и типов

Аудит проверял реальные вызовы, а не только исходный код:

1. На входе `extract_controls`, `extract_pymfe` и `run_custom_landmarkers` X/y
   сравнивались с точным outer train по значениям, dtype, порядку и индексам.
   Проверялась непересекаемость с outer validation/test.
2. `_pymfe_input` получал тот же train DataFrame; переданные categorical indices
   совпадали с утверждённым списком. Проверены 10 событий подготовки на проход.
   У mushroom в outer train **1497** строк `stalk-root=0`; после подготовки PyMFE
   они остались строковой категорией `"0"`, не NaN.
3. Каждый fit pipeline сравнивался с ожидаемым inner train, каждый score —
   с ожидаемым inner validation. В каждом полном custom-проходе проверены
   **150 fit и 150 score событий**, по 15 на датасет.
4. Внутри ColumnTransformer фиксировались `SimpleImputer.fit`,
   `StandardScaler.fit`, `OneHotEncoder.fit`. Проверялись число строк и доступные
   DataFrame-индексы. Во время predict/score новый preprocessing fit не допускается.
5. Внутренние scaler-вызовы самой LDA учитывались отдельно: они выполнялись
   внутри уже проверенного model.fit на его классовых подмножествах.

Индексы и границы folds перепроверены независимым чтением audit JSON;
hash индексов каждого события совпадает с ожидаемым StratifiedKFold.
Controls `log_n` и PyMFE `nr_inst` соответствуют outer train, а `cat_fraction` —
явной типизации. Все остальные числовые признаки сохраняют исходные типы loader.
Декларированная типизация не заменялась подбором категорий по cardinality.

До вычислений зафиксированы SHA-256 и mtime **164 защищённых файлов**:
131 CSV в PMLB (130 задач и catalog), по десять raw-копий/metadata/manifests,
два config и registry. После каждого прохода они совпали с baseline.
Отдельно проверено соответствие источников утверждённым SHA-256.
`git diff --exit-code -- data/PMLB config/research.yaml data/splits config/pmlb_pilot.yaml`
не показал различий для отслеживаемых файлов; игнорируемые manifests дополнительно
проверены по сохранённым checksum и mtime.

## 8. Проблемы, исправления и сохранённые неуспешные попытки

**CLI argparse.** Первый вызов остановился с
`ValueError: mutually exclusive arguments must be optional`: позиционный
`dataset_ids` был помещён в mutually-exclusive group. В `scripts/_research_cli.py`
он вынесен из группы; явная проверка сохраняет контракт «ID или --all-local».
Добавлены пять regression tests без вычислений. Config и методика не менялись.

**Слишком широкая проверка аудитора.** В первом custom-проходе все десять LDA
остановлены assert аудитора. sklearn `LDA(shrinkage="auto")` использует
`StandardScaler.fit_transform` в `_cov` на каждом классовом подмножестве.
Требовать там размер всего inner train было неверно. Исправлена только область
проверки аудитора: строгая проверка preprocessing привязана к ColumnTransformer,
а внутренние преобразования estimator отдельно учитываются внутри checked fit.
Параметры LDA, данные, метрика и folds не менялись. После исправления выполнены
два полных успешных custom-прохода.

Первичная таблица **50 строк (40 ok / 10 error)**, её run_status и audit сохранены
в `results/logs/pmlb_features/baseline/landmarkers*`; первоначальная CLI-ошибка —
в `first/features_audit.json` с отдельным пояснением. Итоговые upsert-таблицы содержат
текущее успешное состояние, но snapshots неуспешных попыток не удалены.

NaN/undefined и warnings PyMFE не скрывались и не «исправлялись» произвольными
числами. Ошибок dataset/namespace при успешном extraction нет; сохранение NaN
предусмотрено протоколом и не считается пропущенным feature record.

## 9. Воспроизводимость и автоматические проверки

Полное сравнение baseline/repeat features: **2070/2070 строк совпадают точно**,
включая места NaN, статусы, ключи, версии и provenance с предупреждениями.
Сравнение corrected/repeat custom: **50/50 mean/std совпадают точно**, как и
metric/config ID/n_folds/status/provenance. Исключены из equality только два
таймера: `fit_time_mean`, `fit_time_std`.

Конфигурации результатов:

| Namespace | config_id |
|---|---|
| controls | `80d140e1ea8d9c63d507cd8f9d621d2f7870eab07612b405076b17a85ec1d3af` |
| paper_features | `1940d0d5a38665bf9bb3cce4548bac17ecbd0eba6cb763b784aa54d467446761` |
| custom_landmarking | `33dc3a3936886ba5021f153067956da190f4eae69eb7281611a50ec1eea0d3bd` |

Внутри новой `.venv` выполнены проверки loader/import/splits и CLI:

```text
74 passed in 14.08s
```

Команда: `.venv/Scripts/python.exe -B -m pytest tests/test_research_cli.py tests/test_pmlb_import.py tests/test_dataset_loading.py -q --basetemp .cache/pytest-features -o cache_dir=.cache/pytest-cache`.
Эти tests не запускали целевые классификаторы и не выполняли meta-learning.
Верификатор `scripts/verify_pmlb_features.py` дополнительно проверил реальные
результаты, полноту names/configs, runtime audits и неизменность входов.

## 10. Файлы результатов

Основные таблицы записаны существующим `upsert_table`, с явными ключами и config IDs:

| Путь | Содержимое |
|---|---|
| `data/derived/meta_features/meta_features.parquet` | 2070 строк: 5 controls + 202 PyMFE на задачу, включая undefined и provenance |
| `data/derived/meta_features/run_status.parquet` | 20 успешных namespace-строк текущего прохода |
| `data/derived/landmarkers/custom_landmarkers.parquet` | 50 успешных model/metric/config-строк с mean/std scores и fit times |
| `data/derived/landmarkers/run_status.parquet` | 10 успешных dataset/custom-строк текущего прохода |
| `results/tables/pmlb_pilot_features/landmarker_scores.csv` | Все 50 числовых результатов без большого provenance JSON |
| `results/tables/pmlb_pilot_features/feature_health.csv` | Счётчики finite/NaN/Inf/undefined/warnings по датасетам |
| `results/tables/pmlb_pilot_features/undefined_features.csv` | Все 41 undefined record с identity/name/config |
| `results/tables/pmlb_pilot_features/warnings.csv` | Все 40 dataset/source/message-пар предупреждений |
| `results/tables/pmlb_pilot_features/dataset_compute_times.csv` | Время controls/PyMFE/custom и суммарное fit time по задачам |
| `results/tables/pmlb_pilot_features/dataset_identity.csv` | Сопоставление original name и ID, split, CSV SHA-256, fingerprints |
| `results/tables/pmlb_pilot_features/feature_manifest.parquet` | 207 уникальных feature definitions, namespaces/groups/versions/config IDs |

В `results/logs/pmlb_features/` также сохранены:

- `protected_before.json`, `environment.json`, `requirements.freeze.txt`,
  `verification.json` (включая SHA-256 реализаций);
- папки `first`, `baseline`, `corrected`, `repeat`: журналы CLI, runtime audit JSON,
  Parquet-snapshots результатов и статусов каждого выполненного этапа;
- пояснения к двум диагностическим неуспешным попыткам.

Текущие артефакты и snapshots локальны и игнорируются существующим `.gitignore`.
Для повторной проверки сохранённых результатов не нужен новый fit:
`.venv/Scripts/python.exe -B scripts/verify_pmlb_features.py` в настроенном окружении.

## 11. Готовность к результатам целевых классификаторов Дианы

**Feature-сторона готова к объединению по ключам**, без переобучения extractor:

- `(dataset_source="local", dataset_id="pmlb_*", split_id="holdout_0")` на уровне
  первичных артефактов; `(dataset_source, dataset_id)` — для dataset-level joins.
- Состав и fingerprints фиксированы; mapping исходных имён доступен в
  `dataset_identity.csv`. Target classifiers должны использовать те же данные,
  типизацию и manifests; совпадение проверяется до join.
- Для каждого датасета доступны 207 ordinary/control features и 10 custom
  score mean/std: потенциально **217 predictor features**. Fit times сохранены
  для анализа стоимости и по умолчанию не включаются builder в predictors.
- Все feature names и model records полны; 41 NaN остаются явными. Их возможная
  импутация должна fit-иться позже внутри training folds meta-evaluation,
  а не на всех десяти/всём объединённом meta-dataset.
- Для join пригоден существующий `build_meta_dataset` после получения реальных
  gap records и выбора одного сравнения/config. Fake targets не создавались.

Сейчас `data/derived/candidate_scores/` ещё пуст: **полного метадатасета с target gap
пока нет**. Диане потребуются реальные candidate validation/test scores с model
identity/version, metric/direction, split provenance и status. Метрика target gap
фиксируется отдельно и не обязана совпадать с диагностической Balanced Accuracy;
внутри сравнения кандидаты должны иметь согласованные метрику и покрытие splits.
Для существующей нормализации нужен явно заданный multi-model reference cohort,
а не только два сравниваемых scores.

Никакого выбора состава по полученным оценкам, запуска целевых моделей,
построения вымышленных gaps или обучения meta-predictor на этом этапе не было.
