# Аудит репликации arXiv:2605.28418

**Обновление архитектуры, 21 октября 2026:** все 21 canonical mapping, семейства
и шесть сравнений реализованы; добавлены configurable PyMFE, пять custom landmarks,
local splits, candidate executor и LODO/TabPFN-2.6 wrapper. Аудит ниже сохраняет
исходные границы доказанного: original PyMFE set и metric discrepancy не уточнены.
Реализация paper-style extraction не превращает её в exact replication.
Текущие статусы и проверки: [STAGE3_REPORT.md](STAGE3_REPORT.md),
интерфейсы: [RESEARCH_PIPELINE.md](RESEARCH_PIPELINE.md).

## Объём текущего этапа

Аудит, импорт опубликованных таблиц результатов, идентификация версий и вычисление
dataset-level gap. Полная численная репликация, обучение метапредиктора и расширения
E1–E5 на этом этапе не запускаются. Локальный каталог датасетов сохраняется.

Используются четыре обозначения достоверности:

- **CONFIRMED FROM PAPER** — непосредственно указано в тексте статьи.
- **CONFIRMED FROM CODE / TABARENA** — проверено в официальном исходном коде или артефакте.
- **NOT CONFIRMED** — достаточных сведений не найдено.
- **OUR ASSUMPTION** — явное решение нашей реализации, а не утверждение авторов.

## Источники и версии

1. [Полный текст arXiv v3](https://arxiv.org/html/2605.28418v3), 1 июля 2026:
   *Explaining Tabular Foundation Model Differences Through Meta-Features*.
   Название *Revisiting Metafeatures to Explain Model Differences on Tabular Data*
   встречается для предыдущей версии/библиографической записи этого же arXiv ID.
2. [Официальный TabArena, зафиксированная ревизия](https://github.com/autogluon/tabarena/tree/4b055decdd272314cf44078d98eafc37b105d785).
3. [Официальный PyMFE, зафиксированная ревизия](https://github.com/ealcobaca/pymfe/tree/550648533ffed791cc52682c2e1ee447ca142160).

Точный репозиторий анализа **самой статьи** публично не найден. Проверены ссылки
полного текста и поиск по arXiv ID, обоим названиям, авторам и GitHub. Это результат
ограниченного поиска, не доказательство отсутствия кода. TabArena — репозиторий
benchmark, а не автоматически репозиторий Herre et al.

## CONFIRMED FROM PAPER

Краткая карта методологии; обозначения и формулы воспроизведены для реализации.

| Компонент | Указание статьи |
|---|---|
| Данные | 51 задача TabArena; классификация и регрессия; Appendix A.1, Table 2 |
| Сравнения | NN/tree; non-TFM/TFM; TabICLv2/TabPFN-2.6; дополнительные пары версий |
| Семейство | Выбор варианта по validation error, оценка по test error |
| Единица анализа | Датасет; разности усредняются по его splits |
| Метапризнаки | Базовые свойства, корреляционная избыточность, сводки PyMFE |
| Типы сводок PyMFE | Статистические, информационные, модельные; точный `groups=` отсутствует |
| Контроли | `log_n`, `log_d`, `d_over_n`, `cat_fraction`, `feature_missing_fraction` |
| Метапредиктор | TabPFN-2.6, без HPO |
| Проверка | Leave-one-dataset-out |
| Baselines | Средний обучающий gap; наиболее частое обучающее направление |
| Метрики | MAE gap и sign accuracy; bootstrap по датасетам |
| Ограничения | Мало датасетов, неполные признаки, зависимость от benchmark; robust-отбор до LODO |

Table 3: MLP = FastAI, MNCA (GPU), NN-Torch, RealMLP (GPU), TabM (GPU);
trees = CatBoost, LightGBM, XGBoost, EBM, PerpetualBoost, ExtraTrees, RandomForest.
non-TFM объединяет эти две группы и KNN/LR; TFM = TabDPT (GPU), TabSTAR,
TabICLv2, TabICL (GPU), TabPFN v2 (GPU), TabPFN-2.6, RealTabPFN v2.5.
Кроме первых трёх routing comparisons рассматриваются RealTabPFN v2.5/TabPFN-2.6,
TabPFN v2/TabPFN-2.6, TabICL v1/TabICLv2 на применимых подмножествах данных.

В predictive evaluation статья проверяет **пять** вариантов: controls only,
all features, controls + all, robust only, controls + robust. Это отдельная ось,
её нельзя путать с нашим расширением A/B/C/D. Robust-набор сформирован глобальным
association analysis; для строгого LODO он должен отбираться внутри training fold.

Определения из §3 и Appendix A.2:

```text
r(D,s) = max(median_test_error(D,s) − min_test_error(D,s), 1e−5)
normalized_error = clip((test_error − min_test_error) / r(D,s), 0, 1)
gap(D) = mean_s(normalized_error_A(D,s) − normalized_error_B(D,s))
```

Положительный gap означает преимущество B. Якоря min/median относятся ко всем
рассматриваемым методам split, а не автоматически только к сравниваемой паре.
Appendix A.2 отдельно требует ту же формулу для **validation errors** с собственными
validation min/median. Внутри семейства минимизируется нормализованный validation
error. Clipping может создавать дополнительные ничьи; правило их разрешения
авторы не определяют. Реализация сохраняет отдельные validation и test якоря.

Указанный preprocessing: Inf→NaN; удаление признаков с >20% пропусков,
почти константных и сильно коррелирующих (|Spearman|>0.95). Указанные уровни
association screening: BH FDR 0.05 и bootstrap sign consistency 0.90.

## CONFIRMED FROM CODE / TABARENA

### Точные исторические идентификаторы

| Модель | `ta_name` | `config_type` | `method` в таблице | `ta_suite` |
|---|---|---|---|---|
| TabICLv2 | `TabICLv2` | `TABICLV2` | `TABICLV2 (default)` | `tabarena-2026-02-16` |
| TabPFN-2.6 | `TabPFN-v2.6` | `TABPFN-V2.6` | `TABPFN-V2.6 (default)` | `tabarena-2026-03-18` |

Обе таблицы `hpo_results.parquet` реально загружены: **816 строк и 51 датасет в каждой**.
MD5 совпали с официальным `results_checksums.json` указанной ревизии:

```text
TabICLv2:     d5c6bc101ae8981f58389586d90e6550
TabPFN-2.6:   ba8466d1f4f4adc1ea8d6701c363ed2a
```

Наличие cached results не означает наличие локальных весов TabPFN-2.6 или
лицензионного доступа для будущего метапредиктора. Веса на этом этапе не проверяются
через запуск модели, не скачиваются и не подменяются версией 3.x.

Источники metadata:
[TabICL info.py](https://github.com/autogluon/tabarena/blob/4b055decdd272314cf44078d98eafc37b105d785/packages/tabarena/src/tabarena/models/tabicl/info.py),
[TabPFN info.py](https://github.com/autogluon/tabarena/blob/4b055decdd272314cf44078d98eafc37b105d785/packages/tabarena/src/tabarena/models/tabpfnv2_5/info.py).
Текущий основной roster уже использует более поздний rerun TabICLv2. Наш профиль
фиксирует исторический февральский артефакт и не смешивает эти прогоны.

### Фактические метрики — расхождение с описанием статьи

В обеих загруженных исторических таблицах:

| Тип | `metric` |
|---|---|
| binary | `roc_auc` |
| multiclass | `log_loss` |
| regression | `rmse` |

Appendix A.1 описывает log-loss для классификации вообще. Cached binary ROC-AUC
нельзя переименовать в log-loss или преобразовать в него по одной агрегированной
оценке. Для точного сопоставления требуется объяснение авторов либо другой артефакт
или пересчёт по доступным вероятностным предсказаниям. Такой пересчёт сейчас не выполняется.

`metric_error` и `metric_error_val` — уже ошибки, меньшие значения лучше;
для ROC-AUC это error-представление, а не сама AUC. Знак исходной score-метрики
не следует менять повторно. Метрики сохраняются вместе с каждым split.

Список реально доступных датасетов, их типов, метрик и числа splits выгружается
в `docs/TABARENA_DATASET_INVENTORY.csv`. Названия берутся из артефактов, а не
выдумываются из коротких отображаемых названий Table 2.
Реальный состав: 30 binary, 8 multiclass, 13 regression; у датасетов по 9 или
30 splits. 816 split’ов одной модели нельзя считать 816 независимыми датасетами.

### Реальный API и уровни артефактов

- `MethodMetadata.load_results()` для config-метода читает `hpo_results.parquet`.
- `load_model_results()` читает per-config `model_results.parquet`.
- `load_processed()` открывает репозиторий предсказаний; загрузка может быть многогигабайтной.
- `load_raw()` ещё тяжелее; текущему этапу не нужен.
- [Экспорт CSV](https://github.com/autogluon/tabarena/blob/4b055decdd272314cf44078d98eafc37b105d785/examples/plots/run_export_results_per_split.py):
  `TabArenaContext().compare(output_dir=..., plot=False)`.
- Для научного импорта нужно явно исключать подставленные результаты: `fillna=None`
  при экспорте; импорт также отклоняет `imputed=True`.

`results_per_split.csv` не размещён как готовый общий файл: его создаёт официальный
экспорт. Наш минимальный адаптер также читает маленькие опубликованные Parquet
того же results-уровня, не устанавливая весь TabArena. Никакой код моделей не запускается.
Исходные таблицы сохраняют `method_subtype`, `config_type`, `ta_name`, `ta_suite`;
в экспортном CSV некоторые из этих полей могут отсутствовать — тогда их нужно
передать явным mapping, а не угадывать версии.

### PyMFE: что уже реализовано библиотекой

[Исходный `DEFAULT_GROUP`](https://github.com/ealcobaca/pymfe/blob/550648533ffed791cc52682c2e1ee447ca142160/pymfe/_internal.py):
`general`, `info-theory`, `statistical`, `model-based`, **`landmarking`**.
`relative` требует landmarking; `complexity` — отдельная группа, не входящая в default.

В [landmarking.py](https://github.com/ealcobaca/pymfe/blob/550648533ffed791cc52682c2e1ee447ca142160/pymfe/landmarking.py)
найдены `best_node`, `random_node`, `worst_node`, `linear_discr`, `naive_bayes`,
`one_nn`, `elite_nn`. Реализации включают decision stumps, LDA, GaussianNB и 1-NN.
В проверенной версии defaults: score=accuracy, 10 CV folds, shuffle=False,
lm_sample_frac=1.0; mean/sd — стандартные сводки. Это параметры **библиотеки**,
не подтверждённые настройки статьи.

## NOT CONFIRMED

- Полный точный список PyMFE groups/features/summaries и версия PyMFE авторов.
- Использовали ли авторы `landmarking`, `relative`, `complexity` или `groups="default"`.
- Точный порог «почти константный», база логарифмов, некоторые определения контролей.
- Как метапризнаки training splits сводились к одной записи датасета.
- Полный неизменяемый roster артефактов, включённый в min/median нормализации:
  какие snapshots, variants, ансамбли и дополнительные методы учитывались.
- Каким артефактом объясняется binary metric discrepancy.
- Полностью точная конфигурация и checkpoint TabPFN-2.6 **Regressor** для метапредиктора.

Следствие: нельзя объявить A «точным original feature set», а B — его гарантированно
новым расширением. A/B/C/D остаются экспериментальным дизайном до уточнения A.

## OUR ASSUMPTION / решения реализации

- При расчёте gap нужен **явно заданный reference cohort**; нормализация только по
  паре по умолчанию запрещена. Состав и coverage сохраняются вместе с результатом.
- Для демонстрации реконструирован roster 21 метода из семейств Table 3,
  выбрав исторические metadata до первой подачи статьи. Это не доказательство
  тождества normalization universe авторов; профиль так и помечается.
- В manifest явно выбран `exclude_imputed_or_unknown`. В 21 таблице найдено
  7 254 строки с imputed=True или неизвестным флагом. Они исключены до strict
  schema loader с отчётом по artifact/subtype/reason. У 8 исторических default
  серий флаг отсутствует на уровне строк; их значения не объявляются False.
  Официальный код в некоторых внутренних путях использует fillna(0); наше
  консервативное исключение неизвестных флагов — отдельное OUR ASSUMPTION.
  В исходных двух новых default-таблицах **вся колонка** imputed отсутствует:
  это pre-comparison cached results; они не содержат leaderboard substitutions.
  После subtype filter default/tuned остаётся 24 732 reference-строки,
  19–21 доступная модель на split. Все 816 split’ов основной пары сохранены.
- Поля `dataset_source=tabarena/local` входят в ключ. Существующий loader и registry
  local-датасетов не меняются и не запускают кандидатов.
- Все неизвестные настройки будущего feature extraction задаются явно; PyMFE
  default не используется как доказательство групп статьи.
- Будущий feature preprocessing обучается внутри LODO. Это строгая проверка;
  вариант с опубликованным global robust-screening будет отмечен отдельно.
- Tie normalized validation error: лексикографический exact identifier; NaN и Inf ошибок
  кандидатов, дубли ключей и imputed результаты не заменяются искусственными значениями.
- Минимальные контроли используют ln(n), ln(d), d/n, долю нечисловых признаков
  (включая bool), долю пропущенных ячеек X_train. Эти определения — рабочая
  операционализация; точное совпадение со статьёй не подтверждено.

## Компонент статьи → модуль проекта

| Ответственность | Наш модуль | Этот этап |
|---|---|---|
| Cached tables / export schema | `tabarena/adapter.py`, `results_loader.py` | Реализовано |
| Точные версии | `tabarena/model_mapping.py` + manifest | Реализовано |
| Нормализация | `performance_gap/normalization.py` | Реализовано |
| Validation-selection / усреднение | `performance_gap/gap.py`, `families.py` | Реализовано |
| Контроли training data | `meta_features/controls.py` | Минимальные функции |
| Исходные PyMFE-признаки | `meta_features/pymfe_features.py`, `feature_sets.py` | Контракт, группы не подтверждены |
| Стандартные/custom landmarks | `landmarkers/` | Контракты, не обучать |
| TabPFN-2.6 gap regressor | `meta_model/tabpfn_predictor.py` | Контракт, не заменять версию |
| Mean / majority | `meta_model/baselines.py` | Реализованы чистые функции |
| Dataset LODO | `meta_model/lodo.py` | Контракт, не запускать |
| MAE/sign/regret | `evaluation/metrics.py` | Формулы, без экспериментов |
| E0–E5 | `evaluation/experiments.py` | Описание серий, не запускать |

## Чем могут отличаться custom landmarkers

LogisticRegression отличается от LDA; дерево заданной глубины — от decision stump;
kNN с k>1 — от 1-NN. Однако отличия конфигурации не гарантируют дополнительный
сигнал. После E0 сравнить A=original, B=A+PyMFE landmarking, C=A+custom,
D=A+обе группы, сохранив метапредиктор, данные, метрики и splits. Для расширения
числа датасетов использовать отдельные E1/E4/E5 и общий provenance.

Текущий успех — подтверждённые cached records, формула и корректная агрегация.
Полная численная репликация таблиц статьи остаётся незавершённой до устранения
перечисленных неопределённостей и выполнения следующих milestones.
