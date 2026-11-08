# Диана: аудит candidate pipeline — этап 0

Дата: 8 ноября 2026. База: `05fd915` (`Update README`). Рабочая ветка:
`diana/candidates`, создана от текущей базы без изменения содержимого working tree.

## Границы аудита

Выполнены только чтение исходников, конфигурации, списка файлов Git, локальных
каталогов и нескольких первых строк CSV. Создан этот документ. Модели, preflight,
pytest, dataset CLI, extraction и benchmark не запускались. Зависимости не
устанавливались, checkpoints/portfolio не скачивались. Часть Риты не изменялась.
Исторические отчёты и сохранённый provenance не являются текущей проверкой READY.

Цель последующих этапов Дианы — четыре classification variants:

| model_family | variant | Целевой вариант |
|---|---|---|
| tabpfn | v2_6 | TabPFN-2.6 Classifier, exact generation |
| tabicl | v2 | TabICLv2, `tabicl-classifier-v2-20260212.ckpt` |
| lightgbm | baseline | Explicit baseline, параметры определить на этапе 1 |
| lightgbm | priortune_t_k25 | Официальный опубликованный portfolio K=25, одна конфигурация |

Checkpoint TabICL и K=25 здесь — **требования задания**, а не результат preflight.
PriorTune — способ выбора конфигурации LightGBM по validation, не отдельная family
и не метамодель проекта. Следующие этапы требуют отдельной команды пользователя.

## Уже существует

### Данные, loader и registry

- В Git зарегистрирован **131 CSV** в `data/PMLB/`. Правило `.gitattributes`:
  `data/PMLB/*.csv filter=lfs diff=lfs merge=lfs -text`.
- В первых строках локальных `iris.csv` и `prnn_crabs.csv` видны реальные CSV
  headers/данные и колонка `target`. Все 131 файла на полноту/targets не проверялись.
- **Специализированного PMLB loader не найдено**: поиск `pmlb` в `src/`, `scripts/`,
  `config/`, `tests/` не выявил интеграции. Наличие CSV не равно наличию loader.
- `datasets/loader.py` реализует `load_dataset(dataset_id, project_root=...)` для
  `data/raw/<dataset_id>/metadata.yaml` + CSV/XLSX/Parquet. Возвращает `Dataset(X,y,metadata)`.
  Target обязателен. Прямой путь к `data/PMLB/*.csv` этот интерфейс не принимает.
- ID локального каталога ограничены snake_case; реальные PMLB filenames содержат
  также uppercase, дефисы и `+`. Прямое использование имени файла как local ID
  не гарантирует совместимости. Нужен согласованный mapping, без переименования raw.
- `data/registry/datasets.csv` имеет фиксированную 14-полевую schema и три записи:
  `_toy_binary`, `_smoke_iris`, `_smoke_mixed`. PMLB datasets в этом индексе отсутствуют.
  Registry восстанавливается из `data/raw/`, не из `data/PMLB/`.

### Splits

- `datasets/splits.py` содержит `make_split`, `check_manifest`, `portion`.
  Проверяются identity, fingerprint, protocol hash, целочисленные positional indices,
  отсутствие пересечений, покрытие строк и наличие всех классов в каждой части.
- `make_split` **создаёт** отсутствующий local holdout manifest: train/validation/test
  60/20/20, random_seed=42 из `config/research.yaml`. Seed хранится внутри `protocol`.
- Локально найдены только manifests трёх smoke datasets в `data/splits/`.
  Список фиксированных 10 PMLB dataset IDs и их shared pilot manifests не найдены
  в проинспектированных source/config/data/splits каталогах.
- `scripts/run_candidate.py` вызывает `make_split` автоматически. Поэтому эту
  команду нельзя использовать для production pilot Дианы без изменения поведения.

### Candidates, TabPFN, TabICL, LightGBM

- `candidates/runner.py`: `CandidateRunner` Protocol, `CandidateResult` dataclass,
  `TabArenaCandidateRunner`, dependency/GPU/checkpoint guards и официальный registry resolver.
- Runner использует внешний TabArena/AutoGluon model registry, один default config,
  feature generator/LabelCleaner, train fit и validation/test scoring. Это прежний
  протокол single-default holdout, не новый четырёхвариантный pilot.
- `tabarena/model_mapping.py` содержит explicit identities 21 исторического метода,
  в том числе `tabpfn_2_6`, `tabicl_v2`, `lightgbm`.
- Для TabPFN-2.6 runner проверяет default classification checkpoint официального
  класса и принимает локальный checkpoint с SHA256. Отдельного
  `TabPFN26Candidate` wrapper нет.
- `meta_model/tabpfn_predictor.py` — **Regressor для метамодели**, не целевой
  классификатор Дианы. Этот файл не использовать как готовый classification wrapper
  и не менять в этой ветке.
- TabICLv2 сейчас представлен identity и обращением к внешнему registry. Отдельного
  wrapper и проверки имени `tabicl-classifier-v2-20260212.ckpt` в candidates нет.
- LightGBM представлен generic registry entry. Standalone baseline wrapper и
  explicit baseline hyperparameters не найдены.
- PriorTune imports/configs/portfolio loader/validation selector в текущем
  `src/`, `scripts/`, `config/`, `tests/` не найдены.
- Старый runner требует CUDA для исторических GPU variants. `auto/mps/cuda/cpu`
  dispatcher отсутствует; работоспособность MPS этим аудитом не проверялась.

### Config, scripts, tests и outputs

- `config/candidates.yaml`: num_cpus=1, num_gpus=1, time_limit=120,
  `official_registry_single_default`, пустые weights/hyperparameters. Это бюджет
  старого runner, не зафиксированная конфигурация нового LightGBM baseline.
- `config/research.yaml`: local splits, настройки частей Риты, meta-predictor.
  Они сохраняются; новый pilot не должен создавать альтернативные splits из этого файла.
- `pyproject.toml` объявляет общие data/sklearn/PyMFE зависимости и optional TabPFN;
  выделенных dependency groups четырёх variants/PriorTune пока нет. Это описание
  зависимостей, не аудит фактически установленных пакетов.
- `scripts/check_candidate_models.py` проверяет старый набор 21 метода, а не
  нужные четыре variants; выводит readiness через старые registry/CUDA guards.
- `tests/test_research_pipeline.py` содержит mocks availability/official adapter,
  проверку split utilities, derived-table keys и другие прежние тесты. Новых
  wrappers/PriorTune K=25/shared-pilot tests нет. Тесты на этом этапе не запускались.
- `artifacts.py`: atomic JSON/Parquet upsert, config hash, environment metadata и
  dataset fingerprint. Базовый `IDENTITY` = source/id/split, **без seed**.
- Локально есть derived smoke tables, старая candidate table и gap tables шести
  сравнений. Их наличие не доказывает готовность новых четырёх variants. Имена
  файлов просмотрены; таблицы не пересчитывались и scores не интерпретировались.
- `data/splits/`, `data/derived/`, `results/`, `.cache/` игнорируются Git с
  исключениями `.gitkeep`. Shared manifests не станут общими через обычный Git add
  автоматически: способ передачи согласовать с Ритой.

## Можно переиспользовать

| Существующая часть | Что использовать позже | Ограничение |
|---|---|---|
| Dataset/validator | X/y, target validation, исходные dtype/NaN | Нужен shared PMLB adapter/mapping |
| check_manifest/portion | Проверки partition/classes и positional slicing | Не вызывать автоматический make_split |
| CandidateRunner/CandidateResult | Общий интерфейс и сохранение status/errors | Расширить variant/seed/timing schema совместимо |
| Exact mapping и checkpoint hashing | Версии, provenance, отказ без точной версии | Не копировать CUDA-only ограничение на новые wrappers |
| artifacts.py | Atomic writes, config hash, fingerprint | Согласовать ключи с seed и schema versions |
| Старые candidate mocks | Dependency injection, no-download unit tests | Дополнить четырьмя variants на соответствующем этапе |
| Существующие output directories | Отдельные smoke/mini-pilot/pilot artifacts | Не смешивать с прежними результатами |

Полностью дублирующий runner/dataset registry не нужен. Новые wrappers должны
расширять существующий candidates boundary. Модельный preprocessing берётся из
официальных реализаций; common AutoGluon preprocessing не навязывается всем четырём.

## Нужно изменить

Все пункты ниже — **будущая работа**, не изменения этапа 0.

1. Убрать автоматическое создание production splits из нового candidate flow:
   читать shared manifests, при отсутствии — `shared pilot manifests are required`.
2. Ввести ключ `dataset_id + split_id + seed` и явный dataset_source; определить
   совместимость существующих ID/protocol/fingerprint полей с manifests Риты.
3. Расширить candidate result schema: family/variant, checkpoint version, метрики,
   отдельные init/fit/validation/tuning/predict/total timings, размеры частей,
   device/package versions, error_type/message. Старые consumers не ломать молча.
4. Отделить score computation от `landmarkers/custom.py`: сейчас candidate runner
   импортирует оттуда `classification_score`, что связывает часть Дианы с Ритой.
   Общий helper/границу согласовать, не редактируя её landmarkers ради кандидатов.
5. Добавить model-specific device и preprocessing contracts; проверить MPS в
   официальных реализациях, а не объявлять READY по факту наличия Mac.
6. Расширить storage/resume/prediction schemas: существующий upsert не является
   resume scheduler, row-level predictions и PriorTune validation trials не хранит.
7. Не считать старый LightGBM из TabArena нужным published PriorTune baseline:
   explicit параметры и selection metric должны быть исследованы на этапе 1.

## Нужно создать

- **Этап 1:** source/API/preflight audit четырёх variants, published portfolio
  provenance, `config/candidates/lightgbm_baseline.yaml`,
  `scripts/candidates/check_models.py`. Ничего из этого на этапе 0 не создано.
- **Этап 2:** shared-manifest reader/integration boundary и
  `docs/DIANA_RITA_INTEGRATION.md` после получения общего списка/схемы.
- **Этап 3:** четыре совместимых wrapper: TabPFN26Candidate, TabICLv2Candidate,
  LightGBMBaselineCandidate, LightGBMPriorTuneCandidate; schema/mocks tests.
- **Этап 4:** официальный portfolio loader, production K=25 validation selection,
  отдельные trials и debug artifacts, validation-only selection tests.
- **Этапы 5–8:** по отдельной команде реальные smoke/mini-pilot/10×4 pilot,
  predictions, resume/failure isolation и dry-run join с Ритой.

Нельзя заранее переносить готовность кода или результаты прежних mocks на
readiness четырёх новых вариантов. Никакие portfolio configurations не генерируются
и не переписываются вручную.

## Потенциальные конфликты с веткой Риты

- `datasets/`, registry, PMLB mapping и split ownership — общий контракт; Диана
  не перемещает raw PMLB в новый каталог и не регистрирует их альтернативным способом.
- `artifacts.IDENTITY`, config/path helpers, `pyproject.toml`, общие tests/docs —
  точки возможных merge/schema conflicts. Не менять общий seed/ID semantics молча.
- Зависимость candidates → landmarkers/custom.py следует устранить согласованно;
  текущая ветка не реализует и не переписывает PyMFE/пять landmarkers/meta-model/LODO.
- Существующий CandidateResult используется tests и score-to-gap bridge: расширение
  требует backward compatibility или явной версии, а не удаления старой abstraction.
- Smoke/debug/pilot результаты и historical TabArena gaps должны иметь различимые
  namespaces/paths, чтобы результаты отладки не стали scientific results.

## Что зависит от shared manifests

Перед первым запуском реального dataset нужны от Риты:

1. Фиксированные 10 dataset_id с точным соответствием CSV PMLB и target mapping.
2. Shared manifests с train/validation/test **positional** indices, единым seed,
   dataset_id/split_id, fingerprint/raw identity и подтверждённым порядком строк.
3. Общая location/schema/способ передачи manifests: текущие local JSON игнорируются Git.
4. Подтверждение наличия всех классов в каждой части; будущий reader проверяет
   пересечения, объединение строк, seed и распределение классов до fit.

Готовые meta-features или landmarks для запуска кандидатов не нужны. Их ожидание
не блокирует этап 1. Отсутствие shared pilot manifests блокирует реальные dataset
запуски, но не source audit/preflight без обучения.

## Релевантное дерево

```text
ML_project/  (локальный checkout VAK_ML)
├── .gitattributes                       PMLB CSV через Git LFS
├── pyproject.toml
├── config/
│   ├── project.yaml                     raw/splits/derived/results paths
│   ├── datasets.yaml
│   ├── candidates.yaml                  прежний generic runner
│   ├── research.yaml                    shared старые настройки, часть Риты
│   └── tabarena_results.yaml             historical cached artifacts
├── data/
│   ├── PMLB/*.csv                       131 tracked CSV
│   ├── raw/<id>/metadata.yaml            прежний local catalog
│   ├── registry/datasets.csv             три smoke IDs
│   ├── splits/<smoke_id>/holdout_0.json   только local smoke manifests
│   └── derived/
│       ├── candidate_scores/
│       ├── meta_features/                часть Риты
│       └── landmarkers/                  часть Риты
├── src/meta_selector/
│   ├── candidates/{runner,results}.py
│   ├── datasets/{loader,validator,registry,models,splits}.py
│   ├── tabarena/{adapter,model_mapping,results_loader}.py
│   ├── meta_model/tabpfn_predictor.py     regressor, часть Риты
│   ├── landmarkers/custom.py             текущая import-зависимость кандидатов
│   ├── artifacts.py
│   └── paths.py
├── scripts/{run_candidate,check_candidate_models}.py
├── tests/{test_research_pipeline,test_dataset_loading,test_tabarena_results}.py
├── docs/DIANA_CANDIDATES_AUDIT.md         этот документ
├── results/{tables,logs,figures}/
└── .cache/tabarena_results/               локальный historical cache
```

## Проверки и проблемы этапа 0

- Source/config/file-tree inspection выполнен; поиск показал отсутствие PMLB
  adapter, PriorTune и четырёх специальных wrappers. Это статические выводы.
- Выполнен scoped Git diff для src/scripts/config/tests/docs/README/attributes:
  до создания аудита изменений в этих путях не показано. Raw datasets не редактировались.
- Обычный `git status` завершился ошибкой Git LFS clean filter: sandbox запретил
  запись во временный `.git/lfs/tmp/...` для `prnn_crabs.csv`. Это **не** доказательство
  повреждения CSV и **не** auto-review rejection. LFS/config/index не исправлялись;
  полный working-tree status не подтверждён. Не использовать обход фильтра с
  последующим `git add`, который мог бы заменить LFS pointers обычными CSV blobs.
- Для ветки использованы Git refs на том же коммите без checkout/LFS hydration.
  Fetch/pull/push/commit и изменение истории в этом этапе не выполнялись.
- Новых runtime проверок зависимостей, MPS, лицензий, official APIs и portfolio
  не было. Они принадлежат **этапу 1**, как и изучение официальных источников ниже.

Официальные источники, которые необходимо изучить до реализации моделей:
[PriorTune paper](https://arxiv.org/abs/2611.05141),
[PriorTune repository](https://github.com/yandex-research/PriorTune),
[TabPFN repository](https://github.com/PriorLabs/TabPFN),
[TabICL repository](https://github.com/soda-inria/tabicl),
[LightGBM documentation](https://lightgbm.readthedocs.io/).
Ссылка на отдельную PriorTune documentation и её соответствие paper/code ещё не
проверены; API, baseline, selection metric и структура portfolio не предполагаются.

**Результат этапа 0: PASS — аудит выполнен, ограничения обнаружены и записаны.**
Это не статус READY четырёх моделей. Для перехода к этапу 1 нужна явная команда
пользователя; никаких работ следующего этапа автоматически не выполняется.
