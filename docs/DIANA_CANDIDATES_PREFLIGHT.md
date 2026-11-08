# Диана: этап 1 — официальные API, окружение и preflight

Дата: 8 ноября 2026. Ветка: `diana/candidates`, база `05fd915`.
Исходная точка — принятый [аудит этапа 0](DIANA_CANDIDATES_AUDIT.md); он не повторялся.

## Итог

**PASS:** повторный preflight на локальной машине, выполненный вручную и
подтверждённый пользователем, дал READY для всех четырёх target variants.
Exact classifier checkpoints TabPFN-2.6 и TabICLv2 доступны локально с SHA256,
совпадающими с зафиксированной официальной metadata. Источники/API и published
portfolio подтверждены; нативный LightGBM исправлен.
Ни один PMLB dataset не загружался в модель; fit/predict, benchmark, PyMFE,
landmarkers, meta-model и LODO не запускались. Shared splits/loader не создавались.

| Variant | Exact version | Source | Device support | READY |
|---|---|---|---|---|
| TabPFN-2.6 | SDK 9.2.0, generation `v2.6` | PriorLabs/TabPFN; Prior-Labs/tabpfn_2_6 | CPU/MPS/CUDA API; локальный device preflight — MPS | READY |
| TabICLv2 | package 2.2.0, classifier v2-20260212 | soda-inria/tabicl; jingang/TabICL | CPU/MPS/CUDA API; локальный device preflight — MPS | READY |
| LightGBM baseline | lightgbm 4.7.0 | Official LightGBM + explicit project YAML | CPU для этого варианта; MPS не используется | READY |
| LightGBM + PriorTune T K=25 | LightGBM 4.7.0 + pinned published resource | mrsergazinov/PriorTune | CPU | READY на уровне dependency/resource preflight |

READY означает import, конструктор API и доступность требуемых локальных ресурсов,
**не успешный fit/inference будущего wrapper**. Wrappers относятся к этапу 3;
validation selection — к этапу 4. Полный inference на MPS не проверялся.

### Ручное завершение preflight

Пользователь сообщил результаты повторной локальной проверки:
TabPFN-2.6 — READY, TabICLv2 — READY, LightGBM — READY, PriorTune K=25 — READY.
Device preflight: **torch = MPS**, **LightGBM/PriorTune = CPU**.
Пути относительно корня проекта:

| Модель | Локальный exact checkpoint | SHA256 |
|---|---|---|
| TabPFN-2.6 | `checkpoints/tabpfn/tabpfn-v2.6-classifier-v2.6_default.ckpt` | `0578fa56f97e11024e31735aaec2c4e7332584b7730242fbaf6c0bbd0299206a` |
| TabICLv2 | `checkpoints/tabicl/tabicl-classifier-v2-20260212.ckpt` | `bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0` |

Это обновление фиксирует предоставленный пользователем результат; новый preflight,
перехеширование или загрузка весов при редактировании отчёта не выполнялись.
Checkpoints, веса и токены в коммит не добавляются. Этап 2 не начинался.

## Проверенное окружение

- Python 3.13.7, Darwin arm64.
- Torch 2.14.1; CUDA отсутствует; MPS built=true и available=true вне sandbox.
- tabpfn 9.2.0, tabicl 2.2.0, lightgbm 4.7.0, huggingface-hub 2.2.0.
- AutoGluon отсутствует. PriorTune не установлен как pip package: официальный
  helper находится в `bin/priortune.py`, его read-only subset API проверен отдельно.
- Первый LightGBM import упал с `Library not loaded: @rpath/libomp.dylib`.
  Выполнено `HOMEBREW_NO_AUTO_UPDATE=1 HOMEBREW_NO_INSTALL_CLEANUP=1 brew install libomp`:
  установлен libomp **23.1.5**, 1.8 MB, `/opt/homebrew/Cellar/libomp/23.1.5`.
  После этого нативный import и конструктор LightGBM проходят. Python-пакеты не обновлялись.
- LightGBM и torch-модели импортируются в разных subprocess. Это изолирует нативные
  OpenMP runtimes; macOS-конфликт отдельно отмечен и в официальном `bin/tune.py`.

В sandbox MPS reported unavailable; тот же минимальный preflight с доступом к
реальному устройству вернул MPS available и правильный результат матричного
умножения 2×2. Поэтому sandbox-результат не выдаётся за отсутствие MPS у Mac.
`auto`: CUDA → MPS → CPU; LightGBM/PriorTune остаются CPU по выбранному варианту.
Backend-specific API probe может консервативно перевести `auto` на CPU; явный
`--device` не подменяется молча. Численная проверка torch не доказывает корректность
всего classifier inference: это отдельная проверка этапа 5.

## Зафиксированные официальные источники

| Источник | Revision |
|---|---|
| [PriorTune](https://github.com/yandex-research/PriorTune/tree/104a255b4ad86755638c183e7749d60a64a6d903) | `104a255b4ad86755638c183e7749d60a64a6d903` |
| [TabPFN](https://github.com/PriorLabs/TabPFN/tree/4992bca68cc6cb4db4551b4668d1b95a201b9006) | `4992bca68cc6cb4db4551b4668d1b95a201b9006` |
| [TabICL](https://github.com/soda-inria/tabicl/tree/d83091b845a8c5f7c087bd54964f755a9e560f45) | `d83091b845a8c5f7c087bd54964f755a9e560f45` |
| [PriorTune resources](https://huggingface.co/datasets/mrsergazinov/PriorTune/tree/8b9cab4c0d6569ddda5cc2a18d78e0de75c526ee) | `8b9cab4c0d6569ddda5cc2a18d78e0de75c526ee` |

Прочитаны [paper v1](https://arxiv.org/html/2611.05141v1),
[project website](https://yandex-research.github.io/PriorTune/), README и HF resource
card; API/метрики сверены с кодом. Website — обзор исследования, не самостоятельный
Python package/API reference. GitHub REST API вернул rate limit: revisions получены
через `git ls-remote`, исходники прочитаны через raw URLs с этими SHA, без clone.
Копии текстов находятся в игнорируемом `.cache/diana_preflight/`.

Latest source revisions — provenance исследования API; они не подменяют версии
реально установленных SDK. Для фактического запуска приоритет имеют explicit
generation/checkpoint и совместимый локально проверенный API, а не latest default.

## TabPFN-2.6: классификатор, не regressor

Подтверждено в установленном SDK и [официальном classifier.py](https://github.com/PriorLabs/TabPFN/blob/4992bca68cc6cb4db4551b4668d1b95a201b9006/src/tabpfn/classifier.py):

```python
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

classifier = TabPFNClassifier.create_default_for_version(
    ModelVersion.V2_6,
    model_path="checkpoints/tabpfn/tabpfn-v2.6-classifier-v2.6_default.ckpt",
    device="mps",  # либо проверенный cpu/cuda
)
```

При первоначальном автоматическом preflight такой конструктор создан без fit,
загрузки checkpoint и сети.
`ModelSource.get_classifier_v2_6()` подтверждает repository и filename; predict и
predict_proba доступны. Environment default/latest generation не используется.
Model-native preprocessing сохраняется за SDK; AutoGluon/LightGBM encoding этому
классификатору не навязывается. Пределы данных и полный preprocessing проверяются
при реализации wrapper, не на реальных данных этапа 1.

Checkpoint metadata:

- Repo: `Prior-Labs/tabpfn_2_6`.
- Revision: `24148873a9a5992b429833d07363690dcffd02a8`.
- File: `tabpfn-v2.6-classifier-v2.6_default.ckpt`, 43 044 699 bytes.
- SHA256: `0578fa56f97e11024e31735aaec2c4e7332584b7730242fbaf6c0bbd0299206a`.

Исходные SHA/размер получены из официальной HF tree/LFS metadata.
При повторном ручном preflight пользователь подтвердил доступность локального
checkpoint по пути выше и совпадающий SHA256; прежний blocker отсутствующих весов снят.
Preflight проверяет локальный путь или exact HF cache entry и SHA; если их нет,
возвращает `unavailable_exact_version`. Это означает отсутствие локальной готовности,
а не отсутствие поколения 2.6 вообще. Подмена на 3.x запрещена.

По [официальному README](https://github.com/PriorLabs/TabPFN/blob/4992bca68cc6cb4db4551b4668d1b95a201b9006/README.md)
downloader использует PriorLabs login/license acceptance; headless-вариант —
`TABPFN_TOKEN`. Установленный `model_loading.py` вызывает `ensure_license_accepted`
до скачивания v2.6. HF metadata одновременно имеет `gated=false`; это **не отменяет**
license/auth flow SDK. Лицензия весов отдельно от code license. При первоначальном
автоматическом preflight auth не проверялся и лицензия за пользователя не принималась.
Текущий PASS основан на сообщённой пользователем локальной readiness; сведения о
токенах и способе прохождения authentication не запрашивались и не записывались.

## TabICLv2

В [официальном API](https://github.com/soda-inria/tabicl/blob/d83091b845a8c5f7c087bd54964f755a9e560f45/src/tabicl/_sklearn/classifier.py)
и установленном package 2.2.0 подтверждён конструктор:

```python
from tabicl import TabICLClassifier

classifier = TabICLClassifier(
    checkpoint_version="tabicl-classifier-v2-20260212.ckpt",
    model_path="checkpoints/tabicl/tabicl-classifier-v2-20260212.ckpt",
    allow_auto_download=False,
    device="mps",
)
```

- Repo: `jingang/TabICL`, revision `4dcd344ece2c00be9e831fdd35bed57b5ad83e19`.
- File: `tabicl-classifier-v2-20260212.ckpt`, 110 368 038 bytes.
- SHA256: `bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0`.
- Локальный путь: `checkpoints/tabicl/tabicl-classifier-v2-20260212.ckpt`;
  доступность и совпадающий SHA256 подтверждены пользователем в ручном preflight.
- Exact constructor, predict/predict_proba подтверждены без загрузки checkpoint.
  `allow_auto_download=False`; v1/v1.1 не выбираются.
- Источник публичный, HF metadata `gated=false`, BSD-3-Clause model card. Auth не
  понадобился для metadata; при первоначальной автоматической проверке веса
  не скачивались. Позднее пользователь подтвердил локальный exact checkpoint.
- API принимает mixed tabular inputs и имеет свой preprocessing/ensemble normalization.
  Не заменяем его общим LightGBM pipeline. Fit/target validation выполняются позже.
- [Официальный device resolver](https://github.com/soda-inria/tabicl/blob/d83091b845a8c5f7c087bd54964f755a9e560f45/src/tabicl/_torch_devices.py)
  также поддерживает XPU, но наш согласованный `auto` ограничен CUDA/MPS/CPU.
  Его guard для виртуализированных Mac учитывается в API probe.

## PriorTune: точный published portfolio

- Repository: `mrsergazinov/PriorTune` (dataset resource), revision указан выше.
- Artifact: **`portfolios/lgbm/T_K25.json`**, 13 979 bytes.
- SHA256: `322a4e87966223ec268cdf390ba8d7a58e3b13d2a2a4a7f4dddd3989ab2688e5`.
- `model=lgbm`, `objective=T`, `K=25`; ровно 25 distinct configs, все
  `tabprep=null`, `tabprep_enabled=false`; каждая содержит `lightgbm` и `model_seed`.
- `config_ids` — множество, в JSON отсортированное; исполняемый порядок — **add_order**:

```text
1452, 3921, 2447, 2005, 3134, 2428, 3599, 3454, 1959, 3585,
926, 1487, 1378, 1460, 3937, 1260, 95, 3044, 3545, 1264,
2372, 3372, 2076, 2737, 410
```

В [официальном helper](https://github.com/yandex-research/PriorTune/blob/104a255b4ad86755638c183e7749d60a64a6d903/bin/priortune.py)
есть `subset('lgbm', rule='T', k=25)` и `autogluon('lgbm', rule='T', k=25)`.
Оба вызваны read-only в отдельном процессе: 25 записей, порядок совпал с add_order,
AutoGluon helper сформировал 25 GBM dicts без обучения или импорта AutoGluon.
Helper SHA256: `c6fdb46d6c1adbc1537b71dde57888192800419bf6cb3be4b6b90bcb621198c4`.
JSON содержит полные selected configs: скачивать 4 000-config bank, scores,
predictions или synthetic datasets для нашего preflight не требуется.

Selection metric из [tune.py](https://github.com/yandex-research/PriorTune/blob/104a255b4ad86755638c183e7749d60a64a6d903/bin/tune.py)
и [ensemble.py](https://github.com/yandex-research/PriorTune/blob/104a255b4ad86755638c183e7749d60a64a6d903/bin/hillclimb/ensemble.py):
binary — **ROC-AUC, maximize**, multiclass — **cross-entropy/log-loss, minimize**.
В T выбирается один validation-best config, при равенстве официальный selector
использует меньший config_id. Balanced Accuracy остаётся общей отчётной метрикой,
не подменяет selection metric. Regression RMSE этому classification pilot не нужен.

Наш вариант — `model_family=lightgbm`, `variant=priortune_t_k25`; plain LightGBM,
не lgbm-prep, не TE/ensemble, не synthetic meta-training и не новый search.
Portfolio не переписывался: исходные JSON bytes сохранены только в `.cache/`.

## LightGBM baseline: явное решение проекта

Examined sources описывают random/meta-tuned portfolios и AutoGluon training
defaults. Единственный полностью специфицированный native baseline, пригодный
для прямого копирования в наш classifier, не найден. Поэтому выбран permitted
fallback **explicit project baseline**, не claimed published paper baseline.

Все выбранные параметры находятся в `config/candidates/lightgbm_baseline.yaml`:

- Package 4.7.0; native CPU LGBMClassifier, gbdt, 31 leaves, max_depth=-1.
- learning_rate=0.05; max 1 000 estimators; validation-only early stopping=50,
  min_delta=0, first_metric_only=true; без refit train+validation.
- min_child_samples=20, min_child_weight=0.001; subsample/colsample=1;
  subsample_freq=0; L1=L2=0; class_weight=null; min_split_gain=0.
- cat_l2=10, cat_smooth=10, max_cat_to_onehot=4, min_data_per_group=100;
  subsample_for_bin=200000, max_bin=255.
- random_state/data_random_seed/feature_fraction_seed/bagging_seed=42;
  n_jobs=1, deterministic=true, force_col_wise=true, verbosity=-1.
- Objective: binary/multiclass задаётся task явно, num_class — по train labels.
  Validation metric: AUC/multi_logloss. Future categorical dictionary fit only on train,
  unknown values become missing; numeric missing values остаются native LightGBM.

Это все **выбранные model/fit параметры**, не попытка вручную перечислить каждый
неактивный engine setting. Остальные options фиксирует exact package version;
будущий fit сохранит effective Booster parameters. Published PriorTune configs и
их model_seed не переопределяются baseline-параметрами.
Бюджет и callbacks native baseline не объявляются тождественными AutoGluon из статьи;
до научных сравнений нужно отдельно отчитать эту разницу.

## Различия paper/code и наш обязательный протокол

1. В статье evaluation использует benchmark/bagging protocol. Пользовательский
   `bin/tune.py` — другой интерфейс: создаёт holdout validation самостоятельно.
   Для pilot Дианы это запрещено: на этапе 2 нужны manifests Риты.
2. `tune.py:write_dataset` factorize-ит категории объединённых train/test данных.
   Это не соответствует нашему stricter train-only preprocessing; эту обвязку
   нельзя переносить без изменения. Model-native transforms должны fit только train.
3. CLI fitting получает test predictions **для каждой** из K конфигураций и потом
   deploy-ит победителя. Даже если выбор зависит от validation, наш протокол строже:
   test оценка только winner. На этапе 4 переиспользуются published configs/metrics,
   а не этот CLI целиком.
4. `priortune.autogluon` формирует GBM configs, но автоматический weighted ensemble
   AutoGluon не является T. Будущий wrapper должен отключить TE и выбрать одну модель.
5. Latest defaults TabPFN/TabICL могут меняться; мы фиксируем поколения явно.
   HF metadata access не доказывает acceptance лицензии downloader.

Таким образом, authoritative источники portfolio/selection — pinned JSON и
официальный helper/source; shared split и train-only/test-only-winner требования
задания имеют приоритет перед удобной CLI-обвязкой. Полная репликация paper training
protocol не заявляется. Никакая адаптация stage 2–4 здесь не реализована.

## Команды и артефакты

```bash
source .venv/bin/activate
python scripts/candidates/check_models.py

# После fresh clone — только один small published JSON, никакие weights:
python scripts/candidates/check_models.py --fetch-portfolio

# С подтверждёнными локальными classifiers, из корня проекта:
python scripts/candidates/check_models.py \
  --tabpfn-checkpoint checkpoints/tabpfn/tabpfn-v2.6-classifier-v2.6_default.ckpt \
  --tabicl-checkpoint checkpoints/tabicl/tabicl-classifier-v2-20260212.ckpt \
  --device auto --json .cache/diana_preflight/report.json
```

CLI offline по умолчанию; единственная download option — `--fetch-portfolio`,
pin revision + SHA256, limit 128 KiB, atomic replace. No weights, no raw, no fitting.
Exact checkpoints проверяются по опубликованным LFS SHA256. При explicit пути с
неверным hash не выполняется скрытый поиск «других» весов.
Exit 0 — все четыре предварительно READY; exit 1 — хотя бы один NOT READY.
Первоначальный автоматический отчёт вне sandbox сохранялся в
`.cache/diana_preflight/report_native.json` и отражал отсутствие checkpoints
(exit 1). Это исторический результат до ручного завершения этапа, а не текущий
blocker. Итоговый PASS фиксируется по сообщению пользователя о повторном локальном
preflight: все четыре variants READY. Новый JSON-отчёт при этом обновлении не
создавался. Sandbox report оставлен отдельно, чтобы не скрывать разницу видимости
MPS. Tokens/auth cache contents в отчёты не выводятся.

При первоначальной реализации этапа созданы: этот документ, `config/candidates/preflight.yaml`,
`config/candidates/lightgbm_baseline.yaml`, `src/meta_selector/candidates/preflight.py`,
`scripts/candidates/check_models.py`. При текущем обновлении изменён только этот
отчёт; исходный код и конфигурация не менялись. Коммиты не создавались,
checkpoints/веса/токены в Git не добавлялись.
Audit stage 0 сохранён; raw PMLB, manifests и часть Риты не изменялись.

## Что требуется дальше

- От Дианы: для завершения этапа 1 дополнительных действий не требуется —
  exact checkpoints доступны локально, повторный preflight подтверждён как PASS.
  Веса и токены не коммитить и токены в чат не присылать.
- От Риты: для завершения этого preflight ничего; на следующем этапе — 10 IDs,
  PMLB/target mapping и shared split manifests/seed. Они не создавались за Риту.
- **STOP. Этап 2 не начинается без команды пользователя.**
