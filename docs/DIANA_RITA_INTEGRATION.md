# Общий контракт Дианы и Риты: PMLB pilot 10

Этап 2, 9 ноября 2026. Ветка `diana/candidates`.
Статус: **PASS** для mapping, identity, target и shared split consumption всех 10 datasets.
Этапы моделей/runner/PriorTune не начинались. Все параметры и индексы взяты из
переданного authoritative bundle; новые splits не создавались.

## Authoritative bundle и безопасный импорт

Источник: `incoming/pmlb_pilot_for_diana.zip`, 1 358 130 bytes.
ZIP SHA256, вычисленный при импорте:
`fc1b34676b13e8af071dd382f598bd9440997c00ae7ec305f8b0efdfd3d47f25`.
Отдельный out-of-band SHA от Риты не передавался; этот hash фиксирует полученный
архив и не выдаётся за независимо предоставленную отправителем подпись.

Состав: 64 файла (4 верхнего уровня, 20 dataset CSV/metadata, 30 TXT indices,
10 JSON manifests), распакованный размер 4 561 333 bytes.
`SHA256SUMS.json` проверен: hashes всех остальных 63 файлов совпали;
SHA checksum-файла:
`577cd4d81fc5591a64e64741878a04a37cb0e384ef08a9e706d53a473f63751a`.

Распаковка только в `.cache/rita_pilot_import/<archive_sha256>/`.
Запрещены absolute/traversal/Windows paths, symlinks, case-colliding/duplicate names,
шифрованные записи и превышение лимитов (256 записей, 32 MiB всего, 8 MiB на файл).
Файлы извлечены без изменения bytes. `verify_package.py` прочитан, но **не исполнялся**;
проверка выполнена нашим importer. CSV из архива остаются только в quarantine,
не копируются в production raw. Для работы используются существующие `data/PMLB/`.

### Дерево архива

```text
pmlb_pilot_for_diana.zip
├── README.md
├── SHA256SUMS.json
├── datasets.csv
├── verify_package.py
├── datasets/
│   ├── pmlb_saheart/
│   │   ├── saheart.csv
│   │   └── metadata.yaml
│   ├── pmlb_tic_tac_toe/
│   │   ├── tic_tac_toe.csv
│   │   └── metadata.yaml
│   ├── pmlb_kr_vs_kp/
│   │   ├── kr_vs_kp.csv
│   │   └── metadata.yaml
│   ├── pmlb_mushroom/
│   │   ├── mushroom.csv
│   │   └── metadata.yaml
│   ├── pmlb_hill_valley_without_noise/
│   │   ├── Hill_Valley_without_noise.csv
│   │   └── metadata.yaml
│   ├── pmlb_twonorm/
│   │   ├── twonorm.csv
│   │   └── metadata.yaml
│   ├── pmlb_penguins/
│   │   ├── penguins.csv
│   │   └── metadata.yaml
│   ├── pmlb_car_evaluation/
│   │   ├── car_evaluation.csv
│   │   └── metadata.yaml
│   ├── pmlb_vehicle/
│   │   ├── vehicle.csv
│   │   └── metadata.yaml
│   ├── pmlb_waveform_21/
│   │   ├── waveform_21.csv
│   │   └── metadata.yaml
├── indices/<каждый из 10 dataset_id>/
│   ├── train.txt
│   ├── validation.txt
│   └── test.txt
└── splits/<каждый из 10 dataset_id>/
    └── holdout_0.json
```

## Canonical mapping и хранение

- `config/pilot_datasets.yaml` — authoritative table, автоматически перенесённая
  из Ритиного `datasets.csv`/metadata. Содержит IDs, **dataset_source=pmlb**, точные
  case-sensitive CSV filenames, target, seed/split_id, размеры, fingerprints,
  categorical semantics, origin_group и исходную metadata.
- `config/pilot_splits/<dataset_id>/holdout_0.json` — **побайтовые копии** оригинальных
  JSON Риты. Они не пересериализованы, индексы и их порядок не менялись.
- `data/PMLB/<file>.csv` — единственное production-расположение данных этого pilot.
  В `data/raw` ничего не переносится, CSV не переименовываются. Existing registry
  smoke datasets и старый local loader сохраняются; 10 metadata.yaml не создавались.
- `.cache/rita_pilot_import/validation_report.json` — локальный read-only validation
  report с class distributions, размерами и fingerprint checks.

Выбрана отдельная **tracked-by-policy директория config/pilot_splits/**, поскольку
это небольшой текстовый scientific protocol. Правила Git уже позволяют её хранить;
`.gitignore` не менялся, весь `data/splits/` по-прежнему игнорируется. Нужные files
можно передавать обычным Git merge между ветками без архивов/больших CSV.
В `.gitattributes` добавлено только `config/pilot_splits/** -text`: это отключает
автоматическую замену LF/CRLF на checkout/add, чтобы byte SHA256 Риты оставались
валидными между ОС. JSON остаётся обычным текстом для diff и не попадает в LFS.
Существующее правило LFS для raw PMLB не изменено.
Суммарный размер JSON manifests — менее 0.5 MiB. Текстовые индексы легко проверить
в diff, но при обновлении возможны конфликты: владелец splits — **Рита**, Диана
их не разрешает путём повторной генерации. Importer отказывается перезаписывать
изменённый существующий контракт. Файлы подготовлены для Git; commit/push этим
этапом не выполнялись.

Альтернативы: открыть весь data/splits через ignore exception смешивает pilot
с локальными smoke manifests; обмен только через игнорируемую cache не создаёт
общего versioned scientific contract. Поэтому выбран config, без глобального
ослабления .gitignore.

## 10 pilot datasets

Во всех строках `target=target`, `seed=42`, `split_id=holdout_0`.
Seeds не выбраны Дианой: они явно переданы в таблице Риты.

| dataset_id | CSV в data/PMLB | rows | features | classes | train | val | test | seed | split_id | status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| pmlb_saheart | saheart.csv | 462 | 9 | 2 | 276 | 93 | 93 | 42 | holdout_0 | PASS |
| pmlb_tic_tac_toe | tic_tac_toe.csv | 958 | 9 | 2 | 574 | 192 | 192 | 42 | holdout_0 | PASS |
| pmlb_kr_vs_kp | kr_vs_kp.csv | 3196 | 36 | 2 | 1917 | 639 | 640 | 42 | holdout_0 | PASS |
| pmlb_mushroom | mushroom.csv | 8124 | 22 | 2 | 4874 | 1625 | 1625 | 42 | holdout_0 | PASS |
| pmlb_hill_valley_without_noise | Hill_Valley_without_noise.csv | 1212 | 100 | 2 | 726 | 243 | 243 | 42 | holdout_0 | PASS |
| pmlb_twonorm | twonorm.csv | 7400 | 20 | 2 | 4440 | 1480 | 1480 | 42 | holdout_0 | PASS |
| pmlb_penguins | penguins.csv | 333 | 7 | 3 | 199 | 67 | 67 | 42 | holdout_0 | PASS |
| pmlb_car_evaluation | car_evaluation.csv | 1728 | 6 | 4 | 1036 | 346 | 346 | 42 | holdout_0 | PASS |
| pmlb_vehicle | vehicle.csv | 846 | 18 | 4 | 507 | 169 | 170 | 42 | holdout_0 | PASS |
| pmlb_waveform_21 | waveform_21.csv | 5000 | 21 | 3 | 3000 | 1000 | 1000 | 42 | holdout_0 | PASS |

Итого 6 binary + 4 multiclass. В данном pilot отсутствующих классов нет.

### Class counts

Числа labels записаны как в CSV, без label remapping. В каждом словаре `label: count`.

| dataset_id | train | validation | test |
|---|---|---|---|
| pmlb_saheart | `{"0": 180, "1": 96}` | `{"0": 61, "1": 32}` | `{"0": 61, "1": 32}` |
| pmlb_tic_tac_toe | `{"0": 199, "1": 375}` | `{"0": 66, "1": 126}` | `{"0": 67, "1": 125}` |
| pmlb_kr_vs_kp | `{"0": 916, "1": 1001}` | `{"0": 305, "1": 334}` | `{"0": 306, "1": 334}` |
| pmlb_mushroom | `{"0": 2524, "1": 2350}` | `{"0": 842, "1": 783}` | `{"0": 842, "1": 783}` |
| pmlb_hill_valley_without_noise | `{"0": 360, "1": 366}` | `{"0": 120, "1": 123}` | `{"0": 120, "1": 123}` |
| pmlb_twonorm | `{"0": 2221, "1": 2219}` | `{"0": 741, "1": 739}` | `{"0": 741, "1": 739}` |
| pmlb_penguins | `{"0": 88, "1": 40, "2": 71}` | `{"0": 29, "1": 14, "2": 24}` | `{"0": 29, "1": 14, "2": 24}` |
| pmlb_car_evaluation | `{"0": 726, "1": 230, "2": 41, "3": 39}` | `{"0": 242, "1": 77, "2": 14, "3": 13}` | `{"0": 242, "1": 77, "2": 14, "3": 13}` |
| pmlb_vehicle | `{"1": 127, "2": 130, "3": 131, "4": 119}` | `{"1": 43, "2": 43, "3": 43, "4": 40}` | `{"1": 42, "2": 44, "3": 44, "4": 40}` |
| pmlb_waveform_21 | `{"0": 994, "1": 988, "2": 1018}` | `{"0": 331, "1": 330, "2": 339}` | `{"0": 332, "1": 329, "2": 339}` |

## Row identity, fingerprints и seed

Все 10 текущих CSV **побайтово совпали** с версиями в Rita bundle:
`SHA256(data/PMLB/file) = datasets.csv.csv_sha256 = metadata.file.sha256`.
Эта сверка подтверждает содержимое, порядок колонок и строк, а не только размеры.
Индексы — positional, zero-based, относительно строк данных **после header CSV**;
это явно указано в README Риты. Применяется `.iloc`, index_col не задаётся.
До slicing нет sorting, filtering, dropna, deduplication, encoding или scaling.
Один и тот же проверенный byte snapshot используется для fingerprint и CSV parser.

У всех 10 `rita_fingerprint_present=true`. В canonical manifest различаются:

- `fingerprint` + `fingerprint_method=sha256_raw_csv_bytes` — байтовая identity,
  совпавшая с уже переданным Ритой CSV SHA256; она не придумана вместо её hash.
- `rita_dataset_fingerprint` + `rita_fingerprint_version=2` — исходный pandas-level
  fingerprint, перенесённый без изменений. Его поле сверено между table и JSON.
  Он **не пересчитывается** как критерий совпадения между pandas 2.3.3 Риты и
  локальным pandas 3.0.6: решающее доказательство — exact CSV byte SHA256.
- `reader_dataframe_fingerprint` в runtime metadata — дополнительный локальный
  guard от изменения X/y после загрузки до slicing. Это fingerprint **нашей**
  representation, не якобы новый hash, вычисленный у Риты.

Если raw SHA изменился — reader FAIL, не использует CSV из cache/archive и не
пересоздаёт split. Неизвестная row basis тоже FAIL. Для другого неполного bundle
импорт останавливается с конкретным missing/invalid field; недостающие научные
параметры не генерируются.

Seed первого этапа — **42**, второго — **43**, сохранён отдельно как
`second_stage_seed`. Manifest `protocol.random_seed=42`, validation/test fractions
0.2/0.2; оригинальные indices определяют реальные размеры с округлением. Они
**не восстанавливаются** по seed. Protocol hash:
`d34dedae75071b7a2a869f0cd5333cd4560dc380d0f0d101b171bff675a5b8cf`.

### Fingerprints текущих CSV

| dataset_id | raw CSV SHA256 | Rita fingerprint present |
|---|---|---|
| pmlb_saheart | `4dec0d1a9a62519922240607c51d5edf6f74371d3725186eb87502f7a0f8c6b7` | true |
| pmlb_tic_tac_toe | `233e81861151be8efd2ec19c677aae9dfdec3696d52dd5ef2d62838ec5119f07` | true |
| pmlb_kr_vs_kp | `ab37abd66c3b6e3dee49648a43c0e409ca1965fa997744ca4aa91ed0af345d07` | true |
| pmlb_mushroom | `99b8ad897e555c5babe12c357a5d2261108ff816019fe56d3aae9fca09167a16` | true |
| pmlb_hill_valley_without_noise | `0f6439d66eabb0db8e80dab4a95d239d44a8b3190175e15b92d42fc349594e07` | true |
| pmlb_twonorm | `6e0baa44a10475fdeb7d0bed7ad7b08aa043ec0ae6ca62c82672f39fbc3c81a7` | true |
| pmlb_penguins | `f9989e6cf026364b44cc02426334c886f49c231d5229f83a849f670fb54ed40d` | true |
| pmlb_car_evaluation | `95a92209c4bf63240e808749c4d610d19dc6de0c20305b932eec7dadbbd52b0b` | true |
| pmlb_vehicle | `6f9bf82f4c27f90aba20bed16b71caaf91dd1a6443d3a8fbe6483706fd49ce02` | true |
| pmlb_waveform_21 | `3baf536f849c7031c7076484b7c7e5bae7331dd29d82ab97a6e44a7ee4197703` | true |

## Явное соответствие source labels и join contract

Bundle README и оригинальные JSON Риты содержат `dataset_source=local`.
Задание требует canonical `dataset_source=pmlb`. Это различие не скрывается:

- исходные manifests оставлены с `local`, без изменения bytes;
- canonical entry/adapter возвращает `pmlb` и хранит `rita_dataset_source=local`;
- top-level `source_label_mapping` фиксирует соответствие;
- canonical join **ровно `(dataset_id, split_id, seed)`**, без join по порядку строк.

IDs `pmlb_*`, split_id и оба seeds не изменены. Если у Риты таблицы по-прежнему
используют local, их не следует отбрасывать при join из-за source label; потребуется
это явно описанное соответствие, а не новый несовместимый ID. Dataset_source
остаётся provenance; разные sources без известного mapping не отождествляются.
Никакой dry-run join features/candidate results сейчас не выполнялся — это этап 8.

## Read-only API

```python
from meta_selector.datasets.shared_pilot import load_pmlb_dataset, load_shared_split

dataset = load_pmlb_dataset("pmlb_saheart")
split = load_shared_split("pmlb_saheart", "holdout_0", seed=42, dataset=dataset)
assert split.join_key == ("pmlb_saheart", "holdout_0", 42)
X_train, y_train = split.portion(dataset, "train")
X_validation, y_validation = split.portion(dataset, "validation")
X_test, y_test = split.portion(dataset, "test")
```

`load_pmlb_dataset` использует только authoritative mapping, возвращает совместимый
Dataset(X,y,metadata) с дополнительными origin/seed/categorical/fingerprint полями.
Target исключается строго по точному имени; похожие feature names не удаляются.
Размеры, target, classification, >=2 класса, header uniqueness и fingerprint проверяются.

Categorical columns берутся **из metadata Риты**, а не guessing по dtype/cardinality.
Сейчас X возвращается raw; категории представлены отдельной annotation, без fit
vocabulary на full dataset. Для mushroom stalk-root=0 сохраняется как unknown
**категория**, не превращается в NaN. Ordinal factors car_evaluation помечены
категориальными. Кодирование/категориальные словари будущих моделей fit только train.

`load_shared_split` читает только config manifests, проверяет их exact SHA,
ID/source/seed/protocol, bounds, duplicate positions, pairwise disjoint/full coverage,
размеры и все классы. Возвращает indices/seed/metadata, не вызывает `make_split`.
Отсутствие manifest — `shared pilot manifests are required`, без fallback.
При missing class — requiring review/FAIL; ни исправлений, ни training автоматически.
Старый `scripts/run_candidate.py` не используется: он всё ещё вызывает local make_split
и не является новым production pilot API. Его адаптация — последующий этап.

## PriorTune contract (только правило)

Ровно K=25 опубликованных configs обучаются на **train**, оцениваются на
**validation**, winner выбирается только по validation. Test не используется
для sorting, early stopping, best iteration или winner selection; только выбранный
winner получает final test evaluation. Selection metric остаётся официальной
ROC-AUC для binary / log-loss для multiclass; общая BA не подменяет её.
Runner не реализовывался: это этап 4.

## Проверки и воспроизведение

```bash
source .venv/bin/activate
# Без моделей: проверенный quarantine import и подготовка текстовых manifests.
python scripts/candidates/import_pilot_bundle.py   --archive incoming/pmlb_pilot_for_diana.zip   --report .cache/rita_pilot_import/validation_report.json
# После получения versioned config-файлов архив для reader не нужен:
python scripts/candidates/import_pilot_bundle.py --check-only
# Только unit tests данных/контракта; без моделей, PyMFE и landmarking.
python -m pytest -q tests/test_shared_pilot.py tests/test_dataset_loading.py
```

Проверки 10 реальных raw CSV выполнены read-only отдельной командой. Unit tests
используют tiny synthetic bundle (seed 77/88), который не является production
pilot. Вложенный fake verifier никогда не исполняется. Тесты покрывают mapping,
точное target exclusion, source normalization, неизменность bytes manifests/raw,
seed/split joins, неизвестные row basis, integrity, bounds, duplicates, disjointness,
coverage, missing classes, no fallback, отсутствие preprocessing перед slicing,
повторный импорт без overwrite и безопасную распаковку.

Результат: **79 tests passed** — 39 новых unit tests shared pilot и 40 прежних
проверок local dataset loading. Общий pytest не запускался:
в нём есть тесты части Риты с fitting landmarkers, запрещённым текущим этапом.
Проверены также неизменность SHA ZIP и 10 raw CSV, побайтовое совпадение 10 JSON
с архивом и Git attributes (JSON не в LFS, автоматической EOL-конверсии нет).
Исходные JSON используют CRLF; `whitespace=cr-at-eol` позволяет Git проверять
их без ложных trailing-whitespace сообщений, сохраняя исходные bytes и SHA256.
Полный `git diff --check` в sandbox наткнулся на ранее известную ошибку Git LFS
clean filter для `data/PMLB/prnn_crabs.csv` (запись в `.git/lfs/tmp` запрещена).
LFS не исправлялся; это не ошибка проверки pilot CSV. Для файлов этого этапа
использованы scoped diff/whitespace checks без обработки raw LFS paths.

### Ownership и границы

Рита — владелец списка pilot, labels/categorical semantics, splits, seed, row identity.
Диана — потребитель этого контракта; исправления scientific info запрашиваются у
Риты, а не генерируются здесь. Нужные поля переданы; запросов недостающих данных нет.
Этап 1 сохранён PASS, веса/токены не затрагивались. Stage 2 не коммитил datasets,
checkpoints или tokens, не запускал модели. Этап 3 **не начат** и требует отдельной команды.
