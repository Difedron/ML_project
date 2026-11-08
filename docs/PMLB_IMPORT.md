# PMLB: импорт и подготовленные разбиения пилота

Дата проверки: 2026-11-08. Все десять задач из
[PMLB_PILOT_SELECTION.md](PMLB_PILOT_SELECTION.md) импортированы, зарегистрированы
и имеют проверенный `holdout_0`. PyMFE и модели не запускались.

## Команды

Из корня checkout, с установленными pandas, PyYAML, numpy и sklearn:

```powershell
# Только копирование проверенных CSV, создание metadata и регистрация:
python scripts/import_pmlb.py

# Импорт и создание/повторное использование фиксированных разбиений:
python scripts/import_pmlb.py --create-splits
```

Фактически выполнена в текущем окружении команда:

```powershell
py -3.12 -B scripts/import_pmlb.py --create-splits
```

Доступны `--project-root` и `--split-id` (по умолчанию `holdout_0`). При изменении
протокола существующие manifests не заменяются: используется существующая ошибка
`make_split`, требующая нового split ID. Повторный импорт при неизменных входах
переиспользует CSV, metadata и manifests, не переписывая их даже при совпадении байтов.
Реестр и производный JSON-отчёт записываются детерминированно.

## Контракт импорта

- `config/pmlb_pilot.yaml` — явный список ровно десяти утверждённых задач,
  стабильные ID `pmlb_<original_name.lower()>`, ожидаемые SHA-256, размерности,
  categorical columns, группы происхождения и ссылки на описания.
- Импортёр не обходит каталог автоматически и не подключает `catalog.csv`,
  резервные задачи, `chess` или другие родственные версии.
- До копирования проверяются SHA-256 всех источников и существующие назначения.
  Отличающиеся payload/metadata и посторонние файлы в папке назначения приводят
  к ошибке с сохранением существующих файлов.
- CSV копируется побайтово, проверяется до атомарной публикации, затем ещё раз
  сверяется с исходником. Исходные `data/PMLB/*.csv` не открываются для записи.
- Metadata сохраняют исходное имя файла, `source.type=pmlb`, `source.external_id`,
  ссылку, `file.sha256`, `categorical_columns` и provenance с исходным путём,
  отчётом выбора, первичным описанием и origin group.
- `metadata_revision` относится к справочным PMLB metadata, а не является
  утверждением о версии локального payload; локальную версию фиксирует SHA-256.
- Валидация и регистрация выполняются существующими `register_dataset`,
  `validate_dataset`/`load_dataset`. Schema реестра не менялась.
- Разбиения создаёт существующий `make_split`; проверки выполняют
  `check_manifest` и `portion`. Нового алгоритма разбиения нет.
- Как и остальные writers проекта, импорт рассчитан на одного writer в checkout.
  При прерванном импорте повторный запуск достраивает отсутствующие файлы,
  если уже скопированные файлы совпадают с утверждёнными.

## Типизация

`categorical_columns` — необязательное поле metadata. Отсутствие поля эквивалентно
пустому списку. Указанные признаки получают pandas dtype `category` до валидации
и fingerprint. Значения, порядок строк, числовые типы остальных признаков и target
сохраняются. Список не может содержать target, дубликаты, пустые/несуществующие имена.
Автоматического вывода категорий по cardinality нет. Поддержаны CSV/XLSX/Parquet.

Для mushroom все 22 признака категориальны; `stalk-root=0` сохранён как отдельный
уровень unknown в **2480 строках**, пропусков NaN в этом столбце нет. Эта политика
также записана в `metadata.yaml:provenance.unknown_categories`. Никакая импутация
при импорте/загрузке не выполняется.

Необязательное `file.sha256` проверяется loader перед чтением. При несовпадении
валидация возвращает `invalid`; без поля сохраняется прежнее поведение loader.

## Фактические результаты

В каждой задаче все исходные строки сохранены; NaN в X/y = 0.
`Числ./кат.` — число признаков после утверждённой типизации, до OHE.

| Dataset ID | Статус | n | d | k | Числ./кат. | Train | Validation | Test |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `pmlb_saheart` | valid | 462 | 9 | 2 | 8 / 1 | 276 | 93 | 93 |
| `pmlb_tic_tac_toe` | valid | 958 | 9 | 2 | 0 / 9 | 574 | 192 | 192 |
| `pmlb_kr_vs_kp` | valid | 3196 | 36 | 2 | 0 / 36 | 1917 | 639 | 640 |
| `pmlb_mushroom` | valid | 8124 | 22 | 2 | 0 / 22 | 4874 | 1625 | 1625 |
| `pmlb_hill_valley_without_noise` | valid | 1212 | 100 | 2 | 100 / 0 | 726 | 243 | 243 |
| `pmlb_twonorm` | valid | 7400 | 20 | 2 | 20 / 0 | 4440 | 1480 | 1480 |
| `pmlb_penguins` | valid | 333 | 7 | 3 | 5 / 2 | 199 | 67 | 67 |
| `pmlb_car_evaluation` | valid | 1728 | 6 | 4 | 0 / 6 | 1036 | 346 | 346 |
| `pmlb_vehicle` | valid | 846 | 18 | 4 | 18 / 0 | 507 | 169 | 170 |
| `pmlb_waveform_21` | valid | 5000 | 21 | 3 | 21 / 0 | 3000 | 1000 | 1000 |

Протокол взят из `config/research.yaml`: seed 42 для отделения test, seed 43
для отделения validation, доли примерно 60/20/20 с округлением sklearn.
Protocol hash: `d34dedae75071b7a2a869f0cd5333cd4560dc380d0f0d101b171bff675a5b8cf`.

Для всех десяти проверены:

1. Попарное отсутствие пересечений train/validation/test и покрытие всех строк.
2. Наличие всех классов в каждой outer-части.
3. Возможность 3-fold StratifiedKFold только на outer train, с сохранением всех
   классов в inner train/validation. Создавались только индексы, модели не fit-ились.
4. При повторном импорте — совпадение байтов и mtime исходных CSV, копий, metadata
   и manifests; совпадение байтов реестра и import report.
5. Независимое пересоздание тех же manifests для всех десяти реальных задач
   в изолированном тестовом корне с теми же данными, dtype, порядком и окружением.
6. Совпадение SHA-256 исходных и импортированных CSV с утверждённым отчётом выбора.
   Дополнительно `git diff --exit-code -- data/PMLB` завершился без различий.

## Сохранённые файлы

- Код/контракт: `scripts/import_pmlb.py`, `src/meta_selector/datasets/pmlb.py`,
  `config/pmlb_pilot.yaml`.
- Дополнены `src/meta_selector/datasets/models.py` и `loader.py`,
  документация `docs/DATASETS.md`; добавлены этот документ и `tests/test_pmlb_import.py`.
- Для каждого ID из таблицы: `data/raw/<id>/metadata.yaml` и CSV под исходным именем.
- Обновлён `data/registry/datasets.csv`: добавлены 10 valid-записей, прежние три
  smoke/toy-записи сохранены.
- Для каждого ID: `data/splits/<id>/holdout_0.json` с indices, protocol/seed,
  dataset fingerprint, protocol hash и environment.
- `results/logs/pmlb_import.json`: SHA-256, validation summary, типы, распределения
  классов, размеры частей, fingerprints и ссылки на manifests.
- `results/logs/pmlb_preparation_verification.json`: фактические результаты повторной
  проверки, SHA-256 manifests и минимальная поддержка классов в inner folds.

CSV-копии, split manifests, JSON-результаты и `.cache/` остаются локальными
игнорируемыми артефактами согласно существующему `.gitignore`. Metadata и код
доступны для контроля версий. Исходный отчёт выбора не изменён.

## Автоматические тесты

Фактически выполнено **69 тестов, все успешны, предупреждений нет**:

```text
69 passed in 16.56s
```

Запускались только `tests/test_dataset_loading.py` и `tests/test_pmlb_import.py`:
валидация/чтение, импорт, типы, повторные запуски и разбиения. Полный suite,
содержащий fit моделей и PyMFE extraction, на этом этапе не запускался.

Команда проверки в текущем окружении Windows (все временные данные внутри checkout):

```powershell
$env:PYTHONPATH = "$PWD/.cache/test-deps"
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
$env:TEMP = "$PWD/.cache/tmp"
$env:TMP = $env:TEMP
py -3.12 -B -m pytest tests/test_pmlb_import.py tests/test_dataset_loading.py -q --basetemp ".cache/pytest-pmlb" -o cache_dir=".cache/pytest-cache"
```

Python 3.12.0, numpy 2.1.2, pandas 2.3.3, sklearn 1.9.1. Тестовые зависимости
pytest 9.2.0, openpyxl 3.1.5 и pyarrow 26.0.1 установлены локально в
`.cache/test-deps`, pip cache и временный каталог также находятся в `.cache/`.
Датасеты не скачивались. Эти зависимости нужны для проверки поддерживаемых
форматов, а не для вычислительных экспериментов.

Тесты покрывают совместимость старых metadata, явные int/float-категории,
сохранение числовых измерений, настоящих NaN и нулевого уровня, ошибки declarations
и checksum, конфликтующие файлы, продолжение частичного импорта, запуск CLI,
повторное использование и независимое пересоздание manifests, отклонение изменения
данных/типов/протокола и интеграцию всех десяти реальных локальных CSV.

Повторяемость manifests проверена в этом окружении; сохранённые версии библиотек
нужны для интерпретации воспроизводимости в других окружениях.
