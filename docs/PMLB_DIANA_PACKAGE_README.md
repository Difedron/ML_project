# Общие данные PMLB для запусков Дианы

Пакет содержит 10 утверждённых задач (6 binary + 4 multiclass), исходные CSV,
metadata и **существующие** разбиения `holdout_0`. Все CSV/metadata/manifests
скопированы без изменения байтов. Метапризнаков и результатов моделей здесь нет.

## Проверка после распаковки

Нужен только Python 3.11+; устанавливать Meta Selector, PyMFE или sklearn не нужно:

```bash
python -B verify_package.py
```

Скрипт работает независимо от текущей рабочей папки. Ожидается JSON с `status: ok`,
`dataset_count: 10`, `verified_files: 64`, `index_files: 30`.
Проверяются SHA-256, состав файлов, размеры CSV, точное совпадение TXT-индексов
с manifests, отсутствие пересечений, полное покрытие строк и все классы в каждой части.
Исходные данные не изменяются; новые разбиения не строятся.

`SHA256SUMS.json` содержит хеши всех остальных 63 файлов (включая этот README
и verifier). Сам checksum-файл не может включать собственный хеш. SHA-256 всего
ZIP передаётся отправителем отдельно и фиксирует также checksum-файл.
При наличии внешнего verification JSON можно дополнительно проверить его хеш:

```bash
python -B verify_package.py --checksums-sha256 <checksums_sha256_из_внешнего_отчёта>
```

## Структура

```text
datasets.csv                         общая таблица и пути внутри пакета
datasets/<dataset_id>/<original>.csv  исходные CSV, заголовок не считается объектом
datasets/<dataset_id>/metadata.yaml   target, categorical_columns, происхождение
splits/<dataset_id>/holdout_0.json     исходный manifest, включая fingerprint
indices/<dataset_id>/train.txt        одна позиция в строке, без заголовка
indices/<dataset_id>/validation.txt
indices/<dataset_id>/test.txt
```

Индексы **позиционные, от нуля**, относятся к строкам данных после заголовка CSV.
Их порядок точно повторяет списки manifest. Используйте `.iloc`, не `.loc`.
Не сортируйте, не фильтруйте и не удаляйте дубликаты/пропуски перед выбором строк.

Основной seed holdout — **42** (отделение test), seed второго этапа — **43**
(отделение validation от train+validation). Они явно записаны в `datasets.csv`.
Нужно использовать приложенные индексы, а не восстанавливать их по seed:
разные окружения/округления могут дать другие разбиения.

## Пример загрузки в своём проекте (pandas)

```python
from pathlib import Path
import json
import pandas as pd

root = Path("/path/to/unpacked/package")
item = pd.read_csv(root / "datasets.csv").set_index("dataset_id").loc["pmlb_saheart"]
frame = pd.read_csv(root / item.csv_path)  # обычный CSV; не задавать index_col
y = frame[item.target_column]
X = frame.drop(columns=[item.target_column])
for column in json.loads(item.categorical_columns):
    X[column] = X[column].astype("category")
parts = {}
for name in ("train", "validation", "test"):
    positions = [int(v) for v in (root / item[f"{name}_indices_path"]).read_text().splitlines()]
    parts[name] = (X.iloc[positions].copy(), y.iloc[positions].copy())
X_train, y_train = parts["train"]
X_validation, y_validation = parts["validation"]
X_test, y_test = parts["test"]
```

Target всегда берётся из metadata/таблицы, не определяется автоматически.
`categorical_columns` в таблице — JSON-массив; категории не выводятся из dtype
или cardinality. Остальные признаки сохраняют числовой смысл.
У mushroom **`stalk-root=0` — отдельная категория unknown**, её нельзя автоматически
заменять на NaN. Порядковые факторы car_evaluation в согласованном представлении
считаются категориями. При адаптации к API моделей сохраняйте эту семантику;
кодирование, imputation и scaling обучайте только на train.

Validation служит выбору настроек, если это предусмотрено протоколом; test —
только итоговой оценке. Не объединяйте части и не создавайте новые разбиения.
Запуски TabPFN-2.6, TabICLv2, LightGBM и LightGBM + PriorTune K=25 выполняются
в вашем проекте; в пакете нет моделей, весов или кода их обучения.

Сохраняйте ключи `dataset_source=local`, `dataset_id=pmlb_*`, `split_id=holdout_0`,
версии моделей, метрики и направление оптимизации. `dataset_fingerprint` и
`protocol_hash` доступны в таблице. Verifier проверяет их сохранённые поля и байтовую
идентичность CSV/metadata/manifests; pandas-зависимый fingerprint заново не вычисляется.
Пути `provenance.original_path` в metadata — исторические пути Meta Selector;
для загрузки используйте пути из `datasets.csv` или файл рядом с metadata.
