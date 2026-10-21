# Этап 3: архитектура исследования и проверка pipeline

Реализовано 21 октября 2026. Существующие datasets/registry, cached results, gap formula и исходный аудит сохранены.
Полный benchmark 21 × 150 не запускался. Точная численная репликация статьи не заявляется.

## A. Все 21 модели статьи

Cached artifacts доступны для всех 21 методов и проверены по прежним MD5. Local runner означает готовность
официального backend на текущей машине, а не наличие Python Protocol. Сейчас TabArena/AutoGluon отсутствуют,
CUDA GPU отсутствует; checkpoints не настроены. Поэтому ни один кандидат локально не объявлен готовым.

| model | canonical_id | family | TabArena artifact | local runner | dependency status |
|---|---|---|---|---|---|
| FastaiMLP | fastai | MLP-based,non-TFM | yes | no | missing dependencies: tabarena, autogluon.tabular, fastai |
| ModernNCA | modernnca | MLP-based,non-TFM | yes | no | missing dependencies: tabarena, autogluon.tabular, category_encoders |
| TorchMLP | nn_torch | MLP-based,non-TFM | yes | no | missing dependencies: tabarena, autogluon.tabular |
| RealMLP | realmlp | MLP-based,non-TFM | yes | no | missing dependencies: tabarena, autogluon.tabular, pytabkit |
| TabM | tabm | MLP-based,non-TFM | yes | no | missing dependencies: tabarena, autogluon.tabular, tabm, rtdl_num_embeddings |
| CatBoost | catboost | Tree-based,non-TFM | yes | no | missing dependencies: tabarena, autogluon.tabular |
| LightGBM | lightgbm | Tree-based,non-TFM | yes | no | missing dependencies: tabarena, autogluon.tabular |
| XGBoost | xgboost | Tree-based,non-TFM | yes | no | missing dependencies: tabarena, autogluon.tabular, xgboost |
| EBM | ebm | Tree-based,non-TFM | yes | no | missing dependencies: tabarena, autogluon.tabular, interpret |
| PerpetualBooster | perpetualboost | Tree-based,non-TFM | yes | no | missing dependencies: tabarena, autogluon.tabular, perpetual |
| ExtraTrees | extra_trees | Tree-based,non-TFM | yes | no | missing dependencies: tabarena, autogluon.tabular |
| RandomForest | random_forest | Tree-based,non-TFM | yes | no | missing dependencies: tabarena, autogluon.tabular |
| KNN | knn | non-TFM | yes | no | missing dependencies: tabarena, autogluon.tabular |
| Linear | logistic_regression | non-TFM | yes | no | missing dependencies: tabarena, autogluon.tabular |
| TabDPT | tabdpt | TFM | yes | no | missing dependencies: tabarena, autogluon.tabular, tabdpt |
| TabSTAR | tabstar | TFM | yes | no | missing dependencies: tabarena, autogluon.tabular, tabstar |
| TabICL | tabicl_v1 | TFM | yes | no | missing dependencies: tabarena, autogluon.tabular |
| TabICLv2 | tabicl_v2 | TFM | yes | no | missing dependencies: tabarena, autogluon.tabular |
| TabPFNv2 | tabpfn_v2 | TFM | yes | no | missing dependencies: tabarena, autogluon.tabular |
| TabPFN-2.6 | tabpfn_2_6 | TFM | yes | no | missing dependencies: tabarena, autogluon.tabular |
| RealTabPFN-2.5 | realtabpfn_v2_5 | TFM | yes | no | missing dependencies: tabarena, autogluon.tabular |

Полные причины, включая GPU/checkpoint/license и compute: [CANDIDATE_AVAILABILITY.csv](CANDIDATE_AVAILABILITY.csv).
`check_candidate_models.py` не загружает веса и не обучает модели. Live registry и dependencies проверяются
перед fit; исторический TabPFN v2 может отсутствовать в новом registry — это unavailable, не замена на 3.x.

## B. Все шесть сравнений

| ID | comparison | dataset observations | paired splits | excluded datasets |
|---|---|---:|---:|---:|
| C1 | mlp_vs_tree | 51 | 816 | 0 |
| C2 | non_tfm_vs_tfm | 51 | 816 | 0 |
| C3 | tabicl_v2_vs_tabpfn_2_6 | 51 | 816 | 0 |
| C4 | realtabpfn_v2_5_vs_tabpfn_2_6 | 51 | 816 | 0 |
| C5 | tabpfn_v2_vs_tabpfn_2_6 | 33 | 654 | 18 |
| C6 | tabicl_v1_vs_tabicl_v2 | 36 | 576 | 15 |

C1: MLP/tree; C2: non-TFM/TFM; C3: TabICLv2/TabPFN-2.6;
C4: RealTabPFN2.5/TabPFN-2.6; C5: TabPFNv2/TabPFN-2.6; C6: TabICLv1/TabICLv2.
Версионные проверки используют явное whole-dataset coverage restriction. Список исключений и причины
сохранены в provenance каждого сравнения; подмножества не выдаются за подтверждённые авторские.

Normalization reference остаётся исторической реконструкцией: default/tuned, 24 732 records,
19–21 метод на split, imputed/unknown исключены. Binary ROC-AUC vs paper log-loss discrepancy сохранена.

## C–D. Папки и точные пути

```text
VAK_ML/
├── config/
│   ├── project.yaml
│   ├── datasets.yaml
│   ├── tabarena_results.yaml
│   ├── research.yaml
│   └── candidates.yaml
├── data/
│   ├── raw/<dataset_id>/          original file + metadata.yaml
│   ├── registry/datasets.csv
│   ├── processed/
│   ├── splits/<dataset_id>/holdout_0.json
│   └── derived/
│       ├── meta_features/meta_features.parquet
│       ├── landmarkers/custom_landmarkers.parquet
│       ├── landmarkers/pymfe_landmarkers.parquet
│       └── candidate_scores/candidate_scores.parquet
├── src/meta_selector/
│   ├── datasets/                 existing loader/registry + splits
│   ├── tabarena/                 verified external results boundary
│   ├── performance_gap/          families + six comparison definitions
│   ├── meta_features/            controls + PyMFE extraction
│   ├── landmarkers/              fold-local preprocessing + five models
│   ├── candidates/               official backend runner + score bridge
│   ├── meta_dataset/             keyed, validated joins
│   ├── meta_model/               exact 2.6 wrapper + baselines + LODO
│   └── evaluation/               metrics + executable E0–E5 definitions
├── scripts/
├── tests/
├── docs/
└── results/tables/<comparison_id>/
```

**150 датасетов:** `data/raw/<dataset_id>/`. **Их metadata:** `data/raw/<dataset_id>/metadata.yaml`.
**Пять простых моделей:** `data/derived/landmarkers/custom_landmarkers.parquet`.
**Meta-features:** `data/derived/meta_features/meta_features.parquet`.
**Candidate scores:** `data/derived/candidate_scores/candidate_scores.parquet`.
Все derived таблицы имеют dataset_source/dataset_id/split_id и provenance; payloads/derived не коммитятся.

## E. Пять custom landmarkers

| ID | Parameters | Scaling |
|---|---|---|
| logistic_regression | C=1, max_iter=1000, seed=42 | yes |
| decision_tree | max_depth=3, seed=42 | no |
| knn | n_neighbors=5 | yes |
| gaussian_nb | var_smoothing=1e-9 | no |
| lda | solver=lsqr, shrinkage=auto | yes |

Конфигурация: `config/research.yaml`; default metric=balanced_accuracy, 3 shuffled stratified folds,
числовые median imputation, категории mode + OHE(ignore unknown). Каждый inner fold fit-ит свой pipeline.
Score mean/std и fit-time mean/std сохранены отдельно от стандартных PyMFE landmarks.

## F. Выполненные проверки

`pytest -q`: **135 passed**. В том числе прежние 86 тестов и новые проверки:
180 metadata folders (178 без payload и 2 с маленьким payload), всех mappings/families/comparisons,
checksums/availability/version guards, split reproducibility/fingerprint/no raw mutation,
5 custom models и metrics, preprocessing leakage, batch error isolation, atomic derived tables,
meta-dataset joins/config/coverage, LODO dataset/group isolation и training-fold preprocessing.
Тест exact SDK factory использует fake SDK/checkpoint только как unit test; это не результат реального TabPFN.

Небольшой интеграционный запуск: `_toy_binary` (30 строк), `_smoke_iris` (150), `_smoke_mixed` (90).
- `meta_features/meta_features.parquet`: 621 rows; statuses {'ok': 602, 'undefined': 19}.
- `landmarkers/custom_landmarkers.parquet`: 15 rows; statuses {'ok': 15}.
- `landmarkers/pymfe_landmarkers.parquet`: 42 rows; statuses {'ok': 42}.
- `candidate_scores/candidate_scores.parquet`: 1 rows; statuses {'unavailable': 1}.

19 undefined PyMFE features оставлены NaN со статусом undefined; raw NaN/категории не удалялись.
Candidate status unavailable проверен явной single-model командой; это не успешный candidate benchmark.
TabPFN meta-command возвращает unavailable_exact_version и exit 2 без локального checkpoint.
LODO проверен тестовым predictor: настоящее foundation-model обучение не выполнялось.
`git diff --check` и Git-ignore проверки derived artifacts проходят.

## G. Следующие команды

```bash
source .venv/bin/activate
python scripts/check_candidate_models.py
python scripts/build_performance_gap.py --all
python scripts/create_smoke_datasets.py
python scripts/extract_meta_features.py _toy_binary _smoke_iris _smoke_mixed
python scripts/run_landmarkers.py _toy_binary _smoke_iris _smoke_mixed
pytest -q
```

Будущие явные batch команды `extract_meta_features.py --all-local` и `run_landmarkers.py --all-local`
обрабатывают только valid local classification registry entries. На этом этапе в local registry только
три smoke datasets; pipeline проверен именно на них. 150 внешних датасетов не скачивались и не обучались.

Подробные команды CandidateRunner/TabPFN/LODO и ограничения внешней интеграции:
[RESEARCH_PIPELINE.md](RESEARCH_PIPELINE.md). Нельзя запускать E0 с тремя local feature rows и 51 TabArena
gap rows: builder отклоняет отсутствие соответствующих benchmark features. Сначала нужны реальные training
features по фиксированным splits, совместимый backend и точные checkpoints.

Полный benchmark и научное сравнение E0–E5 не запускались. Текущий этап завершён.
