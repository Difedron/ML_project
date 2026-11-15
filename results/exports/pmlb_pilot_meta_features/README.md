# Метапризнаки PMLB для Дианы

10 строк, 217 числовых predictors (202 PyMFE + 5 controls + 5 моделей × mean/std),
4 идентификатора: dataset_id, split_id, seed, dataset_source. Seed — идентификатор,
не predictor. split_id=holdout_0, seed=42, dataset_source=local.
Имена и порядок всех колонок приведены ниже и в manifest.json.
PyMFE — paper-style reconstruction, точный набор статьи не подтверждён.

Все значения взяты из существующих Parquet, без повторных вычислений.
Только outer train (~60%, точные размеры в manifest); оценки моделей —
Balanced Accuracy по трём внутренним StratifiedKFold, std с ddof=0.
41 исходный NaN сохранён: в CSV явно NaN, в Parquet null. Не импутировать при импорте.
Времена обучения, candidate scores, gap и метки победителя не включены.

CSV UTF-8 можно читать без окружения Meta Selector. Для точного float64 round-trip:
`features = pandas.read_csv("pmlb_pilot_meta_features_217.csv", float_precision="round_trip")`.
Parquet требует только совместимого reader (например pandas + pyarrow).
PyMFE, sklearn и установка проекта для чтения не нужны. Передайте все четыре файла.

## Замечание о локальных raw-копиях

data/raw CSV имеют другие окончания строк и не совпадают с утверждёнными byte SHA.
Проверено равенство содержимого после нормализации CRLF/LF; файлы не исправлялись.
Экспорт сверялся с неизменёнными, hash-approved CSV из data/PMLB и fingerprint
существующих manifests. Все 10 fingerprints пересчитаны и совпали.
Обычный local loader на текущих raw-копиях останавливается с SHA-256 mismatch;
это не препятствует чтению готового экспорта и работе Дианы с data/PMLB.

## Соединение

Явное соответствие provenance: Рита local → Диана pmlb. Исходный source здесь сохранён.
Join строго по (dataset_id, split_id, seed), никогда по порядку строк.
До join сверить SHA-256 CSV, dataset_fingerprint, protocol_hash и manifest_sha256
с manifest.json и контрактом Дианы; неизвестные source labels отклонять.
Для 40 результатов Дианы проверить ровно 4 различных алгоритма на каждый из 10 ключей,
уникальность (dataset_id, split_id, seed, algorithm), отсутствие лишних/пропущенных ключей.
Затем `scores.merge(features, on=["dataset_id", "split_id", "seed"],
how="left", validate="many_to_one", indicator=True)`; все 40 _merge должны быть both.
Колонки dataset_source из двух таблиц сохраняются с явными suffixes.
Сами 40 результатов этим экспортом не проверяются и не создаются.

## Воспроизведение экспорта

Из корня проекта: `.venv/Scripts/python.exe -B scripts/export_pmlb_meta_features.py`.
Скрипт читает источники и snapshots, проверяет provenance и контракт Дианы,
пишет только этот каталог. SHA-256 входов и outputs сохранены в manifest.json
(manifest не содержит собственный hash).

## Колонки

- `dataset_id`
- `split_id`
- `seed`
- `dataset_source`
- `feature::controls::cat_fraction`
- `feature::controls::d_over_n`
- `feature::controls::feature_missing_fraction`
- `feature::controls::log_d`
- `feature::controls::log_n`
- `feature::custom_landmarking::decision_tree_balanced_accuracy_mean`
- `feature::custom_landmarking::decision_tree_balanced_accuracy_std`
- `feature::custom_landmarking::gaussian_nb_balanced_accuracy_mean`
- `feature::custom_landmarking::gaussian_nb_balanced_accuracy_std`
- `feature::custom_landmarking::knn_balanced_accuracy_mean`
- `feature::custom_landmarking::knn_balanced_accuracy_std`
- `feature::custom_landmarking::lda_balanced_accuracy_mean`
- `feature::custom_landmarking::lda_balanced_accuracy_std`
- `feature::custom_landmarking::logistic_regression_balanced_accuracy_mean`
- `feature::custom_landmarking::logistic_regression_balanced_accuracy_std`
- `feature::paper_features::attr_conc.max`
- `feature::paper_features::attr_conc.mean`
- `feature::paper_features::attr_conc.median`
- `feature::paper_features::attr_conc.min`
- `feature::paper_features::attr_conc.sd`
- `feature::paper_features::attr_ent.max`
- `feature::paper_features::attr_ent.mean`
- `feature::paper_features::attr_ent.median`
- `feature::paper_features::attr_ent.min`
- `feature::paper_features::attr_ent.sd`
- `feature::paper_features::attr_to_inst`
- `feature::paper_features::can_cor.max`
- `feature::paper_features::can_cor.mean`
- `feature::paper_features::can_cor.median`
- `feature::paper_features::can_cor.min`
- `feature::paper_features::can_cor.sd`
- `feature::paper_features::cat_to_num`
- `feature::paper_features::class_conc.max`
- `feature::paper_features::class_conc.mean`
- `feature::paper_features::class_conc.median`
- `feature::paper_features::class_conc.min`
- `feature::paper_features::class_conc.sd`
- `feature::paper_features::class_ent`
- `feature::paper_features::cor.max`
- `feature::paper_features::cor.mean`
- `feature::paper_features::cor.median`
- `feature::paper_features::cor.min`
- `feature::paper_features::cor.sd`
- `feature::paper_features::cov.max`
- `feature::paper_features::cov.mean`
- `feature::paper_features::cov.median`
- `feature::paper_features::cov.min`
- `feature::paper_features::cov.sd`
- `feature::paper_features::eigenvalues.max`
- `feature::paper_features::eigenvalues.mean`
- `feature::paper_features::eigenvalues.median`
- `feature::paper_features::eigenvalues.min`
- `feature::paper_features::eigenvalues.sd`
- `feature::paper_features::eq_num_attr`
- `feature::paper_features::freq_class.max`
- `feature::paper_features::freq_class.mean`
- `feature::paper_features::freq_class.median`
- `feature::paper_features::freq_class.min`
- `feature::paper_features::freq_class.sd`
- `feature::paper_features::g_mean.max`
- `feature::paper_features::g_mean.mean`
- `feature::paper_features::g_mean.median`
- `feature::paper_features::g_mean.min`
- `feature::paper_features::g_mean.sd`
- `feature::paper_features::gravity`
- `feature::paper_features::h_mean.max`
- `feature::paper_features::h_mean.mean`
- `feature::paper_features::h_mean.median`
- `feature::paper_features::h_mean.min`
- `feature::paper_features::h_mean.sd`
- `feature::paper_features::inst_to_attr`
- `feature::paper_features::iq_range.max`
- `feature::paper_features::iq_range.mean`
- `feature::paper_features::iq_range.median`
- `feature::paper_features::iq_range.min`
- `feature::paper_features::iq_range.sd`
- `feature::paper_features::joint_ent.max`
- `feature::paper_features::joint_ent.mean`
- `feature::paper_features::joint_ent.median`
- `feature::paper_features::joint_ent.min`
- `feature::paper_features::joint_ent.sd`
- `feature::paper_features::kurtosis.max`
- `feature::paper_features::kurtosis.mean`
- `feature::paper_features::kurtosis.median`
- `feature::paper_features::kurtosis.min`
- `feature::paper_features::kurtosis.sd`
- `feature::paper_features::leaves`
- `feature::paper_features::leaves_branch.max`
- `feature::paper_features::leaves_branch.mean`
- `feature::paper_features::leaves_branch.median`
- `feature::paper_features::leaves_branch.min`
- `feature::paper_features::leaves_branch.sd`
- `feature::paper_features::leaves_corrob.max`
- `feature::paper_features::leaves_corrob.mean`
- `feature::paper_features::leaves_corrob.median`
- `feature::paper_features::leaves_corrob.min`
- `feature::paper_features::leaves_corrob.sd`
- `feature::paper_features::leaves_homo.max`
- `feature::paper_features::leaves_homo.mean`
- `feature::paper_features::leaves_homo.median`
- `feature::paper_features::leaves_homo.min`
- `feature::paper_features::leaves_homo.sd`
- `feature::paper_features::leaves_per_class.max`
- `feature::paper_features::leaves_per_class.mean`
- `feature::paper_features::leaves_per_class.median`
- `feature::paper_features::leaves_per_class.min`
- `feature::paper_features::leaves_per_class.sd`
- `feature::paper_features::lh_trace`
- `feature::paper_features::mad.max`
- `feature::paper_features::mad.mean`
- `feature::paper_features::mad.median`
- `feature::paper_features::mad.min`
- `feature::paper_features::mad.sd`
- `feature::paper_features::max.max`
- `feature::paper_features::max.mean`
- `feature::paper_features::max.median`
- `feature::paper_features::max.min`
- `feature::paper_features::max.sd`
- `feature::paper_features::mean.max`
- `feature::paper_features::mean.mean`
- `feature::paper_features::mean.median`
- `feature::paper_features::mean.min`
- `feature::paper_features::mean.sd`
- `feature::paper_features::median.max`
- `feature::paper_features::median.mean`
- `feature::paper_features::median.median`
- `feature::paper_features::median.min`
- `feature::paper_features::median.sd`
- `feature::paper_features::min.max`
- `feature::paper_features::min.mean`
- `feature::paper_features::min.median`
- `feature::paper_features::min.min`
- `feature::paper_features::min.sd`
- `feature::paper_features::mut_inf.max`
- `feature::paper_features::mut_inf.mean`
- `feature::paper_features::mut_inf.median`
- `feature::paper_features::mut_inf.min`
- `feature::paper_features::mut_inf.sd`
- `feature::paper_features::nodes`
- `feature::paper_features::nodes_per_attr`
- `feature::paper_features::nodes_per_inst`
- `feature::paper_features::nodes_per_level.max`
- `feature::paper_features::nodes_per_level.mean`
- `feature::paper_features::nodes_per_level.median`
- `feature::paper_features::nodes_per_level.min`
- `feature::paper_features::nodes_per_level.sd`
- `feature::paper_features::nodes_repeated.max`
- `feature::paper_features::nodes_repeated.mean`
- `feature::paper_features::nodes_repeated.median`
- `feature::paper_features::nodes_repeated.min`
- `feature::paper_features::nodes_repeated.sd`
- `feature::paper_features::nr_attr`
- `feature::paper_features::nr_bin`
- `feature::paper_features::nr_cat`
- `feature::paper_features::nr_class`
- `feature::paper_features::nr_cor_attr`
- `feature::paper_features::nr_disc`
- `feature::paper_features::nr_inst`
- `feature::paper_features::nr_norm`
- `feature::paper_features::nr_num`
- `feature::paper_features::nr_outliers`
- `feature::paper_features::ns_ratio`
- `feature::paper_features::num_to_cat`
- `feature::paper_features::p_trace`
- `feature::paper_features::range.max`
- `feature::paper_features::range.mean`
- `feature::paper_features::range.median`
- `feature::paper_features::range.min`
- `feature::paper_features::range.sd`
- `feature::paper_features::roy_root`
- `feature::paper_features::sd.max`
- `feature::paper_features::sd.mean`
- `feature::paper_features::sd.median`
- `feature::paper_features::sd.min`
- `feature::paper_features::sd.sd`
- `feature::paper_features::sd_ratio`
- `feature::paper_features::skewness.max`
- `feature::paper_features::skewness.mean`
- `feature::paper_features::skewness.median`
- `feature::paper_features::skewness.min`
- `feature::paper_features::skewness.sd`
- `feature::paper_features::sparsity.max`
- `feature::paper_features::sparsity.mean`
- `feature::paper_features::sparsity.median`
- `feature::paper_features::sparsity.min`
- `feature::paper_features::sparsity.sd`
- `feature::paper_features::t_mean.max`
- `feature::paper_features::t_mean.mean`
- `feature::paper_features::t_mean.median`
- `feature::paper_features::t_mean.min`
- `feature::paper_features::t_mean.sd`
- `feature::paper_features::tree_depth.max`
- `feature::paper_features::tree_depth.mean`
- `feature::paper_features::tree_depth.median`
- `feature::paper_features::tree_depth.min`
- `feature::paper_features::tree_depth.sd`
- `feature::paper_features::tree_imbalance.max`
- `feature::paper_features::tree_imbalance.mean`
- `feature::paper_features::tree_imbalance.median`
- `feature::paper_features::tree_imbalance.min`
- `feature::paper_features::tree_imbalance.sd`
- `feature::paper_features::tree_shape.max`
- `feature::paper_features::tree_shape.mean`
- `feature::paper_features::tree_shape.median`
- `feature::paper_features::tree_shape.min`
- `feature::paper_features::tree_shape.sd`
- `feature::paper_features::var.max`
- `feature::paper_features::var.mean`
- `feature::paper_features::var.median`
- `feature::paper_features::var.min`
- `feature::paper_features::var.sd`
- `feature::paper_features::var_importance.max`
- `feature::paper_features::var_importance.mean`
- `feature::paper_features::var_importance.median`
- `feature::paper_features::var_importance.min`
- `feature::paper_features::var_importance.sd`
- `feature::paper_features::w_lambda`
