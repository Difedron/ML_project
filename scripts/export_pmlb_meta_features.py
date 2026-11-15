"""Export saved pilot predictors only; never extract features, fit models or create splits."""
import hashlib
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from meta_selector.datasets.loader import read_metadata
from meta_selector.datasets.models import Dataset
from meta_selector.artifacts import config_hash
from meta_selector.datasets.splits import check_manifest
from meta_selector.paths import DEFAULT_PROJECT_ROOT as ROOT, read_yaml_mapping, project_paths

MODELS = ['logistic_regression', 'decision_tree', 'knn', 'gaussian_nb', 'lda']
KEYS = ['dataset_id', 'split_id', 'seed']
IDENTITY = ['dataset_source', 'dataset_id', 'split_id']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reshape(features, custom):
    """Use the builder's naming convention without gap labels or split averaging."""
    long = custom.melt(id_vars=IDENTITY + ['model_id'], value_vars=['score_mean', 'score_std'],
                       var_name='summary', value_name='value')
    long['source'] = 'custom_landmarking'
    long['name'] = long.model_id + '_balanced_accuracy_' + long.summary.str.removeprefix('score_')
    long = pd.concat([features, long], ignore_index=True)
    require(not long.duplicated(IDENTITY + ['source', 'name']).any(), 'Duplicate predictor records')
    long['column'] = 'feature::' + long.source + '::' + long.name
    return long.pivot(index=IDENTITY, columns='column', values='value').reset_index().rename_axis(columns=None)


def export():
    specs = read_yaml_mapping(ROOT / 'config/pmlb_pilot.yaml')['datasets']
    config = read_yaml_mapping(ROOT / 'config/research.yaml')
    diana = read_yaml_mapping(ROOT / 'config/pilot_datasets.yaml')
    canonical = {s['dataset_id']: s for s in diana['datasets']}
    ids = [s['dataset_id'] for s in specs]
    require(len(ids) == len(set(ids)) == 10 and set(ids) == set(canonical), 'Approved ID mismatch')
    fpath = ROOT / 'data/derived/meta_features/meta_features.parquet'
    cpath = ROOT / 'data/derived/landmarkers/custom_landmarkers.parquet'
    # Guard every existing scientific input/result, including old logs and exports.
    protected = [p for folder in ('data', 'config', 'results/logs', 'results/tables')
                 for p in (ROOT / folder).rglob('*') if p.is_file()]
    before = {p: (sha(p), p.stat().st_mtime_ns) for p in protected}
    f, c = pd.read_parquet(fpath), pd.read_parquet(cpath)
    require(len(f) == 2070 and len(c) == 50, 'Unexpected source row counts')
    for table in (f, c):
        require(set(table.dataset_id) == set(ids), 'Source dataset mismatch')
        require(table.dataset_source.eq('local').all() and table.split_id.eq('holdout_0').all(), 'Source identity mismatch')
        require(table.error_message.eq('').all(), 'Source errors present')
    require(set(f.source) == {'controls', 'paper_features'}, 'Unexpected namespaces')
    require(not f.duplicated(IDENTITY + ['source', 'name']).any(), 'Duplicate features/configurations')
    require(not c.duplicated(IDENTITY + ['model_id']).any(), 'Duplicate models/configurations')
    require(f.status.isin(['ok', 'undefined']).all(), 'Failed features')
    require(f.status.eq('undefined').equals(f.value.isna()) and f.value.isna().sum() == 41, 'NaN/status mismatch')
    require(not np.isinf(f.value).any(), 'Infinite features')
    require(c.status.eq('ok').all() and c.n_folds.eq(3).all() and c.metric.eq('balanced_accuracy').all(), 'Invalid custom results')
    require(np.isfinite(c[['score_mean', 'score_std']]).all().all(), 'Nonfinite scores')
    require(c.score_mean.between(0, 1).all() and c.score_std.ge(0).all(), 'Invalid score range')
    require(c.config_id.eq(config_hash(config['custom_landmarkers'])).all(), 'Custom config mismatch')
    for source, group in f.groupby('source'):
        expected = config['paper_features'] if source == 'paper_features' else dict(
            log_base='e', missing='cells', categorical='nonnumeric_including_bool', version='1')
        require(group.config_id.eq(config_hash(expected)).all(), 'Feature config mismatch')
    # Compare primary artifacts with complete, successful saved snapshots.
    for table, snapshot, keys in ((f, 'features', IDENTITY + ['source', 'name']),
                                   (c, 'landmarkers', IDENTITY + ['model_id'])):
        saved = pd.read_parquet(ROOT / f'results/logs/pmlb_features/repeat/{snapshot}.parquet')
        pd.testing.assert_frame_equal(table.sort_values(keys).reset_index(drop=True),
                                      saved.sort_values(keys).reset_index(drop=True), check_exact=True)
    identities = []
    for spec in specs:
        did = spec['dataset_id']
        # Read the hash-approved canonical CSV, preserving the original loader's typing.
        metadata = read_metadata(did, project_paths(ROOT))
        csv = ROOT / f"data/PMLB/{spec['original_name']}.csv"
        require(sha(csv) == spec['sha256'] == metadata.file.sha256, 'CSV hash mismatch')
        frame = pd.read_csv(csv, sep=metadata.file.separator, encoding=metadata.file.encoding)
        for column in metadata.categorical_columns:
            frame[column] = frame[column].astype('category')
        dataset = Dataset(did, frame.drop(columns=[metadata.target]), frame[metadata.target], metadata)
        mp = ROOT / f'data/splits/{did}/holdout_0.json'
        manifest = json.loads(mp.read_text(encoding='utf-8'))
        check_manifest(manifest, dataset)  # Validates fingerprint and complete disjoint partition; no split generation.
        require(manifest['split_id'] == 'holdout_0' and manifest['protocol'] == config['splits'] and
                manifest['protocol']['random_seed'] == 42, 'Split protocol mismatch')
        csv = ROOT / f"data/PMLB/{spec['original_name']}.csv"
        require(sha(csv) == spec['sha256'] == dataset.metadata.file.sha256, 'CSV hash mismatch')
        raw = ROOT / f"data/raw/{did}/{spec['original_name']}.csv"
        require(raw.read_bytes().replace(b'\r\n', b'\n') == csv.read_bytes().replace(b'\r\n', b'\n'),
                'Raw CSV differs beyond line endings')
        entry = canonical[did]
        shared = ROOT / 'config/pilot_splits' / entry['shared_manifest']
        require(shared.read_bytes() == mp.read_bytes() and sha(shared) == entry['shared_manifest_sha256'], 'Diana split mismatch')
        require(entry['seed'] == 42 and entry['split_id'] == 'holdout_0' and
                entry['dataset_source'] == 'pmlb' and entry['rita_dataset_source'] == 'local', 'Diana identity mismatch')
        require(entry['fingerprint'] == sha(csv) and entry['rita_dataset_fingerprint'] == manifest['dataset_fingerprint'] and
                entry['protocol_hash'] == manifest['protocol_hash'], 'Diana fingerprint/protocol mismatch')
        ft, ct = f[f.dataset_id.eq(did)], c[c.dataset_id.eq(did)]
        require(ft.source.value_counts().to_dict() == {'paper_features': 202, 'controls': 5}, 'Feature coverage mismatch')
        require(set(ct.model_id) == set(MODELS) and len(ct) == 5, 'Model coverage mismatch')
        for row in pd.concat([ft, ct]).itertuples():
            context = json.loads(row.provenance)
            require(context['input_role'] == 'outer_training_only', 'Invalid input role')
            if row.source == 'custom_landmarking':
                require(context['config'] == config['custom_landmarkers'] and context['cv_std_ddof'] == 0, 'CV config mismatch')
                context = context['split']
            require(context['train_indices'] == manifest['indices']['train'] and
                    context['dataset_fingerprint'] == manifest['dataset_fingerprint'] and
                    context['protocol_hash'] == manifest['protocol_hash'], 'Training provenance mismatch')
        ntrain = len(manifest['indices']['train'])
        require(ft.loc[ft.name.eq('nr_inst'), 'value'].item() == ntrain, 'Training size mismatch')
        # Read runtime evidence from the completed run; never execute its wrapper.
        for phase, expected in (('features', 2), ('landmarkers', 1)):
            audit = json.loads((ROOT / f'results/logs/pmlb_features/repeat/{phase}_audit.json').read_text())
            require(audit['exit_code'] == 0 and audit['protected_inputs_unchanged'], 'Failed audit')
            events = [e for e in audit['events'] if e['dataset_id'] == did and e['kind'] == 'outer_input']
            train_hash = hashlib.sha256(np.asarray(manifest['indices']['train'], dtype=np.int64).tobytes()).hexdigest()
            require(len(events) == expected and all(e['rows'] == ntrain and e['index_hash'] == train_hash
                                                    for e in events), 'Runtime training input mismatch')
            if phase == 'landmarkers':
                fits = [e for e in audit['events'] if e['dataset_id'] == did and e['kind'] == 'inner_fit']
                require(len(fits) == 15 and all(e['status'] == 'ok' for e in fits), 'Incomplete CV audit')
        identities.append(dict(dataset_id=did, dataset_source='local', split_id='holdout_0', seed=42,
                               csv_sha256=sha(csv), dataset_fingerprint=manifest['dataset_fingerprint'],
                               protocol_hash=manifest['protocol_hash'], manifest_sha256=sha(mp),
                               train_size=ntrain, n_rows=len(dataset.y), nan_count=int(ft.value.isna().sum()),
                               raw_copy_sha256=sha(raw), raw_copy_byte_identical=sha(raw) == sha(csv),
                               raw_copy_equivalent_after_newline_normalization=True))
    wide = reshape(f, c)
    wide = wide.merge(pd.DataFrame(identities)[IDENTITY + ['seed']], on=IDENTITY, validate='one_to_one')
    predictors = sorted(col for col in wide if col.startswith('feature::'))
    wide = wide[KEYS + ['dataset_source'] + predictors].sort_values('dataset_id').reset_index(drop=True)
    require(wide.shape == (10, 221) and len(predictors) == 217, 'Export shape mismatch')
    require(sum(p.startswith('feature::custom_landmarking::') for p in predictors) == 10, 'Custom feature count mismatch')
    require(not wide.columns.duplicated().any() and not wide.duplicated(KEYS).any(), 'Duplicate output')
    require(all(pd.api.types.is_numeric_dtype(wide[p]) for p in predictors), 'Nonnumeric predictors')
    require(wide[predictors].isna().sum().sum() == 41 and not np.isinf(wide[predictors]).any().any(), 'Output NaN/Inf mismatch')
    # Verify every exported value independently against its exact original record.
    by_id = wide.set_index('dataset_id')
    for row in f.itertuples():
        actual = by_id.loc[row.dataset_id, f'feature::{row.source}::{row.name}']
        require((pd.isna(actual) and pd.isna(row.value)) or actual == row.value, 'Feature value mismatch')
    for row in c.itertuples():
        for stat in ('mean', 'std'):
            require(by_id.loc[row.dataset_id, f'feature::custom_landmarking::{row.model_id}_balanced_accuracy_{stat}'] ==
                    getattr(row, 'score_' + stat), 'Custom value mismatch')
    output = ROOT / 'results/exports/pmlb_pilot_meta_features'
    output.mkdir(parents=True, exist_ok=True)
    stem = output / 'pmlb_pilot_meta_features_217'
    wide.to_parquet(stem.with_suffix('.parquet'), index=False)
    wide.to_csv(stem.with_suffix('.csv'), index=False, float_format='%.17g', na_rep='NaN')
    for restored in (pd.read_parquet(stem.with_suffix('.parquet')),
                     pd.read_csv(stem.with_suffix('.csv'), float_precision='round_trip',
                                 dtype={p: 'float64' for p in predictors})):
        pd.testing.assert_frame_equal(wide, restored, check_exact=True)
    readme = '''# Метапризнаки PMLB для Дианы

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

'''
    (output / 'README.md').write_text(readme + '\n'.join('- `' + col + '`' for col in wide) + '\n', encoding='utf-8')
    require(all((sha(p), p.stat().st_mtime_ns) == state for p, state in before.items()), 'Protected files changed')
    report = dict(schema_version=1, rows=10, predictor_count=217, identifier_columns=KEYS + ['dataset_source'],
                  predictor_columns=predictors, nan_count=41, inf_count=0, models=MODELS,
                  metric='balanced_accuracy', inner_folds=3, std_ddof=0,
                  source_label_mapping={'rita': 'local', 'diana': 'pmlb'}, join_keys=KEYS,
                  datasets=identities, feature_config_ids=f.groupby('source').config_id.first().to_dict(),
                  custom_config_id=c.config_id.iloc[0],
                  inputs={p.relative_to(ROOT).as_posix(): state[0] for p, state in before.items()},
                  outputs={p.name: sha(p) for p in output.iterdir() if p.name in
                           ['README.md', stem.with_suffix('.csv').name, stem.with_suffix('.parquet').name]},
                  checks=dict(source_values_exact=True, csv_parquet_roundtrip_exact=True,
                              repeat_snapshots_exact=True, training_only_provenance=True,
                              diana_contract_matches=True, protected_files_unchanged=True),
                  candidate_results_join_performed=False)
    (output / 'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'PASS: {len(wide)} datasets x {len(predictors)} predictors; 41 NaN; exact CSV/Parquet; {len(before)} protected files unchanged')
    return wide


if __name__ == '__main__':
    export()
