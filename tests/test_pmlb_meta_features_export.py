"""Export-only checks: no fitting, extraction or mutation of scientific inputs."""
import importlib.util
from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('meta_export', ROOT / 'scripts/export_pmlb_meta_features.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def sources():
    return (pd.read_parquet(ROOT / 'data/derived/meta_features/meta_features.parquet'),
            pd.read_parquet(ROOT / 'data/derived/landmarkers/custom_landmarkers.parquet'))


def test_row_permutation_keeps_keyed_values_and_nan():
    f, c = sources()
    expected = module.reshape(f, c)
    shuffled = module.reshape(f.sample(frac=1, random_state=9), c.sample(frac=1, random_state=3))
    pd.testing.assert_frame_equal(expected, shuffled, check_exact=True)
    assert expected.isna().sum().sum() == 41


@pytest.mark.parametrize('duplicate_custom', [False, True])
def test_duplicate_sources_rejected(duplicate_custom):
    f, c = sources()
    if duplicate_custom:
        c = pd.concat([c, c.iloc[:1]], ignore_index=True)
    else:
        f = pd.concat([f, f.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match='Duplicate predictor'):
        module.reshape(f, c)


def test_saved_export_exact_roundtrip_and_checksums():
    import json
    output = ROOT / 'results/exports/pmlb_pilot_meta_features'
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    for name, checksum in manifest['outputs'].items():
        assert module.sha(output / name) == checksum
    predictors = manifest['predictor_columns']
    csv = pd.read_csv(output / 'pmlb_pilot_meta_features_217.csv', float_precision='round_trip',
                      dtype={p: 'float64' for p in predictors})
    parquet = pd.read_parquet(output / 'pmlb_pilot_meta_features_217.parquet')
    pd.testing.assert_frame_equal(csv, parquet, check_exact=True)
    assert parquet.shape == (10, 221)
    assert len(predictors) == 217
