import math

import pandas as pd
import pytest

from meta_selector.evaluation.metrics import gap_mae, selection_regret, sign_accuracy
from meta_selector.meta_features.controls import compute_controls
from meta_selector.meta_model.baselines import majority_training_sign, mean_training_gap


def test_controls_preserve_categories_nan_and_input():
    train = pd.DataFrame({"number": [1., None], "category": ["a", "b"], "flag": [True, False]})
    original = train.copy(deep=True)
    controls = compute_controls(train)
    assert controls == {"log_n": math.log(2), "log_d": math.log(3), "d_over_n": 1.5,
                        "cat_fraction": 2 / 3, "feature_missing_fraction": 1 / 6}
    pd.testing.assert_frame_equal(train, original)


def test_training_baselines_and_gap_metrics():
    assert mean_training_gap([-.1, .3]) == pytest.approx(.1)
    assert majority_training_sign([-.1, .2, .3]) == 1
    assert majority_training_sign([0, 0, -.1]) == 0
    assert gap_mae([.1, -.2], [0, .1]) == pytest.approx(.2)
    assert sign_accuracy([.1, 0, -.2], [.2, 0, .1]) == pytest.approx(2 / 3)
    assert selection_regret(.9, .8, greater_is_better=True) == pytest.approx(.1)
    assert selection_regret(.1, .2, greater_is_better=False) == pytest.approx(.1)


@pytest.mark.parametrize("values", [[], [float("nan")], [[1., 2.]]])
def test_baselines_require_training_vector(values):
    with pytest.raises(ValueError):
        mean_training_gap(values)


def test_metrics_reject_incompatible_inputs():
    with pytest.raises(ValueError):
        gap_mae([1], [1, 2])
    with pytest.raises(ValueError):
        sign_accuracy([float("inf")], [0])
    with pytest.raises(ValueError):
        selection_regret(.2, .1, greater_is_better=False)
