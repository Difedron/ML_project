"""Numerical protocol tests with explicitly synthetic test fixtures."""
import numpy as np
import pandas as pd
import pytest

from meta_selector.performance_gap.families import Family
from meta_selector.performance_gap.gap import calculate_dataset_gaps, select_by_validation
from meta_selector.performance_gap.normalization import normalize_test_errors, normalize_validation_errors


A, B = Family("A", ("a",)), Family("B", ("b",))


def selected(reference, family):
    normalized, _ = normalize_validation_errors(reference, reference, reference_label="test")
    return select_by_validation(normalized, family)


def reference_for(errors=(.1, .3, .5), split=0, dataset="demo", source="tabarena"):
    return pd.DataFrame([dict(dataset_source=source, dataset_id=dataset, split_id=split,
                              metric="log_loss", problem_type="binary", exact_model_id=model + "::default",
                              model_id=model, test_error=error, validation_error=error, imputed=False)
                         for model, error in zip(("a", "b", "c"), errors)])


def gap(reference, targets=None):
    targets = reference if targets is None else targets
    return calculate_dataset_gaps(targets, A, B, reference=reference, reference_label="synthetic-test-only")


def test_normalization_formula_clipping_and_epsilon():
    reference = reference_for()
    output, provenance = normalize_test_errors(reference, reference, reference_label="test")
    assert output.normalized_test_error.tolist() == [0., 1., 1.]
    assert provenance["epsilon"] == 1e-5
    equal = reference_for((.1, .1, .1))
    assert normalize_test_errors(equal, equal, reference_label="test")[0].normalized_test_error.eq(0).all()
    near_equal = reference_for((0., 1e-7, 2e-7))
    np.testing.assert_allclose(normalize_test_errors(near_equal, near_equal, reference_label="test")[0]
                               .normalized_test_error, [0, .01, .02])


def test_normalize_per_split_then_average_one_record_per_dataset():
    reference = pd.concat([reference_for(), reference_for((.9, .6, .1), split=1)], ignore_index=True)
    result = gap(reference)
    np.testing.assert_allclose(result.split_gaps.gap, [-1, 0])
    assert len(result.dataset_gaps) == 1
    assert result.dataset_gaps.gap.item() == -.5
    assert result.dataset_gaps.n_splits.item() == 2
    assert result.provenance["sign"].startswith("positive favors B")


def test_selection_uses_validation_not_test_and_ties_are_deterministic():
    reference = reference_for()
    extra = reference.iloc[:1].copy()
    extra["exact_model_id"] = "a::tuned"
    extra["validation_error"] = .05
    extra["test_error"] = .9
    reference = pd.concat([reference, extra], ignore_index=True)
    assert selected(reference, A).exact_model_id.item() == "a::tuned"
    assert gap(reference).dataset_gaps.gap.item() > 0
    reference.loc[reference.exact_model_id.eq("a::tuned"), "test_error"] = 100
    assert selected(reference, A).exact_model_id.item() == "a::tuned"
    reference.loc[reference.model_id.eq("a"), "validation_error"] = .1
    assert selected(reference.sample(frac=1, random_state=3), A).exact_model_id.item() == "a::default"


def test_validation_clipping_ties_use_explicit_tie_policy():
    reference = reference_for((.1, .3, .5))
    reference.loc[reference.model_id.eq("a"), "validation_error"] = .9
    extra = reference.iloc[:1].copy()
    extra["exact_model_id"] = "a::tuned"
    extra["validation_error"] = .8
    extra["test_error"] = .8
    # Add many good validation methods: both A variants clip to 1, so raw .8 must not win.
    others = reference_for((.01, .02, .03))
    others["model_id"] = ["d", "e", "f"]
    others["exact_model_id"] = others.model_id + "::default"
    reference = pd.concat([reference, extra, others], ignore_index=True)
    assert selected(reference, A).exact_model_id.item() == "a::default"


def test_sources_are_part_of_dataset_identity_and_no_split_pseudoreplication():
    reference = pd.concat([reference_for(), reference_for(split=1),
                           reference_for((.3, .1, .5), source="local")], ignore_index=True)
    result = gap(reference)
    assert len(result.dataset_gaps) == 2
    assert result.dataset_gaps.set_index("dataset_source").n_splits.to_dict() == {"local": 1, "tabarena": 2}
    assert result.dataset_gaps.set_index("dataset_source").gap.to_dict() == {"local": 1., "tabarena": -1.}


def test_missing_family_split_is_an_error():
    reference = pd.concat([reference_for(), reference_for(split=1)], ignore_index=True)
    with pytest.raises(ValueError, match="Unequal family split coverage"):
        gap(reference, reference[~(reference.model_id.eq("b") & reference.split_id.eq(1))])


def test_two_models_are_not_a_reference_universe():
    reference = reference_for().iloc[:2]
    with pytest.raises(ValueError, match="Pair-only"):
        gap(reference)


def test_reference_must_include_identical_candidate_records():
    reference = reference_for()
    targets = reference.copy()
    targets.loc[0, "test_error"] = .2
    with pytest.raises(ValueError, match="same candidate records"):
        gap(reference, targets)


@pytest.mark.parametrize("mutation", ["duplicate", "nonfinite", "imputed"])
def test_invalid_normalization_inputs(mutation):
    reference = reference_for()
    if mutation == "duplicate":
        reference = pd.concat([reference, reference.iloc[:1]])
    elif mutation == "nonfinite":
        reference.loc[0, "test_error"] = np.nan
    else:
        reference.loc[0, "imputed"] = True
    with pytest.raises(ValueError):
        gap(reference)


def test_gap_cannot_mix_metrics_or_family_members():
    reference = reference_for()
    with pytest.raises(ValueError, match="overlap"):
        calculate_dataset_gaps(reference, A, Family("other", ("a",)), reference=reference, reference_label="test")
    reference.loc[0, "metric"] = "rmse"
    with pytest.raises(ValueError, match="incompatible metrics"):
        gap(reference)


def test_reference_hash_is_order_independent():
    reference = reference_for()
    first = normalize_test_errors(reference, reference, reference_label="test")[1]
    shuffled = reference.sample(frac=1, random_state=42)
    second = normalize_test_errors(shuffled, shuffled, reference_label="test")[1]
    assert first["reference_sha256"] == second["reference_sha256"]
