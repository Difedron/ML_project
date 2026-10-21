from dataclasses import dataclass

import pandas as pd
import numpy as np

from meta_selector.tabarena.results_loader import OBSERVATION_KEYS, SPLIT_KEYS
from .families import Family
from .normalization import normalize_test_errors, normalize_validation_errors


@dataclass
class GapResult:
    split_gaps: pd.DataFrame
    dataset_gaps: pd.DataFrame
    provenance: dict


def select_by_validation(results: pd.DataFrame, family: Family) -> pd.DataFrame:
    chosen = results[results.model_id.isin(family.members)].copy()
    if chosen.empty:
        raise ValueError(f"No results for exact family {family.name}: {family.members}")
    if "normalized_validation_error" not in chosen:
        raise ValueError("Normalize validation errors against a declared reference before family selection")
    if not np.isfinite(chosen[["validation_error", "normalized_validation_error"]].to_numpy(dtype=float)).all():
        raise ValueError("Validation selection cannot use missing/non-finite errors")
    # No test quantity participates in selection or tie breaking.
    chosen = chosen.sort_values(["normalized_validation_error", "exact_model_id"], kind="stable")
    return chosen.drop_duplicates(SPLIT_KEYS).sort_values(SPLIT_KEYS).reset_index(drop=True)


def calculate_dataset_gaps(results: pd.DataFrame, family_a: Family, family_b: Family, *,
                           reference: pd.DataFrame, reference_label: str,
                           epsilon: float = 1e-5) -> GapResult:
    if set(family_a.members) & set(family_b.members):
        raise ValueError("Compared families must not overlap")
    if family_a.name == family_b.name:
        raise ValueError("Compared families need different names")
    for key, group in results.groupby(OBSERVATION_KEYS):
        if group.metric.nunique() != 1 or group.problem_type.nunique() != 1:
            raise ValueError(f"A dataset mixes incompatible metrics: {key}")
    candidates = results[results.model_id.isin(family_a.members + family_b.members)].copy()
    validated, validation_provenance = normalize_validation_errors(
        candidates, reference, reference_label=reference_label, epsilon=epsilon)
    a = select_by_validation(validated, family_a)
    b = select_by_validation(validated, family_b)
    coverage = a[SPLIT_KEYS].merge(b[SPLIT_KEYS], on=SPLIT_KEYS, how="outer", indicator=True)
    if coverage._merge.ne("both").any():
        missing = coverage.loc[coverage._merge.ne("both"), SPLIT_KEYS + ["_merge"]].to_dict("records")
        raise ValueError(f"Unequal family split coverage; no silent dropping/imputation: {missing[:5]}")
    normalized, provenance = normalize_test_errors(pd.concat([a, b], ignore_index=True), reference,
                                                   reference_label=reference_label, epsilon=epsilon)
    columns = SPLIT_KEYS + ["exact_model_id", "test_error", "validation_error", "normalized_validation_error",
                            "validation_error_min", "validation_error_median", "normalized_test_error",
                            "n_reference_models", "n_reference_variants", "error_min", "error_median"]
    na = normalized[normalized.model_id.isin(family_a.members)][columns]
    nb = normalized[normalized.model_id.isin(family_b.members)][columns]
    splits = na.merge(nb, on=SPLIT_KEYS, validate="one_to_one", suffixes=("_a", "_b"))
    splits["gap"] = splits.normalized_test_error_a - splits.normalized_test_error_b
    splits["raw_error_gap"] = splits.test_error_a - splits.test_error_b
    by_dataset = OBSERVATION_KEYS + ["metric", "problem_type"]
    datasets = splits.groupby(by_dataset, as_index=False).agg(
        gap=("gap", "mean"), raw_error_gap=("raw_error_gap", "mean"),
        n_splits=("split_id", "nunique"), normalized_error_a=("normalized_test_error_a", "mean"),
        normalized_error_b=("normalized_test_error_b", "mean"),
        n_reference_models_min=("n_reference_models_a", "min"),
        n_reference_models_max=("n_reference_models_a", "max"))
    if datasets.duplicated(OBSERVATION_KEYS).any():
        raise ValueError("Expected exactly one meta-observation per dataset/source")
    for table in (splits, datasets):
        table["family_a"] = family_a.name
        table["family_b"] = family_b.name
        table["normalization_reference"] = reference_label
    provenance.update({"family_a": family_a.members, "family_b": family_b.members,
                       "sign": "positive favors B; negative favors A; zero is a tie",
                       "n_datasets": len(datasets), "n_splits": len(splits),
                       "validation_reference_sha256": validation_provenance["reference_sha256"],
                       "selection": "minimum normalized_validation_error; exact_model_id lexical tie break"})
    return GapResult(splits.sort_values(SPLIT_KEYS).reset_index(drop=True),
                     datasets.sort_values(OBSERVATION_KEYS).reset_index(drop=True), provenance)
