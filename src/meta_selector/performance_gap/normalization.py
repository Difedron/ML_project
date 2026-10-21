"""Appendix A.2 formula, with an explicitly declared reference universe."""
import hashlib

import numpy as np
import pandas as pd

from meta_selector.tabarena.results_loader import ROW_KEYS, SPLIT_KEYS


def _normalize_errors(target: pd.DataFrame, reference: pd.DataFrame, *, reference_label: str,
                      error_column: str, epsilon: float) -> tuple[pd.DataFrame, dict]:
    if not reference_label.strip():
        raise ValueError("Declare the normalization reference_label")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive and finite")
    required = set(ROW_KEYS + [error_column, "model_id"])
    for name, table in [("target", target), ("reference", reference)]:
        if table.empty or not required <= set(table):
            raise ValueError(f"{name}: empty or incomplete result schema")
        if table.duplicated(ROW_KEYS).any():
            raise ValueError(f"{name}: duplicate exact result records")
        if not np.isfinite(table[error_column].to_numpy(dtype=float)).all():
            raise ValueError(f"{name}: non-finite {error_column}")
        if "imputed" in table and table.imputed.any():
            raise ValueError("Imputed errors cannot be normalization anchors")
    matched = target[ROW_KEYS + [error_column]].merge(
        reference[ROW_KEYS + [error_column]], on=ROW_KEYS, how="left", validate="one_to_one",
        suffixes=("_target", "_reference"), indicator=True)
    if matched._merge.ne("both").any() or not np.allclose(
            matched[error_column + "_target"], matched[error_column + "_reference"], rtol=1e-12, atol=1e-12):
        raise ValueError(f"Reference must contain the same candidate records and {error_column}")
    anchors = reference.groupby(SPLIT_KEYS, as_index=False).agg(
        error_min=(error_column, "min"), error_median=(error_column, "median"),
        n_reference_variants=("exact_model_id", "nunique"), n_reference_models=("model_id", "nunique"))
    output = target.merge(anchors, on=SPLIT_KEYS, how="left", validate="many_to_one")
    if output.error_min.isna().any():
        raise ValueError("Reference lacks required dataset/split/metric anchors")
    if (output.n_reference_models < 3).any():
        raise ValueError("Pair-only normalization is forbidden: supply a declared multi-model reference cohort")
    denominator = (output.error_median - output.error_min).clip(lower=epsilon)
    output["normalized_" + error_column] = ((output[error_column] - output.error_min) / denominator).clip(0, 1)
    output["normalization_reference"] = reference_label
    fingerprint_table = reference.sort_values(ROW_KEYS)[ROW_KEYS + [error_column]]
    fingerprint = hashlib.sha256(fingerprint_table.to_csv(index=False, float_format="%.17g").encode()).hexdigest()
    provenance = {"reference_label": reference_label, "epsilon": epsilon,
                  "reference_models": sorted(reference.model_id.unique().tolist()),
                  "reference_variants": sorted(reference.exact_model_id.unique().tolist()),
                  "reference_rows": len(reference), "reference_sha256": fingerprint,
                  "error_column": error_column,
                  "formula": "clip((error-min)/(max(median-min, epsilon)), 0, 1)"}
    if error_column == "validation_error":
        output = output.rename(columns={column: "validation_" + column for column in (
            "error_min", "error_median", "n_reference_variants", "n_reference_models")})
    return output, provenance


def normalize_test_errors(target: pd.DataFrame, reference: pd.DataFrame, *, reference_label: str,
                          epsilon: float = 1e-5) -> tuple[pd.DataFrame, dict]:
    return _normalize_errors(target, reference, reference_label=reference_label,
                             error_column="test_error", epsilon=epsilon)


def normalize_validation_errors(target: pd.DataFrame, reference: pd.DataFrame, *, reference_label: str,
                                epsilon: float = 1e-5) -> tuple[pd.DataFrame, dict]:
    """Appendix A.2: use separate validation anchors, never test anchors for selection."""
    return _normalize_errors(target, reference, reference_label=reference_label,
                             error_column="validation_error", epsilon=epsilon)
