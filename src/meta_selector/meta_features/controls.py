"""Minimal controls. Operational definitions are assumptions recorded in the audit."""
import math

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


def compute_controls(X_train: pd.DataFrame) -> dict[str, float]:
    """Caller must supply only the benchmark training portion, excluding its target."""
    n, d = X_train.shape
    if n == 0 or d == 0:
        raise ValueError("Controls require nonempty training features")
    numeric = sum(is_numeric_dtype(dtype) and not is_bool_dtype(dtype) for dtype in X_train.dtypes)
    return {"log_n": math.log(n), "log_d": math.log(d), "d_over_n": d / n,
            "cat_fraction": (d - numeric) / d,
            "feature_missing_fraction": float(X_train.isna().sum().sum() / (n * d))}
