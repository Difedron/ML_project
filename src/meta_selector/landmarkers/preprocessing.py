"""Fold-local mixed-table preprocessing; fit is always called inside inner CV."""
import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def clean_frame(frame):
    result = frame.copy()
    for column in result:
        if is_numeric_dtype(result[column].dtype) and not is_bool_dtype(result[column].dtype):
            result[column] = pd.to_numeric(result[column]).astype(float).replace([np.inf, -np.inf], np.nan)
        else:
            result[column] = result[column].map(lambda value: np.nan if pd.isna(value) else str(value)).astype(object)
    return result


class BoundedOneHotEncoder(OneHotEncoder):
    def __init__(self, max_dense_cells=20000000):
        self.max_dense_cells = max_dense_cells
        super().__init__(handle_unknown="ignore", sparse_output=False)

    def fit(self, X, y=None):
        super().fit(X, y)
        self._check_size(X)
        return self

    def _check_size(self, X):
        if len(X) * sum(len(values) for values in self.categories_) > self.max_dense_cells:
            raise ValueError("One-hot dense representation exceeds max_dense_cells")

    def transform(self, X):
        self._check_size(X)
        return super().transform(X)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


def make_preprocessor(X, *, scale=False, max_dense_cells=20000000):
    numeric = [name for name, dtype in X.dtypes.items() if is_numeric_dtype(dtype) and not is_bool_dtype(dtype)]
    categorical = [name for name in X if name not in numeric]
    steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale:
        steps.append(("scale", StandardScaler()))
    columns = ColumnTransformer([
        ("numeric", Pipeline(steps), numeric),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encode", BoundedOneHotEncoder(max_dense_cells)),
        ]), categorical),
    ], sparse_threshold=0)
    return Pipeline([("clean", FunctionTransformer(clean_frame, validate=False)), ("columns", columns)])
