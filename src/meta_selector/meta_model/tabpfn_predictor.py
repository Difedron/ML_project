"""Exact TabPFN-2.6 regressor; offline checkpoint is mandatory, no generation fallback."""
from typing import Protocol, Self

import numpy as np
import pandas as pd
import hashlib
import importlib.util
from pathlib import Path


REQUIRED_MODEL_VERSION = "2.6"
REQUIRED_ESTIMATOR = "TabPFNRegressor"


class GapPredictor(Protocol):
    def fit(self, features: pd.DataFrame, gap: np.ndarray) -> Self: ...
    def predict(self, features: pd.DataFrame) -> np.ndarray: ...

class ExactVersionUnavailable(RuntimeError):
    status = "unavailable_exact_version"


class TabPFN26Predictor:
    def __init__(self, config):
        self.config = config

    def availability(self):
        try:
            if str(self.config["model_version"]) != "2.6":
                raise ValueError("Only exact model_version=2.6 is supported")
            if importlib.util.find_spec("tabpfn") is None:
                raise ValueError("Missing tabpfn dependency")
            path = Path(self.config.get("checkpoint") or "")
            if not path.is_file():
                raise ValueError("Configure a local TabPFN-2.6 regressor checkpoint; license/authentication may be required to obtain it")
            if path.name != "tabpfn-v2.6-regressor-v2.6_default.ckpt":
                raise ValueError("Expected the exact official 2.6 default regressor filename")
            expected = self.config.get("checkpoint_sha256") or ""
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if len(expected) != 64 or actual != expected:
                raise ValueError("Supply and verify trusted checkpoint_sha256; it must identify the official 2.6 artifact")
            import os
            import tempfile
            os.environ.setdefault("SKB_DATA_DIRECTORY", str(Path(tempfile.gettempdir()) / "meta-selector-skrub"))
            from tabpfn import TabPFNRegressor
            from tabpfn.constants import ModelVersion
            if not hasattr(ModelVersion, "V2_6") or not hasattr(TabPFNRegressor, "create_default_for_version"):
                raise ValueError("SDK lacks exact V2_6 factory")
            return dict(status="available", reason="", checkpoint=str(path.resolve()), sha256=actual)
        except Exception as error:
            return dict(status="unavailable_exact_version", reason=f"{type(error).__name__}: {error}")

    def fit(self, features, gap):
        status = self.availability()
        if status["status"] != "available":
            raise ExactVersionUnavailable(status["reason"])
        from tabpfn import TabPFNRegressor
        from tabpfn.constants import ModelVersion
        self.model_ = TabPFNRegressor.create_default_for_version(
            ModelVersion.V2_6, model_path=status["checkpoint"],
            device=self.config["device"], random_state=self.config["random_seed"])
        self.model_.fit(features, np.asarray(gap, dtype=float))
        from tabpfn.architectures.tabpfn_v2_6 import TabPFNV2p6Config
        configs = getattr(self.model_, "configs_", [])
        if not configs or not all(isinstance(config, TabPFNV2p6Config) for config in configs):
            del self.model_
            raise ExactVersionUnavailable("Loaded architecture is not TabPFN-2.6; no fallback is permitted")
        return self

    def predict(self, features):
        if not hasattr(self, "model_"):
            raise ValueError("Fit the exact-version predictor before predict")
        return np.asarray(self.model_.predict(features), dtype=float)
