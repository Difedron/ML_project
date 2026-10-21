"""Official TabArena model registry / AutoGluon executor, with no algorithm reimplementations."""
from dataclasses import asdict, dataclass, field
import hashlib
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import time
from typing import Protocol
from packaging.requirements import Requirement

import numpy as np
import pandas as pd

from meta_selector.artifacts import config_hash, environment
from meta_selector.landmarkers.custom import classification_score
from meta_selector.tabarena.model_mapping import MODELS

HISTORICAL_REQUIREMENTS = {
    "tabdpt": ("tabdpt<1.2",),
    "realmlp": ("pytabkit>=1.5,<2",),
    "tabm": ("tabm>=0.0.3", "rtdl_num_embeddings>=0.0.12"),
    "tabicl_v1": ("tabicl>=2.0.0",), "tabicl_v2": ("tabicl>=2.0.0",),
    "tabpfn_2_6": ("tabpfn>=8.0.0",), "realtabpfn_v2_5": ("tabpfn>=8.0.0",),
}


@dataclass
class CandidateResult:
    dataset_id: str
    split_id: str
    model_id: str
    model_version: str | None
    exact_model_id: str
    validation_metric: str
    test_metric: str
    dataset_source: str = "local"
    validation_score: float = np.nan
    test_score: float = np.nan
    fit_time: float = np.nan
    predict_time: float = np.nan
    status: str = "unavailable"
    error_message: str = ""
    provenance: dict = field(default_factory=dict)

    def to_record(self):
        row = asdict(self)
        row["provenance"] = json.dumps(row["provenance"], sort_keys=True)
        return row


class CandidateRunner(Protocol):
    def available_models(self) -> list[str]: ...
    def run(self, model_id, X_train, y_train, X_validation, y_validation, X_test, y_test,
            *, dataset_id, split_id, metric="balanced_accuracy") -> CandidateResult: ...


def has_module(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ModuleNotFoundError, ValueError):
        return False


def resolve_official(model_id):
    from tabarena.models.utils import get_configs_generator_from_name
    return get_configs_generator_from_name(MODELS[model_id].registry_name)


def verified_weight(config, model_id):
    entry = config.get("weights", {}).get(model_id, {})
    path = Path(entry.get("path", ""))
    expected = entry.get("sha256", "")
    if not path.is_file() or len(expected) != 64:
        raise ValueError("Local checkpoint + SHA256 not configured; license/authentication may be required to obtain weights")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != expected:
        raise ValueError("Checkpoint SHA256 mismatch")
    return path.resolve(), entry


class TabArenaCandidateRunner:
    def __init__(self, config, *, resolver=None, dependency_probe=None, gpu_probe=None, version_probe=None):
        self.config = config
        self.resolver = resolver or resolve_official
        self.dependency_probe = dependency_probe or has_module
        self.gpu_probe = gpu_probe or self._cuda
        self.version_probe = version_probe or importlib.metadata.version

    @staticmethod
    def _cuda():
        if not has_module("torch"):
            return False
        import torch
        return torch.cuda.is_available()

    def availability(self, model_id):
        definition = MODELS[model_id]
        deps = ("tabarena", "autogluon.tabular") + definition.dependencies
        missing = [name for name in deps if not self.dependency_probe(name)]
        reasons = (["missing dependencies: " + ", ".join(missing)] if missing else [])
        for declaration in HISTORICAL_REQUIREMENTS.get(model_id, ()):
            requirement = Requirement(declaration)
            if not self.dependency_probe(requirement.name):
                continue
            try:
                if self.version_probe(requirement.name) not in requirement.specifier:
                    reasons.append(f"unavailable_exact_version: requires {declaration}")
            except importlib.metadata.PackageNotFoundError:
                reasons.append(f"Cannot verify dependency version: {declaration}")
        if definition.compute == "gpu" and (self.config.get("num_gpus", 0) < 1 or not self.gpu_probe()):
            reasons.append("historical GPU variant requires CUDA GPU")
        if definition.checkpoint_required:
            try:
                _, entry = verified_weight(self.config, model_id)
                if model_id not in {"tabpfn_2_6", "realtabpfn_v2_5"} and not entry.get("parameter"):
                    reasons.append("Official checkpoint hyperparameter not configured")
            except ValueError as error:
                reasons.append(str(error))
        if not missing:
            try:
                generator = self.resolver(model_id)
                if not generator.manual_configs:
                    reasons.append("official registry default config unavailable")
                expected = {"tabpfn_2_6": "tabpfn-v2.6-classifier-v2.6_default.ckpt",
                            "realtabpfn_v2_5": "tabpfn-v2.5-classifier-v2.5_default.ckpt"}.get(model_id)
                if expected and getattr(generator.model_cls, "default_classification_model", None) != expected:
                    reasons.append("unavailable_exact_version: official registry checkpoint generation differs")
            except Exception as error:
                reasons.append(f"official registry unavailable: {type(error).__name__}: {error}")
        return dict(model=definition.registry_name, canonical_id=model_id, available=not reasons,
                    compute=definition.compute, dependency=", ".join(deps), reason="; ".join(reasons))

    def available_models(self):
        return [name for name in MODELS if self.availability(name)["available"]]

    def run(self, model_id, X_train, y_train, X_validation, y_validation, X_test, y_test,
            *, dataset_id, split_id, metric="balanced_accuracy"):
        definition = MODELS[model_id]
        provenance = dict(config=self.config, environment=environment(), model_definition=asdict(definition),
                          problem_type="binary" if y_train.nunique() == 2 else "multiclass",
                          protocol=self.config["protocol"], paper_exact_training_protocol=False)
        result = CandidateResult(dataset_id=dataset_id, split_id=split_id, model_id=model_id,
                                 model_version=definition.version,
                                 exact_model_id=f"local::{model_id}::{config_hash(provenance)}",
                                 validation_metric=metric, test_metric=metric, provenance=provenance)
        availability = self.availability(model_id)
        if not availability["available"]:
            result.error_message = availability["reason"]
            if "unavailable_exact_version" in result.error_message:
                result.status = "unavailable_exact_version"
            return result
        old_offline = os.environ.get("HF_HUB_OFFLINE")
        try:
            # No implicit checkpoint/authentication downloads in this executor.
            os.environ["HF_HUB_OFFLINE"] = "1"
            from autogluon.core.data import LabelCleaner
            from autogluon.features.generators import AutoMLPipelineFeatureGenerator
            generator = self.resolver(model_id)
            cls = generator.model_cls
            params = dict(generator.manual_configs[0])
            params.update(self.config.get("hyperparameters", {}).get(model_id, {}))
            if definition.checkpoint_required:
                path, entry = verified_weight(self.config, model_id)
                if model_id in {"tabpfn_2_6", "realtabpfn_v2_5"}:
                    params["zip_model_path"] = [str(path), None]
                else:
                    parameter = entry.get("parameter")
                    if not parameter:
                        raise ValueError("Configure the official checkpoint hyperparameter explicitly")
                    params[parameter] = str(path)
            task = "binary" if y_train.nunique() == 2 else "multiclass"
            cleaner = LabelCleaner.construct(problem_type=task, y=y_train)
            train_y = cleaner.transform(y_train)
            validation_y, test_y = cleaner.transform(y_validation), cleaner.transform(y_test)
            if validation_y.isna().any() or test_y.isna().any():
                raise ValueError("Validation/test contain labels absent from training")
            features = AutoMLPipelineFeatureGenerator()
            start = time.perf_counter()
            train = features.fit_transform(X_train)
            validation, test = features.transform(X_validation), features.transform(X_test)
            model = cls(problem_type=task, hyperparameters=params)
            model.fit(X=train, y=train_y, X_val=validation, y_val=validation_y,
                      num_cpus=self.config["num_cpus"],
                      num_gpus=self.config["num_gpus"] if definition.compute == "gpu" else 0,
                      time_limit=self.config["time_limit"])
            result.fit_time = time.perf_counter() - start

            class ProbabilityBridge:
                classes_ = np.sort(train_y.unique())
                def predict(self, X):
                    return model.predict(X=X)
                def predict_proba(self, X):
                    values = np.asarray(model.predict_proba(X=X))
                    return np.column_stack([1 - values, values]) if values.ndim == 1 else values

            start = time.perf_counter()
            bridge = ProbabilityBridge()
            result.validation_score = classification_score(bridge, validation, validation_y, metric, bridge.classes_)
            result.test_score = classification_score(bridge, test, test_y, metric, bridge.classes_)
            result.predict_time = time.perf_counter() - start
            result.status = "ok"
            result.provenance.update(official_class=f"{cls.__module__}.{cls.__name__}", hyperparameters=params)
        except Exception as error:
            result.status = "error"
            result.error_message = f"{type(error).__name__}: {error}"
        finally:
            if old_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = old_offline
        return result
