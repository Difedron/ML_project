"""Audit wrapper around the existing feature/landmarker CLIs; no new estimators or splits."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import runpy
import sys
import time
from unittest.mock import patch
import warnings

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from meta_selector import batch, load_dataset
from meta_selector.artifacts import atomic_json, environment, research_config, research_paths, upsert_table
from meta_selector.datasets.pmlb import sha256_file
from meta_selector.datasets.splits import check_manifest
from meta_selector.landmarkers import custom
from meta_selector.meta_features import extraction
from meta_selector.paths import DEFAULT_PROJECT_ROOT as ROOT, read_yaml_mapping


LOGS = ROOT / "results/logs/pmlb_features"


def protected_state(specs):
    paths = list((ROOT / "data/PMLB").glob("*.csv"))
    paths += [ROOT / name for name in ("config/research.yaml", "config/pmlb_pilot.yaml", "data/registry/datasets.csv")]
    for spec in specs:
        folder = ROOT / "data/raw" / spec["dataset_id"]
        paths += [folder / "metadata.yaml", folder / f"{spec['original_name']}.csv",
                  ROOT / "data/splits" / spec["dataset_id"] / "holdout_0.json"]
    return {p.relative_to(ROOT).as_posix(): dict(sha256=sha256_file(p), mtime_ns=p.stat().st_mtime_ns)
            for p in paths}


def initialize():
    assert Path(sys.prefix).resolve() == (ROOT / ".venv").resolve(), "Use the project .venv"
    assert sys.version_info[:2] == (3, 12)
    for key in ("TEMP", "TMP", "PIP_CACHE_DIR", "MPLCONFIGDIR", "JOBLIB_TEMP_FOLDER", "NUMBA_CACHE_DIR", "XDG_CACHE_HOME"):
        assert Path(os.environ[key]).resolve().is_relative_to(ROOT), f"External cache/temp: {key}"
    specs = read_yaml_mapping(ROOT / "config/pmlb_pilot.yaml")["datasets"]
    assert len(specs) == 10
    config = research_config(ROOT)
    data = {}
    for spec in specs:
        dataset = load_dataset(spec["dataset_id"], project_root=ROOT)
        manifest_path = ROOT / "data/splits" / dataset.dataset_id / "holdout_0.json"
        # Fail before calling the batch CLI if any approved manifest is missing/changed.
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        check_manifest(manifest, dataset)
        assert manifest["protocol"] == config["splits"]
        assert dataset.metadata.file.sha256 == spec["sha256"]
        assert list(dataset.metadata.categorical_columns) == spec["categorical_columns"]
        data[dataset.dataset_id] = (dataset, manifest)
    current = protected_state(specs)
    baseline = LOGS / "protected_before.json"
    if baseline.exists():
        assert json.loads(baseline.read_text()) == current, "Protected inputs changed since first run"
    else:
        atomic_json(baseline, current)
    packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    atomic_json(LOGS / "environment.json", dict(executable=sys.executable, environment=environment(),
                                               packages=packages,
                                               threads={k: os.environ.get(k) for k in (
                                                   "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}))
    return specs, config, data, current


def index_hash(index):
    return hashlib.sha256(np.asarray(index, dtype=np.int64).tobytes()).hexdigest()


def execute(phase, run_id, selected=None):
    specs, config, data, protected = initialize()
    ids = selected or [s["dataset_id"] for s in specs]
    assert len(ids) == len(set(ids)) and set(ids) <= set(data)
    output = LOGS / run_id
    audit_path = output / f"{phase}_audit.json"
    if audit_path.exists() or (output / f"{phase}.log").exists():
        raise ValueError("Choose a new run ID; previous diagnostics are preserved")
    output.mkdir(parents=True, exist_ok=True)
    events, timings = [], []
    state = {"context": None, "fit": None, "last_fit": None, "preprocessing_active": False}

    def outer(X, y, identity):
        dataset, manifest = data[identity["dataset_id"]]
        positions = manifest["indices"]["train"]
        pd.testing.assert_frame_equal(X, dataset.X.iloc[positions])
        if y is not None:
            pd.testing.assert_series_equal(y, dataset.y.iloc[positions])
        assert identity["split_id"] == "holdout_0"
        assert not set(X.index) & set(manifest["indices"]["validation"] + manifest["indices"]["test"])
        events.append(dict(kind="outer_input", dataset_id=dataset.dataset_id, rows=len(X),
                           index_hash=index_hash(X.index), categorical_columns=list(X.select_dtypes("category"))))
        return dict(dataset_id=dataset.dataset_id, X=X, y=y)

    def operation(original, namespace):
        def wrapped(X, y=None, **kwargs):
            context = outer(X, y, kwargs["identity"])
            context["namespace"] = namespace
            if namespace == "custom":
                cv = StratifiedKFold(n_splits=config["custom_landmarkers"]["n_splits"], shuffle=True,
                                     random_state=config["custom_landmarkers"]["random_seed"])
                context["folds"] = list(cv.split(X, y))
                context["counts"] = {}
            state["context"] = context
            start = time.perf_counter()
            entry = dict(dataset_id=context["dataset_id"], namespace=namespace, status="error")
            caught = []
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    result = original(X, **kwargs) if namespace == "controls" else original(X, y, **kwargs)
                entry.update(status="error" if result.status.eq("error").any() else "ok", rows=len(result))
                return result
            except Exception as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                entry.update(seconds=time.perf_counter() - start,
                             outer_warnings=[dict(category=w.category.__name__, message=str(w.message)) for w in caught])
                timings.append(entry)
                state["context"] = None
        return wrapped

    original_prepare = extraction._pymfe_input

    def prepared(X):
        context = state["context"]
        pd.testing.assert_frame_equal(X, context["X"])
        values, cats = original_prepare(X)
        assert cats == [i for i, name in enumerate(X) if name in X.select_dtypes("category")]
        event = dict(kind="pymfe_prepared", dataset_id=context["dataset_id"], rows=len(X), categorical_indices=cats)
        if context["dataset_id"] == "pmlb_mushroom":
            unknown = X["stalk-root"].eq(0).to_numpy()
            assert np.all(values[unknown, list(X).index("stalk-root")] == "0")
            event["preserved_unknown_training_rows"] = int(unknown.sum())
        events.append(event)
        return values, cats

    original_make = custom.make_landmarker

    def make_checked(name, X, model_config):
        context = state["context"]
        pd.testing.assert_frame_equal(X, context["X"])
        model = original_make(name, X, model_config)
        original_fit = model.fit

        def fit_checked(X_fit, y_fit, **kwargs):
            number = context["counts"].get(name, 0)
            train, held = context["folds"][number]
            pd.testing.assert_frame_equal(X_fit, context["X"].iloc[train])
            pd.testing.assert_series_equal(y_fit, context["y"].iloc[train])
            assert not set(X_fit.index) & set(context["X"].iloc[held].index)
            event = dict(kind="inner_fit", dataset_id=context["dataset_id"], model_id=name, fold=number,
                         rows=len(X_fit), held_rows=len(held), index_hash=index_hash(X_fit.index), preprocessing=[])
            state["fit"] = event
            state["last_fit"] = (model, number)
            try:
                result = original_fit(X_fit, y_fit, **kwargs)
                assert any(e["transformer"] == "SimpleImputer" for e in event["preprocessing"])
                event["status"] = "ok"
                context["counts"][name] = number + 1
                return result
            except Exception as exc:
                event.update(status="error", error=f"{type(exc).__name__}: {exc}")
                raise
            finally:
                events.append(event)
                state["fit"] = None
        model.fit = fit_checked
        return model

    original_score = custom.classification_score

    def score_checked(model, X, y, metric, classes):
        context = state["context"]
        last_model, number = state["last_fit"]
        assert last_model is model
        _, held = context["folds"][number]
        pd.testing.assert_frame_equal(X, context["X"].iloc[held])
        pd.testing.assert_series_equal(y, context["y"].iloc[held])
        result = original_score(model, X, y, metric, classes)
        events.append(dict(kind="inner_score", dataset_id=context["dataset_id"], fold=number,
                           rows=len(X), index_hash=index_hash(X.index), score=result))
        return result

    def transformer_fit(original, name):
        def checked(instance, X, *args, **kwargs):
            context = state["context"]
            if context is not None:
                fit = state["fit"]
                if context["namespace"] == "custom":
                    assert fit is not None, "Preprocessing fit outside inner training scope"
                    if state["preprocessing_active"]:
                        assert len(X) == fit["rows"], "Fold-local preprocessing row count mismatch"
                        if isinstance(X, pd.DataFrame):
                            assert index_hash(X.index) == fit["index_hash"], "Fold-local preprocessing indices mismatch"
                        fit["preprocessing"].append(dict(transformer=name, rows=len(X), columns=X.shape[1]))
                    else:
                        # LDA(shrinkage='auto') internally standardizes each class subset.
                        # These are estimator-internal fits, not ColumnTransformer preprocessing.
                        assert 0 < len(X) <= fit["rows"]
                        fit.setdefault("estimator_internal_fits", []).append(
                            dict(transformer=name, rows=len(X), columns=X.shape[1]))
                elif name == "SimpleImputer":
                    assert len(X) == len(context["X"])
                    assert isinstance(X, pd.DataFrame) and X.index.equals(context["X"].index)
            return original(instance, X, *args, **kwargs)
        return checked

    original_columns_fit = ColumnTransformer.fit_transform

    def columns_fit(instance, X, *args, **kwargs):
        previous = state["preprocessing_active"]
        state["preprocessing_active"] = True
        try:
            return original_columns_fit(instance, X, *args, **kwargs)
        finally:
            state["preprocessing_active"] = previous

    original_split = batch.make_split

    def existing_split(dataset, **kwargs):
        manifest_path = ROOT / "data/splits" / dataset.dataset_id / "holdout_0.json"
        assert manifest_path.is_file(), "Never generate a missing split in this pilot"
        before = sha256_file(manifest_path)
        result = original_split(dataset, **kwargs)
        assert sha256_file(manifest_path) == before
        return result

    script = "extract_meta_features.py" if phase == "features" else "run_landmarkers.py"
    argv = [str(ROOT / "scripts" / script), *ids]
    if phase == "features":
        argv.append("--without-standard-landmarking")
    status = 1
    start = time.perf_counter()
    with (output / f"{phase}.log").open("x", encoding="utf-8") as log:
        try:
            with redirect_stdout(log), redirect_stderr(log), \
                    patch.object(batch, "make_split", existing_split), \
                    patch.object(batch, "extract_controls", operation(batch.extract_controls, "controls")), \
                    patch.object(batch, "extract_pymfe", operation(batch.extract_pymfe, "paper_features")), \
                    patch.object(batch, "run_custom_landmarkers", operation(batch.run_custom_landmarkers, "custom")), \
                    patch.object(extraction, "_pymfe_input", prepared), \
                    patch.object(custom, "make_landmarker", make_checked), \
                    patch.object(custom, "classification_score", score_checked), \
                    patch.object(ColumnTransformer, "fit_transform", columns_fit), \
                    patch.object(SimpleImputer, "fit", transformer_fit(SimpleImputer.fit, "SimpleImputer")), \
                    patch.object(StandardScaler, "fit", transformer_fit(StandardScaler.fit, "StandardScaler")), \
                    patch.object(OneHotEncoder, "fit", transformer_fit(OneHotEncoder.fit, "OneHotEncoder")), \
                    patch.object(sys, "argv", argv):
                runpy.run_path(argv[0], run_name="__main__")
        except SystemExit as exc:
            status = int(exc.code or 0)
        finally:
            elapsed = time.perf_counter() - start
            unchanged = protected_state(specs) == protected
            atomic_json(audit_path, dict(run_id=run_id, phase=phase, argv=argv, exit_code=status,
                                         seconds=elapsed, protected_inputs_unchanged=unchanged,
                                         namespace_timings=timings, events=events))
            assert unchanged, "CSV/config/metadata/manifests were modified"
    base = research_paths(ROOT)["derived"]
    source = base / ("meta_features/meta_features.parquet" if phase == "features" else "landmarkers/custom_landmarkers.parquet")
    if source.exists():
        frame = pd.read_parquet(source)
        frame = frame[frame.dataset_id.isin(ids)].copy()
        keys = (["dataset_source", "dataset_id", "split_id", "source", "name", "config_id"] if phase == "features"
                else ["dataset_source", "dataset_id", "split_id", "model_id", "metric", "config_id"])
        upsert_table(output / f"{phase}.parquet", frame, keys)
    status_source = base / ("meta_features" if phase == "features" else "landmarkers") / "run_status.parquet"
    if status_source.exists():
        frame = pd.read_parquet(status_source)
        frame = frame[frame.dataset_id.isin(ids)].copy()
        upsert_table(output / f"{phase}_status.parquet", frame,
                     ["dataset_source", "dataset_id", "split_id", "namespace", "config_id"])
    print(f"{run_id}/{phase}: exit={status}, wall={elapsed:.3f}s, audit events={len(events)}, protected={unchanged}")
    print((output / f"{phase}.log").read_text(encoding="utf-8"))
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=["features", "landmarkers"])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset-id", action="append", help="Explicit approved subset for diagnostic retries")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9_]+", args.run_id):
        parser.error("Use a simple run ID")
    return execute(args.phase, args.run_id, args.dataset_id)


if __name__ == "__main__":
    raise SystemExit(main())
