"""CLI selection regression tests without extraction or estimator fitting."""
import importlib
import sys

import pandas as pd
import pytest

from meta_selector.paths import DEFAULT_PROJECT_ROOT


@pytest.fixture
def cli(monkeypatch):
    monkeypatch.syspath_prepend(str(DEFAULT_PROJECT_ROOT / "scripts"))
    return importlib.import_module("_research_cli")


@pytest.mark.parametrize("selection,expected", [
    (["pmlb_saheart"], ["pmlb_saheart"]),
    (["pmlb_saheart", "pmlb_penguins"], ["pmlb_saheart", "pmlb_penguins"]),
    (["--all-local"], ["registered"]),
])
def test_selection_and_feature_flag(cli, monkeypatch, selection, expected):
    seen = []
    monkeypatch.setattr(cli, "local_ids", lambda root: ["registered"])
    def process(kind, ids, **kwargs):
        seen.append((kind, ids, kwargs))
        return pd.DataFrame([dict(dataset_id=ids[0], namespace="controls", status="ok", rows=5, error_message="")])
    monkeypatch.setattr(cli, "process_local", process)
    monkeypatch.setattr(sys, "argv", ["extract_meta_features.py", *selection, "--without-standard-landmarking"])
    assert cli.batch_main("meta_features") == 0
    assert seen[0][0:2] == ("meta_features", expected)
    assert seen[0][2]["include_standard"] is False


@pytest.mark.parametrize("selection", [[], ["pmlb_saheart", "--all-local"]])
def test_invalid_selection_fails_before_computation(cli, monkeypatch, selection):
    monkeypatch.setattr(cli, "process_local", lambda *args, **kwargs: pytest.fail("Must not compute"))
    monkeypatch.setattr(sys, "argv", ["run_landmarkers.py", *selection])
    with pytest.raises(SystemExit) as exc:
        cli.batch_main("landmarkers")
    assert exc.value.code == 2
