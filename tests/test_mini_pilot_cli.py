"""Test the real CLI and Parquet upsert with simulated native subprocesses."""
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from meta_selector import artifacts
from meta_selector.candidates import CandidateResult
from meta_selector.paths import DEFAULT_PROJECT_ROOT


@pytest.fixture
def cli(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(DEFAULT_PROJECT_ROOT / "scripts/candidates"))
    import _isolated_candidates as isolated
    import run_mini_pilot as mini
    import run_saheart_smoke as smoke

    validated, calls = [], []
    failures = {}
    real_load_contract = isolated.load_contract
    monkeypatch.setattr(artifacts, "research_paths", lambda root: {"derived": tmp_path})
    monkeypatch.setattr(isolated, "load_contract", lambda name, expected: validated.append((name, expected)))

    def run(command, *, cwd, env, stdout, stderr, timeout):
        model = command[command.index("--model") + 1]
        dataset = command[command.index("--dataset") + 1] if "--dataset" in command else "pmlb_saheart"
        scope = "candidate_smoke" if dataset == "pmlb_saheart" else "candidate_mini_pilot"
        table = tmp_path / scope / "candidate_results.parquet"
        if calls and calls[-1][0] == scope:
            _, previous_dataset, previous_model = calls[-1]
            previous_rows = pd.read_parquet(table)
            assert ((previous_rows.dataset_id == previous_dataset) & (previous_rows.model_id == previous_model)).any()
        calls.append((scope, dataset, model))
        assert env["HF_HUB_OFFLINE"] == "1" and timeout == isolated.WORKER_TIMEOUT_SECONDS
        failure = failures.get((dataset, model))
        if failure == "crash":
            return SimpleNamespace(returncode=-6)
        family, variant, version = isolated.MODELS[model]
        record = CandidateResult(dataset_id=dataset, split_id="holdout_0", seed=42, dataset_source="pmlb",
            model_id=model, model_family=family, variant=variant, model_version=version, exact_model_id="test-id",
            validation_metric="roc_auc", test_metric="roc_auc", status="ok", total_time_sec=len(calls)).to_record()
        if failure == "wrong_dataset":
            record["dataset_id"] = "pmlb_other"
        response = Path(command[command.index("--worker-result") + 1])
        response.write_text(json.dumps(record), encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(isolated.subprocess, "run", run)
    return SimpleNamespace(isolated=isolated, mini=mini, smoke=smoke, validated=validated, calls=calls,
                           failures=failures, output=tmp_path, real_load_contract=real_load_contract)


def test_mini_pilot_order_failure_containment_and_upsert(cli):
    cli.failures[("pmlb_penguins", "tabpfn_2_6")] = "crash"
    cli.failures[("pmlb_tic_tac_toe", "lightgbm")] = "wrong_dataset"
    assert cli.mini.main(["--all"]) == 1
    assert cli.validated == list(cli.mini.CONTRACTS.items())
    assert cli.calls == [("candidate_mini_pilot", dataset, model)
                         for dataset in cli.mini.CONTRACTS for model in cli.isolated.MODELS]
    output = cli.output / "candidate_mini_pilot"
    path = output / "candidate_results.parquet"
    rows = pd.read_parquet(path)
    assert len(rows) == 12 and rows.status.eq("ok").sum() == 10
    crashed = rows.query('dataset_id == "pmlb_penguins" and model_id == "tabpfn_2_6"').iloc[0]
    assert crashed.status == "error" and crashed.error_type == "WorkerProcessError"
    assert crashed.validation_metric == "log_loss"
    assert json.loads(crashed.provenance)["error_stage"] == "worker"
    assert "pmlb_other" not in set(rows.dataset_id)
    summary = json.loads((output / "run_summary.json").read_text())
    assert (summary["attempts"], summary["successful"], summary["failed"]) == (12, 10, 2)
    assert summary["wall_clock_sec"] > 0

    del cli.failures[("pmlb_penguins", "tabpfn_2_6")]
    assert cli.mini.main(["--dataset", "pmlb_penguins", "--model", "tabpfn_2_6"]) == 0
    rows = pd.read_parquet(path)
    assert len(rows) == 12 and rows.status.eq("ok").sum() == 11
    assert not rows.duplicated(["dataset_id", "split_id", "seed", "model_family", "variant"]).any()
    updated = rows.query('dataset_id == "pmlb_penguins" and model_id == "tabpfn_2_6"').iloc[0]
    assert updated.status == "ok" and updated.total_time_sec == 13
    assert json.loads((output / "run_summary.json").read_text()) == summary


def test_stage5_cli_keeps_fixed_saheart_dataset_and_output(cli):
    assert cli.smoke.main(["--model", "lightgbm"]) == 0
    assert cli.calls == [("candidate_smoke", "pmlb_saheart", "lightgbm")]
    assert cli.validated == [("pmlb_saheart", (462, 9, 2, 276, 93, 93))]
    output = cli.output / "candidate_smoke"
    assert (output / "lightgbm.log").is_file()
    assert pd.read_parquet(output / "candidate_results.parquet").dataset_id.item() == "pmlb_saheart"


@pytest.mark.parametrize("args", [[], ["--dataset", "pmlb_saheart"], ["--dataset", "pmlb_mushroom"],
                                  ["--all", "--model", "lightgbm"]])
def test_cli_requires_explicit_allowed_dataset_scope(cli, args):
    with pytest.raises(SystemExit) as error:
        cli.mini.main(args)
    assert error.value.code == 2 and cli.calls == []


def test_contract_mismatch_on_last_dataset_stops_before_any_worker(cli, monkeypatch):
    from meta_selector.datasets import shared_pilot

    # Keep the real dimension gate, while making just the final shared split malformed.
    monkeypatch.setattr(cli.isolated, "load_contract", cli.real_load_contract)
    seen = []

    def load_dataset(dataset_id):
        rows, features, classes, *_ = cli.mini.CONTRACTS[dataset_id]
        return SimpleNamespace(X=pd.DataFrame(0, index=range(rows), columns=range(features)),
                               y=pd.Series([row % classes for row in range(rows)]))

    def load_split(dataset_id, split_id, *, seed, dataset):
        assert (split_id, seed) == ("holdout_0", 42)
        seen.append(dataset_id)
        _, _, _, train, validation, test = cli.mini.CONTRACTS[dataset_id]
        if dataset_id == "pmlb_hill_valley_without_noise":
            test -= 1
        return SimpleNamespace(train_indices=range(train), validation_indices=range(validation), test_indices=range(test))

    monkeypatch.setattr(shared_pilot, "load_pmlb_dataset", load_dataset)
    monkeypatch.setattr(shared_pilot, "load_shared_split", load_split)
    monkeypatch.setattr(cli.isolated.subprocess, "run", lambda *args, **kwargs: pytest.fail("No worker before contract validation"))
    with pytest.raises(ValueError, match=r"pmlb_hill_valley_without_noise/holdout_0/42: expected"):
        cli.mini.main(["--all"])
    assert seen == list(cli.mini.CONTRACTS)
