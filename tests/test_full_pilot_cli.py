"""Stage-7 planning, resume receipts and diagnostic output without native estimators."""
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from meta_selector import artifacts
from meta_selector.candidates import CandidateResult
from meta_selector.datasets.shared_pilot import file_sha256
from meta_selector.paths import DEFAULT_PROJECT_ROOT


@pytest.fixture
def cli(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(DEFAULT_PROJECT_ROOT / "scripts/candidates"))
    import _isolated_candidates as isolated
    import _full_pilot_artifacts as outputs
    import run_full_pilot as full

    events = []
    monkeypatch.setattr(artifacts, "research_paths", lambda root: {"derived": tmp_path})
    monkeypatch.setattr(isolated, "load_contract", lambda dataset, dimensions: events.append(("verify", dataset)))
    return SimpleNamespace(isolated=isolated, outputs=outputs, full=full, events=events, root=tmp_path)


def test_full_cli_exact_scope_sequential_persistence_and_resume(cli, monkeypatch):
    expected_datasets = ("pmlb_saheart", "pmlb_tic_tac_toe", "pmlb_kr_vs_kp", "pmlb_mushroom",
        "pmlb_hill_valley_without_noise", "pmlb_twonorm", "pmlb_penguins", "pmlb_car_evaluation",
        "pmlb_vehicle", "pmlb_waveform_21")
    models = ("lightgbm", "lightgbm_priortune_t_k25", "tabpfn_2_6", "tabicl_v2")
    assert tuple(cli.full.CONTRACTS) == expected_datasets
    assert tuple(cli.isolated.MODELS) == models
    base = cli.root / "candidate_full_pilot"
    path = base / "candidate_results.parquet"
    last = None

    class Batch:
        protocol_id = "b" * 64

        def __init__(self, root, output, contracts, verified):
            assert list(verified) == list(expected_datasets)

        def prepare(self):
            pass

        def can_resume(self, dataset, model):
            frame = pd.read_parquet(path)
            return frame.loc[frame.dataset_id.eq(dataset) & frame.model_id.eq(model), "status"].item() == "ok"

        def stamp(self, record, dataset, model):
            return record

        def audit(self, *, wall_clock_sec, reused):
            assert wall_clock_sec > 0
            cli.events.append(("audit", reused))
            return int(pd.read_parquet(path).status.ne("ok").any())

    def run(entrypoint, dataset, dimensions, model, log_dir, *, explicit_dataset, protocol_id):
        nonlocal last
        if last:
            saved = pd.read_parquet(path)
            assert ((saved.dataset_id == last[0]) & (saved.model_id == last[1])).any()
        assert explicit_dataset and protocol_id == Batch.protocol_id and log_dir.name == dataset
        fail = last is None
        last = dataset, model
        cli.events.append(("run", dataset, model))
        family, variant, version = cli.isolated.MODELS[model]
        return CandidateResult(dataset, "holdout_0", model, version, "exact-id", "roc_auc", "roc_auc",
            dataset_source="pmlb", seed=42, model_family=family, variant=variant, status="error" if fail else "ok",
            error_type="WorkerProcessError" if fail else "", error_message="crash" if fail else "",
            provenance={"error_stage": "initialization" if fail else None}).to_record()

    monkeypatch.setattr(cli.full, "FullPilotArtifacts", Batch)
    monkeypatch.setattr(cli.isolated, "run_isolated", run)
    monkeypatch.setattr("meta_selector.datasets.splits.make_split", lambda *a, **k: pytest.fail("No fallback split"))
    assert cli.full.main(["--all"]) == 1
    assert cli.events == ([ ("verify", name) for name in expected_datasets]
        + [("run", dataset, model) for dataset in expected_datasets for model in models] + [("audit", 0)])
    saved = pd.read_parquet(path)
    assert len(saved) == 40 and saved.status.eq("error").sum() == 1
    assert saved.loc[saved.status.eq("error"), "error_message"].item() == "crash"
    cli.events.clear()
    assert cli.full.main(["--all", "--resume"]) == 0
    assert [event for event in cli.events if event[0] == "run"] == [("run", "pmlb_saheart", "lightgbm")]
    assert cli.events[-1] == ("audit", 39)
    assert len(pd.read_parquet(path)) == 40


def test_full_cli_blocks_before_any_worker_on_last_dimension_mismatch(cli, monkeypatch):
    def load(dataset, dimensions):
        cli.events.append(("verify", dataset))
        if dataset == "pmlb_waveform_21":
            raise ValueError("shared validation size mismatch")

    monkeypatch.setattr(cli.isolated, "load_contract", load)
    monkeypatch.setattr(cli.isolated, "run_isolated", lambda *args, **kwargs: pytest.fail("worker started before all checks"))
    with pytest.raises(ValueError, match="shared validation size mismatch"):
        cli.full.main(["--dataset", "pmlb_saheart", "--model", "lightgbm"])
    assert cli.events == [("verify", name) for name in cli.full.CONTRACTS]


def test_full_cli_rejects_a_dataset_outside_fixed_scope(cli):
    with pytest.raises(SystemExit) as error:
        cli.full.main(["--dataset", "pmlb_magic"])
    assert error.value.code == 2 and cli.events == []


def test_resume_rejects_stale_protocol_weights_and_logs(cli):
    batch = cli.outputs.FullPilotArtifacts.__new__(cli.outputs.FullPilotArtifacts)
    dataset, model = "pmlb_saheart", "lightgbm"
    batch.output, batch.table, batch.protocol_id = cli.root, cli.root / "candidates.parquet", "protocol-one"
    batch.python, batch.versions = "3.13", {"lightgbm": "4.7.0"}
    batch.contracts = {dataset: (462, 9, 2, 276, 93, 93)}
    batch.catalog = {dataset: (SimpleNamespace(categorical_columns=("Famhist",), y=pd.Series([0, 1])), None)}
    split = {"fingerprint": "csv-sha", "shared_manifest_sha256": "manifest-sha"}
    batch.expected = {(dataset, model): dict(dataset_id=dataset, model_id=model, model_family="lightgbm", variant="baseline",
        split_id="holdout_0", seed=42, exact_model_id="exact-model-one", device="cpu", shared_split=split)}
    log = batch._log(dataset, model)
    log.parent.mkdir()
    log.write_text("lightgbm: fit_train\nlightgbm: score_validation\nlightgbm: final_test\n")
    pv = dict(shared_split=split, full_pilot={"protocol_id": batch.protocol_id, "log_sha256": file_sha256(log)},
              categorical_columns=["Famhist"], preprocessing_fit_on="train_only", estimator_classes=[0, 1], label_classes=[0, 1])
    row = CandidateResult(dataset, "holdout_0", model, "4.7.0", "exact-model-one", "roc_auc", "roc_auc",
        dataset_source="pmlb", seed=42, model_family="lightgbm", variant="baseline", device="cpu", status="ok",
        python_version=batch.python, package_versions=batch.versions, validation_score=.7, balanced_accuracy=.6,
        roc_auc=.7, log_loss=.5, model_init_time_sec=.1, fit_time_sec=.1, validation_time_sec=.1,
        tuning_time_sec=0., predict_time_sec=.1, total_time_sec=.4, n_features=9, n_classes=2,
        n_train=276, n_validation=93, n_test=93, provenance=pv).to_record()

    def save(**changes):
        pd.DataFrame([{**row, **changes}]).to_parquet(batch.table, index=False)

    save()
    assert batch.can_resume(dataset, model)
    for changes in ({"status": "error"}, {"exact_model_id": "old-model"}, {"split_id": "other"},
                    {"provenance": json.dumps({**pv, "full_pilot": {**pv["full_pilot"], "protocol_id": "stale-code"}})},
                    {"provenance": json.dumps({**pv, "shared_split": {"fingerprint": "old-data"}})}):
        save(**changes)
        assert not batch.can_resume(dataset, model)
    save()
    log.write_text("lightgbm: final_test\n")
    assert not batch.can_resume(dataset, model)


def test_best_diagnostic_preserves_exact_ties_and_excludes_partial_groups(cli):
    batch = cli.outputs.FullPilotArtifacts.__new__(cli.outputs.FullPilotArtifacts)
    batch.output, batch.contracts = cli.root, {"tie": None, "partial": None}
    records = []
    for dataset_id in batch.contracts:
        for i, (family, variant, _) in enumerate(cli.isolated.MODELS.values()):
            records.append(dict(dataset_id=dataset_id, model_family=family, variant=variant, device="cpu",
                status="error" if dataset_id == "partial" and i == 3 else "ok", validation_metric="roc_auc",
                balanced_accuracy=.9 if i < 2 else .6, roc_auc=.8, log_loss=.6, validation_score=.7, total_time_sec=.1))
    batch._diagnostics(pd.DataFrame(records))
    tie, partial = json.loads((cli.root / "best_by_balanced_accuracy.json").read_text())
    assert tie["best_variant_by_test_balanced_accuracy"] is None and tie["tie"] is True
    assert tie["tied_variants"] == ["lightgbm/baseline", "lightgbm/priortune_t_k25"]
    assert partial["best_balanced_accuracy"] is None and partial["values"]["tabicl/v2"] is None
    matrix = pd.read_parquet(cli.root / "performance_matrix.parquet").set_index("dataset_id")
    assert matrix.loc["tie", "gap_to_best__tabpfn/v2_6"] == pytest.approx(.3)
    assert pd.isna(matrix.loc["partial", "gap_to_best__lightgbm/baseline"])


def test_priortune_resume_checks_current_trial_set_and_multiclass_winner(cli):
    from meta_selector.candidates.priortune import PORTFOLIO_PIN, PortfolioConfiguration, effective_parameters

    batch = cli.outputs.FullPilotArtifacts.__new__(cli.outputs.FullPilotArtifacts)
    dataset, model = "pmlb_penguins", cli.outputs.PRIOR
    batch.output, batch.table = cli.root, cli.root / "candidates.parquet"
    batch.trials_path, batch.protocol_id = cli.root / "trials.parquet", "protocol-current"
    batch.python, batch.versions = "3.13", {"lightgbm": "4.7.0"}
    batch.contracts = {dataset: (333, 7, 3, 199, 67, 67)}
    batch.catalog = {dataset: (SimpleNamespace(categorical_columns=(), y=pd.Series([0, 1, 2])), None)}
    batch.portfolio = tuple(PortfolioConfiguration(i, i, {"num_leaves": 5}) for i in range(25))
    split = {"fingerprint": "csv-sha", "shared_manifest_sha256": "manifest-sha"}
    batch.expected = {(dataset, model): dict(dataset_id=dataset, model_id=model, model_family="lightgbm",
        variant="priortune_t_k25", split_id="holdout_0", seed=42, exact_model_id="exact", device="cpu", shared_split=split)}
    log = batch._log(dataset, model)
    log.parent.mkdir()
    log.write_text("".join(f"PriorTune config {i}: validation log_loss={1 if i in (3, 4) else 2}, status=ok\n"
                           for i in range(25)) + f"selected config 3 after 25 validation trials\n{model}: final_test\n")
    trial_rows = [dict(dataset_id=dataset, split_id="holdout_0", seed=42, model_family="lightgbm", variant="priortune_t_k25",
        portfolio_id="lgbm/T/K25", portfolio_revision=PORTFOLIO_PIN["revision"], portfolio_sha256=PORTFOLIO_PIN["sha256"],
        exact_model_id="exact", full_pilot_protocol_id=batch.protocol_id, config_id=config.config_id,
        selection_metric="log_loss", selection_direction="minimize", status="ok",
        validation_score=1.0 if config.config_id in (3, 4) else 2.0,
        effective_parameters=json.dumps(effective_parameters(config, 42, 3))) for config in batch.portfolio]
    pv = dict(shared_split=split, full_pilot={"protocol_id": batch.protocol_id, "log_sha256": file_sha256(log)},
        categorical_columns=[], preprocessing_fit_on="train_only", estimator_classes=[0, 1, 2], label_classes=[0, 1, 2],
        priortune=dict(selected_config_id=3, selected_validation_score=1., selected_validation_metric="log_loss",
            selection_direction="minimize", successful_trials=25, failed_trials=0, failure_policy="require_all_25",
            tie_break_rule="smallest_config_id", effective_parameters=effective_parameters(batch.portfolio[3], 42, 3)))
    row = CandidateResult(dataset, "holdout_0", model, "4.7.0", "exact", "log_loss", "log_loss",
        dataset_source="pmlb", seed=42, model_family="lightgbm", variant="priortune_t_k25", device="cpu", status="ok",
        python_version=batch.python, package_versions=batch.versions, validation_score=1., balanced_accuracy=.6,
        roc_auc=.7, log_loss=.5, model_init_time_sec=.1, fit_time_sec=.1, validation_time_sec=.1,
        tuning_time_sec=.3, predict_time_sec=.1, total_time_sec=.4, n_features=7, n_classes=3,
        n_train=199, n_validation=67, n_test=67, provenance=pv).to_record()
    pd.DataFrame([row]).to_parquet(batch.table, index=False)
    pd.DataFrame(trial_rows).to_parquet(batch.trials_path, index=False)
    assert batch.can_resume(dataset, model)
    for altered in (trial_rows[:-1], [{**trial, "full_pilot_protocol_id": "historical"} for trial in trial_rows],
                    [{**trial, "validation_score": 0.5} if trial["config_id"] == 4 else trial for trial in trial_rows]):
        pd.DataFrame(altered).to_parquet(batch.trials_path, index=False)
        assert not batch.can_resume(dataset, model)
