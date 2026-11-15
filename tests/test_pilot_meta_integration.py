"""Stage-8 checks using the tracked Rita export and synthetic CandidateResult rows."""
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from shutil import copytree
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from meta_selector.artifacts import config_hash
from meta_selector.datasets.shared_pilot import file_sha256, read_pilot_manifest
from meta_selector.integration import pilot
from meta_selector.integration.roles import LANDMARKERS, OUTCOMES, VARIANTS, VARIANT_IDS, feature_diagnostics, load_column_roles


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def inputs():
    rita, roles, manifest = pilot.read_rita_export(ROOT)
    shared, rows, expected = {}, [], []
    for entry in read_pilot_manifest(project_root=ROOT)["datasets"]:
        dataset = entry["dataset_id"]
        split = dict(fingerprint=entry["fingerprint"], rita_dataset_fingerprint=entry["rita_dataset_fingerprint"],
                     shared_manifest_sha256=entry["shared_manifest_sha256"], row_order="raw_csv_data_rows_zero_based")
        sizes = entry["split_sizes"]
        shared[dataset] = dict(csv_sha256=entry["fingerprint"], dataset_fingerprint=entry["rita_dataset_fingerprint"],
            manifest_sha256=entry["shared_manifest_sha256"], row_order=split["row_order"], n_features=entry["n_features"],
            n_classes=entry["n_classes"], train_size=sizes["train"], validation_size=sizes["validation"], test_size=sizes["test"])
        for position, ((family, variant), (name, model_id)) in enumerate(VARIANTS.items()):
            ba = (0.7, 0.6, 0.8, 0.9)[position]
            if dataset == "pmlb_mushroom":
                ba = (0.998, 0.5, 1.0, 1.0)[position]
            if dataset == "pmlb_penguins":
                ba = (1.0, 0.7, 0.8, 1.0)[position]
            identity = dict(dataset_id=dataset, split_id="holdout_0", seed=42, model_family=family, variant=variant,
                            model_id=model_id, exact_model_id=f"pilot::{model_id}::{name}", device="cpu")
            measures = dict(balanced_accuracy=ba, roc_auc=0.81 + position * 0.02, log_loss=0.2 + position * 0.1)
            expected.append({**identity, "shared_split": split})
            row = dict(**identity, **measures, schema_version=2, status="ok", dataset_source="pmlb", validation_score=.75,
                validation_metric="roc_auc" if entry["n_classes"] == 2 else "log_loss", total_time_sec=position + 0.5,
                n_train=sizes["train"], n_validation=sizes["validation"], n_test=sizes["test"],
                n_features=entry["n_features"], n_classes=entry["n_classes"],
                provenance=dict(join_key=[dataset, "holdout_0", 42], shared_split=split, test_scores=measures,
                                priortune={"selected_config_id": 95}))
            rows.append(row)
    protocol = dict(schema_version=1, expected=expected)
    protocol["protocol_id"] = config_hash(protocol)
    for row in rows:
        row["provenance"]["full_pilot"] = {"protocol_id": protocol["protocol_id"]}
        row["provenance"] = json.dumps(row["provenance"])
    summary = dict(protocol_id=protocol["protocol_id"], status="PASS", successful_candidate_runs=40, failed_candidate_runs=0)
    return SimpleNamespace(rita=rita, roles=roles, manifest=manifest, shared=shared, scores=pd.DataFrame(rows),
                           protocol=protocol, summary=summary)


def test_export_roles_and_ten_unique_identities(inputs):
    assert inputs.rita.shape == (10, 221) and inputs.rita.dataset_id.is_unique
    assert set(inputs.rita.dataset_id) == set(pilot.PILOT_IDS)
    assert len(inputs.roles.meta_only_feature_columns) == 207
    assert len(inputs.roles.landmarker_feature_columns) == 10
    assert len(inputs.roles.meta_plus_landmarker_feature_columns) == 217
    assert inputs.rita[[column for pair in LANDMARKERS.values() for column in pair]].notna().all().all()
    assert len(LANDMARKERS) == 5


@pytest.mark.parametrize("artifact", [pilot.RITA_CSV, pilot.RITA_PARQUET])
def test_bad_export_checksum_fails(tmp_path, artifact):
    directory = tmp_path / pilot.RITA_DIR
    copytree(ROOT / pilot.RITA_DIR, directory)
    with (directory / artifact).open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="Rita export checksum mismatch"):
        pilot.read_rita_export(tmp_path)


def test_exact_csv_parquet_equivalence_even_with_updated_csv_sha(tmp_path):
    directory = tmp_path / pilot.RITA_DIR
    copytree(ROOT / pilot.RITA_DIR, directory)
    csv = directory / pilot.RITA_CSV
    table = pd.read_csv(csv, float_precision="round_trip")
    table.iloc[0, 4] = 99.0
    table.to_csv(csv, index=False, na_rep="NaN")
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["outputs"][pilot.RITA_CSV] = file_sha256(csv)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="CSV and Parquet differ"):
        pilot.read_rita_export(tmp_path)


def test_valid_candidate_artifact_has_40_successful_rows(inputs):
    assert len(inputs.scores) == 40
    assert pilot.validate_diana(inputs.scores, inputs.protocol, inputs.summary, inputs.shared) == inputs.protocol["protocol_id"]


@pytest.mark.parametrize("case,error", [
    ("duplicate", "duplicate logical"), ("missing variant", "exactly the four"),
    ("unexpected variant", "exactly the four"), ("failed", "40 successful"), ("null status", "40 successful"),
    ("mismatch dataset", "unexpected keys"), ("wrong seed", "unexpected keys"), ("unknown source", "dataset_source"),
    ("bad protocol", "protocol or summary"), ("bad fingerprint", "shared split or protocol"),
    ("bad model", "exact candidate identity"), ("NaN", "invalid numeric"), ("changed BA", "test scores disagree"),
    ("wrong test shape", "candidate sizes mismatch"),
])
def test_candidate_contract_rejects_bad_input(inputs, case, error):
    scores, protocol = inputs.scores.copy(), inputs.protocol.copy()
    if case == "missing variant":
        scores = scores.iloc[1:]
    elif case == "duplicate":
        scores = pd.concat([scores, scores.iloc[:1]])
    elif case == "bad protocol":
        protocol["protocol_id"] = "changed"
    elif case == "bad fingerprint":
        pv = json.loads(scores.loc[0, "provenance"])
        pv["shared_split"]["fingerprint"] = "changed"
        scores.loc[0, "provenance"] = json.dumps(pv)
    else:
        col, value = {"failed": ("status", "error"), "null status": ("status", None), "wrong seed": ("seed", 41),
                     "mismatch dataset": ("dataset_id", "pmlb_other"), "unknown source": ("dataset_source", "local"),
                     "unexpected variant": ("variant", "other"), "bad model": ("exact_model_id", "old"),
                     "NaN": ("roc_auc", np.nan), "changed BA": ("balanced_accuracy", 0.3),
                     "wrong test shape": ("n_test", 1)}[case]
        scores.loc[0, col] = value
    with pytest.raises(ValueError, match=error):
        pilot.validate_diana(scores, protocol, inputs.summary, inputs.shared)


def test_keyed_join_pivots_all_metrics_and_preserves_original_source_and_nan(inputs):
    joined, cardinality = pilot.join_outcomes(inputs.rita.sample(frac=1, random_state=5),
                                            inputs.scores.sample(frac=1, random_state=7), inputs.roles)
    assert cardinality == {"matched": 10, "unmatched_rita": 0, "unmatched_diana": 0, "duplicate_identities": 0}
    assert joined.shape == (10, 256) and joined.dataset_id.is_unique
    assert joined.dataset_source_rita.eq("local").all()
    assert joined.dataset_source_diana.eq("pmlb").all() and joined.dataset_source_normalized.eq("pmlb").all()
    keyed = joined.set_index("dataset_id")
    for row in inputs.scores.itertuples():
        name = VARIANTS[(row.model_family, row.variant)][0]
        for metric, prefix in OUTCOMES.items():
            assert keyed.loc[row.dataset_id, f"{prefix}_{name}"] == getattr(row, metric)
            assert keyed[f"{prefix}_{name}"].dtype == np.dtype("float64")
    predictors = inputs.manifest["predictor_columns"]
    pd.testing.assert_frame_equal(keyed[predictors], inputs.rita.set_index("dataset_id")[predictors], check_exact=True)


@pytest.mark.parametrize("case,error", [
    ("duplicate", "duplicate join identity"), ("missing feature row", "unmatched Diana"),
    ("missing candidate group", "unmatched Rita"), ("unknown source", "Unknown source mapping"),
])
def test_join_does_not_drop_or_relabel_bad_rows(inputs, case, error):
    rita, candidates = inputs.rita.copy(), inputs.scores.copy()
    if case == "duplicate":
        rita = pd.concat([rita, rita.iloc[:1]])
    elif case == "missing feature row":
        rita = rita.iloc[1:]
    elif case == "missing candidate group":
        candidates = candidates[candidates.dataset_id.ne("pmlb_saheart")]
    else:
        rita.loc[0, "dataset_source"] = "pmlb"
    with pytest.raises(ValueError, match=error):
        pilot.join_outcomes(rita, candidates, inputs.roles)


def test_best_target_exact_ties_gaps_and_mushroom_verified_outcome(inputs):
    frame, _ = pilot.join_outcomes(inputs.rita, inputs.scores, inputs.roles)
    indexed = frame.set_index("dataset_id")
    assert indexed.loc["pmlb_saheart", "best_target"] == "tabicl_v2"
    assert indexed.loc["pmlb_mushroom", "ba_lightgbm_priortune_t_k25"] == 0.5
    assert indexed.loc["pmlb_mushroom", "gap_lightgbm_priortune_t_k25"] == 0.5
    assert indexed.loc["pmlb_mushroom", "best_target_candidates"] == ["tabpfn_v2_6", "tabicl_v2"]
    assert indexed.loc["pmlb_penguins", "best_target_candidates"] == ["lightgbm_baseline", "tabicl_v2"]
    assert indexed.loc[["pmlb_mushroom", "pmlb_penguins"], "best_target"].isna().all()
    assert frame.best_target_is_tie.sum() == 2 and frame.best_target_count.max() == 2
    gaps = frame[[f"gap_{variant}" for variant in VARIANT_IDS]]
    assert gaps.ge(0).all().all() and gaps.eq(0).any(axis=1).all()


def test_feature_frame_uses_only_allowlisted_inputs(inputs):
    frame, _ = pilot.join_outcomes(inputs.rita, inputs.scores, inputs.roles)
    roles = inputs.roles
    plain = set(roles.feature_frame(frame).columns)
    augmented = set(roles.feature_frame(frame, include_landmarkers=True).columns)
    forbidden = set(roles.candidate_outcome_columns + roles.target_columns + roles.diagnostic_columns + roles.identity_columns)
    assert not plain & forbidden and not augmented & forbidden
    assert augmented - plain == set(roles.landmarker_feature_columns)
    assert not plain & set(roles.landmarker_feature_columns)
    contaminated = replace(roles, meta_only_feature_columns=(*roles.meta_only_feature_columns, "best_target"),
                           target_columns=tuple(c for c in roles.target_columns if c != "best_target"))
    with pytest.raises(ValueError, match="Leakage guard"):
        contaminated.validate(contaminated.columns)


def test_feature_health_reports_inf_constant_duplicates_and_unexpected_types(inputs):
    frame, _ = pilot.join_outcomes(inputs.rita, inputs.scores, inputs.roles)
    columns = inputs.roles.meta_only_feature_columns[:5]
    a, b, c, d, e = columns
    frame[a] = np.nan
    frame[b] = 2.0
    frame[c] = frame[b]
    frame[d] = [np.inf, -np.inf, *([np.nan] * 8)]
    frame[e] = "unexpected"
    health = feature_diagnostics(frame, inputs.roles).set_index("column")
    assert health.loc[a, "all_nan"] and health.loc[a, "missing_fraction"] == 1
    assert health.loc[b, "constant"] and health.loc[c, "duplicate_of"] == b
    assert health.loc[d, "inf_count"] == 2 and health.loc[d, "missing_count"] == 8
    assert health.loc[e, "non_numeric"] and len(health) == 217


def test_crlf_only_raw_sha_drift_is_diagnostic(tmp_path, monkeypatch, inputs):
    manifest = json.loads(json.dumps(inputs.manifest))
    entries = read_pilot_manifest(project_root=ROOT)
    shapes = {entry["dataset_id"]: entry for entry in entries["datasets"]}
    raw = b"col,target\r\n1,0\r\n"
    now, then = sha256(raw).hexdigest(), sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
    by_dataset = {entry["dataset_id"]: entry for entry in manifest["datasets"]}
    for name, entry in shapes.items():
        path = Path("data/raw") / name / entry["file"]
        (tmp_path / path).parent.mkdir(parents=True)
        (tmp_path / path).write_bytes(raw)
        by_dataset[name].update(csv_sha256=now, raw_copy_sha256=then)
        manifest["inputs"][path.as_posix()] = then

    def split(name, *args, **kwargs):
        info, sizes = by_dataset[name], shapes[name]["split_sizes"]
        return SimpleNamespace(fingerprint=now, rita_dataset_fingerprint=info["dataset_fingerprint"], dataset_source="pmlb",
            metadata=dict(original_manifest={"protocol_hash": info["protocol_hash"]},
                shared_manifest_sha256=info["manifest_sha256"], row_order="raw_csv_data_rows_zero_based", rita_dataset_source="local"),
            train_indices=range(sizes["train"]), validation_indices=range(sizes["validation"]), test_indices=range(sizes["test"]))

    monkeypatch.setattr(pilot, "read_pilot_manifest", lambda **kw: entries)
    monkeypatch.setattr(pilot, "load_shared_split", split)
    _, report = pilot.verify_shared_contract(tmp_path, manifest)
    assert report["status"] == pilot.STALE_RAW_NOTE and report["impact_on_integration"] == "NONE"
    assert len(report["datasets"]) == 10
    assert all(row["matches_canonical_csv"] for row in report["datasets"])
    assert all(path.read_bytes() == raw for path in (tmp_path / "data/raw").rglob("*.csv"))
    manifest["datasets"][0]["protocol_hash"] = "changed"
    original_split = pilot.load_shared_split
    monkeypatch.setattr(pilot, "load_shared_split", lambda name, *args, **kw: corrupt_split(original_split, inputs, name))
    with pytest.raises(ValueError, match="shared identity mismatch"):
        pilot.verify_shared_contract(tmp_path, manifest)


def corrupt_split(loader, inputs, name):
    value = loader(name)
    value.metadata["original_manifest"]["protocol_hash"] = next(
        row["protocol_hash"] for row in inputs.manifest["datasets"] if row["dataset_id"] == name)
    return value


def test_integration_writes_checked_parquet_and_preserves_inputs(tmp_path, monkeypatch, inputs):
    copytree(ROOT / pilot.RITA_DIR, tmp_path / pilot.RITA_DIR)
    diana = tmp_path / pilot.DIANA_DIR
    diana.mkdir(parents=True)
    inputs.scores.to_parquet(diana / "candidate_results.parquet", index=False)
    (diana / "protocol.json").write_text(json.dumps(inputs.protocol))
    (diana / "run_summary.json").write_text(json.dumps(inputs.summary))
    original = {path: file_sha256(path) for base in (tmp_path / pilot.RITA_DIR, diana) for path in base.iterdir()}
    monkeypatch.setattr(pilot, "verify_shared_contract", lambda *_: (inputs.shared,
                        {"status": pilot.STALE_RAW_NOTE, "impact_on_integration": "NONE", "datasets": []}))
    manifest = pilot.build_integration(tmp_path)
    assert {path: file_sha256(path) for path in original} == original
    output = tmp_path / pilot.OUTPUT_DIR
    row = pd.read_parquet(output / pilot.INTEGRATED)
    assert row.shape == (10, 256) and row.best_target.isna().sum() == 2
    roles = load_column_roles(output / "manifest.json")
    assert roles.feature_frame(row, include_landmarkers=True).isna().sum().sum() == 41
    assert manifest["rita"]["artifact_integrity"] == "PASS" and manifest["leakage_guard"] == "PASS"
    assert manifest["rita"]["raw_sha_provenance"]["status"] == pilot.STALE_RAW_NOTE
    assert all(file_sha256(output / name) == digest for name, digest in manifest["outputs"].items())
