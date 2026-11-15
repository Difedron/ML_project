"""Validate and join the immutable stage-7 pilot outputs and Rita's feature export."""
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from meta_selector.artifacts import atomic_json, config_hash
from meta_selector.datasets.shared_pilot import (JOIN_KEYS, contained_path, file_sha256, load_shared_split,
                                               read_pilot_manifest)
from meta_selector.paths import DEFAULT_PROJECT_ROOT
from .roles import (ColumnRoles, DIAGNOSTICS, LANDMARKERS, LANDMARKER_MODELS, OUTCOMES, VARIANTS, VARIANT_IDS,
                    feature_diagnostics, load_column_roles)


RITA_DIR = Path("results/exports/pmlb_pilot_meta_features")
DIANA_DIR = Path("data/derived/candidate_full_pilot")
OUTPUT_DIR = Path("data/derived/meta_integration")
RITA_PARQUET = "pmlb_pilot_meta_features_217.parquet"
RITA_CSV = "pmlb_pilot_meta_features_217.csv"
INTEGRATED = "pmlb_pilot_integrated.parquet"
KEYS = list(JOIN_KEYS)
EXPORT_IDENTIFIERS = [*KEYS, "dataset_source"]
SOURCE_MAPPING = {"rita": "local", "diana": "pmlb"}
STALE_RAW_NOTE = "KNOWN STALE METADATA due to CRLF restoration"
PILOT_IDS = ("pmlb_saheart", "pmlb_tic_tac_toe", "pmlb_kr_vs_kp", "pmlb_mushroom",
             "pmlb_hill_valley_without_noise", "pmlb_twonorm", "pmlb_penguins", "pmlb_car_evaluation",
             "pmlb_vehicle", "pmlb_waveform_21")
EXPECTED_KEYS = {(dataset, "holdout_0", 42) for dataset in PILOT_IDS}
VARIANT_KEYS = [*KEYS, "model_family", "variant"]
EXPECTED_VARIANTS = {(*key, *variant) for key in EXPECTED_KEYS for variant in VARIANTS}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _records(frame, columns):
    return set(frame.loc[:, columns].itertuples(index=False, name=None))


def _validate_identity(frame, source, label, *, candidates=False):
    needed = [*KEYS, "dataset_source"]
    require(frame.columns.is_unique and set(needed) <= set(frame), f"{label}: invalid identity schema")
    require(not frame[needed].isna().any().any() and pd.api.types.is_integer_dtype(frame.seed),
            f"{label}: null or non-integer identity")
    require(frame.dataset_source.eq(source).all(), f"{label}: unsupported dataset_source")
    if not candidates:
        require(not frame.duplicated(KEYS).any(), f"{label}: duplicate join identity")
    keys = _records(frame, KEYS)
    require(keys == EXPECTED_KEYS,
            f"{label}: missing keys {sorted(EXPECTED_KEYS - keys)}, unexpected keys {sorted(keys - EXPECTED_KEYS)}")


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rita_export(root):
    directory = root / RITA_DIR
    manifest = _json(directory / "manifest.json")
    require(manifest.get("schema_version") == 1 and manifest.get("rows") == 10
            and manifest.get("predictor_count") == 217 and manifest.get("identifier_columns") == EXPORT_IDENTIFIERS
            and manifest.get("join_keys") == KEYS and manifest.get("source_label_mapping") == SOURCE_MAPPING,
            "Rita export manifest has an unsupported schema or join contract")
    require(manifest.get("models") == list(LANDMARKER_MODELS.values()) and manifest.get("metric") == "balanced_accuracy"
            and (manifest.get("inner_folds"), manifest.get("std_ddof")) == (3, 0), "Rita landmarker contract mismatch")
    require(all(manifest.get("checks", {}).get(name) is True for name in
                ("training_only_provenance", "source_values_exact", "csv_parquet_roundtrip_exact", "diana_contract_matches")),
            "Rita manifest lacks training or export integrity checks")
    payloads = {name: (directory / name).read_bytes() for name in (RITA_PARQUET, RITA_CSV, "README.md")}
    for name, data in payloads.items():
        require(sha256(data).hexdigest() == manifest.get("outputs", {}).get(name), f"Rita export checksum mismatch: {name}")
    predictors = manifest["predictor_columns"]
    columns = [*EXPORT_IDENTIFIERS, *predictors]
    readme_columns = [line[3:-1] for line in payloads["README.md"].decode("utf-8").splitlines()
                      if line.startswith("- `") and line.endswith("`")]
    require(len(columns) == len(set(columns)) == 221 and all(isinstance(col, str) for col in columns)
            and readme_columns == columns, "Rita README and manifest column schemas differ")
    roles = ColumnRoles.from_predictors(predictors)
    counts = Counter(name.split("::")[1] for name in predictors)
    require(counts == {"paper_features": 202, "controls": 5, "custom_landmarking": 10}, "Rita predictor role counts differ")
    parquet = pd.read_parquet(BytesIO(payloads[RITA_PARQUET]))
    require(parquet.shape == (10, 221) and parquet.columns.tolist() == columns, "Rita Parquet schema mismatch")
    require(all(parquet[name].dtype == np.dtype("float64") for name in predictors), "Rita predictors must be numeric float64")
    csv = pd.read_csv(BytesIO(payloads[RITA_CSV]), encoding="utf-8", float_precision="round_trip",
                      dtype=parquet.dtypes.to_dict())
    try:
        pd.testing.assert_frame_equal(parquet, csv, check_exact=True)
    except AssertionError as error:
        raise ValueError("Rita CSV and Parquet differ in schema, values or missing positions") from error
    _validate_identity(parquet, SOURCE_MAPPING["rita"], "Rita export")
    pred = parquet[predictors]
    require(int(pred.isna().sum().sum()) == manifest["nan_count"]
            and int(np.isinf(pred.to_numpy()).sum()) == manifest["inf_count"], "Rita missing-value counts differ from manifest")
    require(np.isfinite(parquet[list(roles.landmarker_feature_columns)].to_numpy()).all(),
            "Rita export lacks a finite score for one of the five landmarkers")
    dataset_records = pd.DataFrame(manifest["datasets"])
    _validate_identity(dataset_records, SOURCE_MAPPING["rita"], "Rita manifest")
    missing = parquet.set_index(KEYS)[predictors].isna().sum(axis=1)
    require(dataset_records.set_index(KEYS).nan_count.sort_index().equals(missing.sort_index()),
            "Rita per-dataset missing counts differ from manifest")
    return parquet, roles, manifest


def verify_shared_contract(root, rita_manifest):
    catalog = read_pilot_manifest(project_root=root)
    require(catalog.get("source_label_mapping") == {"rita": "local", "canonical": "pmlb"}, "Unknown canonical source mapping")
    entries = {entry["dataset_id"]: entry for entry in catalog["datasets"]}
    require(set(entries) == set(PILOT_IDS), "Canonical catalog does not contain the ten expected datasets")
    rita = {entry["dataset_id"]: entry for entry in rita_manifest["datasets"]}
    shared, raw_checks = {}, []
    for dataset_id in PILOT_IDS:
        entry, reference = entries[dataset_id], rita[dataset_id]
        split = load_shared_split(dataset_id, "holdout_0", seed=42, project_root=root)
        original = split.metadata["original_manifest"]
        expected = dict(csv_sha256=split.fingerprint, dataset_fingerprint=split.rita_dataset_fingerprint,
                        protocol_hash=original["protocol_hash"], manifest_sha256=split.metadata["shared_manifest_sha256"],
                        n_rows=entry["n_rows"], train_size=len(split.train_indices))
        require(all(reference.get(key) == value for key, value in expected.items())
                and split.dataset_source == "pmlb" and split.metadata["rita_dataset_source"] == "local",
                f"Rita/canonical shared identity mismatch: {dataset_id}")
        shared[dataset_id] = {**expected, "row_order": split.metadata["row_order"], "n_features": entry["n_features"],
                              "n_classes": entry["n_classes"], "validation_size": len(split.validation_indices),
                              "test_size": len(split.test_indices)}
        raw_path = contained_path(root, f"data/raw/{dataset_id}/{entry['file']}")
        saved_sha = reference["raw_copy_sha256"]
        require(rita_manifest["inputs"].get(raw_path.relative_to(root).as_posix()) == saved_sha,
                f"Rita manifest raw SHA entries disagree internally: {dataset_id}")
        raw = raw_path.read_bytes() if raw_path.is_file() else None
        current_sha = sha256(raw).hexdigest() if raw is not None else None
        lf_sha = sha256(raw.replace(b"\r\n", b"\n")).hexdigest() if raw is not None else None
        if current_sha == saved_sha:
            status = "MATCH"
        elif current_sha == split.fingerprint and lf_sha == saved_sha:
            status = STALE_RAW_NOTE
        else:
            status = "LOCAL RAW PROVENANCE UNVERIFIED"
        raw_checks.append(dict(dataset_id=dataset_id, path=raw_path.relative_to(root).as_posix(), status=status,
                               export_time_sha256=saved_sha, current_sha256=current_sha, lf_sha256=lf_sha,
                               matches_canonical_csv=current_sha == split.fingerprint))
    statuses = {record["status"] for record in raw_checks}
    status = ("LOCAL RAW PROVENANCE UNVERIFIED" if "LOCAL RAW PROVENANCE UNVERIFIED" in statuses
              else STALE_RAW_NOTE if STALE_RAW_NOTE in statuses else "MATCH")
    return shared, dict(status=status, impact_on_integration="NONE", datasets=raw_checks)


def validate_diana(frame, protocol, summary, shared):
    _validate_identity(frame, SOURCE_MAPPING["diana"], "Diana results", candidates=True)
    required = {*VARIANT_KEYS, "schema_version", "model_id", "exact_model_id", "status", "provenance", "device",
                *OUTCOMES, *DIAGNOSTICS, "n_train", "n_validation", "n_test", "n_features", "n_classes"}
    require(required <= set(frame), "Diana CandidateResult schema is incomplete")
    require(not frame.duplicated(VARIANT_KEYS).any(), "Diana has duplicate logical candidate rows")
    require(len(frame) == 40 and _records(frame, VARIANT_KEYS) == EXPECTED_VARIANTS,
            "Diana requires 40 rows and exactly the four expected variants for each dataset")
    require(frame.status.fillna("").eq("ok").all() and frame.schema_version.fillna(-1).eq(2).all(),
            "Diana requires 40 successful CandidateResult v2 rows")
    numeric = [*OUTCOMES, "validation_score", "total_time_sec"]
    require(all(pd.api.types.is_numeric_dtype(frame[col]) for col in numeric)
            and np.isfinite(frame[numeric].to_numpy(dtype=float, na_value=np.nan)).all(), "Diana has invalid numeric metrics")
    require(frame.balanced_accuracy.between(0, 1).all() and frame.roc_auc.between(0, 1).all()
            and (frame[["log_loss", "total_time_sec", "validation_score"]] >= 0).all().all(), "Diana metrics out of range")
    protocol_id = protocol.get("protocol_id")
    require(protocol.get("schema_version") == 1 and protocol_id == config_hash({k: v for k, v in protocol.items() if k != "protocol_id"})
            and summary.get("protocol_id") == protocol_id and summary.get("status") == "PASS"
            and summary.get("successful_candidate_runs") == 40 and summary.get("failed_candidate_runs") == 0,
            "Canonical stage-7 protocol or summary mismatch")
    expected = {tuple(row[key] for key in VARIANT_KEYS): row for row in protocol["expected"]}
    require(len(protocol["expected"]) == 40 and set(expected) == EXPECTED_VARIANTS,
            "Stage-7 protocol does not cover exactly the 40 candidate identities")
    for row in frame.to_dict("records"):
        key = tuple(row[key] for key in VARIANT_KEYS)
        dataset_id = row["dataset_id"]
        spec, planned = shared[dataset_id], expected[key]
        variant_id, model_id = VARIANTS[key[-2:]]
        pv = json.loads(row["provenance"])
        shared_identity = dict(fingerprint=spec["csv_sha256"], rita_dataset_fingerprint=spec["dataset_fingerprint"],
                               shared_manifest_sha256=spec["manifest_sha256"], row_order=spec["row_order"])
        require(pv.get("join_key") == list(key[:3]) and pv.get("shared_split") == planned.get("shared_split") == shared_identity
                and pv.get("full_pilot", {}).get("protocol_id") == protocol_id,
                f"Diana shared split or protocol mismatch: {dataset_id}/{variant_id}")
        require(row["model_id"] == planned.get("model_id") == model_id
                and isinstance(row["exact_model_id"], str) and row["exact_model_id"].startswith(f"pilot::{model_id}::")
                and row["exact_model_id"] == planned.get("exact_model_id") and row["device"] == planned.get("device"),
                f"Diana exact candidate identity mismatch: {dataset_id}/{variant_id}")
        require([row[key] for key in ("n_train", "n_validation", "n_test", "n_features", "n_classes")]
                == [spec[key] for key in ("train_size", "validation_size", "test_size", "n_features", "n_classes")],
                f"Diana candidate sizes mismatch: {dataset_id}/{variant_id}")
        require(row["validation_metric"] == ("roc_auc" if spec["n_classes"] == 2 else "log_loss"),
                f"Diana validation metric mismatch: {dataset_id}/{variant_id}")
        require(all(pv.get("test_scores", {}).get(metric) == row[metric] for metric in OUTCOMES),
                f"Diana saved test scores disagree with provenance: {dataset_id}/{variant_id}")
    mushroom = frame[(frame.dataset_id == "pmlb_mushroom") & (frame.variant == "priortune_t_k25")].iloc[0]
    require(mushroom.balanced_accuracy == 0.5 and json.loads(mushroom.provenance)["priortune"]["selected_config_id"] == 95,
            "Mushroom verified PriorTune outcome must stay at config 95 and test BA 0.5")
    return protocol_id


def join_outcomes(features, candidates, roles):
    f_keys, c_keys = _records(features, KEYS), _records(candidates, KEYS)
    require(not features.duplicated(KEYS).any(), "Rita has duplicate join identity")
    require(not candidates.duplicated(VARIANT_KEYS).any(), "Diana has duplicate logical candidate rows")
    unmatched_rita, unmatched_diana = sorted(f_keys - c_keys), sorted(c_keys - f_keys)
    require(not unmatched_rita and not unmatched_diana,
            f"Join mismatch: unmatched Rita={unmatched_rita}, unmatched Diana={unmatched_diana}")
    tagged = candidates.assign(variant_id=[VARIANTS[pair][0] for pair in zip(candidates.model_family, candidates.variant)])
    measures = {**OUTCOMES, **{name: name for name in DIAGNOSTICS}}
    wide = pd.concat([tagged.pivot(index=KEYS, columns="variant_id", values=metric).add_prefix(f"{prefix}_")
                      for metric, prefix in measures.items()], axis=1)
    diana_source = candidates[[*KEYS, "dataset_source"]].drop_duplicates().set_index(KEYS)
    wide = wide.join(diana_source.rename(columns={"dataset_source": "dataset_source_diana"}), validate="one_to_one")
    joined = features.rename(columns={"dataset_source": "dataset_source_rita"}).merge(
        wide.reset_index(), on=KEYS, how="outer", validate="one_to_one", indicator=True)
    require(len(joined) == 10 and joined._merge.eq("both").all(), "Expected ten matched datasets")
    require(joined.dataset_source_rita.eq(SOURCE_MAPPING["rita"]).all()
            and joined.dataset_source_diana.eq(SOURCE_MAPPING["diana"]).all(), "Unknown source mapping during join")
    joined["dataset_source_normalized"] = joined.dataset_source_rita.map({"local": "pmlb"})
    ba_columns = [f"ba_{name}" for name in VARIANT_IDS]
    joined["best_balanced_accuracy"] = joined[ba_columns].max(axis=1)
    maxima = joined[ba_columns].eq(joined.best_balanced_accuracy, axis=0)
    joined["best_target_candidates"] = [[variant for variant, matched in zip(VARIANT_IDS, flags) if matched]
                                         for flags in maxima.itertuples(index=False, name=None)]
    joined["best_target_count"] = maxima.sum(axis=1)
    joined["best_target_is_tie"] = joined.best_target_count.gt(1)
    joined["best_target"] = joined.best_target_candidates.map(lambda values: values[0] if len(values) == 1 else None)
    for variant in VARIANT_IDS:
        joined[f"gap_{variant}"] = joined.best_balanced_accuracy - joined[f"ba_{variant}"]
    joined = joined.loc[:, list(roles.columns)].sort_values(KEYS).reset_index(drop=True)
    roles.validate(joined.columns)
    return joined, dict(matched=len(joined), unmatched_rita=0, unmatched_diana=0, duplicate_identities=0)


def build_integration(project_root=None):
    root = Path(project_root or DEFAULT_PROJECT_ROOT).resolve()
    features, roles, rita_manifest = read_rita_export(root)
    shared, raw = verify_shared_contract(root, rita_manifest)
    source = root / DIANA_DIR
    candidate_path = source / "candidate_results.parquet"
    payload = candidate_path.read_bytes()
    scores = pd.read_parquet(BytesIO(payload))
    protocol_id = validate_diana(scores, _json(source / "protocol.json"), _json(source / "run_summary.json"), shared)
    frame, cardinality = join_outcomes(features, scores, roles)
    health = feature_diagnostics(frame, roles)
    targets = frame.loc[:, ["dataset_id", "best_target", "best_target_is_tie", "best_target_candidates"]]
    health_summary = dict(missing_cells=int(health.missing_count.sum()), missing_columns=int(health.missing_count.gt(0).sum()),
                          inf_cells=int(health.inf_count.sum()), constant_columns=health.loc[health.constant, "column"].tolist(),
                          all_nan_columns=health.loc[health.all_nan, "column"].tolist(),
                          duplicate_by_value=dict(health.loc[health.duplicate_of.notna(), ["column", "duplicate_of"]].itertuples(index=False, name=None)),
                          non_numeric_columns=health.loc[health.non_numeric, "column"].tolist())
    manifest = dict(schema_version=1, rows=len(frame), columns=len(frame.columns),
        rita=dict(commit="7ff999d", export_commit="7d5943c", primary_artifact=(RITA_DIR / RITA_PARQUET).as_posix(),
                  manifest_path=(RITA_DIR / "manifest.json").as_posix(), manifest_sha256=file_sha256(root / RITA_DIR / "manifest.json"),
                  export_checksums=rita_manifest["outputs"], artifact_integrity="PASS", rows=len(features),
                  columns=len(features.columns), raw_sha_provenance=raw),
        diana=dict(stage7_commit="6e8038b", artifact=(DIANA_DIR / candidate_path.name).as_posix(),
                   artifact_sha256=sha256(payload).hexdigest(), candidate_protocol_id=protocol_id, rows=len(scores),
                   protocol_path=(DIANA_DIR / "protocol.json").as_posix(), protocol_sha256=file_sha256(source / "protocol.json")),
        join_keys=KEYS, join_cardinality=cardinality, source_label_mapping={**SOURCE_MAPPING, "normalized": "pmlb"},
        shared_datasets=shared, column_roles=roles.as_dict(), landmarker_mapping={key: list(val) for key, val in LANDMARKERS.items()},
        n_meta_only_features=len(roles.meta_only_feature_columns), n_landmarker_feature_columns=len(roles.landmarker_feature_columns),
        n_meta_plus_landmarker_features=len(roles.meta_plus_landmarker_feature_columns),
        tie_policy="exact equality of test balanced accuracy, null best_target on tie", selection_metric="test_balanced_accuracy",
        ties=targets[targets.best_target_is_tie].drop(columns=["best_target"]).to_dict("records"),
        unique_winner_distribution=frame.best_target.dropna().value_counts().to_dict(),
        missing_value_summary=health_summary, leakage_guard="PASS")
    output = contained_path(root, OUTPUT_DIR.as_posix())
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output) as directory:
        temporary = Path(directory)
        frame.to_parquet(temporary / INTEGRATED, index=False)
        health.to_parquet(temporary / "feature_diagnostics.parquet", index=False)
        atomic_json(temporary / "meta_only_features.json", list(roles.meta_only_feature_columns))
        atomic_json(temporary / "meta_plus_landmarkers_features.json", list(roles.meta_plus_landmarker_feature_columns))
        for path in sorted(temporary.iterdir()):
            manifest.setdefault("outputs", {})[path.name] = file_sha256(path)
        atomic_json(temporary / "manifest.json", manifest)
        saved = pd.read_parquet(temporary / INTEGRATED)
        # Arrow reads list elements as arrays. Compare the lists in order as sequences.
        pd.testing.assert_frame_equal(frame.drop(columns="best_target_candidates"),
                                      saved.drop(columns="best_target_candidates"), check_exact=True)
        require(saved.best_target_candidates.map(tuple).equals(frame.best_target_candidates.map(tuple)),
                "Parquet round trip changed the tied target lists")
        for include_landmarkers in (False, True):
            load_column_roles(temporary / "manifest.json").feature_frame(saved, include_landmarkers=include_landmarkers)
        for name in (*manifest["outputs"], "manifest.json"):
            (temporary / name).replace(output / name)
    return manifest
