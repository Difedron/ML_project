"""Verify saved pilot outputs and runtime audit evidence; never fit or split data."""
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from meta_selector.artifacts import _atomic, atomic_json, config_hash, upsert_table
from meta_selector.datasets.pmlb import sha256_file
from run_pmlb_features_pilot import LOGS, ROOT, index_hash, initialize


def ordered(frame, keys):
    assert not frame.duplicated(keys).any()
    return frame.sort_values(keys).reset_index(drop=True)


def main():
    specs, config, data, protected = initialize()
    for spec in specs:
        assert sha256_file(ROOT / "data/PMLB" / f"{spec['original_name']}.csv") == spec["sha256"]
    ids = [spec["dataset_id"] for spec in specs]
    feature_keys = ["dataset_source", "dataset_id", "split_id", "source", "name", "config_id"]
    custom_keys = ["dataset_source", "dataset_id", "split_id", "model_id", "metric", "config_id"]
    features = ordered(pd.read_parquet(ROOT / "data/derived/meta_features/meta_features.parquet"), feature_keys)
    custom = ordered(pd.read_parquet(ROOT / "data/derived/landmarkers/custom_landmarkers.parquet"), custom_keys)
    assert set(features.dataset_id) == set(custom.dataset_id) == set(ids)
    assert len(features) == 2070 and len(custom) == 50
    assert features.dataset_source.eq("local").all() and custom.dataset_source.eq("local").all()
    assert features.split_id.eq("holdout_0").all() and custom.split_id.eq("holdout_0").all()
    assert set(features.source) == {"controls", "paper_features"}
    assert not (ROOT / "data/derived/landmarkers/pymfe_landmarkers.parquet").exists()
    assert features.groupby("source").config_id.nunique().eq(1).all()
    assert custom.config_id.eq(config_hash(config["custom_landmarkers"])).all()
    assert features.loc[features.source.eq("paper_features"), "config_id"].eq(config_hash(config["paper_features"])).all()
    assert custom.status.eq("ok").all() and custom.n_folds.eq(3).all()
    assert custom.metric.eq("balanced_accuracy").all()
    assert custom.score_mean.between(0, 1).all() and custom.score_std.ge(0).all()
    assert np.isfinite(custom[["score_mean", "score_std", "fit_time_mean", "fit_time_std"]]).all().all()
    assert custom[["fit_time_mean", "fit_time_std"]].ge(0).all().all()
    assert custom.error_message.eq("").all()
    assert features.status.isin(["ok", "undefined"]).all()
    assert features.status.eq("undefined").equals(~np.isfinite(features.value))

    first_features = ordered(pd.read_parquet(LOGS / "baseline/features.parquet"), feature_keys)
    repeat_features = ordered(pd.read_parquet(LOGS / "repeat/features.parquet"), feature_keys)
    pd.testing.assert_frame_equal(first_features, repeat_features, check_exact=True)
    pd.testing.assert_frame_equal(features, repeat_features, check_exact=True)
    first_custom = ordered(pd.read_parquet(LOGS / "corrected/landmarkers.parquet"), custom_keys)
    repeat_custom = ordered(pd.read_parquet(LOGS / "repeat/landmarkers.parquet"), custom_keys)
    score_columns = [c for c in custom if c not in {"fit_time_mean", "fit_time_std"}]
    pd.testing.assert_frame_equal(first_custom[score_columns], repeat_custom[score_columns], check_exact=True)
    pd.testing.assert_frame_equal(custom, repeat_custom, check_exact=True)
    initial_custom = pd.read_parquet(LOGS / "baseline/landmarkers.parquet")
    assert len(initial_custom) == 50 and initial_custom.status.eq("error").sum() == 10
    assert initial_custom.loc[initial_custom.status.eq("error"), "model_id"].eq("lda").all()

    audits = {}
    for run, phase in (("baseline", "features"), ("corrected", "landmarkers"), ("repeat", "features"), ("repeat", "landmarkers")):
        audit = json.loads((LOGS / run / f"{phase}_audit.json").read_text())
        assert audit["exit_code"] == 0 and audit["protected_inputs_unchanged"]
        events = audit["events"]
        assert sum(e["kind"] == "outer_input" for e in events) == (20 if phase == "features" else 10)
        fits = [e for e in events if e["kind"] == "inner_fit"]
        scores = [e for e in events if e["kind"] == "inner_score"]
        assert len(fits) == len(scores) == (150 if phase == "landmarkers" else 0)
        for dataset_id in ids:
            dataset, manifest = data[dataset_id]
            train = manifest["indices"]["train"]
            X, y = dataset.X.iloc[train], dataset.y.iloc[train]
            for event in (e for e in events if e["dataset_id"] == dataset_id and e["kind"] == "outer_input"):
                assert event["rows"] == len(X) and event["index_hash"] == index_hash(X.index)
                assert event["categorical_columns"] == list(dataset.metadata.categorical_columns)
            if phase == "features":
                prepared = [e for e in events if e["dataset_id"] == dataset_id and e["kind"] == "pymfe_prepared"]
                assert len(prepared) == 1 and prepared[0]["rows"] == len(X)
            else:
                cv = list(StratifiedKFold(3, shuffle=True, random_state=42).split(X, y))
                observed = [e for e in fits if e["dataset_id"] == dataset_id]
                assert len(observed) == 15
                for event in observed:
                    inner, held = cv[event["fold"]]
                    assert event["status"] == "ok" and event["rows"] == len(inner)
                    assert event["index_hash"] == index_hash(X.iloc[inner].index)
                    assert event["preprocessing"] and all(p["rows"] == len(inner) for p in event["preprocessing"])
                    assert not set(X.iloc[inner].index) & set(X.iloc[held].index)
                observed_scores = [e for e in scores if e["dataset_id"] == dataset_id]
                assert len(observed_scores) == 15
                for event in observed_scores:
                    _, held = cv[event["fold"]]
                    assert event["rows"] == len(held) and event["index_hash"] == index_hash(X.iloc[held].index)
        audits[f"{run}/{phase}"] = audit

    health, warnings_rows, identities = [], [], []
    for dataset_id in ids:
        dataset, manifest = data[dataset_id]
        table = features[features.dataset_id.eq(dataset_id)]
        assert table.source.value_counts().to_dict() == {"paper_features": 202, "controls": 5}
        controls = table[table.source.eq("controls")].set_index("name").value
        n = len(manifest["indices"]["train"])
        assert controls["log_n"] == np.log(n)
        assert controls["cat_fraction"] == len(dataset.metadata.categorical_columns) / dataset.X.shape[1]
        assert controls["feature_missing_fraction"] == 0
        assert table.loc[table.name.eq("nr_inst"), "value"].item() == n
        undefined = table.loc[table.status.eq("undefined"), "name"].tolist()
        contexts = table.drop_duplicates("source")
        feature_warnings = []
        for row in contexts.itertuples():
            context = json.loads(row.provenance)
            assert context["input_role"] == "outer_training_only"
            assert context["train_indices"] == manifest["indices"]["train"]
            assert context["dataset_fingerprint"] == manifest["dataset_fingerprint"]
            if row.source == "paper_features":
                assert context["manifest"] == config["paper_features"]
            for warning in context.get("warnings", []):
                warnings_rows.append(dict(dataset_id=dataset_id, source=row.source, model_id="", warning=warning))
                feature_warnings.append(warning)
        for row in custom[custom.dataset_id.eq(dataset_id)].itertuples():
            context = json.loads(row.provenance)
            assert context["split"]["train_indices"] == manifest["indices"]["train"]
            assert context["config"] == config["custom_landmarkers"]
            assert context["input_role"] == "outer_training_only" and context["cv_std_ddof"] == 0
            for warning in context.get("warnings", []):
                warnings_rows.append(dict(dataset_id=dataset_id, source=row.source, model_id=row.model_id, warning=warning))
        health.append(dict(dataset_id=dataset_id, pymfe_features=202, controls=5, total_features=207,
                           finite=int(np.isfinite(table.value).sum()), nan=int(table.value.isna().sum()),
                           inf=int(np.isinf(table.value).sum()), undefined=len(undefined),
                           warnings=len(feature_warnings), undefined_names=";".join(undefined)))
        identities.append(dict(dataset_source="local", dataset_id=dataset_id, original_name=dataset.metadata.name,
                               split_id="holdout_0", csv_sha256=dataset.metadata.file.sha256,
                               dataset_fingerprint=manifest["dataset_fingerprint"], protocol_hash=manifest["protocol_hash"]))

    output = ROOT / "results/tables/pmlb_pilot_features"
    exports = dict(landmarker_scores=custom.drop(columns="provenance"),
                   feature_health=pd.DataFrame(health),
                   warnings=pd.DataFrame(warnings_rows, columns=["dataset_id", "source", "model_id", "warning"]),
                   dataset_identity=pd.DataFrame(identities),
                   undefined_features=features.loc[features.status.eq("undefined"), feature_keys + ["value", "status"]])
    compute_times = []
    for dataset_id in ids:
        times = {entry["namespace"]: entry["seconds"] for phase in ("features", "landmarkers")
                 for entry in audits[f"repeat/{phase}"]["namespace_timings"] if entry["dataset_id"] == dataset_id}
        compute_times.append(dict(dataset_id=dataset_id, controls_seconds=times["controls"],
                                  pymfe_seconds=times["paper_features"], custom_cv_seconds=times["custom"],
                                  total_custom_fit_seconds=float(custom.loc[custom.dataset_id.eq(dataset_id), "fit_time_mean"].sum() * 3)))
    exports["dataset_compute_times"] = pd.DataFrame(compute_times)
    for name, table in exports.items():
        _atomic(output / f"{name}.csv", lambda temporary, t=table: t.to_csv(temporary, index=False, float_format="%.17g"))
    feature_manifest = features[["source", "name", "group", "extractor_version", "config_id"]].drop_duplicates()
    upsert_table(output / "feature_manifest.parquet", feature_manifest, ["source", "name", "config_id"])
    history = []
    for path in sorted(LOGS.glob("*/*_audit.json")):
        record = json.loads(path.read_text())
        history.append(dict(run_id=record["run_id"], phase=record["phase"], exit_code=record["exit_code"], seconds=record["seconds"]))
    summary = dict(datasets=10, pymfe_features_per_dataset=202, controls_per_dataset=5,
                   total_feature_rows=len(features), finite_feature_rows=int(np.isfinite(features.value).sum()),
                   nan=int(features.value.isna().sum()), inf=int(np.isinf(features.value).sum()),
                   undefined=int(features.status.eq("undefined").sum()), final_custom_ok=50, final_custom_error=0,
                   feature_repeat_exact=True, custom_scores_and_provenance_repeat_exact=True,
                   original_csv_metadata_config_and_manifests_unchanged=True,
                   runtime_inner_fits_verified_per_complete_pass=150,
                   feature_warning_dataset_message_pairs=len(warnings_rows),
                   distinct_warning_messages=len({r["warning"] for r in warnings_rows}),
                   initial_auditor_failures=dict(custom_ok=40, custom_error=10, model_id="lda"),
                   history=history, feature_config_ids=features.groupby("source").config_id.first().to_dict(),
                   custom_config_id=custom.config_id.iloc[0],
                   feature_health=health,
                   final_total_fit_seconds=float(custom.fit_time_mean.sum() * 3),
                   custom_timings_not_expected_to_repeat=True,
                   protected_files=len(protected),
                   implementation_sha256={name: sha256_file(ROOT / name) for name in (
                       "scripts/_research_cli.py", "src/meta_selector/batch.py",
                       "src/meta_selector/meta_features/extraction.py", "src/meta_selector/landmarkers/custom.py",
                       "src/meta_selector/landmarkers/preprocessing.py", "scripts/run_pmlb_features_pilot.py")})
    atomic_json(LOGS / "verification.json", summary)
    frozen = subprocess.run([sys.executable, "-B", "-m", "pip", "freeze", "--exclude-editable"],
                            capture_output=True, text=True, check=True).stdout
    _atomic(LOGS / "requirements.freeze.txt", lambda temporary: temporary.write_text(frozen, encoding="utf-8"))
    report_path = ROOT / "docs/PMLB_FEATURES_REPORT.md"
    if report_path.exists():
        originals = {spec["original_name"]: spec["dataset_id"] for spec in specs}
        section, checked_cells = None, 0
        for line in report_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("## "):
                section = "scores" if line.startswith("## 5.") else "times" if line.startswith("## 6.") else None
            if section and line.startswith("| "):
                cells = [c.strip() for c in line.split("|")[1:-1]]
                if len(cells) == 6 and cells[0] in originals:
                    table = custom[custom.dataset_id.eq(originals[cells[0]])].set_index("model_id")
                    mean, std, scale, digits = (("score_mean", "score_std", 1, 4) if section == "scores"
                                                else ("fit_time_mean", "fit_time_std", 1000, 2))
                    for model, cell in zip(config["custom_landmarkers"]["models"], cells[1:]):
                        observed = tuple(float(v.strip()) for v in cell.split("±"))
                        expected = tuple(float(f"{table.loc[model, field] * scale:.{digits}f}") for field in (mean, std))
                        assert observed == expected, f"Report/table mismatch: {cells[0]}, {model}"
                        checked_cells += 1
        assert checked_cells == 100, "Report must contain all 50 scores and 50 fit times"
        print("Verified all 100 report score/timing cells against saved results.")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"feature_health", "implementation_sha256"}}, indent=2))
    models = config["custom_landmarkers"]["models"]
    for label, mean_col, std_col, multiplier in (("BALANCED ACCURACY", "score_mean", "score_std", 1),
                                                 ("FIT TIME MS", "fit_time_mean", "fit_time_std", 1000)):
        print(label)
        for dataset_id in ids:
            table = custom[custom.dataset_id.eq(dataset_id)].set_index("model_id")
            cells = [f"{table.loc[m, mean_col] * multiplier:.4f} +/- {table.loc[m, std_col] * multiplier:.4f}" for m in models]
            print("| " + dataset_id + " | " + " | ".join(cells) + " |")
    print("FEATURE HEALTH")
    print(pd.DataFrame(health).drop(columns="undefined_names").to_string(index=False))
    print("WARNINGS", Counter(r["warning"] for r in warnings_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
