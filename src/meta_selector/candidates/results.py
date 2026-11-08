"""Direction-aware bridge from explicit local candidate scores to canonical gap records."""
import json
import csv
from dataclasses import fields
from pathlib import Path
import numpy as np
import pandas as pd

from meta_selector import load_dataset
from meta_selector.artifacts import research_config, research_paths
from meta_selector.datasets.loader import _check_headers
from meta_selector.datasets.pmlb import sha256_file
from meta_selector.datasets.splits import check_manifest
from meta_selector.paths import project_paths, read_yaml_mapping
from meta_selector.performance_gap.families import PMLB_PILOT_MODELS
from meta_selector.tabarena.results_loader import ROW_KEYS
from .runner import CandidateResult


def to_gap_records(scores):
    if scores.empty or scores.status.ne("ok").any():
        raise ValueError("Only successful real candidate results can become gap records")
    if scores.dataset_source.ne("local").any():
        raise ValueError("This bridge expects local candidate results")
    if not scores.validation_metric.eq(scores.test_metric).all():
        raise ValueError("Validation and test metrics must match")
    result = scores.copy()
    result["metric"] = result.test_metric
    maximize = result.metric.isin(["balanced_accuracy", "macro_f1", "roc_auc"])
    if not (maximize | result.metric.eq("log_loss")).all():
        raise ValueError("Declare the score direction for this metric")
    result["test_error"] = np.where(maximize, 1 - result.test_score, result.test_score)
    result["validation_error"] = np.where(maximize, 1 - result.validation_score, result.validation_score)
    if not np.isfinite(result[["test_error", "validation_error"]]).all().all():
        raise ValueError("Missing/non-finite scores cannot be imputed")
    result["problem_type"] = result.provenance.map(lambda value: json.loads(value)["problem_type"])
    result["imputed"] = False
    if result.duplicated(ROW_KEYS).any():
        raise ValueError("Duplicate candidate result keys")
    return result


class CandidateResultsValidationError(ValueError):
    """Rejected input with a serializable diagnostic; no scores are fabricated/dropped."""
    def __init__(self, issues, scores=None):
        self.report = dict(status="rejected", issues=issues,
                           rows=0 if scores is None else len(scores), rejected_runs=[])
        if scores is not None and "status" in scores:
            self.report["status_counts"] = {str(k): int(v) for k, v in scores.status.value_counts(dropna=False).items()}
            columns = [c for c in ("dataset_id", "split_id", "model_id", "status", "error_message") if c in scores]
            self.report["rejected_runs"] = scores.loc[scores.status.ne("ok"), columns].fillna("").astype(str).to_dict("records")
        super().__init__("; ".join(issues))


def validate_pmlb_candidate_results(scores, *, project_root=None):
    """Validate one complete final 10 x 4 cohort against existing typed data/manifests.

    Scores are raw Balanced Accuracy, not errors/percentages. All 40 rows must be ok;
    failures are reported, never silently excluded from the four-method reference.
    """
    scores = scores.copy().reset_index(drop=True)
    required = {field.name for field in fields(CandidateResult)}
    if scores.columns.has_duplicates:
        raise CandidateResultsValidationError(["Duplicate CandidateResult columns"])
    if not required <= set(scores):
        raise CandidateResultsValidationError([f"Missing CandidateResult columns: {sorted(required - set(scores))}"], scores)
    issues = []
    for column in ("dataset_source", "dataset_id", "split_id", "model_id", "exact_model_id",
                   "validation_metric", "test_metric", "status"):
        if not scores[column].map(lambda v: isinstance(v, str) and bool(v.strip())).all():
            issues.append(f"{column} must contain nonempty strings")
    if issues:
        raise CandidateResultsValidationError(issues, scores)
    scores["error_message"] = scores.error_message.fillna("").astype(str)
    paths = project_paths(project_root)
    pilot = read_yaml_mapping(paths.root / "config/pmlb_pilot.yaml")["datasets"]
    expected_ids = {spec["dataset_id"] for spec in pilot}
    if len(pilot) != 10 or len(expected_ids) != 10:
        raise CandidateResultsValidationError(["Expected ten unique approved PMLB datasets"], scores)
    expected_keys = {(dataset_id, model) for dataset_id in expected_ids for model in PMLB_PILOT_MODELS}
    observed_keys = set(zip(scores.dataset_id, scores.model_id))
    if len(scores) != 40 or observed_keys != expected_keys:
        issues.append(f"Expected exactly 40 final records; missing={sorted(expected_keys - observed_keys)}, "
                      f"unexpected={sorted(observed_keys - expected_keys)}")
    if scores.duplicated(["dataset_id", "split_id", "model_id"]).any():
        issues.append("Duplicate final model records (PriorTune trials must be a separate log)")
    if scores.duplicated(["dataset_id", "split_id", "exact_model_id"]).any():
        issues.append("Duplicate exact_model_id within a dataset/split")
    if scores.groupby("exact_model_id").model_id.nunique().gt(1).any():
        issues.append("An exact_model_id cannot identify multiple model variants")
    if not scores.dataset_source.eq("local").all():
        issues.append("dataset_source must be local")
    if not scores.split_id.eq("holdout_0").all():
        issues.append("split_id must be holdout_0")
    if not (scores.validation_metric.eq("balanced_accuracy") & scores.test_metric.eq("balanced_accuracy")).all():
        issues.append("Both metrics must be balanced_accuracy for the entire cohort")
    if "imputed" in scores and not scores.imputed.astype(str).str.lower().isin(["false", "0"]).all():
        issues.append("Imputed/unknown scores cannot enter the final reference cohort")
    if not scores.status.eq("ok").all():
        issues.append("Non-successful statuses: the complete four-final-method cohort is required")
        if (scores.status.ne("ok") & scores.error_message.eq("")).any():
            issues.append("Failed rows must include an error_message")
    if (scores.status.eq("ok") & scores.error_message.ne("")).any():
        issues.append("Successful rows must have an empty error_message")
    for column in ("validation_score", "test_score", "fit_time", "predict_time"):
        boolean = scores[column].map(lambda value: isinstance(value, (bool, np.bool_)))
        numeric = pd.Series(pd.to_numeric(scores[column], errors="coerce").to_numpy(dtype=float, na_value=np.nan),
                            index=scores.index)
        if column.endswith("score"):
            valid = np.isfinite(numeric) & numeric.between(0, 1) & ~boolean
            if (scores.status.eq("ok") & ~valid).any():
                issues.append(f"{column} must be a finite Balanced Accuracy in [0, 1]")
        else:
            # Timings can be unknown, but supplied values must be finite seconds >= 0.
            valid = scores[column].isna() | (np.isfinite(numeric) & numeric.ge(0) & ~boolean)
            if not valid.all():
                issues.append(f"{column} must be nonnegative finite seconds or null")
        scores[column] = numeric
    contexts = []
    for number, value in enumerate(scores.provenance):
        try:
            context = json.loads(value)
            if not isinstance(context, dict):
                raise ValueError("expected a JSON object")
        except (ValueError, TypeError) as exc:
            issues.append(f"row {number}: invalid provenance JSON string ({exc})")
            context = {}
        if context.get("is_final") is not True:
            issues.append(f"row {number}: provenance.is_final must be true")
        if scores.model_id.iloc[number] == "lightgbm_priortune_k25":
            prior = context.get("priortune")
            if not isinstance(prior, dict) or type(prior.get("k")) is not int or prior["k"] != 25:
                issues.append(f"row {number}: provenance.priortune.k must be integer 25")
        contexts.append(context)
    if issues:
        raise CandidateResultsValidationError(issues, scores)

    protocol = research_config(paths.root)["splits"]
    for spec in pilot:
        dataset_id = spec["dataset_id"]
        try:
            dataset = load_dataset(dataset_id, project_root=paths.root)
            manifest_path = research_paths(paths.root)["splits"] / dataset_id / "holdout_0.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            check_manifest(manifest, dataset)
            if manifest["protocol"] != protocol or manifest["split_id"] != "holdout_0":
                raise ValueError("Existing manifest differs from current split protocol/ID")
            if (dataset.metadata.file.sha256 != spec["sha256"]
                    or list(dataset.metadata.categorical_columns) != spec["categorical_columns"]):
                raise ValueError("Dataset checksum/categorical typing differs from approved pilot")
            expected = dict(dataset_fingerprint=manifest["dataset_fingerprint"],
                            protocol_hash=manifest["protocol_hash"], csv_sha256=spec["sha256"],
                            split_manifest_sha256=sha256_file(manifest_path),
                            problem_type="binary" if spec["n_classes"] == 2 else "multiclass")
            for number in scores.index[scores.dataset_id.eq(dataset_id)]:
                for field, value in expected.items():
                    if contexts[number].get(field) != value:
                        issues.append(f"row {number}, {dataset_id}: provenance.{field} mismatch")
        except (OSError, ValueError, KeyError) as exc:
            issues.append(f"{dataset_id}: cannot validate existing dataset/split ({exc})")
    if issues:
        raise CandidateResultsValidationError(issues, scores)
    return scores


def load_pmlb_candidate_results(path, *, project_root=None):
    """Read a local CSV/Parquet CandidateResult table; no TabArena conversion/download."""
    path = Path(path)
    try:
        if path.suffix.lower() == ".csv":
            with path.open(encoding="utf-8-sig", newline="") as stream:
                _check_headers(next(csv.reader(stream), []))
            scores = pd.read_csv(path, encoding="utf-8-sig")
        elif path.suffix.lower() == ".parquet":
            scores = pd.read_parquet(path)
        else:
            raise ValueError("Use a .csv or .parquet CandidateResult file")
    except (OSError, ValueError) as exc:
        raise CandidateResultsValidationError([f"Cannot read local results: {exc}"]) from exc
    return validate_pmlb_candidate_results(scores, project_root=project_root)
