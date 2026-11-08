"""Build historical TabArena gaps or strictly validated local PMLB pilot gaps."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from meta_selector.artifacts import _atomic, atomic_json, config_hash, research_config, research_paths
from meta_selector.candidates.results import (CandidateResultsValidationError, load_pmlb_candidate_results,
                                              to_gap_records)
from meta_selector.datasets.pmlb import sha256_file
from meta_selector.meta_dataset import build_meta_dataset
from meta_selector.paths import project_paths, read_yaml_mapping
from meta_selector.performance_gap.comparisons import COMPARISONS, PMLB_COMPARISONS, restrict_coverage
from meta_selector.performance_gap.families import PMLB_PILOT_MODELS
from meta_selector.performance_gap.gap import calculate_dataset_gaps
from meta_selector.tabarena import TabArenaAdapter


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Pinned artifact catalog YAML")
    parser.add_argument("--fetch", action="store_true", help="Download missing small reference tables")
    parser.add_argument("--output-dir", help="Default: results/tables/tabicl_v2_vs_tabpfn_2_6")
    parser.add_argument("--local-scores", type=Path, help="CSV/Parquet CandidateResult: complete 10 x 4 PMLB pilot")
    parser.add_argument("--project-root", type=Path, help="Root of the local PMLB project")
    parser.add_argument("--build-meta-dataset", action="store_true", help="Local only: join existing PyMFE/controls/custom tables")
    parser.add_argument("--features", type=Path, help="Local feature Parquet override (requires --build-meta-dataset)")
    parser.add_argument("--custom", type=Path, help="Local custom-landmarker Parquet override")
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--comparison", choices=sorted(set(COMPARISONS) | set(PMLB_COMPARISONS)))
    choice.add_argument("--all", action="store_true", help="All six comparisons for the selected backend")
    args = parser.parse_args(argv)
    if args.local_scores:
        if args.fetch or args.config:
            parser.error("--local-scores cannot use TabArena --fetch/--config")
        if args.comparison and args.comparison not in PMLB_COMPARISONS:
            parser.error("This comparison is not defined for the four-method PMLB pilot")
        if (args.features or args.custom) and not args.build_meta_dataset:
            parser.error("Feature overrides require --build-meta-dataset")
        return build_local(args)
    if args.project_root or args.build_meta_dataset or args.features or args.custom:
        parser.error("Local preparation options require --local-scores")
    if args.comparison and args.comparison not in COMPARISONS:
        parser.error("This comparison requires --local-scores")
    adapter = TabArenaAdapter(args.config)
    specification = adapter.config["normalization_reference"]
    ids = specification["artifact_ids"]
    if args.fetch:
        for artifact_id in ids:
            adapter.fetch(artifact_id)
    reference = adapter.load(ids)
    exclusions = reference.attrs.get("excluded_results", [])
    reference = reference[reference.method_subtype.isin(specification["method_subtypes"])].copy()
    if set(reference.artifact_id) != set(ids):
        raise ValueError("The declared reference roster is incomplete after subtype filtering")
    selected = list(COMPARISONS.values()) if args.all else [
        COMPARISONS[args.comparison or "tabicl_v2_vs_tabpfn_2_6"]]
    for comparison in selected:
        build_one(adapter, reference, specification, exclusions, comparison, args.output_dir, args.all)
    return 0


def _local_features(args, root, scores):
    """Bind existing feature records to the validated holdout before calling the builder."""
    config = research_config(root)
    base = research_paths(root)["derived"]
    features = pd.read_parquet(args.features or base / "meta_features/meta_features.parquet")
    custom = pd.read_parquet(args.custom or base / "landmarkers/custom_landmarkers.parquet")
    ids = set(scores.dataset_id)
    features = features[features.dataset_source.eq("local") & features.dataset_id.isin(ids)
                        & features.source.isin(["controls", "paper_features"])].copy()
    custom = custom[custom.dataset_source.eq("local") & custom.dataset_id.isin(ids)].copy()
    for table in (features, custom):
        if (table[["dataset_source", "dataset_id", "split_id", "provenance", "config_id"]].isna().any().any()
                or set(table.dataset_id) != ids or not table.split_id.eq("holdout_0").all()):
            raise ValueError("Feature/custom dataset or split coverage differs from the candidate cohort")
    if not features.status.isin(["ok", "undefined"]).all():
        raise ValueError("Feature extraction status must be ok or explicit undefined")
    if set(features.source) != {"controls", "paper_features"} or features.groupby("source").config_id.nunique().gt(1).any():
        raise ValueError("Select one complete controls/paper_features configuration")
    if not features.loc[features.source.eq("paper_features"), "config_id"].eq(config_hash(config["paper_features"])).all():
        raise ValueError("PyMFE configuration differs from research.yaml")
    if (custom[["metric", "source", "status", "n_folds", "score_mean", "score_std"]].isna().any().any()
            or not custom.metric.eq("balanced_accuracy").all()
            or not custom.source.eq("custom_landmarking").all()
            or not custom.config_id.eq(config_hash(config["custom_landmarkers"])).all()
            or not custom.status.eq("ok").all() or not custom.n_folds.eq(3).all()
            or not np.isfinite(custom[["score_mean", "score_std"]]).all().all()
            or not custom.score_mean.between(0, 1).all() or not custom.score_std.ge(0).all()):
        raise ValueError("Custom landmarkers must be successful, configured 3-fold Balanced Accuracy results")
    for dataset_id in ids:
        models = custom.loc[custom.dataset_id.eq(dataset_id), "model_id"]
        if len(models) != 5 or set(models) != set(config["custom_landmarkers"]["models"]):
            raise ValueError(f"Incomplete/duplicate custom models: {dataset_id}")
        expected = json.loads(scores.loc[scores.dataset_id.eq(dataset_id), "provenance"].iloc[0])
        manifest_path = research_paths(root)["splits"] / dataset_id / "holdout_0.json"
        train = json.loads(manifest_path.read_text(encoding="utf-8"))["indices"]["train"]
        for table, is_custom in ((features, False), (custom, True)):
            for value in table.loc[table.dataset_id.eq(dataset_id), "provenance"].unique():
                context = json.loads(value)
                context = context["split"] if is_custom else context
                if (context.get("dataset_fingerprint") != expected["dataset_fingerprint"]
                        or context.get("protocol_hash") != expected["protocol_hash"]
                        or context.get("train_indices") != train
                        or context.get("input_role") != "outer_training_only"):
                    raise ValueError(f"Feature/custom fingerprint or outer-training provenance mismatch: {dataset_id}")
    return features, custom


def build_local(args):
    paths = project_paths(args.project_root)
    root = paths.root
    base = (root / args.output_dir).resolve() if args.output_dir else paths.results / "tables/pmlb_pilot"
    if not base.resolve().is_relative_to(root) or not base.resolve().is_relative_to(paths.results.resolve()):
        raise ValueError("Local outputs must remain under the project's results directory")
    selected = list(PMLB_COMPARISONS.values()) if args.all else [
        PMLB_COMPARISONS[args.comparison or "tabicl_v2_vs_tabpfn_2_6"]]
    destinations = {base / "candidate_import.json", base / "candidate_results.parquet"}
    destinations |= {base / comparison.id / name for comparison in selected
                     for name in ("dataset_gaps.csv", "split_gaps.csv", "provenance.json", "meta_dataset.parquet")}
    if any(path and path.resolve() in destinations for path in (args.local_scores, args.features, args.custom)):
        raise ValueError("An output path would overwrite an input file")
    stage = "candidate_validation"
    try:
        scores = load_pmlb_candidate_results(args.local_scores, project_root=root)
        reference = to_gap_records(scores)
        stage = "feature_validation"
        features, custom = _local_features(args, root, scores) if args.build_meta_dataset else (None, None)
        stage = "gap_and_meta_preparation"
        specs = read_yaml_mapping(root / "config/pmlb_pilot.yaml")["datasets"]
        origins = {s["dataset_id"]: s["origin_group"] for s in specs}
        label = "pmlb_pilot_four_final_balanced_accuracy_v1"
        prepared = []
        for comparison in selected:
            result = calculate_dataset_gaps(reference, comparison.family_a, comparison.family_b,
                                            reference=reference, reference_label=label)
            if set(result.dataset_gaps.dataset_id) != set(scores.dataset_id):
                raise ValueError("Gap calculation did not retain every approved dataset")
            for table in (result.split_gaps, result.dataset_gaps):
                table["comparison_id"] = comparison.id
            result.dataset_gaps["origin_group"] = result.dataset_gaps.dataset_id.map(origins)
            meta = (build_meta_dataset(result.dataset_gaps, [features], namespaces=["controls", "paper_features"],
                                       custom=custom, custom_metric="balanced_accuracy") if args.build_meta_dataset else None)
            result.provenance.update(comparison_id=comparison.id, dataset_source="local", metric="balanced_accuracy",
                                     candidate_input_sha256=sha256_file(args.local_scores),
                                     pilot_config_sha256=sha256_file(root / "config/pmlb_pilot.yaml"),
                                     reference_protocol="one final record per dataset/split/model; four methods; no tuning trials",
                                     meta_dataset_built=meta is not None)
            prepared.append((comparison, result, meta))
    except CandidateResultsValidationError as exc:
        atomic_json(base / "candidate_import.json", dict(exc.report, stage=stage, input_file=str(args.local_scores)))
        print(f"Rejected local results: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, KeyError, TypeError) as exc:
        atomic_json(base / "candidate_import.json", dict(status="rejected", stage=stage, issues=[str(exc)]))
        print(f"Local preparation failed: {exc}", file=sys.stderr)
        return 1
    # Publish only after the entire requested cohort/comparisons/optional joins passed.
    _atomic(base / "candidate_results.parquet", lambda temporary: scores.to_parquet(temporary, index=False))
    for comparison, result, meta in prepared:
        output = base / comparison.id
        _atomic(output / "split_gaps.csv", lambda temporary, t=result.split_gaps: t.to_csv(temporary, index=False))
        _atomic(output / "dataset_gaps.csv", lambda temporary, t=result.dataset_gaps: t.to_csv(temporary, index=False))
        atomic_json(output / "provenance.json", result.provenance)
        if meta is not None:
            _atomic(output / "meta_dataset.parquet", lambda temporary, t=meta: t.to_parquet(temporary, index=False))
        print(f"{comparison.id}: {len(result.dataset_gaps)} datasets; reference=4 final variants; "
              f"meta_dataset={meta is not None}; positive gap favors {comparison.family_b.name}")
    atomic_json(base / "candidate_import.json", dict(status="accepted", rows=len(scores), metric="balanced_accuracy",
                datasets=sorted(scores.dataset_id.unique()), model_ids=list(PMLB_PILOT_MODELS), split_id="holdout_0",
                input_sha256=sha256_file(args.local_scores), comparisons=[c.id for c in selected]))
    return 0


def build_one(adapter, reference, specification, exclusions, comparison, output_dir, multiple):
    cohort, excluded_datasets = restrict_coverage(reference, comparison)
    result = calculate_dataset_gaps(cohort, comparison.family_a, comparison.family_b, reference=reference,
                                    reference_label=specification["label"])
    for table in (result.split_gaps, result.dataset_gaps):
        table["comparison_id"] = comparison.id
    result.provenance.update({
        "comparison_id": comparison.id,
        "coverage_policy": comparison.coverage,
        "excluded_datasets": excluded_datasets,
        "tabarena_revision": adapter.config["tabarena_revision"],
        "paper_exact_snapshot_confirmed": specification["paper_exact_snapshot_confirmed"],
        "reference_subtypes": specification["method_subtypes"],
        "imputation_policy": adapter.config["imputation_policy"],
        "excluded_results_before_subtype_filter": exclusions,
        "artifacts": [adapter.artifacts[key] for key in specification["artifact_ids"]],
        "limitation": "Historical roster reconstruction; binary cached metric is ROC-AUC error, "
                      "whereas Appendix A describes classification log-loss. Not a completed E0 replication.",
    })
    output = Path(output_dir).resolve() if output_dir else adapter.project_root / "results/tables"
    if multiple or not output_dir:
        output = output / comparison.id
    output.mkdir(parents=True, exist_ok=True)
    result.split_gaps.to_csv(output / "split_gaps.csv", index=False)
    result.dataset_gaps.to_csv(output / "dataset_gaps.csv", index=False)
    (output / "provenance.json").write_text(json.dumps(result.provenance, indent=2) + "\n", encoding="utf-8")
    print(f"{comparison.paper_id} {comparison.id}: datasets={len(result.dataset_gaps)}; paired splits={len(result.split_gaps)}")
    print(f"Reference: {specification['label']} ({len(specification['artifact_ids'])} published model tables)")
    print("Paper-exact snapshot: unconfirmed. This is a gap construction check, not an E0 evaluation.")
    print(f"Positive gap favors {comparison.family_b.name}; negative gap favors {comparison.family_a.name}.")
    print(result.dataset_gaps[["dataset_id", "metric", "n_splits", "gap"]].head(10).to_string(index=False))
    print(f"Saved to: {output}")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, KeyError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
