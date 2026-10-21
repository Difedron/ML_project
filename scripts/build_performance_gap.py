"""Build a dataset-level gap using the explicit, reconstructed historical reference roster."""
import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from meta_selector.performance_gap.comparisons import COMPARISONS, restrict_coverage
from meta_selector.performance_gap.gap import calculate_dataset_gaps
from meta_selector.tabarena import TabArenaAdapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Pinned artifact catalog YAML")
    parser.add_argument("--fetch", action="store_true", help="Download missing small reference tables")
    parser.add_argument("--output-dir", help="Default: results/tables/tabicl_v2_vs_tabpfn_2_6")
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--comparison", choices=list(COMPARISONS))
    choice.add_argument("--all", action="store_true", help="Build all six Table-3 comparisons")
    args = parser.parse_args()
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
        main()
    except (ValueError, KeyError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
