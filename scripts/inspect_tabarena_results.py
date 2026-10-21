"""Inspect published aggregate results; --fetch explicitly downloads only small tables."""
import argparse
import sys
import pandas as pd

import _bootstrap  # noqa: F401
from meta_selector.paths import read_yaml_mapping
from meta_selector.tabarena import TabArenaAdapter, load_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Pinned artifact catalog YAML")
    parser.add_argument("--fetch", action="store_true", help="Download missing small aggregate tables")
    parser.add_argument("--reference", action="store_true", help="Inspect the entire declared reference roster")
    parser.add_argument("--csv", help="Inspect an external official results_per_split.csv")
    parser.add_argument("--identity-map", help="Explicit method -> exact metadata YAML for CSV imports")
    args = parser.parse_args()
    if args.csv:
        if args.fetch or args.reference:
            parser.error("--csv cannot be combined with --fetch or --reference")
        identities = read_yaml_mapping(args.identity_map) if args.identity_map else None
        frame = load_results(args.csv, identity_map=identities)
    else:
        adapter = TabArenaAdapter(args.config)
        ids = (adapter.config["normalization_reference"]["artifact_ids"] if args.reference
               else list(adapter.config["pair"].values()))
        if args.fetch:
            for artifact_id in ids:
                adapter.fetch(artifact_id)
        frame = adapter.load(ids)
    print(f"Rows: {len(frame)}; datasets: {frame.dataset_id.nunique()}; "
          f"dataset/splits: {len(frame[['dataset_source', 'dataset_id', 'split_id']].drop_duplicates())}")
    if frame.attrs.get("excluded_results"):
        print("\nExplicit exclusions (imputed or unknown; see configured policy):")
        print(pd.DataFrame(frame.attrs["excluded_results"]).to_string(index=False))
    print("\nExact model identifiers and versions:")
    columns = ["model_id", "model_version", "ta_suite", "ta_name", "method", "method_subtype"]
    print(frame[columns].drop_duplicates().to_string(index=False))
    print("\nDatasets, splits and metrics:")
    print(frame.groupby(["dataset_source", "dataset_id", "problem_type", "metric"], as_index=False)
          .agg(splits=("split_id", "nunique")).to_string(index=False))
    print("\nReal cached rows (up to three per model):")
    print(frame.groupby("model_id", sort=True).head(3)[
        ["dataset_id", "split_id", "model_id", "metric", "validation_error", "test_error"]
    ].to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
