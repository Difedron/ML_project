import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from meta_selector.artifacts import research_config
from meta_selector.batch import local_ids, process_local
from meta_selector.paths import read_yaml_mapping


def batch_main(kind):
    parser = argparse.ArgumentParser(description=f"Compute {kind} on local outer-training portions only")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("dataset_ids", nargs="*")
    selection.add_argument("--all-local", action="store_true")
    parser.add_argument("--project-root")
    parser.add_argument("--config", help="Research YAML override")
    parser.add_argument("--split-id", default="holdout_0")
    if kind == "meta_features":
        parser.add_argument("--without-standard-landmarking", action="store_true")
    args = parser.parse_args()
    config = read_yaml_mapping(Path(args.config)) if args.config else research_config(args.project_root)
    ids = local_ids(args.project_root) if args.all_local else args.dataset_ids
    if not ids:
        parser.error("No valid local classification datasets selected")
    report = process_local(kind, ids, root=args.project_root, config=config, split_id=args.split_id,
                           include_standard=not getattr(args, "without_standard_landmarking", False))
    print(report[["dataset_id", "namespace", "status", "rows", "error_message"]].to_string(index=False))
    return 1 if report.status.eq("error").any() else 0
