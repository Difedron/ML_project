"""Import the ten approved local PMLB datasets; optionally prepare persisted holdout splits."""
import argparse
import sys

import _bootstrap  # noqa: F401
from meta_selector.datasets.pmlb import import_pmlb


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", help="Project checkout containing data/PMLB and config/pmlb_pilot.yaml")
    parser.add_argument("--create-splits", action="store_true", help="Create/reuse splits from config/research.yaml")
    parser.add_argument("--split-id", default="holdout_0")
    args = parser.parse_args(argv)
    try:
        records = import_pmlb(project_root=args.project_root, create_splits=args.create_splits, split_id=args.split_id)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"PMLB import failed: {error}", file=sys.stderr)
        return 1
    print("dataset_id status rows features classes categorical train validation test")
    for row in records:
        sizes = row.get("split_sizes", {})
        print(row["dataset_id"], row["status"], row["n_rows"], row["n_features"], row["n_classes"],
              row["n_categorical_features"], *[sizes.get(name, "-") for name in ("train", "validation", "test")])
    print(f"Verified {len(records)} datasets; report: results/logs/pmlb_import.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
