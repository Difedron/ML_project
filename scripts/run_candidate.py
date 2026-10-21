"""Explicit one-model, one-dataset execution; never a 21 x 150 implicit benchmark."""
import argparse
from pathlib import Path
import pandas as pd
import _bootstrap  # noqa: F401
from meta_selector import load_dataset
from meta_selector.artifacts import IDENTITY, research_paths, upsert_table
from meta_selector.candidates.runner import TabArenaCandidateRunner
from meta_selector.datasets.splits import make_split, portion
from meta_selector.paths import DEFAULT_PROJECT_ROOT, read_yaml_mapping
from meta_selector.tabarena.model_mapping import MODELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_id")
    parser.add_argument("--model", required=True, choices=list(MODELS))
    parser.add_argument("--config", default=str(DEFAULT_PROJECT_ROOT / "config/candidates.yaml"))
    parser.add_argument("--metric", default="balanced_accuracy", choices=["balanced_accuracy", "macro_f1", "roc_auc", "log_loss"])
    parser.add_argument("--split-id", default="holdout_0")
    args = parser.parse_args()
    dataset = load_dataset(args.dataset_id)
    manifest = make_split(dataset, split_id=args.split_id)
    parts = [portion(dataset, manifest, name) for name in ("train", "validation", "test")]
    runner = TabArenaCandidateRunner(read_yaml_mapping(Path(args.config)))
    result = runner.run(args.model, *[part for pair in parts for part in pair],
                        dataset_id=args.dataset_id, split_id=args.split_id, metric=args.metric)
    result.provenance["split_manifest"] = manifest
    upsert_table(research_paths()["derived"] / "candidate_scores/candidate_scores.parquet",
                 pd.DataFrame([result.to_record()]), IDENTITY + ["model_id", "exact_model_id", "test_metric"])
    print(f"{args.model}: {result.status}; {result.error_message}")
    return 0 if result.status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
