"""Explicit LODO entry point; requires real features/gaps and a verified 2.6 checkpoint."""
import argparse
from pathlib import Path
import pandas as pd
import _bootstrap  # noqa: F401
from meta_selector.artifacts import atomic_json, research_config
from meta_selector.evaluation.experiments import EXPERIMENT_SPECS, prepare_experiment
from meta_selector.meta_model.lodo import evaluate_lodo
from meta_selector.meta_model.tabpfn_predictor import TabPFN26Predictor
from meta_selector.paths import read_yaml_mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=list(EXPERIMENT_SPECS), required=True)
    parser.add_argument("--gaps", required=True, help="Single comparison dataset_gaps.csv")
    parser.add_argument("--features", action="append", required=True, help="Long-form feature Parquet, repeatable")
    parser.add_argument("--custom", help="Custom landmarker Parquet")
    parser.add_argument("--feature-config", action="append", default=[], help="Explicit source=config_id, repeatable")
    parser.add_argument("--custom-config-id", help="Choose one custom extractor configuration")
    parser.add_argument("--config", help="Research YAML")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config = read_yaml_mapping(Path(args.config)) if args.config else research_config()
    output = Path(args.output_dir)
    predictor = TabPFN26Predictor(config["meta_predictor"])
    status = predictor.availability()
    if status["status"] != "available":
        atomic_json(output / "status.json", status)
        print(f"{status['status']}: {status['reason']}")
        return 2
    selections = dict(item.split("=", 1) for item in args.feature_config)
    tables = [pd.read_parquet(path) for path in args.features]
    for index, table in enumerate(tables):
        for source, selected_id in selections.items():
            table = table[table.source.ne(source) | table.config_id.eq(selected_id)]
        tables[index] = table
    custom = pd.read_parquet(args.custom) if args.custom else None
    if custom is not None and args.custom_config_id:
        custom = custom[custom.config_id.eq(args.custom_config_id)]
    meta = prepare_experiment(EXPERIMENT_SPECS[args.experiment], pd.read_csv(args.gaps), tables,
                              custom=custom, custom_metric=config["custom_landmarkers"]["scoring"])
    predictions, summary = evaluate_lodo(meta, predictor_factory=lambda: TabPFN26Predictor(config["meta_predictor"]))
    output.mkdir(parents=True, exist_ok=True)
    meta.to_parquet(output / "meta_dataset.parquet", index=False)
    predictions.to_csv(output / "lodo_predictions.csv", index=False)
    atomic_json(output / "summary.json", dict(**summary, experiment=args.experiment, configuration=config,
                                               paper_exact_replication=False))
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
