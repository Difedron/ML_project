import argparse
from pathlib import Path
import pandas as pd
import _bootstrap  # noqa: F401
from meta_selector.candidates.runner import TabArenaCandidateRunner
from meta_selector.paths import DEFAULT_PROJECT_ROOT, read_yaml_mapping
from meta_selector.tabarena import TabArenaAdapter
from meta_selector.tabarena.model_mapping import MODELS
from meta_selector.performance_gap.families import MLP, TREE, NON_TFM, TFM


def main():
    parser = argparse.ArgumentParser(description="Read-only model availability check; no weights or fitting")
    parser.add_argument("--config", default=str(DEFAULT_PROJECT_ROOT / "config/candidates.yaml"))
    parser.add_argument("--output", help="Optional CSV report")
    args = parser.parse_args()
    runner = TabArenaCandidateRunner(read_yaml_mapping(Path(args.config)))
    adapter = TabArenaAdapter()
    catalog = {item["model_id"]: key for key, item in adapter.artifacts.items()}
    rows = []
    for model_id in MODELS:
        row = runner.availability(model_id)
        row["family"] = ",".join(family.name for family in (MLP, TREE, NON_TFM, TFM) if model_id in family.members)
        try:
            adapter.verify(catalog[model_id])
            row["artifact_available"] = True
        except (KeyError, ValueError, FileNotFoundError):
            row["artifact_available"] = False
        rows.append(row)
    frame = pd.DataFrame(rows)
    print(frame.to_string(index=False))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
