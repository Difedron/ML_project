"""Stage 5 only: run the four candidates sequentially in separate native processes."""
import argparse
import json
import logging
import math
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
DATASET, SPLIT, SEED = "pmlb_saheart", "holdout_0", 42
MODELS = {
    "lightgbm": ("lightgbm", "baseline", "4.7.0"),
    "lightgbm_priortune_t_k25": ("lightgbm", "priortune_t_k25", "4.7.0"),
    "tabpfn_2_6": ("tabpfn", "v2_6", "2.6"),
    "tabicl_v2": ("tabicl", "v2", "2"),
}
WORKER_TIMEOUT_SECONDS = 900


def worker(model_id, result_path):
    from meta_selector.artifacts import atomic_json
    from meta_selector.candidates import PilotCandidateRunner
    from meta_selector.datasets.shared_pilot import load_pmlb_dataset, load_shared_split

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
    for name in ("pilot", "priortune"):
        logging.getLogger(f"meta_selector.candidates.{name}").setLevel(logging.DEBUG)

    dataset = load_pmlb_dataset(DATASET)
    split = load_shared_split(DATASET, SPLIT, seed=SEED, dataset=dataset)
    dimensions = (len(dataset.y), dataset.X.shape[1], dataset.y.nunique(),
                  len(split.train_indices), len(split.validation_indices), len(split.test_indices))
    if dimensions != (462, 9, 2, 276, 93, 93):
        raise ValueError(f"Unexpected SAheart smoke contract: {dimensions}")
    print(f"Verified {split.join_key}: {dimensions}, categories={dataset.categorical_columns}", flush=True)

    result = PilotCandidateRunner(device="auto").run_shared(model_id, dataset, split)
    record = result.to_record()
    record = {key: None if isinstance(value, float) and not math.isfinite(value) else value
              for key, value in record.items()}
    atomic_json(result_path, record)
    print(f"{model_id}: {result.status}; device={result.device}; {result.error_type} {result.error_message}", flush=True)


def worker_environment(output):
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED="1", PYTHONFAULTHANDLER="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               HF_HUB_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1", TABPFN_DISABLE_TELEMETRY="1")
    for name, folder in (("SKB_DATA_DIRECTORY", "skrub"), ("MPLCONFIGDIR", "matplotlib"), ("XDG_CACHE_HOME", "xdg")):
        cache = output / "cache" / folder
        cache.mkdir(parents=True, exist_ok=True)
        env[name] = str(cache)
    return env


def run_isolated(model_id, output):
    from meta_selector.artifacts import environment
    from meta_selector.candidates import CandidateResult

    log_path = output / f"{model_id}.log"
    with TemporaryDirectory(dir=output) as directory, log_path.open("w", encoding="utf-8") as log:
        result_path = Path(directory) / "result.json"
        command = [sys.executable, str(Path(__file__).resolve()), "--model", model_id, "--worker-result", str(result_path)]
        try:
            process = subprocess.run(command, cwd=ROOT, env=worker_environment(output), stdout=log,
                                     stderr=subprocess.STDOUT, timeout=WORKER_TIMEOUT_SECONDS)
            if process.returncode == 0 and result_path.is_file():
                return json.loads(result_path.read_text(encoding="utf-8"))
            error_type, message = "WorkerProcessError", f"Worker exited {process.returncode} or returned no result"
        except (subprocess.TimeoutExpired, OSError, ValueError) as error:
            error_type, message = type(error).__name__, str(error)
        print(f"{error_type}: {message}", file=log)

    family, variant, model_version = MODELS[model_id]
    versions = environment()
    return CandidateResult(DATASET, SPLIT, model_id, model_version, "", "roc_auc", "roc_auc",
                           dataset_source="pmlb", seed=SEED, model_family=family, variant=variant,
                           status="error", error_type=error_type, error_message=message,
                           python_version=versions.pop("python"), package_versions=versions,
                           provenance={"error_stage": "worker", "log": str(log_path)}).to_record()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, help="Repeat just this candidate")
    parser.add_argument("--worker-result", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_result is not None:
        if args.model is None:
            parser.error("worker requires --model")
        worker(args.model, args.worker_result)
        return 0

    import pandas as pd
    from meta_selector.artifacts import research_paths, upsert_table
    from meta_selector.datasets.shared_pilot import JOIN_KEYS

    output = research_paths(ROOT)["derived"] / "candidate_smoke"
    output.mkdir(parents=True, exist_ok=True)
    statuses = []
    for model_id in (args.model,) if args.model else MODELS:
        print(f"Running {model_id}", flush=True)
        record = run_isolated(model_id, output)
        upsert_table(output / "candidate_results.parquet", pd.DataFrame([record]),
                     [*JOIN_KEYS, "model_family", "variant"])
        statuses.append(record["status"])
        print(f"Saved {model_id} ({record['status']}); log: {output / (model_id + '.log')}", flush=True)
    return 0 if all(status == "ok" for status in statuses) else 1


if __name__ == "__main__":
    raise SystemExit(main())
