"""Shared subprocess execution for the candidate pilots."""
import argparse
import json
import logging
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
SPLIT, SEED = "holdout_0", 42
MODELS = {
    "lightgbm": ("lightgbm", "baseline", "4.7.0"),
    "lightgbm_priortune_t_k25": ("lightgbm", "priortune_t_k25", "4.7.0"),
    "tabpfn_2_6": ("tabpfn", "v2_6", "2.6"),
    "tabicl_v2": ("tabicl", "v2", "2"),
}
WORKER_TIMEOUT_SECONDS = 900


def load_contract(dataset_id, expected):
    from meta_selector.datasets.shared_pilot import load_pmlb_dataset, load_shared_split

    dataset = load_pmlb_dataset(dataset_id)
    split = load_shared_split(dataset_id, SPLIT, seed=SEED, dataset=dataset)
    dimensions = (len(dataset.y), dataset.X.shape[1], dataset.y.nunique(),
                  len(split.train_indices), len(split.validation_indices), len(split.test_indices))
    if dimensions != expected:
        raise ValueError(f"{dataset_id}/{SPLIT}/{SEED}: expected {expected}, found {dimensions}")
    return dataset, split


def worker(dataset_id, expected, model_id, result_path, *, protocol_id=None):
    from meta_selector.artifacts import atomic_json
    from meta_selector.candidates import PilotCandidateRunner

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
    for name in ("pilot", "priortune"):
        logging.getLogger(f"meta_selector.candidates.{name}").setLevel(logging.DEBUG)

    dataset, split = load_contract(dataset_id, expected)
    print(f"Verified {split.join_key}: {expected}, categories={dataset.categorical_columns}", flush=True)

    result = PilotCandidateRunner(device="auto").run_shared(model_id, dataset, split, trial_protocol_id=protocol_id)
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


def run_isolated(entrypoint, dataset_id, expected, model_id, output, *, explicit_dataset, protocol_id=None):
    from meta_selector.artifacts import environment
    from meta_selector.candidates import CandidateResult

    log_path = output / f"{model_id}.log"
    family, variant, model_version = MODELS[model_id]
    identity = dict(dataset_id=dataset_id, split_id=SPLIT, seed=SEED, model_family=family, variant=variant,
                    model_id=model_id, dataset_source="pmlb")
    started = perf_counter()
    with TemporaryDirectory(dir=output) as directory, log_path.open("w", encoding="utf-8") as log:
        result_path = Path(directory) / "result.json"
        command = [sys.executable, str(entrypoint), "--model", model_id, "--worker-result", str(result_path)]
        if explicit_dataset:
            command.extend(("--dataset", dataset_id))
        if protocol_id is not None:
            command.extend(("--worker-protocol", protocol_id))
        try:
            process = subprocess.run(command, cwd=ROOT, env=worker_environment(output), stdout=log,
                                     stderr=subprocess.STDOUT, timeout=WORKER_TIMEOUT_SECONDS)
            if process.returncode == 0 and result_path.is_file():
                record = json.loads(result_path.read_text(encoding="utf-8"))
                if (not isinstance(record, dict) or record.get("schema_version") != 2
                        or any(record.get(key) != value for key, value in identity.items()) or not record.get("status")):
                    raise ValueError("Worker record schema or candidate identity does not match the request")
                return record
            error_type, message = "WorkerProcessError", f"Worker exited {process.returncode} or returned no result"
        except (subprocess.TimeoutExpired, OSError, ValueError) as error:
            error_type, message = type(error).__name__, str(error)
        print(f"{error_type}: {message}", file=log)

    versions = environment()
    metric = "roc_auc" if expected[2] == 2 else "log_loss"
    diagnostic = log_path.read_text(encoding="utf-8", errors="replace")
    pattern = rf"{re.escape(model_id)}: (input_validation|package_check|initialization|fit_train|score_validation|final_test|portfolio_check|validation_trials)\b"
    stages = re.findall(pattern, diagnostic)
    devices = re.findall(rf"{re.escape(model_id)} actual device: (cpu|mps|cuda)\b", diagnostic)
    return CandidateResult(**identity, model_version=model_version, exact_model_id="", validation_metric=metric,
                           test_metric=metric, status="error", error_type=error_type, error_message=message,
                           device=devices[-1] if devices else None,
                           total_time_sec=perf_counter() - started, python_version=versions.pop("python"),
                           package_versions=versions, provenance={"error_stage": stages[-1] if stages else "worker",
                                                                  "log": str(log_path)}).to_record()


def run_cli(entrypoint, description, contracts, output_name, *, explicit_dataset, argv=None, full_batch=None):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", choices=MODELS, help="Repeat just this candidate")
    parser.add_argument("--worker-result", type=Path, help=argparse.SUPPRESS)
    if explicit_dataset:
        selection = parser.add_mutually_exclusive_group(required=True)
        selection.add_argument("--dataset", choices=contracts, help="Run or repeat one shared dataset")
        selection.add_argument("--all", action="store_true", help="Run all fixed datasets in order")
    if full_batch is not None:
        parser.add_argument("--resume", action="store_true", help="Reuse only fully validated full-pilot rows")
        parser.add_argument("--audit-only", action="store_true", help="Validate the saved full-pilot artifacts")
        parser.add_argument("--worker-protocol", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    dataset_id = args.dataset if explicit_dataset else next(iter(contracts))
    if args.worker_result is not None:
        if args.model is None or dataset_id is None:
            parser.error("worker requires a single dataset and --model")
        protocol_id = getattr(args, "worker_protocol", None)
        if full_batch is not None and not re.fullmatch(r"[a-f0-9]{64}", protocol_id or ""):
            parser.error("full-pilot worker requires a protocol ID")
        worker(dataset_id, contracts[dataset_id], args.model, args.worker_result, protocol_id=protocol_id)
        return 0
    if dataset_id is None and args.model is not None:
        parser.error("--model requires a single dataset")
    audit_only, resume = getattr(args, "audit_only", False), getattr(args, "resume", False)
    if audit_only and (dataset_id is not None or resume):
        parser.error("--audit-only requires --all and cannot be combined with --resume")

    datasets = (dataset_id,) if dataset_id is not None else tuple(contracts)
    models = (args.model,) if args.model else tuple(MODELS)
    return run_batch(entrypoint, contracts, datasets, models, output_name, explicit_dataset=explicit_dataset,
                     full_batch=full_batch, resume=resume, audit_only=audit_only)


def run_batch(entrypoint, contracts, datasets, models, output_name, *, explicit_dataset,
              full_batch=None, resume=False, audit_only=False):
    import pandas as pd
    from meta_selector.artifacts import atomic_json, research_paths, upsert_table
    from meta_selector.datasets.shared_pilot import JOIN_KEYS

    started = perf_counter()
    # Validate every requested contract before spawning even the first model.
    verified = {}
    for dataset_id in (contracts if full_batch is not None else datasets):
        verified[dataset_id] = load_contract(dataset_id, contracts[dataset_id])
        print(f"Verified {dataset_id}/{SPLIT}/{SEED}: {contracts[dataset_id]}", flush=True)

    output = research_paths(ROOT)["derived"] / output_name
    output.mkdir(parents=True, exist_ok=True)
    batch = full_batch(ROOT, output, contracts, verified) if full_batch is not None else None
    if audit_only:
        return batch.audit()
    if batch is not None:
        batch.prepare()
    statuses = []
    reused = 0
    for dataset_id in datasets:
        log_dir = output / dataset_id if explicit_dataset else output
        log_dir.mkdir(exist_ok=True)
        for model_id in models:
            if batch is not None and resume and batch.can_resume(dataset_id, model_id):
                print(f"Verified existing {dataset_id}/{model_id}", flush=True)
                statuses.append("ok")
                reused += 1
                continue
            print(f"Running {dataset_id}/{model_id}", flush=True)
            record = run_isolated(entrypoint, dataset_id, contracts[dataset_id], model_id, log_dir,
                                  explicit_dataset=explicit_dataset, protocol_id=batch.protocol_id if batch else None)
            if batch is not None:
                record = batch.stamp(record, dataset_id, model_id)
            upsert_table(output / "candidate_results.parquet", pd.DataFrame([record]),
                         [*JOIN_KEYS, "model_family", "variant"])
            statuses.append(record["status"])
            print(f"Saved {dataset_id}/{model_id} ({record['status']}); log: {log_dir / (model_id + '.log')}", flush=True)
    complete_scope = explicit_dataset and len(datasets) == len(contracts) and len(models) == len(MODELS)
    if batch is not None and complete_scope:
        return batch.audit(wall_clock_sec=perf_counter() - started, reused=reused)
    if complete_scope:
        wall_clock = perf_counter() - started
        atomic_json(output / "run_summary.json", dict(datasets=list(datasets), models=list(models),
                    split_id=SPLIT, seed=SEED, attempts=len(statuses), successful=statuses.count("ok"),
                    failed=len(statuses) - statuses.count("ok"), wall_clock_sec=wall_clock))
        print(f"Mini-pilot wall clock: {wall_clock:.3f} s", flush=True)
    return 0 if all(status == "ok" for status in statuses) else 1
