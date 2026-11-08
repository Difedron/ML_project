"""Stage-1 offline import/device/resource checks. No datasets, fit or checkpoint loading."""
import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import urllib.request


ROOT = Path(__file__).resolve().parents[3]
NAMES = {"tabpfn": "tabpfn-v2.6-classifier-v2.6_default.ckpt",
         "tabicl": "tabicl-classifier-v2-20260212.ckpt"}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_portfolio(path, specification):
    """Inspect the published resource only; fitting and winner selection belong to stage 4."""
    if sha256(path) != specification["sha256"]:
        raise ValueError("Published portfolio checksum mismatch")
    record = json.loads(Path(path).read_text())
    if (record["model"], record["objective"], record["K"]) != ("lgbm", "T", 25):
        raise ValueError("Expected plain lgbm/T/K=25")
    ids, order, configs = record["config_ids"], record["add_order"], record["configs"]
    if len(ids) != 25 or len(set(ids)) != 25 or len(order) != 25 or set(order) != set(ids):
        raise ValueError("Portfolio must contain exactly 25 distinct IDs in a published order")
    if len(configs) != 25 or {row["config_id"] for row in configs} != set(ids):
        raise ValueError("Portfolio config/ID mismatch")
    if any(row.get("tabprep_enabled") is not False or row.get("tabprep") is not None for row in configs):
        raise ValueError("TabPrep must be disabled in every configuration")
    if any(not isinstance(row.get("lightgbm"), dict) or "model_seed" not in row for row in configs):
        raise ValueError("Incomplete published configuration")
    return dict(ready=True, model="lgbm", rule="T", k=25, add_order=order,
                sha256=specification["sha256"], tabprep=False, ensemble=False)


def fetch_portfolio(specification, destination):
    """Explicit opt-in download of this one small public JSON, pinned by revision/hash."""
    url = (f"https://huggingface.co/datasets/{specification['repo_id']}/resolve/"
           f"{specification['revision']}/{specification['path']}")
    request = urllib.request.Request(url, headers={"User-Agent": "ML-project candidate preflight"})
    with urllib.request.urlopen(request, timeout=30) as stream:
        data = stream.read(131073)
    if len(data) > 131072 or hashlib.sha256(data).hexdigest() != specification["sha256"]:
        raise ValueError("Portfolio download exceeds size limit or disagrees with pinned SHA256")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(dir=destination.parent, suffix=".part")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
        validate_portfolio(name, specification)
        os.replace(name, destination)
    finally:
        Path(name).unlink(missing_ok=True)


def _worker(kind, payload):
    """Every native import has its own process: avoid macOS OpenMP runtime collisions."""
    if kind == "torch":
        import torch
        probes = {}
        for device in ("cpu", "cuda", "mps"):
            available = device == "cpu" or (torch.cuda.is_available() if device == "cuda"
                                             else torch.backends.mps.is_available())
            if not available:
                probes[device] = dict(available=False, arithmetic_pass=False)
                continue
            try:
                value = torch.eye(2, device=device) @ torch.ones(2, 1, device=device)
                actual = float(value.cpu().sum())
                probes[device] = dict(available=True, arithmetic_pass=actual == 2.0)
            except Exception as error:
                probes[device] = dict(available=True, arithmetic_pass=False, error=type(error).__name__)
        return dict(import_ok=True, version=torch.__version__, mps_built=torch.backends.mps.is_built(), devices=probes)
    if kind == "lightgbm":
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(**payload["params"], objective="binary")
        return dict(import_ok=True, constructor_ok=True, version=importlib.metadata.version("lightgbm"),
                    device="cpu", resolved_model_params=model.get_params())
    if kind == "tabpfn":
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion
        from tabpfn.model_loading import ModelSource
        from tabpfn.utils import infer_devices
        source = ModelSource.get_classifier_v2_6()
        if source.default_filename != NAMES[kind] or source.repo_id != "Prior-Labs/tabpfn_2_6":
            raise ValueError("unavailable_exact_version: SDK model source differs from pinned classifier")
        infer_devices(payload["device"])
        model = TabPFNClassifier.create_default_for_version(
            ModelVersion.V2_6, device=payload["device"], model_path=payload["checkpoint"])
        if str(model.model_path) != payload["checkpoint"]:
            raise ValueError("unavailable_exact_version: explicit checkpoint was not preserved")
        return dict(import_ok=True, constructor_ok=True, exact_model_identifier=ModelVersion.V2_6.value,
                    checkpoint_identifier=source.default_filename, device=payload["device"],
                    predict=callable(model.predict), predict_proba=callable(model.predict_proba))
    if kind == "tabicl":
        from tabicl import TabICLClassifier
        from tabicl._torch_devices import resolve_torch_device, mps_possibly_faulty
        if payload["device"] == "mps" and mps_possibly_faulty():
            raise ValueError("MPS numerics on this virtualized Mac are flagged by official TabICL")
        device = resolve_torch_device(payload["device"])
        model = TabICLClassifier(checkpoint_version=NAMES[kind], model_path=payload["checkpoint"],
                                allow_auto_download=False, device=str(device))
        if model.checkpoint_version != NAMES[kind] or model.allow_auto_download:
            raise ValueError("unavailable_exact_version: TabICLv2 checkpoint configuration differs")
        return dict(import_ok=True, constructor_ok=True, checkpoint_identifier=model.checkpoint_version,
                    device=str(device), predict=callable(model.predict), predict_proba=callable(model.predict_proba))
    raise ValueError("Unknown preflight worker")


def run_worker(kind, payload, *, timeout):
    env = os.environ.copy()
    cache = ROOT / ".cache/diana_preflight/runtime"
    for key, directory in (("SKB_DATA_DIRECTORY", "skrub"), ("MPLCONFIGDIR", "matplotlib"), ("XDG_CACHE_HOME", "xdg")):
        env.setdefault(key, str(cache / directory))
    env.update(HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1", TABPFN_DISABLE_TELEMETRY="1")
    try:
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", kind],
                                input=json.dumps(payload), capture_output=True, text=True, env=env, timeout=timeout)
        if result.returncode:
            return dict(import_ok=False, reason=f"Native worker exited {result.returncode}")
        return json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        return dict(import_ok=False, reason="Import/device probe timed out")
    except (OSError, ValueError) as error:
        return dict(import_ok=False, reason=f"{type(error).__name__}: preflight worker failed")


def choose_device(requested, torch_report):
    if requested not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError("Device must be auto/cpu/mps/cuda")
    choices = ("cuda", "mps", "cpu") if requested == "auto" else (requested,)
    for device in choices:
        if torch_report.get("devices", {}).get(device, {}).get("arithmetic_pass"):
            return device
    return None


def local_checkpoint(specification, explicit_path):
    """Only inspect provided paths / exact HF cache entries; never download weights."""
    from huggingface_hub import try_to_load_from_cache
    filename = specification["filename"]
    if explicit_path:
        path = Path(explicit_path).expanduser().resolve()
        candidates = [path]
    else:
        hit = try_to_load_from_cache(specification["repo_id"], filename, revision=specification["revision"])
        candidates = [Path(hit)] if isinstance(hit, str) else []
        candidates += [Path.home() / ".cache/tabpfn" / filename,
                       Path.home() / "Library/Caches/tabpfn" / filename]
    for path in candidates:
        if path.is_file():
            if sha256(path) != specification["sha256"]:
                return dict(ready=False, path=str(path), reason="Exact checkpoint SHA256 mismatch")
            return dict(ready=True, path=str(path), sha256=specification["sha256"])
    return dict(ready=False, path=str(candidates[0]) if explicit_path else None,
                reason="Exact classifier checkpoint not found locally; no automatic download")


def check_models(config, baseline, *, device=None, checkpoint_paths=None, fetch=False):
    timeout = config["worker_timeout_seconds"]
    versions = {}
    for package in ("torch", "tabpfn", "tabicl", "lightgbm", "huggingface-hub", "autogluon.tabular", "priortune"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    report = dict(python=platform.python_version(), platform=platform.platform(), package_versions=versions,
                  readiness_scope="imports + exact API constructor + verified local resources; inference not tested",
                  source_revisions=config["source_revisions"], models={})
    report["torch"] = run_worker("torch", {}, timeout=timeout)
    requested = device or config["device"]
    selected = choose_device(requested, report["torch"])
    report["requested_device"], report["selected_torch_device"] = requested, selected
    for key in ("tabpfn", "tabicl"):
        spec = config["checkpoints"][key]
        explicit = (checkpoint_paths or {}).get(key) or spec.get("local_path")
        try:
            checkpoint = local_checkpoint(spec, explicit)
        except Exception as error:
            checkpoint = dict(ready=False, reason=f"{type(error).__name__}: checkpoint cache inspection failed")
        probe = run_worker(key, dict(device=selected or "cpu", checkpoint=checkpoint.get("path") or
                                    str(ROOT / ".cache/diana_preflight/missing" / NAMES[key])), timeout=timeout)
        # Auto may conservatively fall back if the official backend rejects an accelerator.
        if requested == "auto" and not probe.get("import_ok") and selected in {"cuda", "mps"}:
            probe = run_worker(key, dict(device="cpu", checkpoint=checkpoint.get("path") or
                                        str(ROOT / ".cache/diana_preflight/missing" / NAMES[key])), timeout=timeout)
            probe["auto_fallback"] = "accelerator API probe failed; used CPU"
        ready = bool(probe.get("constructor_ok") and checkpoint["ready"] and selected)
        report["models"][key] = dict(ready=ready, status="ready" if ready else "unavailable_exact_version",
                                     version=versions[key], checkpoint=checkpoint, api=probe,
                                     checkpoint_identifier=spec["filename"],
                                     reason="; ".join(value for value in [probe.get("reason"),
                                         None if checkpoint["ready"] else checkpoint.get("reason"),
                                         None if selected else "Requested device not available"] if value))
    native = run_worker("lightgbm", dict(params=baseline["model_params"]), timeout=timeout)
    same_version = versions["lightgbm"] == baseline["package_version"]
    report["models"]["lightgbm"] = dict(ready=bool(native.get("constructor_ok") and same_version),
        version=versions["lightgbm"], device="cpu", api=native,
        reason=native.get("reason", "") + ("; baseline package pin mismatch" if not same_version else ""))
    spec = config["portfolio"]
    path = ROOT / spec["cache_path"]
    try:
        if fetch and not path.is_file():
            fetch_portfolio(spec, path)
        portfolio = validate_portfolio(path, spec)
    except Exception as error:
        portfolio = dict(ready=False, reason=f"{type(error).__name__}: {error}")
    report["models"]["priortune"] = dict(ready=bool(portfolio["ready"] and report["models"]["lightgbm"]["ready"]),
        device="cpu", portfolio=portfolio, source=spec, reason="; ".join(value for value in [
            portfolio.get("reason"), None if report["models"]["lightgbm"]["ready"] else "LightGBM native dependency not ready"] if value))
    report["authentication"] = {
        "tabpfn": "Official downloader requires PriorLabs license acceptance; auth not queried. No tokens printed.",
        "tabicl": "Public jingang/TabICL checkpoint; no authentication attempted.",
        "priortune": "Public published JSON; no authentication needed for the verified resource.",
    }
    report["all_ready"] = all(item["ready"] for item in report["models"].values())
    return report


def _worker_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True)
    args = parser.parse_args()
    payload = json.load(sys.stdin)
    try:
        # Libraries may log on import. Never forward uncontrolled logs or environment values.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = _worker(args.worker, payload)
    except Exception as error:
        message = str(error)
        if args.worker == "lightgbm" and "libomp" in message:
            message = "Missing libomp.dylib; install the official macOS OpenMP dependency (brew install libomp)"
        elif args.worker != "lightgbm":
            message = f"{type(error).__name__} during exact API/device check"
        else:
            message = f"{type(error).__name__} during native LightGBM import"
        result = dict(import_ok=False, reason=message)
    print(json.dumps(result))


if __name__ == "__main__":
    _worker_main()
