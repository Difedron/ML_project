"""Small result-table cache. Fetch is explicit and never touches local dataset registration."""
import hashlib
import json
import re
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from meta_selector.paths import DEFAULT_PROJECT_ROOT, read_yaml_mapping
from .results_loader import ROW_KEYS, ResultsSchemaError, standardize_results


class ExportContext(Protocol):
    def compare(self, output_dir: Path, *, plot: bool, fillna: None) -> Any: ...


def export_official_results(context: ExportContext, output_dir: str | Path) -> Path:
    """Checked against pinned TabArena compare(); caller chooses the context/snapshot."""
    output = Path(output_dir)
    context.compare(output_dir=output, plot=False, fillna=None)
    path = output / "results_per_split.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Official export did not create {path}")
    return path


class TabArenaAdapter:
    def __init__(self, config_path: str | Path | None = None):
        self.config_path = Path(config_path).resolve() if config_path else DEFAULT_PROJECT_ROOT / "config/tabarena_results.yaml"
        self.config = read_yaml_mapping(self.config_path)
        if self.config.get("schema_version") != 1:
            raise ValueError("Unsupported TabArena catalog schema")
        self.project_root = self.config_path.parent.parent
        self.cache_dir = (self.project_root / self.config["cache_dir"]).resolve()
        items = self.config["artifacts"]
        self.artifacts = {item["artifact_id"]: item for item in items}
        if len(items) != len(self.artifacts):
            raise ValueError("Duplicate artifact IDs")
        for key, item in self.artifacts.items():
            if not re.fullmatch(r"[a-z][a-z0-9_]*", key):
                raise ValueError("Unsafe artifact ID")
            if not re.fullmatch(r"[0-9a-f]{32}", item["md5"]):
                raise ValueError("Each artifact needs the official pinned MD5")
            if not item["url"].startswith(("https://tabarena.s3.amazonaws.com/", "https://data.tabarena.ai/")):
                raise ValueError("Only official result hosts are accepted")

    def path_for(self, artifact_id: str) -> Path:
        if artifact_id not in self.artifacts:
            raise KeyError(f"Unknown artifact: {artifact_id}")
        return self.cache_dir / f"{artifact_id}.parquet"

    def verify(self, artifact_id: str) -> Path:
        path = self.path_for(artifact_id)
        if not path.is_file():
            raise FileNotFoundError(f"Missing cached artifact {artifact_id}; use explicit --fetch")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "md5").hexdigest()
        if digest != self.artifacts[artifact_id]["md5"]:
            raise ValueError(f"Checksum mismatch for {artifact_id}; do not silently replace the pinned snapshot")
        return path

    def fetch(self, artifact_id: str) -> Path:
        """Fetch at most one small published aggregate table, without weights/predictions."""
        path = self.path_for(artifact_id)
        if path.exists():
            return self.verify(artifact_id)
        item = self.artifacts[artifact_id]
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            request = urllib.request.Request(item["url"], headers={"User-Agent": "meta-selector/0.1 research audit"})
            with urllib.request.urlopen(request, timeout=30) as response:
                with tempfile.NamedTemporaryFile(dir=self.cache_dir, delete=False, suffix=".part") as stream:
                    temporary = Path(stream.name)
                    size = 0
                    for block in iter(lambda: response.read(65536), b""):
                        size += len(block)
                        if size > self.config["max_artifact_bytes"]:
                            raise ValueError("Artifact exceeds the declared small-table download limit")
                        stream.write(block)
            data = temporary.read_bytes()
            if hashlib.md5(data).hexdigest() != item["md5"]:
                raise ValueError(f"Official checksum mismatch for {artifact_id}")
            temporary.replace(path)
            metadata = {**item, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
                        "tabarena_revision": self.config["tabarena_revision"]}
            path.with_suffix(".provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
        finally:
            if temporary:
                temporary.unlink(missing_ok=True)
        return path

    def load(self, artifact_ids=None) -> pd.DataFrame:
        ids = list(artifact_ids) if artifact_ids is not None else list(self.config["pair"].values())
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("Choose nonempty distinct artifact IDs")
        tables = []
        exclusions = []
        for artifact_id in ids:
            item = self.artifacts[artifact_id]
            native = pd.read_parquet(self.verify(artifact_id))
            policy = self.config.get("imputation_policy", "reject")
            if policy not in {"reject", "exclude_imputed_or_unknown"}:
                raise ValueError("Unknown imputation_policy")
            if policy == "exclude_imputed_or_unknown" and "imputed" in native:
                flags = native.imputed.map(lambda value: str(value).lower())
                recognized = flags.isin(["false", "true", "0", "1"]) | native.imputed.isna()
                if not recognized.all():
                    raise ResultsSchemaError("Malformed imputation flags")
                keep = flags.isin(["false", "0"])
                excluded = native.loc[~keep].copy()
                excluded["reason"] = ["unknown_imputation" if pd.isna(value) else "imputed"
                                      for value in excluded.imputed]
                for (reason, subtype), group in excluded.groupby(["reason", "method_subtype"]):
                    exclusions.append(dict(artifact_id=artifact_id, reason=reason,
                                           method_subtype=subtype, rows=len(group)))
                native = native.loc[keep].copy()
            frame = standardize_results(native, artifact_id=artifact_id)
            if set(frame.ta_suite) != {item["ta_suite"]} or set(frame.ta_name) != {item["ta_name"]}:
                raise ResultsSchemaError(f"Artifact identity disagrees with pinned metadata: {artifact_id}")
            # The catalog also covers historical methods whose names do not encode version.
            if set(frame.model_id) != {item["model_id"]}:
                raise ResultsSchemaError(f"Model mapping disagrees with catalog: {artifact_id}")
            tables.append(frame)
        result = pd.concat(tables, ignore_index=True)
        if result.duplicated(ROW_KEYS).any():
            raise ResultsSchemaError("Artifacts overlap on exact result identifiers")
        for dataset, group in result.groupby(["dataset_source", "dataset_id"]):
            if group.metric.nunique() != 1 or group.problem_type.nunique() != 1:
                raise ResultsSchemaError(f"Metric mismatch across artifacts: {dataset}")
        result = result.sort_values(ROW_KEYS).reset_index(drop=True)
        result.attrs["excluded_results"] = exclusions
        return result
