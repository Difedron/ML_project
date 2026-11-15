"""Stage-7 identity, resume validation and diagnostics derived from canonical Parquet."""
import importlib.metadata
import json
import math
import platform
import re

import pandas as pd

from _isolated_candidates import MODELS, SEED, SPLIT
from meta_selector.artifacts import _atomic, atomic_json, config_hash, research_paths, upsert_table
from meta_selector.candidates import PilotCandidateRunner
from meta_selector.candidates.priortune import (PORTFOLIO_PIN, TRIAL_KEYS, effective_parameters, load_portfolio,
                                                selection_rule)
from meta_selector.datasets.shared_pilot import file_sha256


PRIOR = "lightgbm_priortune_t_k25"
METRICS = ("validation_score", "balanced_accuracy", "roc_auc", "log_loss")
TIMES = ("model_init_time_sec", "fit_time_sec", "validation_time_sec", "tuning_time_sec",
         "predict_time_sec", "total_time_sec")
VARIANTS = tuple(f"{family}/{variant}" for family, variant, _ in MODELS.values())


def provenance(row):
    return json.loads(row["provenance"])


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


class FullPilotArtifacts:
    def __init__(self, root, output, contracts, verified):
        self.output, self.contracts = output, contracts
        self.table = output / "candidate_results.parquet"
        self.trials_path = research_paths(root)["derived"] / "priortune_validation/trials.parquet"
        self.catalog = verified
        self.expected = {}
        runner = PilotCandidateRunner(project_root=root, device="auto")
        pre = json.loads((output / "preflight.json").read_text(encoding="utf-8"))
        if not pre.get("all_ready") or pre.get("requested_device") != "auto":
            raise ValueError("Native full-pilot preflight did not pass")
        names = ("numpy", "pandas", "scikit-learn", "torch", "lightgbm", "tabpfn", "tabicl")
        self.versions = {name: importlib.metadata.version(name) for name in names}
        self.python = platform.python_version()
        if pre["python"] != self.python or pre["platform"] != platform.platform():
            raise ValueError("The preflight environment differs from the runner")
        for name in ("torch", "lightgbm", "tabpfn", "tabicl"):
            if pre["package_versions"][name] != self.versions[name]:
                raise ValueError(f"The preflight version differs for {name}")
        candidate = runner.candidates[PRIOR]
        spec = candidate.resources["portfolio"]
        self.portfolio = load_portfolio(root / spec["cache_path"], spec)
        if pre["models"]["priortune"]["portfolio"]["sha256"] != spec["sha256"]:
            raise ValueError("Preflight portfolio SHA256 differs")
        for name in ("tabpfn", "tabicl"):
            if pre["models"][name]["checkpoint"]["sha256"] != candidate.resources["checkpoints"][name]["sha256"]:
                raise ValueError(f"Preflight checkpoint SHA256 differs for {name}")
        for dataset_id, (dataset, split) in verified.items():
            shared = dict(fingerprint=dataset.fingerprint, shared_manifest_sha256=split.metadata["shared_manifest_sha256"],
                          rita_dataset_fingerprint=split.rita_dataset_fingerprint, row_order=split.metadata["row_order"])
            for model_id, (family, variant, _) in MODELS.items():
                device = "cpu" if family == "lightgbm" else pre["models"][family]["api"]["device"]
                exact_id = runner.candidates[model_id].expected_identity(dataset, device, self.versions[family])
                self.expected[dataset_id, model_id] = dict(dataset_id=dataset_id, split_id=SPLIT, seed=SEED,
                    model_family=family, variant=variant, model_id=model_id, exact_model_id=exact_id,
                    device=device, shared_split=shared)
        sources = sorted((root / "src/meta_selector").rglob("*.py"))
        sources += [root / "scripts/candidates" / name for name in
                    ("check_models.py", "_isolated_candidates.py", "_full_pilot_artifacts.py", "run_full_pilot.py")]
        configs = [root / name for name in ("config/candidates/pilot.yaml", "config/pilot_datasets.yaml",
                   candidate.config["baseline_config"], candidate.config["resources_config"])]
        self.protocol = dict(schema_version=1, python=self.python, platform=platform.platform(), packages=self.versions,
                             files={str(path.relative_to(root)): file_sha256(path) for path in sources + configs},
                             checkpoints=candidate.resources["checkpoints"], portfolio=PORTFOLIO_PIN,
                             expected=list(self.expected.values()))
        self.protocol_id = config_hash(self.protocol)

    def prepare(self):
        atomic_json(self.output / "protocol.json", {**self.protocol, "protocol_id": self.protocol_id})
        history = self.output / "stage5_stage6_trials.parquet"
        if self.trials_path.exists() and not history.exists():
            rows = pd.read_parquet(self.trials_path)
            if "full_pilot_protocol_id" in rows:
                rows = rows[rows.full_pilot_protocol_id.isna()]
            if not rows.empty:
                upsert_table(history, rows, list(TRIAL_KEYS))

    def _log(self, dataset_id, model_id):
        return self.output / dataset_id / f"{model_id}.log"

    def stamp(self, row, dataset_id, model_id):
        expected = self.expected[dataset_id, model_id]
        pv = provenance(row)
        if row["status"] == "ok" and (any(row.get(key) != expected[key] for key in ("exact_model_id", "device"))
                                       or pv.get("shared_split") != expected["shared_split"]):
            row.update(status="error", error_type="ProtocolIdentityError", error_message="Model, device or split differs from pre-run identity")
            pv["error_stage"] = "artifact_validation"
        pv.setdefault("shared_split", expected["shared_split"])
        pv["full_pilot"] = dict(protocol_id=self.protocol_id, log_sha256=file_sha256(self._log(dataset_id, model_id)))
        return {**row, "provenance": json.dumps(pv, sort_keys=True)}

    def _trials(self):
        if not self.trials_path.exists():
            return pd.DataFrame()
        rows = pd.read_parquet(self.trials_path)
        if "full_pilot_protocol_id" not in rows:
            return rows.iloc[:0]
        return rows[rows.full_pilot_protocol_id.eq(self.protocol_id)]

    def _check_trials(self, row, pv, trials):
        dataset_id = row["dataset_id"]
        n_classes = self.contracts[dataset_id][2]
        metric, direction, _ = selection_rule(n_classes)
        trials = trials[trials.dataset_id.eq(dataset_id)] if not trials.empty else trials
        if len(trials) != 25 or trials.duplicated(list(TRIAL_KEYS)).any() or not trials.status.eq("ok").all():
            raise ValueError("Expected 25 unique successful validation trials")
        if not (trials.selection_metric.eq(metric).all() and trials.selection_direction.eq(direction).all()
                and trials.split_id.eq(SPLIT).all() and trials.seed.eq(SEED).all()
                and trials.model_family.eq("lightgbm").all() and trials.variant.eq("priortune_t_k25").all()
                and trials.portfolio_id.eq("lgbm/T/K25").all()
                and trials.portfolio_revision.eq(PORTFOLIO_PIN["revision"]).all()
                and trials.portfolio_sha256.eq(PORTFOLIO_PIN["sha256"]).all()
                and trials.exact_model_id.eq(row["exact_model_id"]).all()
                and trials.validation_score.map(finite).all()):
            raise ValueError("Validation trial protocol mismatch")
        expected = {config.config_id: effective_parameters(config, SEED, n_classes) for config in self.portfolio}
        if {t.config_id: json.loads(t.effective_parameters) for t in trials.itertuples()} != expected:
            raise ValueError("Effective trial parameters differ from the published portfolio")
        best = trials.sort_values(["validation_score", "config_id"], ascending=[direction == "minimize", True]).iloc[0]
        saved = pv["priortune"]
        if (saved["selected_config_id"] != best.config_id or saved["selected_validation_score"] != best.validation_score
                or row["validation_score"] != best.validation_score or saved["selected_validation_metric"] != metric
                or saved["selection_direction"] != direction or saved["successful_trials"] != 25 or saved["failed_trials"] != 0
                or saved["failure_policy"] != "require_all_25" or saved["tie_break_rule"] != "smallest_config_id"
                or saved["effective_parameters"] != expected[best.config_id]):
            raise ValueError("PriorTune winner does not match independent validation ranking")
        return int(best.config_id)

    def _check_success(self, row, trials):
        dataset_id, model_id = row["dataset_id"], row["model_id"]
        pv, expected = provenance(row), self.expected[dataset_id, model_id]
        if row["status"] != "ok" or row["schema_version"] != 2 or any(
                row.get(key) != value for key, value in expected.items() if key != "shared_split"):
            raise ValueError("Candidate identity or status mismatch")
        _, features, classes, train, val, test = self.contracts[dataset_id]
        if [row.get(key) for key in ("n_features", "n_classes", "n_train", "n_validation", "n_test")] != [features, classes, train, val, test]:
            raise ValueError("Candidate dimensions mismatch")
        if (pv.get("shared_split") != expected["shared_split"] or pv.get("full_pilot", {}).get("protocol_id") != self.protocol_id
                or row["python_version"] != self.python or any(json.loads(row["package_versions"]).get(k) != v for k,v in self.versions.items())
                or not all(finite(row.get(key)) for key in METRICS + TIMES)
                or row["validation_metric"] != selection_rule(classes)[0] or row["test_metric"] != row["validation_metric"]):
            raise ValueError("Candidate protocol, environment or metrics mismatch")
        dataset, _ = self.catalog[dataset_id]
        if (pv.get("categorical_columns") != list(dataset.categorical_columns) or pv.get("preprocessing_fit_on") != "train_only"
                or pv.get("estimator_classes") != list(range(classes))
                or pv.get("label_classes") != sorted(dataset.y.unique().tolist())):
            raise ValueError("Category, class order or preprocessing contract mismatch")
        family = row["model_family"]
        if family != "lightgbm":
            pin = self.protocol["checkpoints"][family]
            if any(pv.get("checkpoint", {}).get(k) != pin[k] for k in ("repo_id", "revision", "sha256")) or not (
                    row["checkpoint_version"] == pin["filename"] and pv.get("estimator_devices")
                    and all(device.split(":", 1)[0] == expected["device"] for device in pv["estimator_devices"])):
                raise ValueError("Checkpoint or fitted neural device mismatch")
        log_path = self._log(dataset_id, model_id)
        if file_sha256(log_path) != pv["full_pilot"]["log_sha256"]:
            raise ValueError("Candidate log digest mismatch")
        log = log_path.read_text(encoding="utf-8")
        final = f"{model_id}: final_test"
        if log.count(final) != 1:
            raise ValueError("Expected exactly one final test stage")
        if model_id == PRIOR:
            winner = self._check_trials(row, pv, trials)
            selected = f"selected config {winner} after 25 validation trials"
            entries = list(re.finditer(r"PriorTune config (\d+): validation .*status=ok", log))
            if ([int(entry.group(1)) for entry in entries] != [c.config_id for c in self.portfolio] or selected not in log
                    or not entries[-1].end() < log.index(selected) < log.index(final)):
                raise ValueError("PriorTune validation and final test stages are out of order")
        elif not log.index(f"{model_id}: fit_train") < log.index(f"{model_id}: score_validation") < log.index(final):
            raise ValueError("Fit, validation and final test stages are out of order")

    def can_resume(self, dataset_id, model_id):
        if not self.table.exists():
            return False
        rows = pd.read_parquet(self.table)
        rows = rows[rows.dataset_id.eq(dataset_id) & rows.model_id.eq(model_id)]
        if len(rows) != 1:
            return False
        try:
            self._check_success(rows.iloc[0], self._trials() if model_id == PRIOR else pd.DataFrame())
        except (ValueError, KeyError, TypeError, AttributeError, OSError):
            return False
        return True

    def audit(self, *, wall_clock_sec=None, reused=None):
        rows, trials = pd.read_parquet(self.table), self._trials()
        problems = []
        keys = ("dataset_id", "split_id", "seed", "model_family", "variant", "exact_model_id")
        slots = set(zip(rows.dataset_id, rows.model_id))
        if len(rows) != len(self.expected) or slots != set(self.expected) or rows.duplicated(list(keys)).any():
            problems.append("Expected exactly 40 unique candidate rows across the 10 fixed datasets")
        if (not trials.empty and (trials.duplicated(list(TRIAL_KEYS)).any() or any("test" in col for col in trials)
                                  or not set(trials.dataset_id) <= set(self.contracts))):
            problems.append("Unexpected, duplicate or test fields in PriorTune trials")
        errors = []
        for _, row in rows.iterrows():
            dataset_id, model_id = row["dataset_id"], row["model_id"]
            if row["status"] == "ok":
                try:
                    self._check_success(row, trials)
                except (ValueError, KeyError, TypeError, AttributeError, OSError) as error:
                    problems.append(f"{dataset_id}/{model_id}: {error}")
            else:
                pv = provenance(row)
                errors.append(dict(dataset_id=dataset_id, model_id=model_id, error_type=row["error_type"],
                                   error_message=row["error_message"], failure_stage=pv.get("error_stage"),
                                   device=row["device"] if pd.notna(row["device"]) else None))
                expected = self.expected.get((dataset_id, model_id))
                if (expected is None or any(row.get(k) != expected[k] for k in ("split_id", "seed", "model_family", "variant"))
                        or pv.get("full_pilot", {}).get("protocol_id") != self.protocol_id
                        or pv.get("shared_split") != expected["shared_split"]
                        or row["error_type"] == "ProtocolIdentityError"):
                    problems.append(f"{dataset_id}/{model_id}: failed row has a mismatching protocol")
                if model_id == PRIOR and pv.get("priortune", {}).get("failed_trials", 0):
                    log = self._log(dataset_id, model_id).read_text(encoding="utf-8")
                    if f"{model_id}: final_test" in log:
                        problems.append(f"{dataset_id}: PriorTune evaluated test after a failed trial")
        n_trials = 0 if trials.empty else int(trials.status.eq("ok").sum())
        trial_total = len(self.contracts) * PORTFOLIO_PIN["k"]
        if not errors and (len(trials) != trial_total or n_trials != trial_total):
            problems.append("Expected 250 successful validation trials")
        status = "FAIL" if problems else "PARTIAL" if errors else "PASS"
        report_path = self.output / "run_summary.json"
        if wall_clock_sec is None and report_path.exists():
            previous = json.loads(report_path.read_text(encoding="utf-8"))
            if previous.get("protocol_id") == self.protocol_id:
                wall_clock_sec, reused = previous["wall_clock_sec"], previous["reused_candidate_rows"]
        atomic_json(report_path, dict(status=status, protocol_id=self.protocol_id, number_of_datasets=len(self.contracts),
            verified_dataset_ids=list(self.contracts), expected_candidate_runs=len(self.expected),
            successful_candidate_runs=int(rows.status.eq("ok").sum()), failed_candidate_runs=len(errors),
            expected_priortune_trials=trial_total, successful_priortune_trials=n_trials,
            failed_priortune_trials=len(trials)-n_trials, missing_priortune_trials=trial_total-len(trials),
            binary_datasets=sum(dims[2] == 2 for dims in self.contracts.values()),
            multiclass_datasets=sum(dims[2] > 2 for dims in self.contracts.values()), wall_clock_sec=wall_clock_sec,
            reused_candidate_rows=reused, counts_by_device=rows.device.fillna("unknown").value_counts().to_dict(),
            errors=errors, integrity_errors=problems))
        if not problems:
            self._diagnostics(rows)
        print(f"Full-pilot integrity: {status}. Summary: {report_path}", flush=True)
        return {"PASS": 0, "PARTIAL": 1, "FAIL": 2}[status]

    def _diagnostics(self, rows):
        rows = rows.assign(variant_key=rows.model_family + "/" + rows.variant)
        best, gaps = [], {}
        for dataset_id in self.contracts:
            group = rows[rows.dataset_id.eq(dataset_id)]
            eligible = len(group) == 4 and group.status.eq("ok").all()
            vals = {variant: None for variant in VARIANTS}
            vals.update({row.variant_key: float(row.balanced_accuracy) for row in group.itertuples() if row.status == "ok"})
            maximum = max(vals.values()) if eligible else None
            winners = [v for v in VARIANTS if eligible and vals[v] == maximum]
            gaps.update({(dataset_id, variant): maximum-score for variant, score in vals.items()} if eligible else {})
            best.append(dict(dataset_id=dataset_id, best_balanced_accuracy=maximum,
                             best_variant_by_test_balanced_accuracy=winners[0] if len(winners) == 1 else None,
                             tie=len(winners) > 1, tied_variants=winners if len(winners) > 1 else [], values=vals))
        rows["gap_to_best"] = [gaps.get(key) for key in zip(rows.dataset_id, rows.variant_key)]
        fields = (*METRICS, "validation_metric", "total_time_sec", "status", "device", "gap_to_best")
        wide = rows.set_index(["dataset_id", "variant_key"])[list(fields)].unstack().reindex(list(self.contracts))
        wide = wide.reindex(columns=pd.MultiIndex.from_product([fields, VARIANTS]))
        wide.columns = [f"{metric}__{variant}" for metric, variant in wide.columns]
        _atomic(self.output / "performance_matrix.parquet", lambda path: wide.reset_index().to_parquet(path, index=False))
        atomic_json(self.output / "best_by_balanced_accuracy.json", best)
