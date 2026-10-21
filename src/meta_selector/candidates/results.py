"""Direction-aware bridge from explicit local candidate scores to canonical gap records."""
import json
import numpy as np
import pandas as pd

from meta_selector.tabarena.results_loader import ROW_KEYS


def to_gap_records(scores):
    if scores.empty or scores.status.ne("ok").any():
        raise ValueError("Only successful real candidate results can become gap records")
    if scores.dataset_source.ne("local").any():
        raise ValueError("This bridge expects local candidate results")
    if not scores.validation_metric.eq(scores.test_metric).all():
        raise ValueError("Validation and test metrics must match")
    result = scores.copy()
    result["metric"] = result.test_metric
    maximize = result.metric.isin(["balanced_accuracy", "macro_f1", "roc_auc"])
    if not (maximize | result.metric.eq("log_loss")).all():
        raise ValueError("Declare the score direction for this metric")
    result["test_error"] = np.where(maximize, 1 - result.test_score, result.test_score)
    result["validation_error"] = np.where(maximize, 1 - result.validation_score, result.validation_score)
    if not np.isfinite(result[["test_error", "validation_error"]]).all().all():
        raise ValueError("Missing/non-finite scores cannot be imputed")
    result["problem_type"] = result.provenance.map(lambda value: json.loads(value)["problem_type"])
    result["imputed"] = False
    if result.duplicated(ROW_KEYS).any():
        raise ValueError("Duplicate candidate result keys")
    return result
