import numpy as np


def _paired(actual, predicted):
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if actual.ndim != 1 or actual.shape != predicted.shape or not len(actual):
        raise ValueError("Need equal, nonempty 1D arrays")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Metrics require finite values")
    return actual, predicted


def gap_mae(actual, predicted) -> float:
    actual, predicted = _paired(actual, predicted)
    return float(np.abs(actual - predicted).mean())


def sign_accuracy(actual, predicted) -> float:
    """Exact three-way signs including zero; this zero convention needs paper clarification."""
    actual, predicted = _paired(actual, predicted)
    return float((np.sign(actual) == np.sign(predicted)).mean())


def selection_regret(best: float, selected: float, *, greater_is_better: bool) -> float:
    if not np.isfinite([best, selected]).all():
        raise ValueError("Regret requires finite scores")
    value = best - selected if greater_is_better else selected - best
    if value < -1e-12:
        raise ValueError("best/selected order disagrees with metric direction")
    return float(max(0.0, value))
