"""Small training-only baseline functions; no LODO experiment is run here."""
import numpy as np


def _training_gaps(values) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("Need a finite nonempty vector of training-dataset gaps")
    return array


def mean_training_gap(training_gaps) -> float:
    return float(_training_gaps(training_gaps).mean())


def majority_training_sign(training_gaps) -> int:
    """Three-way sign: -1/0/+1. Count ties explicitly; equal counts choose -1,0,+1 order.

    Zero-gap and voting tie conventions are OUR ASSUMPTION, not confirmed paper code.
    """
    signs, counts = np.unique(np.sign(_training_gaps(training_gaps)).astype(int), return_counts=True)
    return int(signs[counts.argmax()])
