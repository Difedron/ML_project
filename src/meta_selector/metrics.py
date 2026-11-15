"""Shared classification scoring. No dependency on candidate or landmarker code."""
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss, roc_auc_score


def classification_score(model, X, y, metric, classes):
    """Historical scoring API, also re-exported by landmarkers.custom."""
    if metric == "balanced_accuracy":
        return float(balanced_accuracy_score(y, model.predict(X)))
    if metric == "macro_f1":
        return float(f1_score(y, model.predict(X), average="macro", zero_division=0))
    probabilities = model.predict_proba(X)
    if metric == "log_loss":
        return float(log_loss(y, probabilities, labels=classes))
    if metric == "roc_auc":
        return float(roc_auc_score(y, probabilities[:, 1] if len(classes) == 2 else probabilities,
                                   labels=classes, multi_class="ovr", average="macro"))
    raise ValueError(f"Unknown classification metric: {metric}")


def aligned_class_probabilities(model, X, n_classes):
    """Align estimator probabilities to canonical integer training labels."""
    classes = np.asarray(model.classes_)
    expected = np.arange(n_classes)
    if classes.shape != expected.shape or set(classes.tolist()) != set(expected.tolist()):
        raise ValueError("Estimator classes do not match training labels")
    probabilities = np.asarray(model.predict_proba(X), dtype=float)
    if probabilities.shape != (len(X), n_classes):
        raise ValueError("Estimator returned an invalid probability shape")
    order = [int(np.flatnonzero(classes == label)[0]) for label in expected]
    return probabilities[:, order]


def _validated_probabilities(y, probabilities, classes):
    y = np.asarray(y)
    probabilities, classes = np.asarray(probabilities, dtype=float), np.asarray(classes)
    if y.ndim != 1 or not len(y):
        raise ValueError("Need nonempty 1D labels")
    if classes.ndim != 1 or len(classes) < 2 or not np.array_equal(classes, np.unique(classes)):
        raise ValueError("Classes must be distinct and sorted")
    if not np.array_equal(np.unique(y), classes):
        raise ValueError("Evaluation portion requires every training class")
    if probabilities.shape != (len(y), len(classes)) or not np.isfinite(probabilities).all():
        raise ValueError("Invalid probability shape or nonfinite probabilities")
    if ((probabilities < 0).any() or (probabilities > 1).any()
            or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6, rtol=1e-6)):
        raise ValueError("Probabilities must be in [0, 1] and sum to one")
    return y, probabilities, classes


def _roc_auc(y, probabilities, classes):
    return float(roc_auc_score(y, probabilities[:, 1] if len(classes) == 2 else probabilities,
                               labels=classes, multi_class="ovr", average="macro"))


def classification_probability_score(y, probabilities, metric):
    """Score only the chosen probability metric on integer-encoded labels."""
    probabilities = np.asarray(probabilities)
    if probabilities.ndim != 2:
        raise ValueError("Expected a two-dimensional probability matrix")
    y, probabilities, classes = _validated_probabilities(y, probabilities, np.arange(probabilities.shape[1]))
    if metric == "roc_auc":
        return _roc_auc(y, probabilities, classes)
    if metric == "log_loss":
        return float(log_loss(y, probabilities, labels=classes))
    raise ValueError(f"Unknown probability metric: {metric}")


def classification_scores(y, predicted, probabilities, classes):
    """Score one prediction snapshot; probability columns must follow sorted classes."""
    y, probabilities, classes = _validated_probabilities(y, probabilities, classes)
    predicted = np.asarray(predicted)
    if predicted.shape != y.shape or not np.isin(predicted, classes).all():
        raise ValueError("Predictions must align with evaluation rows and training labels")
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
        "roc_auc": _roc_auc(y, probabilities, classes),
        "log_loss": float(log_loss(y, probabilities, labels=classes)),
    }
