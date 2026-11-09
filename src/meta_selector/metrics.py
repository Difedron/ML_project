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


def classification_scores(y, predicted, probabilities, classes):
    """Score one prediction snapshot; probability columns must follow sorted classes."""
    y, predicted = np.asarray(y), np.asarray(predicted)
    probabilities, classes = np.asarray(probabilities, dtype=float), np.asarray(classes)
    if y.ndim != 1 or predicted.shape != y.shape or not len(y):
        raise ValueError("Need nonempty, matching 1D labels and predictions")
    if classes.ndim != 1 or len(classes) < 2 or not np.array_equal(classes, np.unique(classes)):
        raise ValueError("Classes must be distinct and sorted")
    if not np.isin(y, classes).all() or not np.isin(predicted, classes).all():
        raise ValueError("Prediction/target labels must belong to training classes")
    if not np.array_equal(np.unique(y), classes):
        raise ValueError("Evaluation portion requires every training class")
    if probabilities.shape != (len(y), len(classes)) or not np.isfinite(probabilities).all():
        raise ValueError("Invalid probability shape or nonfinite probabilities")
    if ((probabilities < 0).any() or (probabilities > 1).any()
            or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6, rtol=1e-6)):
        raise ValueError("Probabilities must be in [0, 1] and sum to one")
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities[:, 1] if len(classes) == 2 else probabilities,
                                       labels=classes, multi_class="ovr", average="macro")),
        "log_loss": float(log_loss(y, probabilities, labels=classes)),
    }
