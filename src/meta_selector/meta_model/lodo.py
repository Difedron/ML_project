"""Dataset-isolated LODO; preprocessing fits only on the remaining datasets."""
import json

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .baselines import mean_training_gap, majority_training_sign
from meta_selector.artifacts import environment
from meta_selector.evaluation.metrics import gap_mae, sign_accuracy


def evaluate_lodo(meta_dataset, *, predictor_factory, feature_columns=None):
    keys = ["dataset_source", "dataset_id"]
    if not set(keys + ["gap"]) <= set(meta_dataset) or meta_dataset.duplicated(keys).any():
        raise ValueError("LODO requires exactly one meta-observation per dataset/source; aggregate splits first")
    if meta_dataset[keys].isna().any().any() or not np.isfinite(meta_dataset.gap).all():
        raise ValueError("LODO needs complete dataset identities and finite gaps")
    features = feature_columns or [name for name in meta_dataset if name.startswith("feature::")]
    if not features or not all(name.startswith("feature::") for name in features):
        raise ValueError("Use explicitly namespaced feature columns, never IDs or target gap")
    if "origin_group" in meta_dataset:
        if meta_dataset.origin_group.isna().any():
            raise ValueError("origin_group must be complete if supplied")
        groups = meta_dataset.origin_group.astype(str)
    else:
        groups = meta_dataset[keys].apply(lambda row: json.dumps(row.tolist()), axis=1)
    if groups.nunique() < 3:
        raise ValueError("LODO needs at least three distinct dataset groups")
    records = []
    for group in sorted(groups.unique()):
        test_mask = groups.eq(group)
        train, test = meta_dataset.loc[~test_mask], meta_dataset.loc[test_mask]
        preprocessing = Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                                  ("scale", StandardScaler())])
        X_train = preprocessing.fit_transform(train[features].replace([np.inf, -np.inf], np.nan))
        X_test = preprocessing.transform(test[features].replace([np.inf, -np.inf], np.nan))
        predictor = predictor_factory()
        predictor.fit(pd.DataFrame(X_train, columns=features, index=train.index), train.gap.to_numpy())
        predictions = np.asarray(predictor.predict(pd.DataFrame(X_test, columns=features, index=test.index)))
        if predictions.shape != (len(test),) or not np.isfinite(predictions).all():
            raise ValueError("Predictor returned malformed/non-finite gaps")
        mean, majority = mean_training_gap(train.gap), majority_training_sign(train.gap)
        provenance = json.dumps(dict(train_datasets=train[keys].to_dict("records"), held_out_group=group,
                                     preprocessing="training-fold median imputation + standard scaling",
                                     feature_columns=features, environment=environment()), sort_keys=True)
        for (_, item), prediction in zip(test.iterrows(), predictions):
            records.append(dict(dataset_source=item.dataset_source, dataset_id=item.dataset_id,
                                actual_gap=item.gap, predicted_gap=float(prediction), baseline_mean_gap=mean,
                                baseline_majority_sign=majority, n_train_datasets=len(train), provenance=provenance))
    predictions = pd.DataFrame(records)
    summary = summarize_predictions(predictions)
    summary["by_source"] = {source: summarize_predictions(group) for source, group in predictions.groupby("dataset_source")}
    return predictions, summary


def summarize_predictions(predictions):
    choice_correct = np.where(predictions.actual_gap > 0, predictions.predicted_gap > 0,
                              np.where(predictions.actual_gap < 0, predictions.predicted_gap <= 0, True))
    return dict(n_datasets=len(predictions), mae=gap_mae(predictions.actual_gap, predictions.predicted_gap),
                   sign_accuracy=sign_accuracy(predictions.actual_gap, predictions.predicted_gap),
                   selection_accuracy=float(np.mean(choice_correct)),
                   baseline_mae=gap_mae(predictions.actual_gap, predictions.baseline_mean_gap),
                   baseline_sign_accuracy=sign_accuracy(predictions.actual_gap, predictions.baseline_majority_sign),
                   # Pairwise normalized regret; zero-gap predictions choose A deterministically.
                   mean_normalized_regret=float(np.mean(np.where(predictions.predicted_gap > 0,
                       np.maximum(-predictions.actual_gap, 0), np.maximum(predictions.actual_gap, 0)))))
