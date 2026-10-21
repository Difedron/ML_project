"""Explicit dataset/source joins; split aggregation is declared, not row-order alignment."""
import pandas as pd

from meta_selector.artifacts import IDENTITY, config_hash

KEYS = ["dataset_source", "dataset_id"]


def build_meta_dataset(gaps, feature_tables, *, namespaces, custom=None, custom_metric="balanced_accuracy"):
    if gaps.empty or gaps.duplicated(KEYS).any():
        raise ValueError("Select one comparison: exactly one gap per dataset/source is required")
    tables = [table for table in feature_tables if not table.empty]
    if not tables:
        raise ValueError("No feature tables")
    features = pd.concat(tables, ignore_index=True)
    features = features[features.source.isin(namespaces)].copy()
    if set(features.source.unique()) != set(namespaces):
        raise ValueError("Missing requested feature namespaces")
    if custom is not None:
        custom = custom[custom.metric.eq(custom_metric)].copy()
        if custom.empty or custom.status.ne("ok").any():
            raise ValueError("Requested custom landmarks are missing or failed")
        long = custom.melt(id_vars=IDENTITY + ["model_id", "config_id"],
                           value_vars=["score_mean", "score_std"], var_name="summary", value_name="value")
        long["source"] = "custom_landmarking"
        long["name"] = long.model_id + "_" + custom_metric + "_" + long.summary.str.removeprefix("score_")
        features = pd.concat([features, long], ignore_index=True)
    required = IDENTITY + ["source", "name", "value", "config_id"]
    if not set(required) <= set(features) or features[IDENTITY].isna().any().any():
        raise ValueError("Incomplete feature identities")
    if "status" in features and features.status.eq("error").any():
        raise ValueError("Feature extraction failed")
    if features.groupby(["source", "name"]).config_id.nunique().gt(1).any():
        raise ValueError("Select one extractor configuration per feature before joining")
    if features.duplicated(IDENTITY + ["source", "name"]).any():
        raise ValueError("Duplicate split/feature records")
    expected_names = features.groupby("source").name.agg(set).to_dict()
    for (_, _, _, source), table in features.groupby(IDENTITY + ["source"]):
        if set(table.name) != expected_names[source]:
            raise ValueError("Missing feature records; undefined values must be explicit rows")
    # Every dataset must have the same split coverage for every selected feature namespace.
    observed = features.groupby(KEYS).apply(lambda table: {
        source: set(table.loc[table.source.eq(source), "split_id"]) for source in table.source.unique()}, include_groups=False)
    for coverage in observed:
        if len({frozenset(value) for value in coverage.values()}) > 1:
            raise ValueError("Feature namespaces have unequal training split coverage")
    features["column"] = "feature::" + features.source + "::" + features.name
    wide = features.pivot(index=IDENTITY, columns="column", values="value").reset_index()
    # Average training-split features into one observation, an explicit reconstruction assumption.
    columns = [name for name in wide if name.startswith("feature::")]
    aggregated = wide.groupby(KEYS, as_index=False)[columns].mean()
    result = gaps.merge(aggregated, on=KEYS, how="left", validate="one_to_one", indicator=True)
    if result._merge.ne("both").any():
        raise ValueError("Some gap datasets have no matching features; no silent row dropping")
    expected_sources = set(namespaces) | ({"custom_landmarking"} if custom is not None else set())
    present = features.groupby(KEYS).source.agg(set)
    for identity in map(tuple, result[KEYS].to_numpy()):
        if present.loc[identity] != expected_sources:
            raise ValueError("Incomplete feature namespaces for a dataset")
    result = result.drop(columns="_merge")
    result["feature_manifest_id"] = config_hash({source: sorted(group.config_id.unique().tolist())
                                                 for source, group in features.groupby("source")})
    result["feature_split_aggregation"] = "mean_over_training_splits"
    return result.sort_values(KEYS).reset_index(drop=True)
