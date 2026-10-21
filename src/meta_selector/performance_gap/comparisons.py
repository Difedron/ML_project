"""Explicit Table-3 comparisons; coverage restrictions are recorded, never implicit."""
from dataclasses import dataclass
from .families import (Family, MLP, TREE, NON_TFM, TFM, TABICL_V1, TABICL_V2,
                       TABPFN_V2, TABPFN_2_6, REALTABPFN_V2_5)


@dataclass(frozen=True)
class Comparison:
    id: str
    paper_id: str
    family_a: Family
    family_b: Family
    coverage: str = "strict"


COMPARISONS = {item.id: item for item in (
    Comparison("mlp_vs_tree", "C1", MLP, TREE),
    Comparison("non_tfm_vs_tfm", "C2", NON_TFM, TFM),
    Comparison("tabicl_v2_vs_tabpfn_2_6", "C3", TABICL_V2, TABPFN_2_6),
    Comparison("realtabpfn_v2_5_vs_tabpfn_2_6", "C4", REALTABPFN_V2_5, TABPFN_2_6, "complete_datasets"),
    Comparison("tabpfn_v2_vs_tabpfn_2_6", "C5", TABPFN_V2, TABPFN_2_6, "complete_datasets"),
    Comparison("tabicl_v1_vs_tabicl_v2", "C6", TABICL_V1, TABICL_V2, "complete_datasets"),
)}


def restrict_coverage(results, comparison):
    """Version checks retain whole datasets with complete A/B coverage of reference splits.

    This explicit conservative policy is a reconstruction, not a confirmed author subset.
    """
    from meta_selector.tabarena.results_loader import SPLIT_KEYS, OBSERVATION_KEYS
    if comparison.coverage == "strict":
        return results, []
    if comparison.coverage != "complete_datasets":
        raise ValueError("Unknown comparison coverage policy")
    excluded, keep = [], []
    for identity, group in results.groupby(OBSERVATION_KEYS):
        expected = set(map(tuple, group[SPLIT_KEYS].drop_duplicates().to_numpy()))
        coverage = [set(map(tuple, group[group.model_id.isin(family.members)][SPLIT_KEYS]
                              .drop_duplicates().to_numpy())) for family in (comparison.family_a, comparison.family_b)]
        if coverage[0] == coverage[1] == expected:
            keep.extend(group.index.tolist())
        else:
            excluded.append(dict(zip(OBSERVATION_KEYS, identity), reason="incomplete_family_split_coverage",
                                 expected_splits=len(expected), a_splits=len(coverage[0]), b_splits=len(coverage[1])))
    if not keep:
        raise ValueError("No complete datasets for this comparison")
    return results.loc[keep].copy(), excluded
