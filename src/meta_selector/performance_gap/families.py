from dataclasses import dataclass


@dataclass(frozen=True)
class Family:
    name: str
    members: tuple[str, ...]

    def __post_init__(self):
        if not self.name or not self.members or len(self.members) != len(set(self.members)):
            raise ValueError("A family needs a name and distinct exact-version model IDs")


TABICL_V2 = Family("TabICLv2", ("tabicl_v2",))
TABPFN_2_6 = Family("TabPFN-2.6", ("tabpfn_2_6",))
MLP = Family("MLP-based", ("fastai", "modernnca", "nn_torch", "realmlp", "tabm"))
TREE = Family("Tree-based", ("catboost", "lightgbm", "xgboost", "ebm", "perpetualboost",
                             "extra_trees", "random_forest"))
NON_TFM = Family("non-TFM", MLP.members + TREE.members + ("knn", "logistic_regression"))
TFM = Family("TFM", ("tabdpt", "tabstar", "tabicl_v2", "tabicl_v1", "tabpfn_v2",
                      "tabpfn_2_6", "realtabpfn_v2_5"))
TABICL_V1 = Family("TabICL v1", ("tabicl_v1",))
TABPFN_V2 = Family("TabPFN v2", ("tabpfn_v2",))
REALTABPFN_V2_5 = Family("RealTabPFN v2.5", ("realtabpfn_v2_5",))
