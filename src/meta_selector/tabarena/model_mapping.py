"""Exact, explicit version mapping. Never match a TabPFN generation by prefix."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelIdentity:
    model_id: str
    version: str | None


@dataclass(frozen=True)
class ModelDefinition:
    canonical_id: str
    ta_name: str
    config_type: str
    registry_name: str
    compute: str
    dependencies: tuple[str, ...]
    version: str | None = None
    checkpoint_required: bool = False


# Historical identities come from the pinned, verified aggregate artifacts.
# registry_name is the exact official display_name/method resolved by TabArena utils.
MODEL_DEFINITIONS = (
    ModelDefinition("fastai", "NeuralNetFastAI", "FASTAI", "FastaiMLP", "cpu", ("fastai",)),
    ModelDefinition("modernnca", "ModernNCA_GPU", "MNCA_GPU", "ModernNCA", "gpu", ("torch", "category_encoders")),
    ModelDefinition("nn_torch", "NeuralNetTorch", "NN_TORCH", "TorchMLP", "cpu", ("torch",)),
    ModelDefinition("realmlp", "RealMLP_GPU", "REALMLP_GPU", "RealMLP", "gpu", ("torch", "pytabkit")),
    ModelDefinition("tabm", "TabM_GPU", "TABM_GPU", "TabM", "gpu", ("torch", "tabm", "rtdl_num_embeddings")),
    ModelDefinition("catboost", "CatBoost", "CAT", "CatBoost", "cpu", ("catboost",)),
    ModelDefinition("lightgbm", "LightGBM", "GBM", "LightGBM", "cpu", ("lightgbm",)),
    ModelDefinition("xgboost", "XGBoost", "XGB", "XGBoost", "cpu", ("xgboost",)),
    ModelDefinition("ebm", "ExplainableBM", "EBM", "EBM", "cpu", ("interpret",)),
    ModelDefinition("perpetualboost", "PerpetualBooster", "PB", "PerpetualBooster", "cpu", ("perpetual",)),
    ModelDefinition("extra_trees", "ExtraTrees", "XT", "ExtraTrees", "cpu", ("sklearn",)),
    ModelDefinition("random_forest", "RandomForest", "RF", "RandomForest", "cpu", ("sklearn",)),
    ModelDefinition("knn", "KNeighbors", "KNN", "KNN", "cpu", ("sklearn",)),
    ModelDefinition("logistic_regression", "LinearModel", "LR", "Linear", "cpu", ("sklearn",)),
    ModelDefinition("tabdpt", "TabDPT_GPU", "TABDPT_GPU", "TabDPT", "gpu", ("torch", "tabdpt"), checkpoint_required=True),
    ModelDefinition("tabstar", "TabSTAR", "TABSTAR", "TabSTAR", "gpu", ("torch", "tabstar"), checkpoint_required=True),
    ModelDefinition("tabicl_v1", "TabICL_GPU", "TABICL_GPU", "TabICL", "gpu", ("torch", "tabicl"), "v1", True),
    ModelDefinition("tabicl_v2", "TabICLv2", "TABICLV2", "TabICLv2", "gpu", ("torch", "tabicl"), "v2", True),
    ModelDefinition("tabpfn_v2", "TabPFNv2_GPU", "TABPFNV2_GPU", "TabPFNv2", "gpu", ("torch", "tabpfn"), "v2", True),
    ModelDefinition("tabpfn_2_6", "TabPFN-v2.6", "TABPFN-V2.6", "TabPFN-2.6", "gpu", ("torch", "tabpfn"), "2.6", True),
    ModelDefinition("realtabpfn_v2_5", "RealTabPFN-v2.5", "REALTABPFN-V2.5", "RealTabPFN-2.5", "gpu", ("torch", "tabpfn"), "2.5", True),
)
MODELS = {model.canonical_id: model for model in MODEL_DEFINITIONS}
_EXACT = {(model.ta_name, model.config_type): ModelIdentity(model.canonical_id, model.version)
          for model in MODEL_DEFINITIONS}


def identify_model(ta_name: str, config_type: str) -> ModelIdentity:
    """Unknown versions keep their exact name, with unknown version; no replacement."""
    return _EXACT.get((ta_name, config_type), ModelIdentity(ta_name, None))


def exact_identifier(ta_suite: str, ta_name: str, method: str) -> str:
    return "::".join((ta_suite, ta_name, method))
