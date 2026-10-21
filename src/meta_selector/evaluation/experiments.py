"""Executable experiment definitions with a shared exact-version predictor/protocol."""

EXPERIMENTS = {
    "E0": "Paper-style features; historical TabArena; TabPFN-2.6 gap regression",
    "E1": "E0 + explicitly admitted local datasets",
    "E2": "E0 + standard PyMFE landmarking",
    "E3": "E0 + five configured custom landmarks",
    "E4": "E0 + local datasets + custom landmarks",
    "E5": "E0 + local datasets + standard and custom landmarks",
}

from dataclasses import dataclass


@dataclass(frozen=True)
class Experiment:
    id: str
    include_local: bool
    include_standard: bool
    include_custom: bool

    @property
    def namespaces(self):
        return ["controls", "paper_features"] + (["pymfe_standard_landmarking"] if self.include_standard else [])


EXPERIMENT_SPECS = {item.id: item for item in (
    Experiment("E0", False, False, False), Experiment("E1", True, False, False),
    Experiment("E2", False, True, False), Experiment("E3", False, False, True),
    Experiment("E4", True, False, True), Experiment("E5", True, True, True),
)}


def prepare_experiment(spec, gaps, feature_tables, *, custom=None, custom_metric="balanced_accuracy"):
    from meta_selector.meta_dataset import build_meta_dataset
    selected = gaps[gaps.dataset_source.isin(["tabarena", "local"] if spec.include_local else ["tabarena"])].copy()
    if not selected.dataset_source.eq("tabarena").any():
        raise ValueError("Paper-style E0–E5 require original TabArena observations")
    if spec.include_local and not selected.dataset_source.eq("local").any():
        raise ValueError("This experiment requires explicitly benchmarked local datasets with candidate gaps")
    if spec.include_custom and custom is None:
        raise ValueError("Custom feature table is required")
    return build_meta_dataset(selected, feature_tables, namespaces=spec.namespaces,
                              custom=custom if spec.include_custom else None, custom_metric=custom_metric)
