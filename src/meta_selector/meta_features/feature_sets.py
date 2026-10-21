"""Future A/B/C/D design; A cannot be called an exact replication without its manifest."""
from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSet:
    name: str
    include_pymfe_landmarking: bool
    include_custom_landmarking: bool


FEATURE_SETS = (
    FeatureSet("A", False, False), FeatureSet("B", True, False),
    FeatureSet("C", False, True), FeatureSet("D", True, True),
)
PAPER_PYMFE_GROUPS = None  # NOT CONFIRMED; do not silently use PyMFE's default groups.

# Paper predictive variants, distinct from our A/B/C/D extension axis.
PAPER_PREDICTIVE_VARIANTS = ("controls", "all", "controls_plus_all", "robust", "controls_plus_robust")
