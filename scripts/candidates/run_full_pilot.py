"""Stage 7: run or audit the ten shared PMLB datasets."""
from pathlib import Path

from _isolated_candidates import run_cli
from _full_pilot_artifacts import FullPilotArtifacts


CONTRACTS = {
    "pmlb_saheart": (462, 9, 2, 276, 93, 93),
    "pmlb_tic_tac_toe": (958, 9, 2, 574, 192, 192),
    "pmlb_kr_vs_kp": (3196, 36, 2, 1917, 639, 640),
    "pmlb_mushroom": (8124, 22, 2, 4874, 1625, 1625),
    "pmlb_hill_valley_without_noise": (1212, 100, 2, 726, 243, 243),
    "pmlb_twonorm": (7400, 20, 2, 4440, 1480, 1480),
    "pmlb_penguins": (333, 7, 3, 199, 67, 67),
    "pmlb_car_evaluation": (1728, 6, 4, 1036, 346, 346),
    "pmlb_vehicle": (846, 18, 4, 507, 169, 170),
    "pmlb_waveform_21": (5000, 21, 3, 3000, 1000, 1000),
}


def main(argv=None):
    return run_cli(Path(__file__).resolve(), __doc__, CONTRACTS, "candidate_full_pilot",
                   explicit_dataset=True, argv=argv, full_batch=FullPilotArtifacts)


if __name__ == "__main__":
    raise SystemExit(main())
