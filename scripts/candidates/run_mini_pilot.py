"""Stage 6: run only the three fixed shared PMLB datasets."""
from pathlib import Path

from _isolated_candidates import run_cli


CONTRACTS = {
    "pmlb_tic_tac_toe": (958, 9, 2, 574, 192, 192),
    "pmlb_penguins": (333, 7, 3, 199, 67, 67),
    "pmlb_hill_valley_without_noise": (1212, 100, 2, 726, 243, 243),
}


def main(argv=None):
    return run_cli(Path(__file__).resolve(), __doc__, CONTRACTS, "candidate_mini_pilot", explicit_dataset=True, argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
