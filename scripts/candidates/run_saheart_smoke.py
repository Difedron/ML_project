"""Stage 5: run the four SAheart candidates in separate native processes."""
from pathlib import Path

from _isolated_candidates import run_cli


def main(argv=None):
    return run_cli(Path(__file__).resolve(), __doc__, {"pmlb_saheart": (462, 9, 2, 276, 93, 93)},
                   "candidate_smoke", explicit_dataset=False, argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
