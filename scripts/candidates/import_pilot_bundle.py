"""Stage 2 only: import verified Rita mapping/splits; no candidate/model execution."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from meta_selector.datasets.pilot_import import import_pilot_bundle
from meta_selector.datasets.shared_pilot import validate_pilot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "incoming/pmlb_pilot_for_diana.zip")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--check-only", action="store_true", help="Read canonical contract and current CSV; no import")
    parser.add_argument("--report", type=Path, help="Optional service JSON report under the project's .cache/")
    args = parser.parse_args()
    if args.report and not args.report.resolve().is_relative_to((args.project_root / ".cache").resolve()):
        parser.error("--report must stay under .cache/; raw/config/source artifacts must not be overwritten")
    try:
        report = (dict(status="PASS", datasets=validate_pilot(project_root=args.project_root)) if args.check_only
                  else import_pilot_bundle(args.archive, project_root=args.project_root))
        print("dataset_id | CSV | rows | features | classes | train | val | test | seed | split_id | status")
        for row in report["datasets"]:
            print(" | ".join(str(row[key]) for key in ("dataset_id", "csv", "rows", "features", "classes",
                                                       "train", "validation", "test", "seed", "split_id", "status")))
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return 0
    except Exception as error:
        print(f"FAIL: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
