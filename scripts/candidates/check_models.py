"""Stage 1 only: offline imports, exact API and device/resource preflight. Never fit."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from meta_selector.candidates.preflight import check_models
from meta_selector.paths import read_yaml_mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/candidates/preflight.yaml")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--tabpfn-checkpoint")
    parser.add_argument("--tabicl-checkpoint")
    parser.add_argument("--fetch-portfolio", action="store_true", help="Only the pinned small public lgbm/T/K25 JSON")
    parser.add_argument("--json", type=Path, help="Optional preflight report file")
    args = parser.parse_args()
    config = read_yaml_mapping(args.config)
    baseline = read_yaml_mapping(ROOT / "config/candidates/lightgbm_baseline.yaml")
    report = check_models(config, baseline, device=args.device, fetch=args.fetch_portfolio,
                          checkpoint_paths={"tabpfn": args.tabpfn_checkpoint, "tabicl": args.tabicl_checkpoint})
    for key, label in (("tabpfn", "TabPFN-2.6"), ("tabicl", "TabICLv2"), ("lightgbm", "LightGBM"), ("priortune", "PriorTune K=25")):
        item = report["models"][key]
        print(f"{label:20} {'READY' if item['ready'] else 'NOT READY'}  {item.get('reason', '')}")
    print(f"\nDevice: requested={report['requested_device']}, torch={report['selected_torch_device']}; LightGBM/PriorTune=cpu")
    print("Python:", report["python"])
    print("Torch:", json.dumps(report["torch"], ensure_ascii=False))
    print("Package versions:", json.dumps(report["package_versions"]))
    for key in ("tabpfn", "tabicl"):
        print(f"Checkpoint {key}:", report["models"][key]["checkpoint_identifier"], report["models"][key]["checkpoint"])
    portfolio = config["portfolio"]
    print("PriorTune portfolio source:", portfolio["repo_id"], portfolio["revision"], portfolio["path"])
    print("PriorTune selection metric:", json.dumps(portfolio["selection_metric_by_task"]))
    print("Authentication:", json.dumps(report["authentication"]))
    print("Readiness scope:", report["readiness_scope"])
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0 if report["all_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
