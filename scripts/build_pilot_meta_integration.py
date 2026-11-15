"""Validate saved pilot inputs and write the dataset-level stage-8 artifacts."""
import _bootstrap  # noqa: F401
from meta_selector.integration.pilot import build_integration


def main():
    manifest = build_integration()
    print(f"Stage 8 integration: {manifest['rows']} rows × {manifest['columns']} columns")
    print(f"Rita export integrity: {manifest['rita']['artifact_integrity']}")
    print(f"Rita raw SHA provenance: {manifest['rita']['raw_sha_provenance']['status']}")
    print(f"Impact on integration: {manifest['rita']['raw_sha_provenance']['impact_on_integration']}")
    print(f"Leakage guard: {manifest['leakage_guard']}")


if __name__ == "__main__":
    main()
