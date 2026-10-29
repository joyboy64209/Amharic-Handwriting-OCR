"""Validate a dataset manifest and print a data-quality report.

Reads a JSONL manifest (default: data/manifests/unified.jsonl) and reports:

* total / valid / rejected sample counts
* reject reasons (missing file, unreadable image, empty label, duplicate)
* class distribution and number of classes
* breakdown by source and task type

Usage (PowerShell, from the project root):
    python -m scripts.validate_manifest [manifest_path]
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_pipeline import print_quality_report, read_jsonl, validate_manifests


def main() -> int:
    manifest_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "data/manifests/unified.jsonl"
    )
    if not manifest_path.exists():
        print(f"[validate_manifest] Manifest not found: {manifest_path}")
        print("  Run: python -m scripts.download_datasets  (to generate manifests first)")
        return 1

    manifests = read_jsonl(manifest_path)
    if not manifests:
        print(f"[validate_manifest] Manifest is empty: {manifest_path}")
        return 1

    report = validate_manifests(manifests, check_files=True)
    print_quality_report(report, title=f"Manifest validation: {manifest_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
