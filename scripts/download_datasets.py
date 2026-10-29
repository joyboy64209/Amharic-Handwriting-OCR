"""Download and prepare all configured Amharic OCR dataset sources.

Reads ``data_sources`` from ``config.yaml`` and, for each enabled source:

* ``local_char``  - validates the local NumPy files exist and converts them
  into PNG images + a JSONL manifest (no download needed).
* ``ahld29k``     - streams the HF parquet via the ``datasets`` library and
  writes extracted PNG images + manifest (streaming, low memory).
* ``fidel``       - downloads the HF repo (LFS-tracked ZIPs) via
  ``huggingface_hub``, reads JSON labels, copies selected images into the
  pipeline image dir, and writes a manifest.

Original datasets are **never modified**.  Processed outputs live under
``data/``.

Usage (PowerShell, from the project root):
    python -m scripts.download_datasets
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import SETTINGS
from src.data_pipeline import (
    build_manifests,
    print_quality_report,
    validate_manifests,
    write_jsonl,
    write_unified_manifest,
)


def main() -> int:
    # data_sources lives in config.yaml (not in the typed Settings dataclass),
    # so re-read the raw YAML for source configuration.
    raw = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8")) or {}
    sources_cfg = raw.get("data_sources", {})

    pipeline_cfg = raw.get("pipeline", {})
    unified_path = Path(
        pipeline_cfg.get("unified_manifest", "data/manifests/unified.jsonl")
    )

    per_source = build_manifests(sources_cfg, only_enabled=True)

    total_written = 0
    for name, manifests in per_source.items():
        manifest_path = Path(sources_cfg[name]["manifest_file"])
        write_jsonl(manifests, manifest_path)
        report = validate_manifests(manifests, check_files=True)
        print_quality_report(report, title=f"{name} manifest")
        total_written += len(manifests)

    unified = write_unified_manifest(per_source, unified_path)
    report = validate_manifests(unified, check_files=True)
    print_quality_report(report, title="unified manifest")

    print(f"[download_datasets] Wrote {total_written} per-source samples + "
          f"{len(unified)} unified entries.")
    print(f"[download_datasets] Unified manifest: {unified_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
