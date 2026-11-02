"""Inspect and prepare the Amharic handwriting dataset.

Reports dataset health, writes a train/val/test split manifest to
``data/manifests/unified.jsonl`` (per config ``pipeline.unified_manifest``),
and prints a summary. Run before training:

    python -m scripts.prepare_dataset
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import SETTINGS
from src.dataset import (
    load_line_records,
    split_char_dataset,
    validate_char_dataset,
    validate_line_dataset,
)
from src.vocabulary import Vocabulary


def main():
    print("=== Amharic OCR dataset preparation ===")
    vocab = Vocabulary.load(SETTINGS.models.cnn_checkpoint)
    print(f"Vocabulary: {vocab.size()} characters (source: {vocab.source})")

    # Character dataset.
    cv = validate_char_dataset()
    print(f"\n-- Character dataset: {SETTINGS.data.char_dataset_dir} --")
    print(f"  OK: {cv.ok}")
    print(f"  Classes: {cv.class_count}, images: {cv.total_images}")
    print(f"  Min/max per class: {cv.min_images}/{cv.max_images}")
    print(f"  Writer-separated split possible: {cv.writer_ids_found}")
    if cv.errors:
        print("  Errors:")
        for e in cv.errors:
            print("   -", e)
    if cv.missing_classes:
        print(f"  Classes with <10 samples: {len(cv.missing_classes)}")

    manifest = split_char_dataset(
        vocab=vocab,
        val_split=SETTINGS.training.val_split,
        test_split=SETTINGS.pipeline.test_split,
        seed=SETTINGS.pipeline.seed,
    )

    # Write unified manifest (train + val + test) to the configured path.
    manifest_path = Path(SETTINGS.pipeline.unified_manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    def _records(samples):
        return [{"image_path": str(p), "label": str(label), "task_type": "char"}
                for p, label in samples]

    all_records = (_records(manifest.train) +
                   _records(manifest.val) +
                   _records(manifest.test))
    with manifest_path.open("w", encoding="utf-8") as fh:
        for rec in all_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nSplit manifest written to {manifest_path}")
    print(f"  train={len(manifest.train)}, val={len(manifest.val)}, "
          f"test={len(manifest.test)}")
    print(f"  strategy={manifest.strategy} "
          f"(writer_separated={manifest.writer_separated})")

    # Line dataset.
    lv = validate_line_dataset()
    print(f"\n-- Line dataset: {SETTINGS.data.line_labels_file} --")
    print(f"  OK: {lv.ok}")
    print(f"  Records: {lv.total_images}")
    if lv.errors:
        for e in lv.errors:
            print("   -", e)
    if not lv.ok:
        print("  Create data/lines.jsonl (path\\ttext) and add line images to data/lines/")
        print("  See docs/DATASET.md.")

    print("\n=== Preparation complete ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())