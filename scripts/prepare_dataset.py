"""Inspect and prepare the Amharic handwriting dataset.

Reports dataset health, writes a train/val split manifest to
data/split_manifest.json, and prints a summary. Run before training:

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

    manifest = split_char_dataset(vocab=vocab)
    manifest_path = Path("data/split_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "strategy": manifest.strategy,
        "writer_separated": manifest.writer_separated,
        "num_classes": manifest.num_classes,
        "num_train": len(manifest.train),
        "num_val": len(manifest.val),
        "val_split": SETTINGS.training.val_split,
        "seed": SETTINGS.training.seed,
        "train": [str(p) + "\t" + str(label) for p, label in manifest.train],
        "val": [str(p) + "\t" + str(label) for p, label in manifest.val],
    }
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"\nSplit manifest written to {manifest_path}")
    print(f"  train={len(manifest.train)}, val={len(manifest.val)}")

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