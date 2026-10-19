"""Train the CRNN sequence recognizer with CTC loss for short Amharic text lines.

NOTE: The upstream project (and this new one) does NOT include labeled text-line
data. Train this model only after you have collected line images and written
data/lines.jsonl (see docs/DATASET.md). Until then, this script validates the
line dataset and exits with a clear message.

Usage:
    python -m scripts.train_crnn
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import SETTINGS
from src.dataset import validate_line_dataset, load_line_records
from src.vocabulary import Vocabulary


def main():
    print("[train_crnn] CRNN sequence recognizer (CTC) for Amharic text lines")
    labels_file = SETTINGS.data.line_labels_file
    images_dir = SETTINGS.data.line_images_dir

    vr = validate_line_dataset(labels_file, images_dir)
    if not vr.ok:
        print("[train_crnn] Line dataset not ready:")
        for e in vr.errors:
            print("   -", e)
        print()
        print("To proceed:")
        print("  1. Collect scanned/photographed line images into data/lines/")
        print("  2. Create data/lines.jsonl: each line is 'relative_path\\tAmharic text'")
        print("     Example: lines/001.png\\tሰላም")
        print("  3. Re-run: python -m scripts.train_crnn")
        print("See docs/DATASET.md for the annotation format and collection instructions.")
        return 1

    records = load_line_records(labels_file)
    vocab = Vocabulary.load(SETTINGS.models.cnn_checkpoint)
    print(f"[train_crnn] Line dataset valid: {len(records)} records, "
          f"vocab={vocab.size()} chars ({vocab.source})")
    print("[train_crnn] CRNN training loop not yet wired (no labeled-line "
          "checkpoint format defined). Implement training in this script once "
          "line data is available, reusing src/models/crnn.py and src/metrics.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())