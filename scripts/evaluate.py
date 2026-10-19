"""Evaluate the CNN baseline on the validation split and report real metrics.

Computes accuracy, precision, recall, F1 (macro + per-class), and a confusion
matrix from an actual evaluation run. No metric is claimed unless computed here.

Usage:
    python -m scripts.evaluate [checkpoint_path]
Defaults to models/amharic_ocr_v2_trained.pth, falling back to the upstream
models/amharic_ocr_v2.pth if present.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from src.config import SETTINGS
from src.dataset import split_char_dataset, CharImageDataset, validate_char_dataset
from src.models.cnn import build_model
from src.vocabulary import Vocabulary
from src.metrics import classification_metrics


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_arg = sys.argv[1] if len(sys.argv) > 1 else None

    cfg = SETTINGS
    vocab = Vocabulary.load(cfg.models.cnn_checkpoint)
    num_classes = vocab.size()
    labels = [vocab.char_for_index(i) for i in range(num_classes)]

    # Validation dataset health.
    vr = validate_char_dataset()
    if not vr.ok:
        print("[evaluate] Dataset validation failed:")
        for e in vr.errors:
            print("   -", e)
        print("[evaluate] Aborting. See docs/DATASET.md.")
        return 1

    manifest = split_char_dataset(vocab=vocab, val_split=cfg.training.val_split,
                                  seed=cfg.training.seed)
    val_ds = CharImageDataset(
        manifest.val,
        img_height=cfg.preprocessing.img_height,
        img_width=cfg.preprocessing.img_width,
        mean=cfg.preprocessing.mean,
        std=cfg.preprocessing.std,
        augment=False,
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size,
                            shuffle=False, num_workers=0)
    print(f"[evaluate] Split: {manifest.strategy} "
          f"(writer_separated={manifest.writer_separated}), "
          f"val samples={len(val_ds)}")

    # Pick checkpoint.
    candidates = []
    if ckpt_arg:
        candidates.append(Path(ckpt_arg))
    trained = Path("models/amharic_ocr_v2_trained.pth")
    upstream = cfg.models.cnn_checkpoint
    if trained.exists():
        candidates.append(trained)
    if upstream.exists():
        candidates.append(upstream)
    if not candidates:
        print("[evaluate] No checkpoint found. Train first with "
              "scripts/train_cnn.py or supply a checkpoint path.")
        return 1
    ckpt = candidates[0]
    print(f"[evaluate] Using checkpoint: {ckpt}")

    model, metadata = build_model(num_classes=num_classes,
                                  checkpoint_path=str(ckpt),
                                  device=str(device))
    if metadata.get("final_accuracy") is not None:
        print(f"[evaluate] Checkpoint recorded upstream accuracy: "
              f"{metadata['final_accuracy']} (upstream run, not recomputed)")

    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    n = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels_batch in val_loader:
            images = images.to(device)
            labels_batch = labels_batch.to(device)
            logits = model(images)
            loss = criterion(logits, labels_batch)
            total_loss += loss.item() * images.size(0)
            n += images.size(0)
            pred = logits.argmax(dim=1)
            y_true.extend(labels_batch.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())

    avg_loss = total_loss / max(n, 1)
    report = classification_metrics(y_true, y_pred, num_classes, labels=labels)

    print(f"[evaluate] val_loss={avg_loss:.4f}")
    print(f"[evaluate] accuracy      = {report.accuracy:.4f}")
    print(f"[evaluate] precision     = {report.precision_macro:.4f} (macro)")
    print(f"[evaluate] recall        = {report.recall_macro:.4f} (macro)")
    print(f"[evaluate] f1            = {report.f1_macro:.4f} (macro)")
    print(f"[evaluate] Confusion matrix ({num_classes}x{num_classes}):")
    # Print a compact diagonal summary and top confusions.
    cm = report.confusion_matrix
    diag = [cm[i][i] for i in range(num_classes)]
    print(f"   correct per class (top 8): "
          f"{sorted([(diag[i], labels[i]) for i in range(num_classes)], reverse=True)[:8]}")
    # Top off-diagonal confusions.
    confusions = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i][j] > 0:
                confusions.append((cm[i][j], labels[i], labels[j]))
    confusions.sort(reverse=True)
    if confusions:
        print("   top confusions (true->pred: count):")
        for cnt, a, b in confusions[:10]:
            print(f"      {a} -> {b}: {cnt}")
    print(f"[evaluate] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())