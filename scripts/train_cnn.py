"""Train the CNN baseline for isolated Amharic characters.

Usage (PowerShell, from the project root):
    python -m scripts.train_cnn

Reads config.yaml. Splits data/uni_dataset (writer-separated when writer ids
are present in filenames), trains, and saves:
    models/amharic_ocr_v2_trained.pth
with model_state_dict, class_to_idx (explicit vocabulary), input_shape,
and validation metrics computed from a real run.

No metrics are claimed unless computed here from the actual val split.
"""

from __future__ import annotations

import os
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import SETTINGS
from src.dataset import CharImageDataset, split_char_dataset, validate_char_dataset
from src.metrics import classification_metrics
from src.models.cnn import AHCNN, build_model
from src.vocabulary import Vocabulary


def atomic_torch_save(obj: dict, path: Path) -> None:
    """Atomically save a PyTorch object to ``path``.

    Writes to a temporary file in the same directory, then renames.  This
    prevents a partial checkpoint from being left behind if the process is
    interrupted during ``torch.save``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as fh:
            torch.save(obj, fh)
        os.replace(tmp_name, str(path))
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        n += images.size(0)
    return total_loss / max(n, 1)

def evaluate(model, loader, criterion, device, num_classes, labels):
    model.eval()
    total_loss = 0.0
    n = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels_batch in loader:
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
    return avg_loss, report

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_cnn] Device: {device}")

    cfg = SETTINGS
    # Deterministic seeding for reproducibility.
    seed = int(cfg.training.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[train_cnn] Seed: {seed}")
    vocab = Vocabulary.load(cfg.models.cnn_checkpoint)
    num_classes = vocab.size()
    print(f"[train_cnn] Vocabulary size: {num_classes} (source: {vocab.source})")

    val_report = validate_char_dataset()
    if not val_report.ok:
        print("[train_cnn] Dataset validation failed:")
        for e in val_report.errors:
            print("   -", e)
        print("[train_cnn] Aborting. See docs/DATASET.md to provide data.")
        return 1
    print(f"[train_cnn] Dataset: {val_report.class_count} classes, "
          f"{val_report.total_images} images, "
          f"min/max per class: {val_report.min_images}/{val_report.max_images}")
    if val_report.missing_classes:
        print(f"[train_cnn] Warning: classes with <10 samples: "
              f"{val_report.missing_classes[:8]}...")

    # Small-dataset guard: need at least 3 samples for train/val/test.
    if val_report.total_images < 3:
        print("[train_cnn] Error: dataset has fewer than 3 images; "
              "cannot create train/val/test splits.")
        print("[train_cnn] Aborting. Add more data or reduce split ratios.")
        return 1

    manifest = split_char_dataset(
        vocab=vocab,
        val_split=cfg.training.val_split,
        seed=cfg.training.seed,
    )
    # Validate labels against the vocabulary before training.
    unknown: set = set()
    for path, label_idx in list(manifest.train) + list(manifest.val):
        ch = vocab.char_for_index(int(label_idx))
        if ch == "?":
            unknown.add(f"index {label_idx}")
    if unknown:
        print("[train_cnn] Error: labels not in vocabulary:")
        for u in sorted(unknown)[:10]:
            print("   -", u)
        print("[train_cnn] Aborting. Update labels or vocabulary before "
              "training.")
        return 1

    print(f"[train_cnn] Split: {manifest.strategy} "
          f"(writer_separated={manifest.writer_separated}), "
          f"test_split handled by prepare_dataset (3-way split)")
    print(f"[train_cnn] train={len(manifest.train)}  val={len(manifest.val)}")

    train_ds = CharImageDataset(
        manifest.train,
        img_height=cfg.preprocessing.img_height,
        img_width=cfg.preprocessing.img_width,
        mean=cfg.preprocessing.mean,
        std=cfg.preprocessing.std,
        augment=True,
        augment_cfg={
            "rotation_degrees": cfg.training.rotation_degrees,
            "scale": cfg.training.scale,
            "shear_degrees": cfg.training.shear_degrees,
            "brightness": cfg.training.brightness,
        },
    )
    val_ds = CharImageDataset(
        manifest.val,
        img_height=cfg.preprocessing.img_height,
        img_width=cfg.preprocessing.img_width,
        mean=cfg.preprocessing.mean,
        std=cfg.preprocessing.std,
        augment=False,
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size,
                            shuffle=False, num_workers=0)

    labels = [vocab.char_for_index(i) for i in range(num_classes)]

    start_from = cfg.models.cnn_checkpoint
    model, metadata = build_model(
        num_classes=num_classes,
        checkpoint_path=str(start_from) if start_from.exists() else None,
        device=str(device),
    )
    if metadata.get("final_accuracy") is not None:
        print(f"[train_cnn] Checkpoint recorded upstream accuracy: "
              f"{metadata['final_accuracy']} (upstream run, not recomputed)")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, verbose=True
    )

    best_val_acc = 0.0
    best_path = Path("models/amharic_ocr_v2_trained.pth")
    best_path.parent.mkdir(parents=True, exist_ok=True)
    early_stop_patience = 5
    epochs_since_improvement = 0

    t0 = time.time()
    for epoch in range(1, cfg.training.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, report = evaluate(model, val_loader, criterion, device,
                                    num_classes, labels)
        print(f"[train_cnn] Epoch {epoch:02d}/{cfg.training.epochs} "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_acc={report.accuracy:.4f} val_f1={report.f1_macro:.4f}")
        if report.accuracy > best_val_acc:
            best_val_acc = report.accuracy
            epochs_since_improvement = 0
            atomic_torch_save({
                "model_state_dict": model.state_dict(),
                "class_to_idx": {ch: i for i, ch in vocab.index_to_char.items()},
                "input_shape": (1, cfg.preprocessing.img_height, cfg.preprocessing.img_width),
                "model_architecture": "AHCNN",
                "final_accuracy": best_val_acc,
                "val_split": cfg.training.val_split,
                "split_strategy": manifest.strategy,
                "writer_separated": manifest.writer_separated,
                "seed": seed,
                "epoch": epoch,
                "num_classes": num_classes,
                "preprocessing": {
                    "img_height": cfg.preprocessing.img_height,
                    "img_width": cfg.preprocessing.img_width,
                    "mean": cfg.preprocessing.mean,
                    "std": cfg.preprocessing.std,
                },
            }, best_path)
            print(f"   -> new best saved to {best_path} (val_acc={best_val_acc:.4f})")
        else:
            epochs_since_improvement += 1
        scheduler.step(report.accuracy)
        if epochs_since_improvement >= early_stop_patience:
            print(f"[train_cnn] Early stopping: no improvement for "
                  f"{early_stop_patience} epochs.")
            break

    dt = time.time() - t0
    print(f"[train_cnn] Done in {dt:.1f}s. Best val accuracy: {best_val_acc:.4f}")
    print(f"[train_cnn] Best checkpoint: {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())