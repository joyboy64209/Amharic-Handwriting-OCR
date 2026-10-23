"""Train the CRNN sequence recognizer with CTC loss for short Amharic text lines.

Reads labeled line images from ``data/lines.jsonl`` (format:
``{"path": "data/lines/line_00001.png", "text": "ሰላም"}``), splits them into
train/validation, trains a CRNN with CTC loss, and saves a checkpoint to
``models/crnn_amharic_line.pth``.

Generate synthetic line data first if you have no real labels:
    python scripts/generate_synthetic_lines.py --num-lines 2000 --img-height 28 --seed 42

Usage (PowerShell, from the project root):
    python -m scripts.train_crnn
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import SETTINGS
from src.dataset import (
    LineTextDataset,
    load_line_records,
    validate_line_dataset,
)
from src.metrics import cer_wer_list
from src.models.crnn import CRNN, ctc_loss, greedy_decode
from src.vocabulary import Vocabulary


def collate_fn(batch):
    """Collate variable-width line images into a padded batch for CTC.

    Each item: (image_tensor, text, target_indices) where target_indices
    already has +1 offset (0 = CTC blank).
    """
    images, texts, targets = zip(*batch)
    # Pad images to the max width in the batch (pad right with white/255).
    max_w = max(img.shape[-1] for img in images)
    padded_images = []
    for img in images:
        c, h, w = img.shape
        if w < max_w:
            pad = torch.full((c, h, max_w - w), 255.0, dtype=img.dtype)
            img = torch.cat([img, pad], dim=2)
        padded_images.append(img)

    # Compute actual timesteps per sequence after the CRNN feature extractor.
    # The conv front end applies 3 x 2x2 pooling; widths that are not
    # divisible by 8 get floored by each step, so we compute the exact
    # output width per image with integer arithmetic using the same
    # formula as the CRNN blocks.
    input_lengths = torch.zeros(len(padded_images), dtype=torch.long)
    for i, img in enumerate(padded_images):
        w = img.shape[-1]
        # block1: 2x2 pooling -> (w - 2) // 2 + 1
        w = (w - 2) // 2 + 1
        # block2: 2x2 pooling
        w = (w - 2) // 2 + 1
        # block3: 2x2 pooling
        w = (w - 2) // 2 + 1
        # block4: 2x2 pooling
        w = (w - 2) // 2 + 1
        input_lengths[i] = w
    input_lengths = input_lengths  # (B,)

    target_lengths = torch.tensor([t.shape[0] for t in targets], dtype=torch.long)
    targets_padded = torch.nn.utils.rnn.pad_sequence(
        targets, batch_first=True, padding_value=0
    )  # (B, T_max)

    images = torch.stack(padded_images, dim=0)  # (B, 1, H, W_max)
    return images, texts, targets_padded, input_lengths, target_lengths


def train_one_epoch(model, loader, optimizer, device, vocab_size):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for images, _, targets, input_lengths, target_lengths in loader:
        images = images.to(device)
        targets = targets.to(device)
        input_lengths = input_lengths.to(device)
        target_lengths = target_lengths.to(device)

        optimizer.zero_grad()
        log_probs = model(images)  # (B, T, vocab_size+1), log_softmax
        # CTCLoss expects (T, B, C)
        log_probs = log_probs.transpose(0, 1)
        loss = ctc_loss(
            log_probs, targets, input_lengths, target_lengths, blank=0
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def evaluate(model, loader, device, vocab):
    """Compute validation loss, CER, and WER from a real forward pass."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    references = []
    hypotheses = []
    with torch.no_grad():
        for images, texts, targets, input_lengths, target_lengths in loader:
            images = images.to(device)
            targets = targets.to(device)
            input_lengths = input_lengths.to(device)
            target_lengths = target_lengths.to(device)

            log_probs = model(images)  # (B, T, C)
            log_probs_t = log_probs.transpose(0, 1)
            loss = ctc_loss(
                log_probs_t, targets, input_lengths, target_lengths, blank=0
            )
            total_loss += loss.item()
            n_batches += 1

            # Greedy decode each item in the batch.
            decoded = greedy_decode(log_probs, blank=0)
            for i, chars in enumerate(decoded):
                hyp_text = "".join(vocab.char_for_index(c) for c in chars)
                references.append(texts[i])
                hypotheses.append(hyp_text)

    avg_loss = total_loss / max(n_batches, 1)
    cer, wer = cer_wer_list(references, hypotheses)
    return avg_loss, cer, wer


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
    num_classes = vocab.size()
    print(f"[train_crnn] Line dataset valid: {len(records)} records, "
          f"vocab={num_classes} chars ({vocab.source})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_crnn] Device: {device}")

    # Build dataset and split.
    full_ds = LineTextDataset(
        records,
        vocab,
        img_height=SETTINGS.preprocessing.img_height,
        mean=SETTINGS.preprocessing.mean,
        std=SETTINGS.preprocessing.std,
        images_dir=images_dir,
    )
    if len(full_ds) == 0:
        print("[train_crnn] No records in line dataset. Aborting.")
        return 1

    n_val = max(1, int(len(full_ds) * SETTINGS.training.val_split))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(SETTINGS.training.seed),
    )
    print(f"[train_crnn] Split: train={len(train_ds)}, val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=SETTINGS.training.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=SETTINGS.training.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # Build model.
    model = CRNN(
        num_classes=num_classes,
        img_height=SETTINGS.preprocessing.img_height,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=SETTINGS.training.lr)

    # Train.
    best_val_cer = float("inf")
    best_path = Path("models/crnn_amharic_line.pth")
    best_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for epoch in range(1, SETTINGS.training.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, num_classes)
        val_loss, val_cer, val_wer = evaluate(model, val_loader, device, vocab)
        print(f"[train_crnn] Epoch {epoch:02d}/{SETTINGS.training.epochs} "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_cer={val_cer:.4f} val_wer={val_wer:.4f}")

        if val_cer < best_val_cer:
            best_val_cer = val_cer
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_to_idx": {ch: i for i, ch in vocab.index_to_char.items()},
                "input_shape": (1, SETTINGS.preprocessing.img_height, SETTINGS.preprocessing.img_width),
                "model_architecture": "CRNN_CTC",
                "final_cer": best_val_cer,
                "final_wer": val_wer,
                "val_split": SETTINGS.training.val_split,
                "seed": SETTINGS.training.seed,
                "epoch": epoch,
                "num_lines": len(records),
            }, str(best_path))
            print(f"   -> new best saved to {best_path} (val_cer={best_val_cer:.4f})")

    dt = time.time() - t0
    print(f"[train_crnn] Done in {dt:.1f}s. Best val CER: {best_val_cer:.4f}")
    print(f"[train_crnn] Best checkpoint: {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())