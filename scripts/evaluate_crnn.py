"""Evaluate the CRNN checkpoint on the held-out test split.

Reports CER and WER computed from real forward-pass predictions.  No metric
is claimed unless computed here from the actual test split.

Usage:
    python -m scripts.evaluate_crnn [checkpoint_path]
Defaults to models/crnn_amharic_line.pth.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

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
    images, texts, targets = zip(*batch)
    max_w = max(img.shape[-1] for img in images)
    padded_images = []
    for img in images:
        c, h, w = img.shape
        if w < max_w:
            pad = torch.full((c, h, max_w - w), 255.0, dtype=img.dtype)
            img = torch.cat([img, pad], dim=2)
        padded_images.append(img)

    input_lengths = torch.zeros(len(padded_images), dtype=torch.long)
    for i, img in enumerate(padded_images):
        w = img.shape[-1]
        w = (w - 2) // 2 + 1
        w = (w - 2) // 2 + 1
        w = (w - 2) // 2 + 1
        w = (w - 2) // 2 + 1
        input_lengths[i] = w

    target_lengths = torch.tensor([t.shape[0] for t in targets], dtype=torch.long)
    targets_padded = torch.nn.utils.rnn.pad_sequence(
        targets, batch_first=True, padding_value=0
    )
    images = torch.stack(padded_images, dim=0)
    return images, texts, targets_padded, input_lengths, target_lengths


def evaluate(model, loader, device, vocab):
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

            log_probs = model(images)
            log_probs_t = log_probs.transpose(0, 1)
            loss = ctc_loss(
                log_probs_t, targets, input_lengths, target_lengths, blank=0
            )
            total_loss += loss.item()
            n_batches += 1

            decoded = greedy_decode(log_probs, blank=0)
            for i, chars in enumerate(decoded):
                hyp_text = "".join(vocab.char_for_index(c) for c in chars)
                references.append(texts[i])
                hypotheses.append(hyp_text)

    avg_loss = total_loss / max(n_batches, 1)
    cer, wer = cer_wer_list(references, hypotheses)
    return avg_loss, cer, wer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_arg = sys.argv[1] if len(sys.argv) > 1 else None

    cfg = SETTINGS
    vocab = Vocabulary.load(cfg.models.cnn_checkpoint)
    num_classes = vocab.size()
    print(f"[evaluate_crnn] Vocabulary size: {num_classes} (source: {vocab.source})")

    # Validate the line dataset.
    labels_file = cfg.data.line_labels_file
    images_dir = cfg.data.line_images_dir
    vr = validate_line_dataset(labels_file, images_dir)
    if not vr.ok:
        print("[evaluate_crnn] Line dataset validation failed:")
        for e in vr.errors:
            print("   -", e)
        print("[evaluate_crnn] Aborting. See docs/DATASET.md.")
        return 1

    records = load_line_records(labels_file)
    if not records:
        print("[evaluate_crnn] No line records found. Aborting.")
        return 1

    full_ds = LineTextDataset(
        records,
        vocab,
        img_height=cfg.preprocessing.img_height,
        mean=cfg.preprocessing.mean,
        std=cfg.preprocessing.std,
        images_dir=images_dir,
    )

    # Use the same deterministic split as training so test matches what
    # train_crnn.py held out.
    test_split = 0.1
    val_split = cfg.training.val_split
    n_test = max(1, int(len(full_ds) * test_split))
    n_val = max(1, int(len(full_ds) * val_split))
    n_train = len(full_ds) - n_val - n_test
    if n_train < 1:
        n_val = max(1, (len(full_ds) - 1) // 2)
        n_test = max(1, len(full_ds) - 1 - n_val)
        n_train = len(full_ds) - n_val - n_test

    rng = torch.Generator().manual_seed(cfg.training.seed)
    perm = torch.randperm(len(full_ds), generator=rng).tolist()
    test_idx = perm[:n_test]

    from torch.utils.data import Subset
    test_ds = Subset(full_ds, test_idx)
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    print(f"[evaluate_crnn] Test samples: {len(test_ds)}")

    # Pick checkpoint.
    candidates = []
    if ckpt_arg:
        candidates.append(Path(ckpt_arg))
    default_ckpt = cfg.models.crnn_checkpoint
    if default_ckpt.exists():
        candidates.append(default_ckpt)
    if not candidates:
        print("[evaluate_crnn] No CRNN checkpoint found. Train first with "
              "scripts/train_crnn.py or supply a checkpoint path.")
        return 1
    ckpt = candidates[0]
    print(f"[evaluate_crnn] Using checkpoint: {ckpt}")

    model = CRNN(
        num_classes=num_classes,
        img_height=cfg.preprocessing.img_height,
    ).to(device)
    state = torch.load(ckpt, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()

    test_loss, test_cer, test_wer = evaluate(model, test_loader, device, vocab)
    print(f"[evaluate_crnn] TEST loss={test_loss:.4f}")
    print(f"[evaluate_crnn] TEST CER={test_cer:.4f}")
    print(f"[evaluate_crnn] TEST WER={test_wer:.4f}")
    print("[evaluate_crnn] Metrics are computed on the held-out test split.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())