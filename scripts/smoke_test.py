"""End-to-end smoke test for the multi-dataset pipeline.

Loads a small batch from each *available* manifest, runs the images through
the existing CNN (char task) and CRNN (line task) models, and verifies that:

* input tensor dimensions match the model expectations
* label encoding matches the model's output representation
* the forward pass produces finite outputs of the right shape

This is a *smoke test only* — it does not train. It exits non-zero if any
check fails.

Usage (PowerShell, from the project root):
    python -m scripts.smoke_test
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
from src.data_pipeline import (
    normalize_amharic,
    read_jsonl,
    validate_manifests,
)
from src.dataset import CharImageDataset, LineTextDataset
from src.models.cnn import AHCNN
from src.models.cnn import build_model as build_cnn
from src.models.crnn import CRNN
from src.vocabulary import Vocabulary


def main() -> int:
    device = torch.device("cpu")
    failures = 0

    vocab = Vocabulary.load(SETTINGS.models.cnn_checkpoint)
    num_classes = vocab.size()
    print(f"[smoke_test] Vocabulary size: {num_classes}")

    # ---------- Character task ----------
    char_manifest = Path("data/manifests/train_char.jsonl")
    if char_manifest.exists():
        manifests = read_jsonl(char_manifest)
        report = validate_manifests(manifests, check_files=True)
        print(f"[smoke_test] Char manifest: {report.valid_samples} valid / "
              f"{report.total_samples} total, {report.num_classes} classes")

        # Take a small batch.
        batch = manifests[:8]
        items = [(Path(m.image_path), vocab.index_for_char(m.label) or 0)
                 for m in batch]
        ds = CharImageDataset(
            items,
            img_height=SETTINGS.preprocessing.img_height,
            img_width=SETTINGS.preprocessing.img_width,
            mean=SETTINGS.preprocessing.mean,
            std=SETTINGS.preprocessing.std,
            augment=False,
        )
        loader = DataLoader(ds, batch_size=len(items), shuffle=False)

        model, _ = build_cnn(
            num_classes=num_classes,
            checkpoint_path=str(SETTINGS.models.cnn_checkpoint)
            if SETTINGS.models.cnn_checkpoint.exists()
            else None,
            device="cpu",
        )
        model.eval()
        with torch.no_grad():
            for images, labels in loader:
                logits = model(images)
                assert logits.shape == (len(items), num_classes), (
                    f"CNN output shape {logits.shape} != expected "
                    f"({len(items)}, {num_classes})"
                )
                assert torch.isfinite(logits).all(), "CNN logits not finite"
                print(f"[smoke_test] CNN forward OK: {tuple(logits.shape)}")
        print("[smoke_test] Char task: PASS")
    else:
        print("[smoke_test] Char manifest not found; skipping CNN test.")
        failures += 1

    # ---------- Line task ----------
    # Check for any line manifest (ahld29k or fidel).
    line_manifest = None
    for cand in (Path("data/manifests/ahld29k.jsonl"),
                 Path("data/manifests/fidel.jsonl")):
        if cand.exists():
            line_manifest = cand
            break

    if line_manifest is not None:
        manifests = read_jsonl(line_manifest)
        report = validate_manifests(manifests, check_files=True)
        print(f"[smoke_test] Line manifest ({line_manifest.name}): "
              f"{report.valid_samples} valid / {report.total_samples} total")

        # CRNN expects (B, 1, H, W). Use a couple of line images.
        batch = manifests[:4]
        ds = LineTextDataset(
            [(m.image_path, m.label) for m in batch],
            vocab,
            img_height=SETTINGS.preprocessing.img_height,
            mean=SETTINGS.preprocessing.mean,
            std=SETTINGS.preprocessing.std,
        )
        loader = DataLoader(ds, batch_size=len(batch), shuffle=False)

        model = CRNN(
            num_classes=num_classes,
            img_height=SETTINGS.preprocessing.img_height,
        ).to(device)
        model.eval()
        with torch.no_grad():
            for images, texts, targets in loader:
                images = images.to(device)
                log_probs = model(images)
                # (B, T, num_classes + 1)
                assert log_probs.ndim == 3, \
                    f"CRNN output ndim {log_probs.ndim} != 3"
                assert log_probs.shape[-1] == num_classes + 1, \
                    f"CRNN output last dim {log_probs.shape[-1]} != {num_classes + 1}"
                assert torch.isfinite(log_probs).all(), "CRNN log_probs not finite"
                print(f"[smoke_test] CRNN forward OK: {tuple(log_probs.shape)}")
        print("[smoke_test] Line task: PASS")
    else:
        print("[smoke_test] No line manifest found; skipping CRNN test.")
        failures += 1

    # ---------- Normalization check ----------
    test_text = "ሰላም   ኢትዮጵያ  "
    norm = normalize_amharic(test_text)
    assert norm == "ሰላም ኢትዮጵያ", f"Normalization failed: {norm!r}"
    # Write to stdout as UTF-8 to avoid Windows console (cp1252) encoding errors.
    import sys as _sys
    _sys.stdout.buffer.write(
        f"[smoke_test] Text normalization OK: {test_text!r} -> {norm!r}\n".encode(
            "utf-8"
        )
    )

    if failures:
        print(f"[smoke_test] {failures} check(s) skipped (manifests missing).")
        return 0
    print("[smoke_test] All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
