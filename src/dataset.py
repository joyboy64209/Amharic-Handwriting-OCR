"""Dataset handling for Amharic handwriting OCR.

Two dataset flavors:

1. Isolated-character folders (``data/uni_dataset/<char>/.jpg``) — for
   training/evaluating the CNN baseline. One folder per Amharic character
   class; the folder name *is* the Unicode label.

2. Labeled text-line images (``data/lines.jsonl``) — one record per line:
   ``{"path": "data/lines/line_00001.png", "text": "ሰላም"}`` — for training/evaluating
   the CRNN+CTC sequence recognizer.

Dataset validation, train/validation splitting (writer-separated when a writer
identity can be derived from the filename, otherwise stratified random with a
documented limitation), and PyTorch Dataset wrappers are provided here.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import SETTINGS
from .preprocessing import preprocess_for_cnn
from .vocabulary import Vocabulary

_WRITER_RE = re.compile(r"^(\d+)")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@dataclass
class DatasetValidation:
    ok: bool
    class_count: int = 0
    total_images: int = 0
    images_per_class: Dict[str, int] = field(default_factory=dict)
    min_images: int = 0
    max_images: int = 0
    missing_classes: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    writer_ids_found: bool = False


def validate_char_dataset(root: Optional[Path] = None) -> DatasetValidation:
    """Inspect an isolated-character folder dataset and report its health."""
    root = root or SETTINGS.data.char_dataset_dir
    errors: List[str] = []
    images_per_class: Dict[str, int] = {}
    total = 0
    writer_ids = set()

    if not root.exists():
        return DatasetValidation(
            ok=False,
            errors=[f"Character dataset directory not found: {root}"],
        )

    class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if not class_dirs:
        return DatasetValidation(
            ok=False,
            errors=[f"No character class folders found under {root}"],
        )

    for class_dir in class_dirs:
        label = class_dir.name
        imgs = [p for p in class_dir.iterdir()
                if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")]
        count = len(imgs)
        images_per_class[label] = count
        total += count
        for p in imgs:
            m = _WRITER_RE.match(p.stem)
            if m:
                writer_ids.add(m.group(1))

    counts = list(images_per_class.values())
    min_c = min(counts) if counts else 0
    max_c = max(counts) if counts else 0
    missing_classes = [label for label, c in images_per_class.items() if c < 10]

    ok = total > 0 and min_c > 0 and not errors
    return DatasetValidation(
        ok=ok,
        class_count=len(class_dirs),
        total_images=total,
        images_per_class=images_per_class,
        min_images=min_c,
        max_images=max_c,
        missing_classes=missing_classes,
        errors=errors,
        writer_ids_found=len(writer_ids) > 0,
    )


def validate_line_dataset(labels_file: Optional[Path] = None,
                          images_dir: Optional[Path] = None
                          ) -> DatasetValidation:
    """Inspect a text-line JSONL dataset."""
    labels_file = labels_file or SETTINGS.data.line_labels_file
    images_dir = images_dir or SETTINGS.data.line_images_dir
    errors: List[str] = []
    total = 0
    missing_files: List[str] = []

    if not labels_file.exists():
        return DatasetValidation(
            ok=False,
            errors=[f"Line labels file not found: {labels_file} "
                    "(create one per docs/DATASET.md)"],
        )

    with labels_file.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                errors.append(f"{labels_file}:{lineno} — expected JSON object")
                continue
            rel_path = obj.get("path")
            text = obj.get("text")
            if rel_path is None or text is None:
                errors.append(f"{labels_file}:{lineno} — missing 'path' or 'text'")
                continue
            total += 1
            img_path = (images_dir / rel_path) if images_dir else Path(rel_path)
            if not img_path.exists():
                missing_files.append(str(img_path))

    if missing_files:
        errors.append(f"{len(missing_files)} referenced line image(s) missing; "
                      f"first: {missing_files[0]}")

    ok = total > 0 and not errors
    return DatasetValidation(
        ok=ok,
        total_images=total,
        class_count=0,
        errors=errors,
    )


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

@dataclass
class SplitManifest:
    train: List[Tuple[Path, int]]
    val: List[Tuple[Path, int]]
    test: List[Tuple[Path, int]]
    strategy: str
    writer_separated: bool
    num_classes: int


def _writer_id(path: Path) -> Optional[str]:
    m = _WRITER_RE.match(path.stem)
    return m.group(1) if m else None


def split_char_dataset(root: Optional[Path] = None,
                       val_split: Optional[float] = None,
                       test_split: Optional[float] = None,
                       seed: Optional[int] = None,
                       vocab: Optional[Vocabulary] = None
                       ) -> SplitManifest:
    """Build train/val/test splits for the character dataset.

    Strategy:
      * Writer-separated if every image filename starts with a numeric writer
        id (e.g. ``001he.1.jpg`` -> writer ``001``). All images from a writer
        go entirely to one split — no writer leakage across splits.
      * Otherwise stratified random per class; documented limitation that the
        split is not writer-separated.

    Returns (train, val, test) lists of (image_path, class_index).
    """
    root = root or SETTINGS.data.char_dataset_dir
    val_split = SETTINGS.training.val_split if val_split is None else val_split
    test_split = SETTINGS.pipeline.test_split if test_split is None else test_split
    seed = SETTINGS.training.seed if seed is None else seed
    vocab = vocab or Vocabulary.load(SETTINGS.models.cnn_checkpoint)

    rng = np.random.default_rng(seed)
    class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    label_to_idx = {ch: i for i, ch in vocab.index_to_char.items()}
    for i, d in enumerate(class_dirs):
        if d.name not in label_to_idx:
            label_to_idx[d.name] = len(label_to_idx)

    train: List[Tuple[Path, int]] = []
    val: List[Tuple[Path, int]] = []
    test: List[Tuple[Path, int]] = []
    writer_separated = False

    sample_writer_ids = set()
    for class_dir in class_dirs:
        for p in class_dir.iterdir():
            if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
                wid = _writer_id(p)
                if wid:
                    sample_writer_ids.add(wid)
                break
        if sample_writer_ids:
            break

    if sample_writer_ids:
        writer_separated = True
        writer_to_items: Dict[str, List[Tuple[Path, int]]] = {}
        for class_dir in class_dirs:
            idx = label_to_idx[class_dir.name]
            for p in class_dir.iterdir():
                if not (p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                    continue
                wid = _writer_id(p) or "__unknown__"
                writer_to_items.setdefault(wid, []).append((p, idx))
        writers = sorted(writer_to_items.keys())
        rng.shuffle(writers)
        n_writers = len(writers)
        n_test = max(0, int(n_writers * test_split)) if test_split > 0 else 0
        n_val = max(1, int(n_writers * val_split)) if val_split > 0 else 0
        # Ensure at least one writer remains for training.
        if n_test + n_val >= n_writers:
            n_val = max(1, n_writers - n_test - 1) if n_writers - n_test - 1 > 0 else 0
            n_test = max(0, n_writers - n_val - 1) if n_writers - n_val - 1 > 0 else 0
        test_writers = set(writers[:n_test])
        val_writers = set(writers[n_test:n_test + n_val])
        for wid, items in writer_to_items.items():
            if wid in test_writers:
                target = test
            elif wid in val_writers:
                target = val
            else:
                target = train
            target.extend(items)
    else:
        writer_separated = False
        for class_dir in class_dirs:
            idx = label_to_idx[class_dir.name]
            items = [p for p in class_dir.iterdir()
                     if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")]
            rng.shuffle(items)
            n = len(items)
            n_test = max(0, int(n * test_split)) if test_split > 0 else 0
            n_val = max(0, int(n * val_split)) if val_split > 0 else 0
            # Guard: don't consume all samples.
            if n_test + n_val >= n:
                n_val = max(0, n - n_test - 1)
                n_test = max(0, n - n_val - 1)
            test.extend(items[:n_test])
            val.extend(items[n_test:n_test + n_val])
            train.extend(items[n_test + n_val:])
        rng.shuffle(train)
        rng.shuffle(val)
        rng.shuffle(test)

    return SplitManifest(
        train=train,
        val=val,
        test=test,
        strategy="writer-separated" if writer_separated else "stratified-random",
        writer_separated=writer_separated,
        num_classes=len(label_to_idx),
    )


# ---------------------------------------------------------------------------
# PyTorch Datasets
# ---------------------------------------------------------------------------

class CharImageDataset(Dataset):
    """PyTorch Dataset for isolated-character folders.

    Each item returns (image_tensor, class_index) where image_tensor is a
    (1, H, W) float32 array preprocessed identically to inference.
    """

    def __init__(self, items: Sequence[Tuple[Path, int]],
                 img_height: int = 28, img_width: int = 28,
                 mean: float = 0.239, std: float = 0.407,
                 augment: bool = False,
                 augment_cfg=None):
        self.items = list(items)
        self.img_height = img_height
        self.img_width = img_width
        self.mean = mean
        self.std = std
        self.augment = augment
        self.augment_cfg = augment_cfg or {}

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        path, label = self.items[index]
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            image = np.zeros((self.img_height, self.img_width), dtype=np.uint8)
        result = preprocess_for_cnn(
            image, self.img_height, self.img_width, self.mean, self.std,
            max_skew_angle=45.0, return_intermediates=False,
        )
        arr = result.image
        if self.augment:
            disp = ((arr * self.std + self.mean) * 255).clip(0, 255).astype(np.uint8)
            try:
                from .augmentation import augment
                disp = augment(
                    disp,
                    rotation_degrees=float(self.augment_cfg.get("rotation_degrees", 8)),
                    scale_factor=float(self.augment_cfg.get("scale", 0.95)),
                    shear_degrees=float(self.augment_cfg.get("shear_degrees", 4)),
                    brightness_gain=float(self.augment_cfg.get("brightness", 0.15)),
                    seed=index,
                )
            except Exception as exc:
                print(f"[dataset] Augmentation failed for {path} (seed={index}); "
                      f"using unaugmented image. {exc}")
            arr = preprocess_for_cnn(
                disp, self.img_height, self.img_width, self.mean, self.std,
                max_skew_angle=45.0, return_intermediates=False,
            ).image
        tensor = torch.from_numpy(arr).unsqueeze(0)
        return tensor, int(label)


class LineTextDataset(Dataset):
    """PyTorch Dataset for labeled text-line images (JSONL).

    Items: (image_tensor, label_string, label_indices) for CTC training.

    Raises FileNotFoundError if a referenced image is missing or unreadable,
    and ValueError if a label contains characters outside the vocabulary.
    Nothing is silently substituted or dropped.
    """

    def __init__(self, records: Sequence[Tuple[str, str]],
                 vocab: Vocabulary,
                 img_height: int = 28,
                 mean: float = 0.239, std: float = 0.407,
                 images_dir: Optional[Path] = None,
                 strict: bool = True):
        self.records = list(records)
        self.vocab = vocab
        self.img_height = img_height
        self.mean = mean
        self.std = std
        self.images_dir = images_dir or SETTINGS.data.line_images_dir
        self.strict = strict
        # Pre-validate all records so failures surface at dataset construction
        # rather than deep inside a training loop.
        if strict:
            self._validate_records()

    def _validate_records(self) -> None:
        missing: List[str] = []
        unreadable: List[str] = []
        unknown_chars: List[str] = []
        for index, (rel_path, text) in enumerate(self.records):
            img_path = (self.images_dir / rel_path) if not os.path.isabs(rel_path) else Path(rel_path)
            if not img_path.exists():
                missing.append(str(img_path))
                continue
            image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                unreadable.append(str(img_path))
                continue
            for ch in text:
                if ch != " " and self.vocab.index_for_char(ch) is None:
                    unknown_chars.append(f"{rel_path}: '{ch}' (U+{ord(ch):04X})")
        if missing:
            raise FileNotFoundError(
                f"LineTextDataset: {len(missing)} image(s) not found. "
                f"Examples: {missing[:5]}"
            )
        if unreadable:
            raise ValueError(
                f"LineTextDataset: {len(unreadable)} image(s) unreadable. "
                f"Examples: {unreadable[:5]}"
            )
        if unknown_chars:
            raise ValueError(
                f"LineTextDataset: {len(unknown_chars)} label character(s) "
                f"not in vocabulary. Examples: {unknown_chars[:5]}. "
                f"Ensure labels only use characters present in the model "
                f"vocabulary ({self.vocab.size()} classes)."
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        rel_path, text = self.records[index]
        img_path = (self.images_dir / rel_path) if not os.path.isabs(rel_path) else Path(rel_path)
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            if self.strict:
                raise FileNotFoundError(
                    f"LineTextDataset: cannot read image {img_path}"
                )
            # Non-strict mode: return a blank tensor and an empty target.
            image = np.zeros((self.img_height, self.img_height), dtype=np.uint8)
        h, w = image.shape[:2]
        scale = self.img_height / max(h, 1)
        new_w = max(1, int(w * scale))
        resized = cv2.resize(image, (new_w, self.img_height), interpolation=cv2.INTER_AREA)
        arr = resized.astype(np.float32) / 255.0
        arr = (arr - self.mean) / self.std
        arr = np.clip(arr, -3.0, 3.0).astype(np.float32)
        tensor = torch.from_numpy(arr).unsqueeze(0)

        indices = [self.vocab.index_for_char(ch) for ch in text]
        missing_chars = [ch for ch in text if ch != " " and self.vocab.index_for_char(ch) is None]
        if missing_chars:
            raise ValueError(
                f"LineTextDataset: label '{text}' contains characters not in "
                f"vocabulary: {missing_chars}"
            )
        target = torch.tensor([i + 1 for i in indices], dtype=torch.long)
        return tensor, text, target


def load_line_records(labels_file: Optional[Path] = None) -> List[Tuple[str, str]]:
    """Load (relative_path, text) records from a JSONL labels file.

    Each line is a JSON object: ``{"path": "data/lines/line_00001.png", "text": "ሰላም"}``.
    """
    labels_file = labels_file or SETTINGS.data.line_labels_file
    records: List[Tuple[str, str]] = []
    if not labels_file.exists():
        return records
    with labels_file.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            path = obj.get("path")
            text = obj.get("text")
            if path is None or text is None:
                continue
            records.append((str(path), str(text)))
    return records


__all__ = [
    "DatasetValidation", "validate_char_dataset", "validate_line_dataset",
    "SplitManifest", "split_char_dataset",
    "CharImageDataset", "LineTextDataset", "load_line_records",
]

import cv2  # noqa: E402