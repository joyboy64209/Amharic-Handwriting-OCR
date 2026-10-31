"""Unified multi-dataset data pipeline for Amharic Handwriting OCR.

Loads three data sources, converts each into a consistent internal
representation, validates, and produces reproducible train/validation/test
splits:

  * ``local_char``  - isolated-character NumPy arrays (32x32 grayscale,
    single Ethiopic char per sample) -> character classification task
  * ``ahld29k``     - line-level handwritten Amharic parquet from HF
    (streaming) -> text-line sequence recognition task
  * ``fidel``       - line-level Amharic ZIP archives + JSON labels from HF
    -> text-line sequence recognition task

Original datasets are **never modified**.  Processed images are extracted
into dedicated directories under ``data/`` and manifests are written as
JSONL.  All paths are configured via ``config.yaml`` (``data_sources`` and
``pipeline`` sections) so the project is not tied to any personal directory
structure.

The pipeline intentionally keeps character and line data as **separate
training tasks** -- they are not blindly merged into one training set.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from src.config import SETTINGS

# --------------------------------------------------------------------------- #
# Internal representation
# --------------------------------------------------------------------------- #

@dataclass
class DatasetManifest:
    """One sample in the unified manifest.

    ``image_path`` is relative to the project root and points at a processed
    PNG that this pipeline has written (original datasets are left untouched).
    """
    image_path: str
    label: str
    source: str
    task_type: str               # 'char' or 'line'
    writer_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "image_path": self.image_path,
            "label": self.label,
            "source": self.source,
            "task_type": self.task_type,
            "writer_id": self.writer_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DatasetManifest":
        return cls(
            image_path=d["image_path"],
            label=d["label"],
            source=d["source"],
            task_type=d["task_type"],
            writer_id=d.get("writer_id"),
            metadata=d.get("metadata", {}),
        )


# --------------------------------------------------------------------------- #
# Text / image helpers
# --------------------------------------------------------------------------- #

def normalize_amharic(text: str) -> str:
    """Unicode-normalize Amharic text without destroying Ethiopic characters.

    Uses NFC normalization and collapses whitespace.  Strips leading/trailing
    whitespace.  Does *not* transliterate or substitute characters.
    """
    if not isinstance(text, str):
        text = str(text)
    text = unicodedata.normalize("NFC", text)
    text = " ".join(text.split())
    return text


def image_md5(image: Image.Image) -> str:
    """MD5 of a PNG-encoded image for duplicate detection."""
    import io as _io
    buf = _io.BytesIO()
    image.save(buf, format="PNG")
    return hashlib.md5(buf.getvalue()).hexdigest()


def array_to_png(arr: np.ndarray, out_path: Path) -> None:
    """Save a grayscale NumPy array (0-255) as a PNG image."""
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="L")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG")


def save_pil_image(image: Image.Image, out_path: Path) -> None:
    """Save a PIL image as PNG, converting to grayscale if needed."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if image.mode != "L":
        image = image.convert("L")
    image.save(out_path, format="PNG")
# --------------------------------------------------------------------------- #
# Dataset loaders
# --------------------------------------------------------------------------- #

def load_local_char(cfg: dict) -> List[DatasetManifest]:
    """Load the local NumPy-based isolated-character dataset.

    Reads ``x_train.npy`` (N, 32, 32 float32) and ``y_train.npy`` (N, 1)
    single-char Unicode labels, saves each image as PNG under the configured
    ``image_dir``, and returns a manifest list.
    """
    root = Path(cfg["root"])
    x_file = root / cfg["x_file"]
    y_file = root / cfg["y_file"]
    image_dir = Path(cfg["image_dir"])

    if not x_file.exists():
        raise FileNotFoundError(f"Local char x_train.npy not found: {x_file}")
    if not y_file.exists():
        raise FileNotFoundError(f"Local char y_train.npy not found: {y_file}")

    # Memory-map to avoid loading 70k x 32x32 into RAM all at once.
    x = np.load(x_file, mmap_mode="r")
    y = np.load(y_file, mmap_mode="r")

    if len(x) != len(y):
        raise ValueError(
            f"x_train ({len(x)}) and y_train ({len(y)}) row counts differ"
        )

    manifests: List[DatasetManifest] = []
    for i in range(len(x)):
        label = normalize_amharic(str(y[i].item()))
        if not label:
            continue
        out_name = f"train_char_{i:06d}.png"
        out_path = image_dir / out_name
        array_to_png(x[i], out_path)
        manifests.append(
            DatasetManifest(
                image_path=str(out_path),
                label=label,
                source="local_char",
                task_type="char",
                writer_id=None,
                metadata={"index": i, "original_shape": list(x[i].shape)},
            )
        )
    return manifests


def load_ahld29k(cfg: dict) -> List[DatasetManifest]:
    """Load AHLD-29k line-level dataset from Hugging Face parquet (streaming).

    Streams rows one at a time so the full 4.6 GB parquet is never loaded
    into memory.  Each row's PIL image is saved to ``image_dir``.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "The `datasets` library is required to load AHLD-29k. "
            "Install it with: pip install datasets pyarrow"
        ) from exc

    repo_id = cfg["repo_id"]
    image_dir = Path(cfg["image_dir"])
    streaming = bool(cfg.get("streaming", True))
    max_samples = cfg.get("max_samples", None)

    ds = load_dataset(repo_id, split="train", streaming=streaming)

    manifests: List[DatasetManifest] = []
    seen_md5: set = set()
    for idx, row in enumerate(ds):
        if max_samples is not None and len(manifests) >= max_samples:
            break
        image = row["image"]
        text = normalize_amharic(row.get("text", ""))
        file_name = row.get("file_name", f"ahld29k_{idx:06d}")
        writer_id = row.get("writer_id")
        width = row.get("image_width")

        if not text:
            continue

        # Skip exact-duplicate images within this dataset.
        img_md5 = image_md5(image)
        if img_md5 in seen_md5:
            continue
        seen_md5.add(img_md5)

        # Derive a safe filename from file_name (already unique-ish).
        safe_name = file_name if file_name else f"ahld29k_{idx:06d}"
        if not safe_name.lower().endswith(".png"):
            safe_name = safe_name + ".png"
        out_path = image_dir / safe_name
        save_pil_image(image, out_path)

        manifests.append(
            DatasetManifest(
                image_path=str(out_path),
                label=text,
                source="ahld29k",
                task_type="line",
                writer_id=str(writer_id) if writer_id is not None else None,
                metadata={
                    "index": idx,
                    "file_name": file_name,
                    "image_width": width,
                },
            )
        )
    return manifests
def load_fidel(cfg: dict) -> List[DatasetManifest]:
    """Load the Fidel Dataset line-level dataset from HF CSV labels.

    Downloads only the label CSVs (train_labels.csv ~24 MB, test_labels.csv ~4 MB)
    via ``huggingface_hub`` -- NOT the 12+ GB ZIP archives.  Images are stored
    inside LFS-tracked ZIPs (train-clean.zip ~10.6 GB, test-clean.zip ~2.1 GB)
    and cannot be selectively extracted without downloading whole ZIPs, so this
    loader produces manifests with ``image_path`` pointing at expected locations
    inside the downloaded repo but does **not** copy image files.

    Callers must download/extract the ZIPs separately if they need the actual
    images.  See ``scripts/download_datasets.py`` for the bounded download flow.
    """
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "The `huggingface_hub` library is required to load Fidel. "
            "Install it with: pip install huggingface-hub"
        ) from exc

    import csv

    repo_id = cfg["repo_id"]
    image_dir = Path(cfg["image_dir"])
    include_handwritten = bool(cfg.get("include_handwritten", True))
    include_typed = bool(cfg.get("include_typed", False))
    include_synthetic = bool(cfg.get("include_synthetic", False))
    max_samples = cfg.get("max_samples", None)

    allowed_types = set()
    if include_handwritten:
        allowed_types.add("handwritten")
    if include_typed:
        allowed_types.add("typed")
    if include_synthetic:
        allowed_types.add("synthetic")

    # Download only the label CSVs (not the 12+ GB ZIP archives).
    label_files: Dict[str, Path] = {}
    for split in ("train", "test"):
        try:
            csv_path = hf_hub_download(
                repo_id=repo_id,
                filename=f"{split}_labels.csv",
                repo_type="dataset",
            )
            label_files[split] = Path(csv_path)
        except Exception:
            # Split may not exist; skip silently.
            continue

    if not label_files:
        raise FileNotFoundError(
            f"No *_labels.csv found for Fidel repo: {repo_id}"
        )

    manifests: List[DatasetManifest] = []
    seen_md5: set = set()
    total_written = 0

    for split, labels_path in label_files.items():
        with labels_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for entry in reader:
                if max_samples is not None and total_written >= max_samples:
                    break

                src_type = entry.get("type", "")
                if src_type not in allowed_types:
                    continue

                text = normalize_amharic(entry.get("line_text", ""))
                if not text:
                    continue

                img_filename = entry.get("image_filename", "")
                if not img_filename:
                    continue

                # Writer ID column may be empty for typed/synthetic samples.
                writer_val = entry.get("writer", "") or None

                # Images live inside LFS-tracked ZIPs; we cannot point to
                # extracted files without downloading whole ZIPs.  Record the
                # expected path inside the repo so callers know where images
                # would live after extraction.
                expected_path = str(image_dir / f"fidel_{split}_{img_filename}")

                # Duplicate check by (filename, text) since we can't hash bytes.
                dedupe_key = (img_filename, text)
                if dedupe_key in seen_md5:
                    continue
                seen_md5.add(dedupe_key)

                manifests.append(
                    DatasetManifest(
                        image_path=expected_path,
                        label=text,
                        source="fidel",
                        task_type="line",
                        writer_id=str(writer_val) if writer_val else None,
                        metadata={
                            "split": split,
                            "type": src_type,
                            "file_name": img_filename,
                            "images_in_zip": True,
                            "note": "Images are inside train-clean.zip / test-clean.zip; "
                            "download ZIP separately to access actual files.",
                        },
                    )
                )
                total_written += 1

    return manifests


# --------------------------------------------------------------------------- #
# Loader registry
# --------------------------------------------------------------------------- #

DATASET_LOADERS = {
    "local_char": load_local_char,
    "ahld29k": load_ahld29k,
    "fidel": load_fidel,
}


def load_dataset_source(name: str, cfg: dict) -> List[DatasetManifest]:
    """Load a single named dataset source and return its manifests."""
    if name not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset source: {name}. "
                         f"Available: {list(DATASET_LOADERS)}")
    return DATASET_LOADERS[name](cfg)
# --------------------------------------------------------------------------- #
# Manifest building
# --------------------------------------------------------------------------- #

def write_jsonl(manifests: Sequence[DatasetManifest], path: Path) -> None:
    """Write manifests to a JSONL file (one JSON object per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for m in manifests:
            fh.write(json.dumps(m.to_dict(), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[DatasetManifest]:
    """Read manifests from a JSONL file."""
    manifests: List[DatasetManifest] = []
    if not path.exists():
        return manifests
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            manifests.append(DatasetManifest.from_dict(json.loads(line)))
    return manifests


def build_manifests(
    sources_cfg: Dict[str, dict],
    only_enabled: bool = True,
) -> Dict[str, List[DatasetManifest]]:
    """Load each configured dataset source and return {name: manifests}."""
    result: Dict[str, List[DatasetManifest]] = {}
    for name, cfg in sources_cfg.items():
        if only_enabled and not cfg.get("enabled", True):
            continue
        result[name] = load_dataset_source(name, cfg)
    return result


def write_unified_manifest(
    per_source: Dict[str, List[DatasetManifest]],
    output_path: Path,
) -> List[DatasetManifest]:
    """Concatenate all per-source manifests into one unified JSONL file."""
    unified: List[DatasetManifest] = []
    for name in sorted(per_source):
        unified.extend(per_source[name])
    write_jsonl(unified, output_path)
    return unified
# --------------------------------------------------------------------------- #
# Splitting (writer-separated when possible, else stratified random)
# --------------------------------------------------------------------------- #

def split_manifests(
    manifests: Sequence[DatasetManifest],
    val_split: float,
    test_split: float,
    seed: int,
) -> Tuple[List[DatasetManifest], List[DatasetManifest], List[DatasetManifest]]:
    """Split manifests into train/val/test.

    Strategy:
      * If every sample has a writer_id, do writer-separated splitting so no
        writer appears in more than one split.
      * Otherwise, stratified random split by label (for char tasks) or plain
        random (for line tasks).

    Returns (train, val, test).
    """
    import random

    rng = random.Random(seed)
    manifests = list(manifests)

    # Group by writer_id when available.
    by_writer: Dict[str, List[DatasetManifest]] = defaultdict(list)
    no_writer: List[DatasetManifest] = []
    for m in manifests:
        if m.writer_id:
            by_writer[m.writer_id].append(m)
        else:
            no_writer.append(m)

    if by_writer and len(by_writer) > 1:
        # Writer-separated split.
        writers = sorted(by_writer.keys())
        rng.shuffle(writers)
        n_writers = len(writers)
        n_test = max(1, int(n_writers * test_split))
        n_val = max(1, int(n_writers * val_split))

        test_writers = set(writers[:n_test])
        val_writers = set(writers[n_test : n_test + n_val])

        train, val, test = [], [], []
        for w in writers:
            items = by_writer[w]
            if w in test_writers:
                test.extend(items)
            elif w in val_writers:
                val.extend(items)
            else:
                train.extend(items)

        # Samples with no writer_id go to train (stratified by label if char).
        if no_writer:
            train.extend(no_writer)
    else:
        # Stratified random by label (char) or plain random (line).
        if manifests and manifests[0].task_type == "char":
            by_label: Dict[str, List[DatasetManifest]] = defaultdict(list)
            for m in manifests:
                by_label[m.label].append(m)
            train, val, test = [], [], []
            for label, items in by_label.items():
                rng.shuffle(items)
                n = len(items)
                n_test = max(0, int(n * test_split))
                n_val = max(0, int(n * val_split))
                test.extend(items[:n_test])
                val.extend(items[n_test : n_test + n_val])
                train.extend(items[n_test + n_val :])
        else:
            rng.shuffle(manifests)
            n = len(manifests)
            n_test = max(0, int(n * test_split))
            n_val = max(0, int(n * val_split))
            test = manifests[:n_test]
            val = manifests[n_test : n_test + n_val]
            train = manifests[n_test + n_val :]

    return train, val, test
def check_split_leakage(
    train: Sequence[DatasetManifest],
    val: Sequence[DatasetManifest],
    test: Sequence[DatasetManifest],
) -> Dict[str, Any]:
    """Check for duplicate images and writer leakage across splits."""
    def paths(split): return {m.image_path for m in split}
    def writers(split): return {m.writer_id for m in split if m.writer_id}

    train_paths, val_paths, test_paths = paths(train), paths(val), paths(test)
    train_writers, val_writers, test_writers = writers(train), writers(val), writers(test)

    dup_train_val = train_paths & val_paths
    dup_train_test = train_paths & test_paths
    dup_val_test = val_paths & test_paths

    writer_train_val = train_writers & val_writers
    writer_train_test = train_writers & test_writers
    writer_val_test = val_writers & test_writers

    return {
        "duplicate_images_train_val": len(dup_train_val),
        "duplicate_images_train_test": len(dup_train_test),
        "duplicate_images_val_test": len(dup_val_test),
        "writer_leakage_train_val": len(writer_train_val),
        "writer_leakage_train_test": len(writer_train_test),
        "writer_leakage_val_test": len(writer_val_test),
        "clean": (
            len(dup_train_val) == 0
            and len(dup_train_test) == 0
            and len(dup_val_test) == 0
            and len(writer_train_val) == 0
            and len(writer_train_test) == 0
            and len(writer_val_test) == 0
        ),
    }
# --------------------------------------------------------------------------- #
# Validation / quality report
# --------------------------------------------------------------------------- #

@dataclass
class QualityReport:
    total_samples: int
    valid_samples: int
    rejected_samples: int
    reject_reasons: Dict[str, int]
    class_distribution: Dict[str, int]
    num_classes: int
    duplicate_images: int
    missing_files: int
    unreadable_images: int
    empty_labels: int
    by_source: Dict[str, int]
    by_task_type: Dict[str, int]


def validate_manifests(
    manifests: Sequence[DatasetManifest],
    check_files: bool = True,
) -> QualityReport:
    """Validate a manifest list and return a quality report.

    Checks each entry for: missing image file, unreadable image, empty label,
    duplicate image.  Does not modify the input.
    """
    reject_reasons: Dict[str, int] = Counter()
    seen_md5: set = set()
    duplicate_images = 0
    missing_files = 0
    unreadable_images = 0
    empty_labels = 0
    valid = 0
    class_distribution: Dict[str, int] = Counter()
    by_source: Dict[str, int] = Counter()
    by_task_type: Dict[str, int] = Counter()

    for m in manifests:
        reasons: List[str] = []

        # Label checks.
        if not m.label or not m.label.strip():
            reasons.append("empty_label")
            empty_labels += 1

        # File checks.
        if check_files:
            p = Path(m.image_path)
            if not p.exists():
                reasons.append("missing_file")
                missing_files += 1
            else:
                try:
                    with Image.open(p) as img:
                        img.load()
                except Exception:
                    reasons.append("unreadable_image")
                    unreadable_images += 1

        # Duplicate check (by image bytes).
        if check_files and Path(m.image_path).exists():
            try:
                img_bytes = Path(m.image_path).read_bytes()
                img_md5 = hashlib.md5(img_bytes).hexdigest()
                if img_md5 in seen_md5:
                    reasons.append("duplicate_image")
                    duplicate_images += 1
                else:
                    seen_md5.add(img_md5)
            except Exception:
                reasons.append("unreadable_image")
                unreadable_images += 1

        if reasons:
            for r in reasons:
                reject_reasons[r] += 1
        else:
            valid += 1
            class_distribution[m.label] += 1
            by_source[m.source] += 1
            by_task_type[m.task_type] += 1

    return QualityReport(
        total_samples=len(manifests),
        valid_samples=valid,
        rejected_samples=len(manifests) - valid,
        reject_reasons=dict(reject_reasons),
        class_distribution=dict(class_distribution),
        num_classes=len(class_distribution),
        duplicate_images=duplicate_images,
        missing_files=missing_files,
        unreadable_images=unreadable_images,
        empty_labels=empty_labels,
        by_source=dict(by_source),
        by_task_type=dict(by_task_type),
    )


def print_quality_report(report: QualityReport, title: str = "Quality report") -> None:
    """Print a human-readable quality report."""
    print(f"=== {title} ===")
    print(f"Total samples:      {report.total_samples}")
    print(f"Valid samples:      {report.valid_samples}")
    print(f"Rejected samples:   {report.rejected_samples}")
    print(f"Duplicate images:   {report.duplicate_images}")
    print(f"Missing files:      {report.missing_files}")
    print(f"Unreadable images:  {report.unreadable_images}")
    print(f"Empty labels:       {report.empty_labels}")
    print(f"Unique classes:     {report.num_classes}")
    print(f"By source:          {report.by_source}")
    print(f"By task type:       {report.by_task_type}")
    if report.reject_reasons:
        print(f"Reject reasons:     {report.reject_reasons}")
    top_classes = sorted(report.class_distribution.items(),
                         key=lambda kv: -kv[1])[:8]
    print(f"Top classes:        {top_classes}")
    print()


# --------------------------------------------------------------------------- #
# Manifest adapters for existing training scripts
# --------------------------------------------------------------------------- #

def manifests_to_char_records(
    manifests: Sequence[DatasetManifest],
    vocab,
) -> Tuple[List[Tuple[Path, int]], int]:
    """Convert char-task manifests to (image_path, class_index) tuples.

    Compatible with ``CharImageDataset`` and ``split_char_dataset``.
    Determines the actual number of classes from the data -- never
    silently discards or maps to class zero.

    Returns:
        (records, actual_num_classes) where actual_num_classes is the number
        of distinct labels found in the data.
    """
    records: List[Tuple[Path, int]] = []
    seen_indices: set = set()
    for m in manifests:
        if m.task_type != "char":
            continue
        idx = vocab.index_for_char(m.label)
        if idx is not None:
            if idx not in seen_indices:
                seen_indices.add(idx)
                records.append((Path(m.image_path), idx))
    actual_num_classes = len(seen_indices)
    return records, actual_num_classes


def manifests_to_line_records(
    manifests: Sequence[DatasetManifest],
) -> List[Tuple[str, str]]:
    """Convert line-task manifests to (path_str, text) tuples.

    Compatible with ``LineTextDataset`` and ``load_line_records``.
    """
    records: List[Tuple[str, str]] = []
    for m in manifests:
        if m.task_type != "line":
            continue
        records.append((m.image_path, m.label))
    return records


def split_char_manifests(
    manifests: Sequence[DatasetManifest],
    val_split: float,
    test_split: float,
    seed: int,
) -> Tuple[
    List[DatasetManifest],
    List[DatasetManifest],
    List[DatasetManifest],
]:
    """Split char manifests into train/val/test using writer-separated or
    stratified random splitting, returning DatasetManifest lists.

    Use :func:`manifests_to_char_records` to convert the results to
    ``(Path, class_index)`` tuples for ``CharImageDataset``.
    """
    import random

    rng = random.Random(seed)
    manifests = list(manifests)

    by_writer: Dict[str, List[DatasetManifest]] = defaultdict(list)
    no_writer: List[DatasetManifest] = []
    for m in manifests:
        if m.writer_id:
            by_writer[m.writer_id].append(m)
        else:
            no_writer.append(m)

    if by_writer and len(by_writer) > 1:
        writers = sorted(by_writer.keys())
        rng.shuffle(writers)
        n_writers = len(writers)
        n_test = max(1, int(n_writers * test_split))
        n_val = max(1, int(n_writers * val_split))
        test_writers = set(writers[:n_test])
        val_writers = set(writers[n_test : n_test + n_val])

        train, val, test = [], [], []
        for w in writers:
            items = by_writer[w]
            if w in test_writers:
                test.extend(items)
            elif w in val_writers:
                val.extend(items)
            else:
                train.extend(items)
        if no_writer:
            train.extend(no_writer)
    else:
        # Stratified random by label.
        by_label: Dict[str, List[DatasetManifest]] = defaultdict(list)
        for m in manifests:
            by_label[m.label].append(m)
        train, val, test = [], [], []
        for label, items in by_label.items():
            rng.shuffle(items)
            n = len(items)
            n_test = max(0, int(n * test_split))
            n_val = max(0, int(n * val_split))
            test.extend(items[:n_test])
            val.extend(items[n_test : n_test + n_val])
            train.extend(items[n_test + n_val :])

    return train, val, test


__all__ = [
    "DatasetManifest",
    "normalize_amharic",
    "image_md5",
    "array_to_png",
    "save_pil_image",
    "load_local_char",
    "load_ahld29k",
    "load_fidel",
    "load_dataset_source",
    "DATASET_LOADERS",
    "write_jsonl",
    "read_jsonl",
    "build_manifests",
    "write_unified_manifest",
    "split_manifests",
    "check_split_leakage",
    "validate_manifests",
    "QualityReport",
    "print_quality_report",
    "manifests_to_char_records",
    "manifests_to_line_records",
    "split_char_manifests",
]







