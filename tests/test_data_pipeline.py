"""Tests for src.data_pipeline — unified multi-dataset data pipeline.

Covers: label normalization, manifest read/write, validation (invalid files,
missing labels, duplicates), splitting, and split-leakage checks. Uses only
stdlib + existing project deps; no network or HF dataset downloads.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import unittest

import cv2
import numpy as np

from src.data_pipeline import (
    DatasetManifest,
    check_split_leakage,
    normalize_amharic,
    read_jsonl,
    split_manifests,
    validate_manifests,
    write_jsonl,
)


class TestNormalizeAmharic(unittest.TestCase):

    def test_nfc_and_whitespace(self):
        self.assertEqual(normalize_amharic("  ሰላ   ም  "), "ሰላ ም")
        self.assertEqual(normalize_amharic("ሰላም"), "ሰላም")

    def test_preserves_ethiopic(self):
        self.assertEqual(normalize_amharic("ሀለው"), "ሀለው")

    def test_empty_and_non_string(self):
        self.assertEqual(normalize_amharic(""), "")
        self.assertEqual(normalize_amharic(123), "123")


class TestManifestReadWrite(unittest.TestCase):

    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "manifest.jsonl"
            manifests = [
                DatasetManifest(
                    image_path="data/train_char/train_char_000001.png",
                    label="ሀ", source="local_char", task_type="char",
                    writer_id=None, metadata={"index": 1},
                ),
                DatasetManifest(
                    image_path="data/ahld29k/P001_S01_Page01_Line01.png",
                    label="ሰላም", source="ahld29k", task_type="line",
                    writer_id="P001", metadata={"index": 0},
                ),
            ]
            write_jsonl(manifests, p)
            back = read_jsonl(p)
            self.assertEqual(len(back), 2)
            self.assertEqual(back[0].label, "ሀ")
            self.assertEqual(back[0].task_type, "char")
            self.assertEqual(back[1].label, "ሰላም")
            self.assertEqual(back[1].writer_id, "P001")
            self.assertEqual(back[1].metadata["index"], 0)

    def test_read_missing_file(self):
        self.assertEqual(read_jsonl(Path("does_not_exist.jsonl")), [])
class TestValidateManifests(unittest.TestCase):

    def _write_png(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        img = np.full((28, 28), 255, dtype=np.uint8)
        cv2.rectangle(img, (5, 5), (22, 22), 0, -1)
        cv2.imwrite(str(path), img)

    def test_all_valid(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "img.png"
            self._write_png(p)
            manifests = [
                DatasetManifest(
                    image_path=str(p), label="ሀ",
                    source="local_char", task_type="char",
                )
            ]
            report = validate_manifests(manifests, check_files=True)
            self.assertEqual(report.valid_samples, 1)
            self.assertEqual(report.rejected_samples, 0)
            self.assertEqual(report.num_classes, 1)

    def test_missing_file(self):
        manifests = [
            DatasetManifest(
                image_path="data/nonexistent.png", label="ሀ",
                source="local_char", task_type="char",
            )
        ]
        report = validate_manifests(manifests, check_files=True)
        self.assertEqual(report.valid_samples, 0)
        self.assertEqual(report.reject_reasons.get("missing_file"), 1)

    def test_empty_label(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "img.png"
            self._write_png(p)
            manifests = [
                DatasetManifest(
                    image_path=str(p), label="",
                    source="local_char", task_type="char",
                ),
                DatasetManifest(
                    image_path=str(p), label="   ",
                    source="local_char", task_type="char",
                ),
            ]
            report = validate_manifests(manifests, check_files=True)
            self.assertEqual(report.valid_samples, 0)
            self.assertEqual(report.reject_reasons.get("empty_label"), 2)

    def test_duplicate_image(self):
        with tempfile.TemporaryDirectory() as td:
            p1 = Path(td) / "a.png"
            p2 = Path(td) / "b.png"
            self._write_png(p1)
            shutil.copy2(p1, p2)
            manifests = [
                DatasetManifest(image_path=str(p1), label="ሀ",
                                source="local_char", task_type="char"),
                DatasetManifest(image_path=str(p2), label="ሁ",
                                source="local_char", task_type="char"),
            ]
            report = validate_manifests(manifests, check_files=True)
            self.assertEqual(report.valid_samples, 1)
            self.assertEqual(report.duplicate_images, 1)
            self.assertEqual(report.reject_reasons.get("duplicate_image"), 1)

    def test_unreadable_image(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "broken.png"
            p.write_text("not an image")
            manifests = [
                DatasetManifest(
                    image_path=str(p), label="ሀ",
                    source="local_char", task_type="char",
                )
            ]
            report = validate_manifests(manifests, check_files=True)
            self.assertEqual(report.valid_samples, 0)
            self.assertEqual(report.reject_reasons.get("unreadable_image"), 1)
class TestSplitting(unittest.TestCase):

    def test_char_stratified_split(self):
        manifests = []
        for i in range(20):
            manifests.append(DatasetManifest(
                image_path=f"img_{i}.png", label="ሀ" if i < 10 else "ሁ",
                source="local_char", task_type="char",
            ))
        train, val, test = split_manifests(
            manifests, val_split=0.2, test_split=0.1, seed=42)
        self.assertEqual(len(train) + len(val) + len(test), 20)
        self.assertGreater(len(train), 0)
        self.assertGreater(len(val), 0)
        self.assertGreater(len(test), 0)

    def test_line_random_split(self):
        manifests = [
            DatasetManifest(
                image_path=f"line_{i}.png", label=f"text {i}",
                source="ahld29k", task_type="line", writer_id=f"W{i % 3}"
            )
            for i in range(12)
        ]
        train, val, test = split_manifests(
            manifests, val_split=0.25, test_split=0.25, seed=7)
        self.assertEqual(len(train) + len(val) + len(test), 12)

    def test_writer_separated_split(self):
        manifests = []
        for w in range(6):
            for j in range(2):
                manifests.append(DatasetManifest(
                    image_path=f"w{w}_s{j}.png", label="ሀ",
                    source="ahld29k", task_type="line", writer_id=f"W{w}",
                ))
        train, val, test = split_manifests(
            manifests, val_split=0.25, test_split=0.25, seed=42)
        train_w = {m.writer_id for m in train if m.writer_id}
        val_w = {m.writer_id for m in val if m.writer_id}
        test_w = {m.writer_id for m in test if m.writer_id}
        self.assertEqual(len(train_w & val_w), 0)
        self.assertEqual(len(train_w & test_w), 0)
        self.assertEqual(len(val_w & test_w), 0)

    def test_split_leakage_check(self):
        manifests = [
            DatasetManifest(image_path=f"img_{i}.png", label="ሀ",
                            source="local_char", task_type="char")
            for i in range(10)
        ]
        train, val, test = split_manifests(
            manifests, val_split=0.2, test_split=0.1, seed=42)
        leakage = check_split_leakage(train, val, test)
        self.assertTrue(leakage["clean"])


if __name__ == "__main__":
    unittest.main()


