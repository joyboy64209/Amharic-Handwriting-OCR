"""Tests for fresh-clone behavior when datasets or checkpoints are missing."""

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import unittest

from src.config import SETTINGS
from src.dataset import validate_char_dataset, validate_line_dataset
from src.vocabulary import Vocabulary


class TestFreshCloneMissingData(unittest.TestCase):
    """A fresh clone with no data should degrade gracefully."""

    def test_character_dataset_missing(self):
        """validate_char_dataset should report ok=False when data/uni_dataset is absent."""
        # Use a path that doesn't exist.
        report = validate_char_dataset(Path("/nonexistent/path/to/uni_dataset"))
        self.assertFalse(report.ok)
        self.assertTrue(len(report.errors) > 0)

    def test_line_dataset_missing(self):
        """validate_line_dataset should report ok=False when labels file absent."""
        report = validate_line_dataset(
            Path("/nonexistent/path/to/lines.jsonl"),
            Path("/nonexistent/path/to/lines"),
        )
        self.assertFalse(report.ok)
        self.assertTrue(len(report.errors) > 0)

    def test_vocabulary_fallback_when_checkpoint_missing(self):
        """Vocabulary.load should fall back to FALLBACK_VOCAB when checkpoint absent."""
        vocab = Vocabulary.load(Path("/nonexistent/checkpoint.pth"))
        self.assertEqual(vocab.source, "fallback")
        self.assertGreater(vocab.size(), 0)

    def test_settings_defaults_point_to_local_paths(self):
        """All default data paths should be relative, not personal absolute paths."""
        self.assertNotIn("C:/", str(SETTINGS.data.char_dataset_dir))
        self.assertNotIn("C:/", str(SETTINGS.data.line_labels_file))
        self.assertNotIn("C:/", str(SETTINGS.data.line_images_dir))
        self.assertNotIn("C:/", str(SETTINGS.models.cnn_checkpoint))
        self.assertNotIn("C:/", str(SETTINGS.models.crnn_checkpoint))


class TestFreshCloneMissingCheckpoints(unittest.TestCase):
    """API and training should report clear errors when checkpoints are missing."""

    def test_cnn_checkpoint_missing_message(self):
        from src.inference import OCRPredictor
        # Temporarily point at a nonexistent checkpoint.
        old = SETTINGS.models.cnn_checkpoint
        SETTINGS.models.cnn_checkpoint = Path("/nonexistent/cnn.pth")
        try:
            predictor = OCRPredictor()
            self.assertFalse(predictor.cnn_ready)
            msg = predictor.missing_model_message("char")
            self.assertIn("cnn", msg.lower())
            self.assertIn("nonexistent", msg.lower())
        finally:
            SETTINGS.models.cnn_checkpoint = old

    def test_crnn_checkpoint_missing_message(self):
        from src.inference import OCRPredictor
        old = SETTINGS.models.crnn_checkpoint
        SETTINGS.models.crnn_checkpoint = Path("/nonexistent/crnn.pth")
        try:
            predictor = OCRPredictor()
            self.assertFalse(predictor.crnn_ready)
            msg = predictor.missing_model_message("line")
            self.assertIn("crnn", msg.lower())
        finally:
            SETTINGS.models.crnn_checkpoint = old


if __name__ == "__main__":
    unittest.main()