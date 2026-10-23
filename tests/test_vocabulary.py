"""Tests for src.vocabulary — explicit Ethiopic char<->index mapping."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tempfile
import unittest

from src.vocabulary import FALLBACK_VOCAB, Vocabulary


class TestVocabulary(unittest.TestCase):

    def test_fallback_size_and_order(self):
        # The fallback is a superset used only when no checkpoint exists; the
        # authoritative 237-class vocabulary comes from the checkpoint
        # (tested in test_load_from_checkpoint).
        self.assertGreaterEqual(len(FALLBACK_VOCAB), 237)
        self.assertEqual(FALLBACK_VOCAB[0], "ሀ")
        # Ordered with no duplicates.
        self.assertEqual(len(set(FALLBACK_VOCAB)), len(FALLBACK_VOCAB))

    def test_from_ordered_chars_roundtrip(self):
        vocab = Vocabulary.from_ordered_chars(["ሀ", "ሁ", "ሂ"], source="test")
        self.assertEqual(vocab.size(), 3)
        self.assertEqual(vocab.char_for_index(0), "ሀ")
        self.assertEqual(vocab.char_for_index(2), "ሂ")
        self.assertEqual(vocab.index_for_char("ሁ"), 1)
        self.assertIsNone(vocab.index_for_char("?"))
        self.assertIn("ሀ", vocab)
        self.assertNotIn("x", vocab)

    def test_load_from_checkpoint(self):
        ckpt = Path("models/amharic_ocr_v2.pth")
        self.assertTrue(ckpt.exists(), "supplied checkpoint must exist")
        vocab = Vocabulary.load(ckpt)
        self.assertEqual(vocab.source, "checkpoint")
        self.assertEqual(vocab.size(), 237)
        self.assertEqual(vocab.char_for_index(0), "ሀ")
        self.assertEqual(vocab.char_for_index(236), "ፖ")
        self.assertEqual(vocab.index_for_char("ሀ"), 0)
        # Bidirectional consistency.
        for i in range(vocab.size()):
            ch = vocab.char_for_index(i)
            self.assertEqual(vocab.index_for_char(ch), i)

    def test_load_fallback_when_missing_checkpoint(self):
        missing = Path("models/does_not_exist.pth")
        vocab = Vocabulary.load(missing)
        self.assertEqual(vocab.source, "fallback")
        self.assertEqual(vocab.size(), len(FALLBACK_VOCAB))

    def test_json_roundtrip(self):
        vocab = Vocabulary.from_ordered_chars(["ሀ", "ሁ", "ሂ"], source="unittest")
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "vocab.json"
            vocab.to_json(p)
            vocab2 = Vocabulary.from_json(p)
        self.assertEqual(vocab2.source, "unittest")
        self.assertEqual(vocab2.size(), 3)
        self.assertEqual(vocab2.char_for_index(1), "ሁ")

    def test_never_uses_filesystem_order(self):
        # The explicit vocab must match the checkpoint's class_to_idx, not
        # whatever order the OS lists directories in.
        vocab = Vocabulary.load(Path("models/amharic_ocr_v2.pth"))
        self.assertEqual(vocab.char_for_index(0), "ሀ")
        self.assertEqual(vocab.char_for_index(1), "ሁ")


if __name__ == "__main__":
    unittest.main()