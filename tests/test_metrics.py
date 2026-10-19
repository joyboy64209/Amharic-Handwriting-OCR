"""Tests for src.metrics — CER/WER and classification metrics."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import unittest

from src.metrics import cer, wer, cer_wer_list, normalize_text
from src.metrics import classification_metrics, confusion_matrix_from_lists


class TestCERWER(unittest.TestCase):

    def test_cer_identical(self):
        self.assertEqual(cer("ሰላም", "ሰላም"), 0.0)

    def test_cer_substitution(self):
        # "ሰላም" is 3 Ethiopic codepoints; substituting the last costs 1 edit.
        self.assertEqual(cer("ሰላም", "ሰላም".replace("ም", "ህ")), 1 / 3)

    def test_cer_empty_reference(self):
        self.assertEqual(cer("", "abc"), 1.0)
        self.assertEqual(cer("", ""), 0.0)

    def test_wer_identical(self):
        self.assertEqual(wer("ሰላም ኢትዮጵያ", "ሰላም ኢትዮጵያ"), 0.0)

    def test_wer_insertion(self):
        self.assertEqual(wer("a b", "a b c"), 1 / 2)

    def test_wer_empty_reference(self):
        self.assertEqual(wer("", "x y"), 1.0)
        self.assertEqual(wer("", ""), 0.0)

    def test_normalize_text(self):
        self.assertEqual(normalize_text("  ሰላ   ም  "), "ሰላ ም")

    def test_cer_wer_list_micro(self):
        refs = ["ሰላም", "ኢትዮጵያ"]
        hyps = ["ሰላም", "ኢትዮጵያ"]
        c, w = cer_wer_list(refs, hyps)
        self.assertEqual(c, 0.0)
        self.assertEqual(w, 0.0)

        refs = ["abc", "de"]
        hyps = ["abx", "d"]
        c, w = cer_wer_list(refs, hyps)
        self.assertGreater(c, 0)
        self.assertGreaterEqual(w, 0)


class TestClassificationMetrics(unittest.TestCase):

    def test_perfect(self):
        y_true = [0, 1, 2, 0, 1]
        y_pred = [0, 1, 2, 0, 1]
        r = classification_metrics(y_true, y_pred, num_classes=3,
                                   labels=["ሀ", "ሁ", "ሂ"])
        self.assertEqual(r.accuracy, 1.0)
        self.assertEqual(r.precision_macro, 1.0)
        self.assertEqual(r.recall_macro, 1.0)
        self.assertEqual(r.f1_macro, 1.0)
        np_cm = r.confusion_matrix
        self.assertEqual(np_cm[0][0], 2)
        self.assertEqual(np_cm[1][1], 2)
        self.assertEqual(np_cm[2][2], 1)

    def test_partial(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 1, 0]
        r = classification_metrics(y_true, y_pred, num_classes=2)
        self.assertEqual(r.accuracy, 0.5)
        # class 0: TP=1, FP=1, FN=1 -> P=0.5, R=0.5, F1=0.5
        # class 1: TP=1, FP=1, FN=1 -> P=0.5, R=0.5, F1=0.5
        self.assertAlmostEqual(r.precision_per_class[0], 0.5)
        self.assertAlmostEqual(r.recall_per_class[0], 0.5)
        self.assertAlmostEqual(r.f1_per_class[0], 0.5)
        self.assertEqual(r.support_per_class, [2, 2])

    def test_confusion_matrix_from_lists(self):
        y_true = [0, 0, 1]
        y_pred = [0, 1, 1]
        cm = confusion_matrix_from_lists(y_true, y_pred, num_classes=2)
        self.assertEqual(cm, [[1, 1], [0, 1]])


if __name__ == "__main__":
    unittest.main()