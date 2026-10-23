"""Tests for src.models.cnn and src.models.crnn — forward output handling."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import unittest

import torch

from src.models.cnn import AHCNN
from src.models.crnn import CRNN, ctc_loss, greedy_decode


class TestCNN(unittest.TestCase):

    def test_forward_output_shape(self):
        model = AHCNN(num_classes=237)
        model.eval()
        x = torch.randn(2, 1, 28, 28)
        with torch.no_grad():
            logits = model(x)
        self.assertEqual(tuple(logits.shape), (2, 237))

    def test_forward_probabilities_in_range(self):
        model = AHCNN(num_classes=10)
        model.eval()
        x = torch.randn(1, 1, 28, 28)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)
        self.assertTrue(torch.all(probs >= 0) and torch.all(probs <= 1))
        self.assertAlmostEqual(float(probs.sum(dim=1).max()), 1.0, places=5)


class TestCRNN(unittest.TestCase):

    def test_forward_output_shape(self):
        model = CRNN(num_classes=237, img_height=28)
        model.eval()
        x = torch.randn(1, 1, 28, 200)  # tall narrow line
        with torch.no_grad():
            log_probs = model(x)
        # last dim = num_classes + 1 (CTC blank)
        self.assertEqual(log_probs.shape[-1], 238)
        self.assertEqual(log_probs.ndim, 3)
        # log-probs should be <= 0
        self.assertTrue(torch.all(log_probs <= 0))

    def test_greedy_decode_removes_blanks_and_repeats(self):
        # (T, C) log-probs; blank=0, char indices 1,2,3
        import torch as t
        log_probs = t.tensor([[
            [  0,-10, -10, -10],  # blank (index 0 max)
            [  0,-10, -10, -10],  # blank
            [-10,  0, -10, -10],  # char 1
            [-10,  0, -10, -10],  # char 1 (repeat -> collapse)
            [  0,-10, -10, -10],  # blank
            [-10, -10,  0, -10],  # char 2
            [-10,  0, -10, -10],  # char 1
        ]], dtype=t.float32).log_softmax(dim=2)
        decoded = greedy_decode(log_probs, blank=0)
        self.assertEqual(decoded[0], [1, 2, 1])

    def test_ctc_loss_runs(self):
        model = CRNN(num_classes=5, img_height=28)
        model.eval()
        x = torch.randn(1, 1, 28, 100)
        with torch.no_grad():
            log_probs = model(x)  # (1, T, 6)
        targets = torch.tensor([1, 2, 3], dtype=torch.long)
        input_lengths = torch.tensor([log_probs.shape[1]], dtype=torch.long)
        target_lengths = torch.tensor([3], dtype=torch.long)
        loss = ctc_loss(log_probs.transpose(0, 1), targets,
                        input_lengths, target_lengths, blank=0)
        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()