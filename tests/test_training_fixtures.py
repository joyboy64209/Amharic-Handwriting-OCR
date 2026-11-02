"""Tiny CNN and CRNN training/evaluation using generated fixtures.

No network access, no external datasets. Verifies the training loop runs end-to-end
on synthetic data and produces valid checkpoints/metrics.
"""

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config import SETTINGS
from src.dataset import CharImageDataset, SplitManifest
from src.metrics import cer_wer_list, classification_metrics
from src.models.cnn import AHCNN, build_model
from src.models.crnn import CRNN, ctc_loss, greedy_decode
from src.vocabulary import FALLBACK_VOCAB, Vocabulary


class TestTinyCNNTraining(unittest.TestCase):
    """End-to-end tiny CNN training on synthetic data."""

    def _make_vocab(self):
        chars = ["ሀ", "ሁ", "ሂ"]
        return Vocabulary.from_ordered_chars(chars, source="test")

    def _make_dummy_data(self, num_samples=20, img_size=28):
        """Create synthetic (image, class_index) pairs."""
        images = []
        labels = []
        for i in range(num_samples):
            img = np.zeros((img_size, img_size), dtype=np.float32)
            cls = i % 3
            img[cls * 5 : cls * 5 + 4, cls * 5 : cls * 5 + 4] = 1.0
            images.append(img)
            labels.append(cls)
        images = np.stack(images)[..., None]
        images = images.transpose(0, 3, 1, 2)
        return torch.from_numpy(images), torch.tensor(labels, dtype=torch.long)

    def test_tiny_cnn_train_and_evaluate(self):
        vocab = self._make_vocab()
        num_classes = vocab.size()
        images, labels = self._make_dummy_data(num_samples=30)
        ds = TensorDataset(images, labels)
        train_loader = DataLoader(ds, batch_size=8, shuffle=True)
        val_loader = DataLoader(ds, batch_size=8, shuffle=False)

        model = AHCNN(num_classes=num_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()

        model.train()
        for epoch in range(2):
            for imgs, labs in train_loader:
                optimizer.zero_grad()
                logits = model(imgs)
                loss = criterion(logits, labs)
                loss.backward()
                optimizer.step()

        model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for imgs, labs in val_loader:
                logits = model(imgs)
                pred = logits.argmax(dim=1)
                y_true.extend(labs.tolist())
                y_pred.extend(pred.tolist())

        report = classification_metrics(y_true, y_pred, num_classes,
                                        labels=[vocab.char_for_index(i) for i in range(num_classes)])
        self.assertGreaterEqual(report.accuracy, 0.0)
        self.assertLessEqual(report.accuracy, 1.0)
        self.assertIsNotNone(report.confusion_matrix)
        self.assertEqual(len(report.confusion_matrix), num_classes)

    def test_checkpoint_has_required_metadata(self):
        vocab = self._make_vocab()
        num_classes = vocab.size()
        model = AHCNN(num_classes=num_classes)
        ckpt = {
            "model_state_dict": model.state_dict(),
            "class_to_idx": {ch: i for i, ch in vocab.index_to_char.items()},
            "input_shape": (1, 28, 28),
            "model_architecture": "AHCNN",
            "final_accuracy": 0.5,
            "val_split": 0.2,
            "seed": 42,
            "epoch": 1,
        }
        with tempfile.TemporaryDirectory() as td:
            ckpt_path = Path(td) / "tiny.pth"
            torch.save(ckpt, ckpt_path)
            model2, metadata = build_model(num_classes=num_classes,
                                           checkpoint_path=str(ckpt_path),
                                           device="cpu")
            self.assertEqual(metadata["input_shape"], (1, 28, 28))
            self.assertEqual(metadata["final_accuracy"], 0.5)
            self.assertEqual(metadata["epoch"], 1)


class TestTinyCRNNTraining(unittest.TestCase):
    """End-to-end tiny CRNN training on synthetic data."""

    def _make_vocab(self):
        chars = ["ሀ", "ሁ", "ሂ"]
        return Vocabulary.from_ordered_chars(chars, source="test")

    def test_crnn_forward_and_ctc_loss(self):
        vocab = self._make_vocab()
        num_classes = vocab.size()
        model = CRNN(num_classes=num_classes, img_height=28)
        model.eval()

        images = torch.randn(2, 1, 28, 64)
        with torch.no_grad():
            log_probs = model(images)  # (2, T, num_classes+1)
        self.assertEqual(log_probs.shape[-1], num_classes + 1)
        self.assertTrue(torch.all(log_probs <= 0))

        targets = torch.tensor([1, 2], dtype=torch.long)
        input_lengths = torch.tensor([log_probs.shape[1], log_probs.shape[1]], dtype=torch.long)
        target_lengths = torch.tensor([1, 1], dtype=torch.long)
        loss = ctc_loss(log_probs.transpose(0, 1), targets, input_lengths, target_lengths, blank=0)
        self.assertTrue(torch.isfinite(loss))

    def test_crnn_decode_roundtrip(self):
        vocab = self._make_vocab()
        num_classes = vocab.size()
        model = CRNN(num_classes=num_classes, img_height=28)
        model.eval()
        images = torch.randn(1, 1, 28, 100)
        with torch.no_grad():
            log_probs = model(images)
        decoded = greedy_decode(log_probs, blank=0)
        self.assertEqual(len(decoded), 1)
        text = "".join(vocab.char_for_index(c) for c in decoded[0])
        for ch in text:
            self.assertTrue(ch in vocab or ch == "?")

    def test_cer_wer_on_fixtures(self):
        refs = ["ሀሁሂ", "ሀሁ", "ሂ"]
        hyps = ["ሀሁሂ", "ሀሂ", "ሀሁሂ"]
        cer, wer = cer_wer_list(refs, hyps)
        self.assertGreaterEqual(cer, 0.0)
        self.assertLessEqual(cer, 1.0)
        self.assertGreaterEqual(wer, 0.0)
        self.assertLessEqual(wer, 1.0)


class TestSmallDatasetHandling(unittest.TestCase):
    """Very small datasets should not crash training/splitting code."""

    def test_split_with_3_samples(self):
        from src.data_pipeline import DatasetManifest, split_manifests
        ms = [DatasetManifest(image_path=f"img_{i}.png", label="ሀ",
                              source="local_char", task_type="char")
              for i in range(3)]
        train, val, test = split_manifests(ms, val_split=0.33, test_split=0.33, seed=42)
        self.assertEqual(len(train) + len(val) + len(test), 3)
        self.assertGreaterEqual(len(train), 1)

    def test_validate_empty_manifest(self):
        from src.data_pipeline import DatasetManifest, validate_manifests
        report = validate_manifests([], check_files=False)
        self.assertEqual(report.valid_samples, 0)
        self.assertEqual(report.rejected_samples, 0)
        self.assertEqual(report.num_classes, 0)


if __name__ == "__main__":
    unittest.main()