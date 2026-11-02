"""Tests for checkpoint save/load round trips and rejection of incompatible checkpoints."""

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import unittest

import torch

from src.models.cnn import AHCNN, build_model, load_checkpoint
from src.models.crnn import CRNN


class TestCheckpointRoundTrip(unittest.TestCase):
    """A checkpoint saved by training should load back into the same architecture."""

    def test_cnn_save_load_roundtrip(self):
        model = AHCNN(num_classes=10)
        state = model.state_dict()
        metadata = {
            "class_to_idx": {"ሀ": 0, "ሁ": 1},
            "input_shape": (1, 28, 28),
            "model_architecture": "AHCNN",
            "epoch": 1,
        }
        with tempfile.TemporaryDirectory() as td:
            ckpt_path = Path(td) / "test.pth"
            torch.save({"model_state_dict": state, **metadata}, ckpt_path)
            model2, meta = build_model(num_classes=10, checkpoint_path=str(ckpt_path), device="cpu")
            self.assertEqual(meta["class_to_idx"], {"ሀ": 0, "ሁ": 1})
            self.assertEqual(meta["input_shape"], (1, 28, 28))
            self.assertEqual(meta["model_architecture"], "AHCNN")
            self.assertEqual(meta["epoch"], 1)

    def test_cnn_load_missing_checkpoint_returns_empty_metadata(self):
        model, meta = build_model(num_classes=10, checkpoint_path=None, device="cpu")
        self.assertEqual(meta, {})

    def test_cnn_load_nonexistent_path_returns_empty_metadata(self):
        model, meta = build_model(num_classes=10, checkpoint_path="/nonexistent/path.pth", device="cpu")
        self.assertEqual(meta, {})

    def test_crnn_state_dict_keys_match(self):
        model = CRNN(num_classes=5, img_height=28)
        state = model.state_dict()
        # All keys should be loadable back into the same architecture.
        model2 = CRNN(num_classes=5, img_height=28)
        model2.load_state_dict(state, strict=True)
        self.assertEqual(model2.num_classes, 5)


class TestIncompatibleCheckpointRejection(unittest.TestCase):
    """Incompatible checkpoints must raise clear errors, not silently load with strict=False."""

    def test_cnn_wrong_num_classes_rejected(self):
        # Build a model with 10 classes, save, then try loading into 5-class model.
        model = AHCNN(num_classes=10)
        with tempfile.TemporaryDirectory() as td:
            ckpt_path = Path(td) / "wrong.pth"
            torch.save({"model_state_dict": model.state_dict()}, ckpt_path)
            model5 = AHCNN(num_classes=5)
            with self.assertRaises(RuntimeError):
                load_checkpoint(model5, str(ckpt_path), device="cpu")

    def test_crnn_checkpoint_with_missing_keys_rejected(self):
        model = CRNN(num_classes=5, img_height=28)
        state = model.state_dict()
        # Remove a key to simulate an incompatible checkpoint.
        del state["block1.conv.weight"]
        with tempfile.TemporaryDirectory() as td:
            ckpt_path = Path(td) / "missing_keys.pth"
            torch.save({"model_state_dict": state}, ckpt_path)
            model2 = CRNN(num_classes=5, img_height=28)
            with self.assertRaises(RuntimeError):
                model2.load_state_dict(torch.load(ckpt_path, map_location="cpu")["model_state_dict"], strict=True)


if __name__ == "__main__":
    unittest.main()