"""Tests for src.preprocessing — preprocessing pipeline + line detection."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import unittest

import cv2
import numpy as np

from src.preprocessing import (
    binarize,
    crop_image,
    crop_to_content,
    denoise,
    deskew,
    detect_text_lines,
    normalize,
    preprocess_for_cnn,
    resize,
    to_grayscale,
)


def make_ink_image(size=(60, 60)):
    """Synthetic white background with a dark filled rectangle (ink)."""
    img = np.full((*size, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (size[0] - 10, size[1] - 10), (0, 0, 0), -1)
    return img


class TestPreprocessing(unittest.TestCase):

    def test_to_grayscale(self):
        img = make_ink_image()
        g = to_grayscale(img)
        self.assertEqual(g.ndim, 2)
        self.assertEqual(g.dtype, np.uint8)
        # already gray
        g2 = to_grayscale(g)
        np.testing.assert_array_equal(g, g2)

    def test_denoise(self):
        img = make_ink_image()
        gray = to_grayscale(img)
        d = denoise(gray)
        self.assertEqual(d.shape, gray.shape)
        self.assertEqual(d.dtype, np.uint8)

    def test_binarize_otsu(self):
        img = make_ink_image()
        g = to_grayscale(img)
        b = binarize(g, method="otsu")
        self.assertTrue(np.all((b == 0) | (b == 255)))

    def test_crop_to_content(self):
        img = make_ink_image()
        g = to_grayscale(img)
        b = binarize(g, method="otsu")
        if np.mean(b) > 127:
            b = 255 - b
        x, y, w, h = crop_to_content(b)
        self.assertGreater(w, 0)
        self.assertGreater(h, 0)

    def test_preprocess_for_cnn_output_shape_and_range(self):
        img = make_ink_image((100, 100))
        res = preprocess_for_cnn(img, 28, 28, mean=0.239, std=0.407)
        self.assertEqual(res.image.shape, (28, 28))
        self.assertEqual(res.image.dtype, np.float32)
        self.assertTrue(np.all(res.image >= -3.0) and np.all(res.image <= 3.0))
        # intermediates present when requested
        res2 = preprocess_for_cnn(img, 28, 28, 0.239, 0.407,
                                  return_intermediates=True)
        self.assertIn("grayscale", res2.intermediates)
        self.assertIn("denoised", res2.intermediates)
        self.assertIn("binary", res2.intermediates)

    def test_detect_text_lines(self):
        img = np.full((200, 400), 255, dtype=np.uint8)
        # draw two horizontal dark bands (two lines)
        cv2.rectangle(img, (20, 40), (380, 55), 0, -1)
        cv2.rectangle(img, (20, 100), (300, 115), 0, -1)
        lines = detect_text_lines(img)
        self.assertGreaterEqual(len(lines), 1)
        for (x, y, w, h) in lines:
            self.assertGreater(h, 0)
            self.assertGreater(w, 0)
            self.assertGreaterEqual(x, 0)
            self.assertGreaterEqual(y, 0)

    def test_normalize_range(self):
        arr = np.zeros((10, 10), dtype=np.uint8)
        n = normalize(arr, mean=0.239, std=0.407)
        self.assertEqual(n.dtype, np.float32)

    def test_resize(self):
        img = np.ones((50, 50), dtype=np.uint8) * 128
        r = resize(img, 28, 28)
        self.assertEqual(r.shape, (28, 28))


if __name__ == "__main__":
    unittest.main()