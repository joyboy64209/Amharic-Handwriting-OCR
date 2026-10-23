"""Tests for src.api.app — FastAPI route validation and error handling.

Uses stdlib unittest only (no httpx/TestClient installed). Route handlers are
invoked directly with fake UploadFile objects; the module-level predictor is
stubbed so no model checkpoint is required.
"""

import asyncio
import io
import sys
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import unittest

import cv2
import numpy as np
from fastapi import HTTPException, status

import src.api.app as app_module


class FakeUploadFile:
    """Minimal stand-in for fastapi.UploadFile used by _read_upload / ocr."""

    def __init__(self, filename: str, content: bytes):
        self.filename = filename
        self.file = io.BytesIO(content)
        self.content_type = "image/png" if filename.lower().endswith(".png") else "image/jpeg"

    async def read(self):
        return self.file.getvalue()


def make_png_image(size=(60, 60)) -> bytes:
    """Encode a synthetic image (dark square on white) as PNG bytes."""
    img = np.full((*size, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (size[0] - 10, size[1] - 10), (0, 0, 0), -1)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


class TestReadUpload(unittest.TestCase):

    def test_empty_file(self):
        file = FakeUploadFile("x.png", b"")
        with self.assertRaises(HTTPException) as ctx:
            app_module._read_upload(file)
        self.assertEqual(ctx.exception.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(ctx.exception.detail["error"], "empty_file")

    def test_invalid_file_type(self):
        file = FakeUploadFile("document.pdf", b"%PDF dummy")
        with self.assertRaises(HTTPException) as ctx:
            app_module._read_upload(file)
        self.assertEqual(ctx.exception.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(ctx.exception.detail["error"], "invalid_file_type")

    def test_unreadable_image(self):
        file = FakeUploadFile("broken.png", b"not a real image at all")
        with self.assertRaises(HTTPException) as ctx:
            app_module._read_upload(file)
        self.assertEqual(ctx.exception.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(ctx.exception.detail["error"], "unreadable_image")

    def test_file_too_large(self):
        old_limit = app_module.SETTINGS.api.max_upload_bytes
        try:
            app_module.SETTINGS.api.max_upload_bytes = 1
            file = FakeUploadFile("small.png", make_png_image())
            with self.assertRaises(HTTPException) as ctx:
                app_module._read_upload(file)
            self.assertEqual(ctx.exception.status_code, status.HTTP_413_CONTENT_TOO_LARGE)
            self.assertEqual(ctx.exception.detail["error"], "file_too_large")
        finally:
            app_module.SETTINGS.api.max_upload_bytes = old_limit

    def test_valid_image_decodes(self):
        file = FakeUploadFile("ok.png", make_png_image())
        image = app_module._read_upload(file)
        self.assertIsInstance(image, np.ndarray)
        self.assertEqual(image.ndim, 3)
        self.assertEqual(image.shape[2], 3)


class TestOcrRoute(unittest.TestCase):

    def setUp(self):
        self.saved_predictor = app_module.predictor
        self.saved_error = app_module.predictor_load_error

    def tearDown(self):
        app_module.predictor = self.saved_predictor
        app_module.predictor_load_error = self.saved_error

    def _run(self, coro):
        return asyncio.run(coro)

    def test_ocr_no_model(self):
        app_module.predictor = None
        app_module.predictor_load_error = "no checkpoint"
        file = FakeUploadFile("ok.png", make_png_image())
        with self.assertRaises(HTTPException) as ctx:
            self._run(app_module.ocr(file, mode="char"))
        self.assertEqual(ctx.exception.status_code, status.HTTP_503_SERVICE_UNAVAILABLE)
        self.assertEqual(ctx.exception.detail["error"], "model_not_available")

    def test_ocr_invalid_mode(self):
        app_module.predictor = MagicMock()
        app_module.predictor_load_error = None
        app_module.predictor.cnn_ready = True
        app_module.predictor.crnn_ready = False
        file = FakeUploadFile("ok.png", make_png_image())
        with self.assertRaises(HTTPException) as ctx:
            self._run(app_module.ocr(file, mode="weird"))
        self.assertEqual(ctx.exception.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(ctx.exception.detail["error"], "invalid_mode")

    def test_ocr_line_mode_unsupported(self):
        app_module.predictor = MagicMock()
        app_module.predictor_load_error = None
        app_module.predictor.cnn_ready = True
        app_module.predictor.crnn_ready = False
        app_module.predictor.missing_model_message = MagicMock(
            return_value="No CRNN checkpoint."
        )
        file = FakeUploadFile("ok.png", make_png_image())
        with self.assertRaises(HTTPException) as ctx:
            self._run(app_module.ocr(file, mode="line"))
        self.assertEqual(ctx.exception.status_code, status.HTTP_503_SERVICE_UNAVAILABLE)
        self.assertEqual(ctx.exception.detail["error"], "model_not_available")
        app_module.predictor.missing_model_message.assert_called_with("line")

    def test_ocr_success_char_mode(self):
        fake_pred = MagicMock()
        fake_pred.char = "ሰ"
        fake_pred.confidence = 0.93
        fake_pred.topk = [("ሰ", 0.93), ("ስ", 0.04)]

        fake_result = MagicMock()
        fake_result.error = None
        fake_result.mode = "char"
        fake_result.text = "ሰ"
        fake_result.confidence = 0.93
        fake_result.low_confidence = False
        fake_result.char_predictions = [fake_pred]
        fake_result.line_boxes = []

        app_module.predictor = MagicMock()
        app_module.predictor_load_error = None
        app_module.predictor.cnn_ready = True
        app_module.predictor.crnn_ready = False
        app_module.predictor.predict = MagicMock(return_value=fake_result)

        file = FakeUploadFile("ok.png", make_png_image())
        data = self._run(app_module.ocr(file, mode="char"))
        self.assertTrue(data["ok"])
        self.assertEqual(data["text"], "ሰ")
        self.assertEqual(data["prediction"], "ሰ")
        self.assertEqual(data["mode"], "char")
        self.assertFalse(data["low_confidence"])

    def test_ocr_inference_error(self):
        app_module.predictor = MagicMock()
        app_module.predictor_load_error = None
        app_module.predictor.cnn_ready = True
        app_module.predictor.crnn_ready = False
        app_module.predictor.predict = MagicMock(side_effect=RuntimeError("boom"))
        file = FakeUploadFile("ok.png", make_png_image())
        with self.assertRaises(HTTPException) as ctx:
            self._run(app_module.ocr(file, mode="char"))
        self.assertEqual(ctx.exception.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
        self.assertEqual(ctx.exception.detail["error"], "inference_error")


class TestHealth(unittest.TestCase):

    def test_health_shape(self):
        data = app_module.health()
        self.assertIn("status", data)
        self.assertIn("cnn_ready", data)
        self.assertIn("crnn_ready", data)
        self.assertIn("default_mode", data)
        self.assertIn("vocab_size", data)
        self.assertIn("vocab_source", data)


if __name__ == "__main__":
    unittest.main()