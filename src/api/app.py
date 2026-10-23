"""FastAPI backend for the Amharic OCR web application.

Endpoints:
    GET  /health     — service readiness, model status
    POST /ocr        — upload an image, get recognized Amharic Unicode text

File type and size are validated. Unreadable images and missing models return
clear, structured errors. No 3D/visualization payloads are served.
"""

from __future__ import annotations

import io
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.config import SETTINGS
from src.inference import OCRPredictor


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load the OCR predictor once at startup (replaces on_event('startup'))."""
    _load_predictor()
    yield


app = FastAPI(
    title="Amharic Handwriting OCR",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global predictor (loaded once at startup).
# ---------------------------------------------------------------------------

predictor: Optional[OCRPredictor] = None
predictor_load_error: Optional[str] = None


def _load_predictor() -> None:
    global predictor, predictor_load_error
    try:
        predictor = OCRPredictor()
        modes = []
        if predictor.cnn_ready:
            modes.append("char")
        if predictor.crnn_ready:
            modes.append("line")
        print(f"[api] OCR predictor ready. Modes: {modes or ['none (no model)']}")
    except Exception as exc:
        predictor_load_error = str(exc)
        print(f"[api] Predictor could not be initialized: {exc}")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _allowed_file(filename: str) -> bool:
    return any(filename.lower().endswith(ext)
               for ext in SETTINGS.api.allowed_extensions)


def _read_upload(upload: UploadFile) -> np.ndarray:
    """Read an uploaded file into a BGR numpy image."""
    if upload.filename is None or not _allowed_file(upload.filename):
        allowed = ", ".join(SETTINGS.api.allowed_extensions)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_file_type",
                "message": f"Unsupported file type. Allowed: {allowed}",
            },
        )

    contents = upload.file.read()
    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "empty_file",
                    "message": "The uploaded file is empty."},
        )
    if len(contents) > SETTINGS.api.max_upload_bytes:
        mb = SETTINGS.api.max_upload_bytes / (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail={
                "error": "file_too_large",
                "message": f"File exceeds {mb:.0f} MB limit.",
            },
        )

    file_bytes = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "unreadable_image",
                "message": "Could not decode the image. Use a valid PNG/JPEG.",
            },
        )
    return image


def _mode_supported(mode: str) -> bool:
    if predictor is None:
        return False
    if mode == "char":
        return predictor.cnn_ready
    if mode == "line":
        return predictor.crnn_ready
    return False

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    cnn_path = SETTINGS.models.cnn_checkpoint
    crnn_path = SETTINGS.models.crnn_checkpoint
    return {
        "status": "ok" if predictor is not None else "error",
        "predictor_loaded": predictor is not None and predictor_load_error is None,
        "load_error": predictor_load_error,
        "cnn_checkpoint": str(cnn_path),
        "cnn_checkpoint_exists": cnn_path.exists(),
        "cnn_ready": bool(predictor and predictor.cnn_ready),
        "crnn_checkpoint": str(crnn_path),
        "crnn_checkpoint_exists": crnn_path.exists(),
        "crnn_ready": bool(predictor and predictor.crnn_ready),
        "default_mode": SETTINGS.inference.default_mode,
        "vocab_size": predictor.vocab.size() if predictor else None,
        "vocab_source": predictor.vocab.source if predictor else None,
    }


@app.post("/ocr")
async def ocr(file: UploadFile = File(...), mode: Optional[str] = None,
              decoder: Optional[str] = None, beam_width: Optional[int] = None):
    """Recognize Amharic text in an uploaded image.

    ``decoder`` selects the CTC decode strategy for line mode:
    ``"greedy"`` (default) or ``"beam"`` (prefix beam search). ``beam_width``
    controls the beam size when ``decoder="beam"``.
    """
    if predictor is None or predictor_load_error is not None:
        msg = predictor_load_error or (
            "No model loaded. "
            f"Expected CNN checkpoint at {SETTINGS.models.cnn_checkpoint}. "
            "Copy a trained checkpoint into models/ or train with "
            "scripts/train_cnn.py."
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "model_not_available", "message": msg},
        )

    mode = mode or SETTINGS.inference.default_mode
    if mode not in ("char", "line"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "invalid_mode",
                    "message": "mode must be 'char' or 'line'"},
        )
    if not _mode_supported(mode):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "model_not_available",
                "message": predictor.missing_model_message(mode),
            },
        )

    try:
        image = _read_upload(file)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "read_error", "message": str(exc)},
        )

    try:
        result = predictor.predict(
            image,
            mode=mode,
            decoder=decoder or "greedy",
            beam_width=beam_width if beam_width is not None else 10,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "inference_error", "message": str(exc)},
        )

    if result.error:
        return {
            "ok": False,
            "mode": result.mode,
            "error": result.error,
            "text": "",
            "confidence": 0.0,
        }

    response = {
        "ok": True,
        "mode": result.mode,
        "text": result.text,
        "confidence": round(float(result.confidence), 4),
        "low_confidence": float(result.confidence) < SETTINGS.inference.low_confidence_threshold,
    }
    if result.char_predictions:
        response["prediction"] = result.char_predictions[0].char
        response["topk"] = result.char_predictions[0].topk
    if result.line_boxes:
        response["lines"] = [
            {"text": t, "confidence": round(c, 4), "box": list(b)}
            for t, c, b in zip(result.line_texts, result.line_confidences, result.line_boxes)
        ]
    return response


# Serve the frontend SPA if present.
frontend_dir = Path(__file__).resolve().parents[2] / "frontend"
if (frontend_dir / "index.html").exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
else:
    @app.get("/")
    def root():
        return {"message": "Amharic OCR API. See /health. Frontend not built."}