"""OCR inference: isolated-character (CNN) and short text-line (CRNN+CTC).

The :class:`OCRPredictor` loads a CNN baseline and/or a CRNN sequence model,
applies the same preprocessing used in training, and returns Amharic Unicode
text with confidence. It never relies on filesystem order for labels: the
character vocabulary is loaded explicitly from the checkpoint's ``class_to_idx``
(or the documented fallback).

If no checkpoint is available, the predictor starts in "no model" mode and
every prediction returns a clear error telling the operator what file to
provide.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from .config import SETTINGS
from .preprocessing import (
    detect_text_lines,
    preprocess_for_cnn,
    to_grayscale,
)
from .vocabulary import Vocabulary
from .models.cnn import AHCNN, build_model as build_cnn
from .models.crnn import CRNN, greedy_decode


@dataclass
class CharPrediction:
    char: str
    confidence: float
    index: int
    topk: List[Tuple[str, float]] = field(default_factory=list)


@dataclass
class OCRResult:
    mode: str
    text: str
    confidence: float
    char_predictions: List[CharPrediction] = field(default_factory=list)
    line_boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    line_texts: List[str] = field(default_factory=list)
    line_confidences: List[float] = field(default_factory=list)
    error: Optional[str] = None
    intermediates: dict = field(default_factory=dict)


class OCRPredictor:
    def __init__(self, config=None, device: Optional[str] = None):
        self.config = config or SETTINGS
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.vocab: Vocabulary = Vocabulary.load(self.config.models.cnn_checkpoint)
        self.num_classes = self.vocab.size()

        self.cnn: Optional[AHCNN] = None
        self.cnn_ready = False
        self.cnn_metadata: dict = {}
        self._load_cnn()

        self.crnn: Optional[CRNN] = None
        self.crnn_ready = False
        self._load_crnn()

    # ------------------------------------------------------------------ #
    # Model loading
    # ------------------------------------------------------------------ #
    def _load_cnn(self) -> None:
        ckpt = self.config.models.cnn_checkpoint
        if ckpt.exists():
            try:
                self.cnn, self.cnn_metadata = build_cnn(
                    num_classes=self.num_classes,
                    checkpoint_path=str(ckpt),
                    device=str(self.device),
                )
                self.cnn_ready = True
            except Exception as exc:
                print(f"[inference] CNN load failed: {exc}")
        else:
            print(f"[inference] CNN checkpoint not found: {ckpt}")

    def _load_crnn(self) -> None:
        ckpt = self.config.models.crnn_checkpoint
        if ckpt.exists():
            try:
                self.crnn = CRNN(
                    num_classes=self.num_classes,
                    img_height=self.config.preprocessing.img_height,
                )
                state = torch.load(ckpt, map_location=self.device)
                if isinstance(state, dict) and "model_state_dict" in state:
                    state = state["model_state_dict"]
                self.crnn.load_state_dict(state, strict=False)
                self.crnn.to(self.device)
                self.crnn.eval()
                self.crnn_ready = True
            except Exception as exc:
                print(f"[inference] CRNN load failed: {exc}")
        else:
            print(f"[inference] CRNN checkpoint not found: {ckpt}")

    def is_ready(self, mode: Optional[str] = None) -> bool:
        mode = mode or self.config.inference.default_mode
        if mode == "line":
            return self.crnn_ready
        return self.cnn_ready

    def missing_model_message(self, mode: Optional[str] = None) -> str:
        mode = mode or self.config.inference.default_mode
        if mode == "line":
            return (
                "No CRNN line-recognition checkpoint found. "
                f"Expected at: {self.config.models.crnn_checkpoint}. "
                "Train one with scripts/train_crnn.py after collecting "
                "labeled line images (see docs/DATASET.md)."
            )
        return (
            "No CNN character checkpoint found. "
            f"Expected at: {self.config.models.cnn_checkpoint}. "
            "Copy the trained checkpoint (e.g. amharic_ocr_v2.pth) into "
            "models/ or train with scripts/train_cnn.py."
        )

    # ------------------------------------------------------------------ #
    # Character (isolated) recognition
    # ------------------------------------------------------------------ #
    def predict_char(self, image: np.ndarray,
                     return_intermediates: bool = False) -> CharPrediction:
        if not self.cnn_ready or self.cnn is None:
            raise RuntimeError(self.missing_model_message("char"))

        cfg = self.config.preprocessing
        result = preprocess_for_cnn(
            image, cfg.img_height, cfg.img_width, cfg.mean, cfg.std,
            max_skew_angle=cfg.max_skew_angle,
            return_intermediates=return_intermediates,
        )
        tensor = torch.from_numpy(result.image).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.cnn(tensor)
            probs = torch.softmax(logits, dim=1)  # (1, num_classes)
        conf, idx = torch.max(probs, 1)
        idx = int(idx.item())
        conf = float(conf.item())

        topk = []
        top_vals, top_idxs = torch.topk(probs, min(5, self.num_classes), dim=1)
        for v, i in zip(top_vals.tolist()[0], top_idxs.tolist()[0]):
            topk.append((self.vocab.char_for_index(i), float(v)))

        return CharPrediction(
            char=self.vocab.char_for_index(idx),
            confidence=conf,
            index=idx,
            topk=topk,
        )

    # ------------------------------------------------------------------ #
    # Line (sequence) recognition
    # ------------------------------------------------------------------ #
    def predict_line(self, image: np.ndarray) -> OCRResult:
        cfg = self.config.preprocessing
        lines = detect_text_lines(image)

        if not self.crnn_ready or self.crnn is None:
            if self.cnn_ready and self.cnn is not None:
                try:
                    pred = self.predict_char(image, return_intermediates=True)
                    return OCRResult(
                        mode="char-fallback",
                        text=pred.char,
                        confidence=pred.confidence,
                        char_predictions=[pred],
                    )
                except Exception as exc:
                    return OCRResult(mode="line", text="", confidence=0.0,
                                     error=str(exc))
            return OCRResult(
                mode="line", text="", confidence=0.0,
                error=self.missing_model_message("line"),
            )

        recognized: List[str] = []
        confidences: List[float] = []

        for (x, y, w, h) in lines:
            line_crop = image[y:y + h, x:x + w]
            gray = to_grayscale(line_crop)
            scale = cfg.img_height / max(h, 1)
            new_w = max(1, int(w * scale))
            resized = cv2.resize(gray, (new_w, cfg.img_height),
                                 interpolation=cv2.INTER_AREA)
            arr = resized.astype(np.float32) / 255.0
            arr = (arr - cfg.mean) / cfg.std
            arr = np.clip(arr, -3.0, 3.0).astype(np.float32)
            tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(
                self.device)

            with torch.no_grad():
                log_probs = self.crnn(tensor)
            chars = greedy_decode(log_probs, blank=0)[0]
            text = "".join(self.vocab.char_for_index(c) for c in chars)
            conf = self._line_confidence(log_probs[0], chars)
            recognized.append(text)
            confidences.append(conf)

        full_text = " ".join(recognized)
        overall = float(np.mean(confidences)) if confidences else 0.0

        return OCRResult(
            mode="line",
            text=full_text,
            confidence=overall,
            line_boxes=lines,
            line_texts=recognized,
            line_confidences=confidences,
        )

    @staticmethod
    def _line_confidence(log_probs: torch.Tensor, chars: List[int]) -> float:
        if not chars:
            return 0.0
        probs = torch.exp(log_probs.detach().cpu())
        T = probs.shape[0]
        confs = []
        step = max(1, T // max(len(chars), 1))
        for i, ch in enumerate(chars):
            t = min(i * step, T - 1)
            confs.append(float(probs[t, ch].item()))
        return float(np.mean(confs)) if confs else 0.0

    # ------------------------------------------------------------------ #
    # Unified entry point
    # ------------------------------------------------------------------ #
    def predict(self, image: np.ndarray, mode: Optional[str] = None,
                return_intermediates: bool = False) -> OCRResult:
        mode = mode or self.config.inference.default_mode
        if mode == "line":
            return self.predict_line(image)
        try:
            pred = self.predict_char(image, return_intermediates=return_intermediates)
            return OCRResult(
                mode="char",
                text=pred.char,
                confidence=pred.confidence,
                char_predictions=[pred],
                intermediates=pred.__dict__ if return_intermediates else {},
            )
        except Exception as exc:
            return OCRResult(mode="char", text="", confidence=0.0,
                             error=str(exc))


__all__ = ["OCRPredictor", "OCRResult", "CharPrediction"]