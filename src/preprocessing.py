"""Image preprocessing for Amharic handwriting OCR.

The same pipeline is used at training time and at inference time so that the
model always sees inputs it was trained on. Intermediate results are kept
(and optionally returned) so they can be inspected or displayed for debugging.

Pipeline (isolated character / CNN input):
    load -> grayscale -> denoise -> deskew -> crop-to-content -> normalize
    -> resize to (img_height, img_width) -> tensor (1, H, W)

Pipeline (text line detection):
    load -> grayscale -> denoise -> binarize -> horizontal projection profile
    -> find line bands -> return bounding boxes (x, y, w, h)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import cv2
import numpy as np


@dataclass
class PreprocessedResult:
    """Holds the final normalized array plus optional intermediates."""

    image: np.ndarray
    intermediates: Dict[str, np.ndarray] = field(default_factory=dict)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert BGR/RGB/RGBA image to single-channel grayscale (uint8)."""
    if image.ndim == 2:
        return image.copy()
    channels = image.shape[2] if image.ndim == 3 else 1
    if channels == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    if channels == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image.copy()


def denoise(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Light Gaussian denoising appropriate for scanned handwriting."""
    if ksize < 1:
        ksize = 1
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(image, (ksize, ksize), 0)


def binarize(image: np.ndarray, method: str = "otsu") -> np.ndarray:
    """Binarize a grayscale image. Returns 0/255 uint8."""
    if method == "otsu":
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "adaptive":
        binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 11,
        )
    else:
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    return binary


def _estimate_skew_angle(binary: np.ndarray, max_angle: float = 45.0) -> float:
    """Estimate skew angle from the binary image using min-area rectangle."""
    coords = cv2.findNonZero(binary)
    if coords is None or len(coords) < 10:
        return 0.0
    rect = cv2.minAreaRect(coords)
    angle = rect[2]
    if angle < -45:
        angle = 90 + angle
    angle = max(-max_angle, min(max_angle, angle))
    return float(angle)


def deskew(image: np.ndarray, binary: np.ndarray, max_angle: float = 45.0) -> np.ndarray:
    """Rotate the grayscale image to correct estimated skew."""
    angle = _estimate_skew_angle(binary, max_angle=max_angle)
    if abs(angle) < 0.5:
        return image.copy()
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


def crop_to_content(binary: np.ndarray, pad: int = 4) -> Tuple[int, int, int, int]:
    """Return (x, y, w, h) bounding box of non-zero pixels in a binary image."""
    coords = cv2.findNonZero(binary)
    if coords is None or len(coords) == 0:
        return (0, 0, binary.shape[1], binary.shape[0])
    x, y, w, h = cv2.boundingRect(coords)
    H, W = binary.shape[:2]
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(W - x, w + 2 * pad)
    h = min(H - y, h + 2 * pad)
    return (int(x), int(y), int(w), int(h))


def crop_image(image: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    """Crop image to (x, y, w, h), padding with border if out of bounds."""
    x, y, w, h = box
    H, W = image.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + w), min(H, y + h)
    cropped = image[y1:y2, x1:x2]
    if cropped.shape[0] < h or cropped.shape[1] < w:
        cropped = cv2.copyMakeBorder(
            cropped,
            max(0, y1 - y), max(0, y2 - H),
            max(0, x1 - x), max(0, x2 - W),
            cv2.BORDER_REPLICATE,
        )
    return cropped


def normalize(image: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Normalize to zero mean / unit std, then clip to a sane float range."""
    arr = image.astype(np.float32) / 255.0
    if std > 0:
        arr = (arr - mean) / std
    return np.clip(arr, -3.0, 3.0).astype(np.float32)


def resize(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize with area interpolation (good for shrinking handwriting)."""
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def preprocess_for_cnn(
    image: np.ndarray,
    img_height: int,
    img_width: int,
    mean: float,
    std: float,
    max_skew_angle: float = 45.0,
    return_intermediates: bool = False,
) -> PreprocessedResult:
    """Full preprocessing pipeline for a single isolated character image.

    Returns a float32 array of shape (H, W) ready to be turned into a
    (1, 1, H, W) torch tensor. If ``return_intermediates`` is True, the
    grayscale/denoised/binarized/deskeved/cropped stages are attached.
    """
    intermediates: Dict[str, np.ndarray] = {}

    gray = to_grayscale(image)
    if return_intermediates:
        intermediates["grayscale"] = gray

    denoised = denoise(gray, ksize=3)
    if return_intermediates:
        intermediates["denoised"] = denoised

    binary = binarize(denoised, method="otsu")
    if return_intermediates:
        intermediates["binary"] = binary

    # Amharic handwriting is typically dark ink on light paper; Otsu may
    # invert depending on the image. Ensure ink is dark (low values) on a
    # light (high values) background for consistent deskew/crop.
    if np.mean(binary) > 127:
        binary = 255 - binary
        if return_intermediates:
            intermediates["binary"] = binary

    deskewed = deskew(denoised, binary, max_angle=max_skew_angle)
    if return_intermediates:
        intermediates["deskewed"] = deskewed

    # Re-binarize the deskewed image for a tight content crop.
    binary_ds = binarize(deskewed, method="otsu")
    if np.mean(binary_ds) > 127:
        binary_ds = 255 - binary_ds
    box = crop_to_content(binary_ds, pad=6)
    cropped = crop_image(deskewed, box)
    if return_intermediates:
        intermediates["cropped"] = cropped

    resized = resize(cropped, img_height, img_width)
    if return_intermediates:
        intermediates["resized"] = resized

    normalized = normalize(resized, mean=mean, std=std)
    return PreprocessedResult(image=normalized, intermediates=intermediates)


def detect_text_lines(
    image: np.ndarray,
    min_line_height: int = 12,
    min_line_width: int = 20,
    line_pad: int = 6,
) -> List[Tuple[int, int, int, int]]:
    """Detect horizontal text-line regions using a projection profile.

    Returns a list of (x, y, w, h) bounding boxes, top-to-bottom. We do NOT
    cut individual characters apart (unreliable for connected handwriting);
    each band is fed whole to the CRNN sequence recognizer.
    """
    gray = to_grayscale(image)
    denoised = denoise(gray, ksize=3)
    binary = binarize(denoised, method="otsu")
    if np.mean(binary) > 127:
        binary = 255 - binary

    dark = (binary < 127).astype(np.uint8)
    projection = dark.sum(axis=1)

    H, W = binary.shape[:2]
    lines: List[Tuple[int, int, int, int]] = []
    in_line = False
    start = 0
    for y in range(H):
        if projection[y] > 0 and not in_line:
            in_line = True
            start = y
        elif projection[y] == 0 and in_line:
            in_line = False
            end = y
            if end - start >= min_line_height:
                x, w = 0, W
                band = binary[start:end, :]
                col_dark = (band < 127).sum(axis=0)
                cols = np.where(col_dark > 0)[0]
                if len(cols) > 0:
                    x = int(max(0, cols[0] - line_pad))
                    w = int(min(W - x, cols[-1] - cols[0] + 1 + 2 * line_pad))
                lines.append((x, start, w, end - start))
    if in_line:
        end = H
        if end - start >= min_line_height:
            lines.append((0, start, W, end - start))

    # Merge bands that are too close together (likely one line split by gaps).
    merged: List[Tuple[int, int, int, int]] = []
    for box in lines:
        if merged and box[1] - (merged[-1][1] + merged[-1][3]) <= max(4, min_line_height // 3):
            prev = merged[-1]
            new_h = (box[1] + box[3]) - prev[1]
            merged[-1] = (prev[0], prev[1], max(prev[2], box[2]), new_h)
        else:
            merged.append(box)

    merged = [b for b in merged if b[3] >= min_line_height and b[2] >= min_line_width]
    return merged


def line_crop_to_cnn_input(
    line_image: np.ndarray,
    img_height: int,
    img_width: int,
    mean: float,
    std: float,
) -> np.ndarray:
    """Preprocess a single text-line crop for the CNN (per-slice classification)."""
    result = preprocess_for_cnn(
        line_image, img_height, img_width, mean, std,
        max_skew_angle=45.0, return_intermediates=False,
    )
    return result.image


__all__ = [
    "PreprocessedResult",
    "to_grayscale",
    "denoise",
    "binarize",
    "deskew",
    "crop_to_content",
    "crop_image",
    "normalize",
    "resize",
    "preprocess_for_cnn",
    "detect_text_lines",
    "line_crop_to_cnn_input",
]