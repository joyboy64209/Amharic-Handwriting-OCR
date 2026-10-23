"""Handwriting-friendly image augmentation.

Transforms are modest so as not to destroy the structure of Amharic
characters: rotation, scale, shear, brightness, and elastic deformation.
All operate on a single grayscale uint8 image and return a grayscale uint8
image of the same size.
"""

from __future__ import annotations

import cv2
import numpy as np


def _clip01(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0, 255).astype(np.uint8)


def rotate(image: np.ndarray, degrees: float, center: str = "image") -> np.ndarray:
    """Rotate by ``degrees`` (fill border with background/white)."""
    h, w = image.shape[:2]
    if center == "image":
        c = (w // 2, h // 2)
    else:
        c = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(c, degrees, 1.0)
    return cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


def scale(image: np.ndarray, factor: float) -> np.ndarray:
    """Scale around center then pad/crop back to original size."""
    h, w = image.shape[:2]
    new_h, new_w = max(1, int(h * factor)), max(1, int(w * factor))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    out = np.full((h, w), 255, dtype=np.uint8)
    y0 = max(0, (h - new_h) // 2)
    x0 = max(0, (w - new_w) // 2)
    y1 = y0 + min(new_h, h - y0)
    x1 = x0 + min(new_w, w - x0)
    out[y0:y1, x0:x1] = resized[: y1 - y0, : x1 - x0]
    return out


def shear(image: np.ndarray, degrees: float) -> np.ndarray:
    """Shear horizontally by ``degrees``."""
    h, w = image.shape[:2]
    angle_rad = np.deg2rad(degrees)
    M = np.float32([[1, np.tan(angle_rad), 0], [0, 1, 0]])
    return cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


def brightness(image: np.ndarray, gain: float) -> np.ndarray:
    """Multiply pixel values by ``gain`` (1.0 = unchanged)."""
    return _clip01(image.astype(np.float32) * gain)


def elastic_deform(
    image: np.ndarray, alpha: float = 30.0, sigma: float = 4.0, seed: int | None = None
) -> np.ndarray:
    """Elastic deformation (mild, handwriting-appropriate).

    Displaces pixels by a smoothed random field. ``alpha`` controls
    displacement magnitude; ``sigma`` controls smoothness.
    """
    rng = np.random.default_rng(seed)
    h, w = image.shape[:2]
    dx = rng.normal(0, 1, (h, w)).astype(np.float32)
    dy = rng.normal(0, 1, (h, w)).astype(np.float32)
    dx = cv2.GaussianBlur(dx, (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur(dy, (0, 0), sigma) * alpha

    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    return cv2.remap(
        image, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


def augment(
    image: np.ndarray,
    rotation_degrees: float = 8.0,
    scale_factor: float = 0.95,
    shear_degrees: float = 4.0,
    brightness_gain: float = 0.15,
    elastic: bool = True,
    seed: int | None = None,
) -> np.ndarray:
    """Apply a random subset of handwriting augmentations.

    Order: rotate -> scale -> shear -> brightness -> elastic. Each transform
    is applied with probability 0.5 (elastic always if requested).
    """
    rng = np.random.default_rng(seed)
    img = image.copy()

    if rng.random() < 0.5:
        img = rotate(img, float(rng.uniform(-rotation_degrees, rotation_degrees)))
    if rng.random() < 0.5:
        img = scale(img, float(rng.uniform(1.0 - scale_factor, 1.0 + scale_factor)))
    if rng.random() < 0.5:
        img = shear(img, float(rng.uniform(-shear_degrees, shear_degrees)))
    if rng.random() < 0.5:
        img = brightness(img, float(rng.uniform(1.0 - brightness_gain, 1.0 + brightness_gain)))
    if elastic and rng.random() < 0.5:
        img = elastic_deform(img, alpha=25.0, sigma=4.0, seed=int(rng.integers(0, 1 << 31)))
    return _clip01(img)


__all__ = ["rotate", "scale", "shear", "brightness", "elastic_deform", "augment"]