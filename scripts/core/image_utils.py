"""Shared image utilities: resize, mask, contrast, component selection.

Provides unified image processing primitives used across generation,
evaluation, and preprocessing scripts.
"""

from __future__ import annotations

import cv2
import numpy as np


# ── Resize ────────────────────────────────────────────────────────────────────

def resize_long_side(
    gray: np.ndarray,
    long_side: int,
    *,
    only_downscale: bool = False,
    up_interp: int = cv2.INTER_LANCZOS4,
    down_interp: int = cv2.INTER_AREA,
    min_side: int = 1,
    eps: float = 0.0,
) -> np.ndarray:
    """Scale so long side == long_side, preserving aspect ratio.

    Args:
        gray: Input grayscale image (H, W) uint8.
        long_side: Target long side. ≤ 0 means no-op.
        only_downscale: If True, skip when current long side ≤ target.
        up_interp: Interpolation for upscaling (default LANCZOS4).
        down_interp: Interpolation for downscaling (default INTER_AREA).
        min_side: Minimum pixel size for output dimensions.
        eps: Skip resize when |scale - 1.0| < eps.
    """
    if long_side <= 0:
        return gray
    h, w = gray.shape[:2]
    cur_long = max(h, w)
    if only_downscale and cur_long <= long_side:
        return gray
    if cur_long == long_side:
        return gray
    scale = long_side / cur_long
    if eps > 0 and abs(scale - 1.0) < eps:
        return gray
    nh = max(min_side, int(round(h * scale)))
    nw = max(min_side, int(round(w * scale)))
    interp = up_interp if scale > 1 else down_interp
    return cv2.resize(gray, (nw, nh), interpolation=interp)


# ── Contrast ──────────────────────────────────────────────────────────────────

def enhance_input_contrast(
    gray: np.ndarray,
    clahe_clip: float = 0.8,
    clahe_grid: int = 8,
) -> np.ndarray:
    """Apply CLAHE if median < 40 (dark images)."""
    if np.median(gray) >= 40:
        return gray
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
    return clahe.apply(gray)


# ── File helpers ──────────────────────────────────────────────────────────────

def is_image(path: object) -> bool:
    """Return True if path has a recognised image extension."""
    from pathlib import Path
    if isinstance(path, Path):
        return path.suffix.lower() in frozenset({'.bmp', '.webp', '.jpg', '.png', '.jpeg'})
    return False


# ── Mask ──────────────────────────────────────────────────────────────────────

def largest_component(binary: np.ndarray) -> np.ndarray:
    """Return the largest connected component as a binary mask."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return np.zeros_like(binary, dtype=np.uint8)
    idx = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    return (labels == idx).astype(np.uint8) * 255


def build_mask(gray: np.ndarray) -> np.ndarray:
    """Build breast tissue mask from grayscale mammogram via Otsu + morphology."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.count_nonzero(th) > th.size * 0.8:
        th = cv2.bitwise_not(th)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)
    return largest_component(th)
