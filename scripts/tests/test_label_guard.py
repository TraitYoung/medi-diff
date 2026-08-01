"""CPU unit tests for scripts.core.label_guard."""
from __future__ import annotations

import numpy as np
from scripts.core.label_guard import (
    erase_background_labels,
    erase_bright_border_labels,
    feather_canvas_edge,
)


def _tissue_canvas(h: int = 256, w: int = 192) -> np.ndarray:
    """Dark background + bright tissue blob (synthetic mammogram-like)."""
    gray = np.full((h, w), 12, dtype=np.uint8)
    gray[40 : h - 20, 20 : w - 30] = 140
    return gray


def test_feather_canvas_edge_darkens_border():
    gray = np.full((64, 64), 200, dtype=np.uint8)
    out = feather_canvas_edge(gray, feather_px=3)
    assert out.dtype == np.uint8
    assert out[0, 0] < gray[0, 0]
    assert out[32, 32] == 200


def test_feather_canvas_edge_zero_is_noop():
    gray = np.full((32, 32), 100, dtype=np.uint8)
    assert np.array_equal(feather_canvas_edge(gray, feather_px=0), gray)


def test_erase_bright_border_labels_off_mode():
    gray = _tissue_canvas()
    gray[0:6, 10:40] = 255
    out = erase_bright_border_labels(gray, mode="off")
    assert np.array_equal(out, gray)


def test_erase_bright_border_labels_threshold_clears_strip():
    gray = _tissue_canvas()
    gray[0:8, :] = 255
    out = erase_bright_border_labels(gray, border_frac=0.05, bright_pct=90.0, mode="threshold")
    assert out[0:4, :].max() < 255


def test_erase_background_labels_returns_same_shape():
    gray = _tissue_canvas()
    # Small bright blob on dark background (label-like)
    gray[8:20, 8:40] = 250
    out = erase_background_labels(gray)
    assert out.shape == gray.shape
    assert out.dtype == np.uint8
