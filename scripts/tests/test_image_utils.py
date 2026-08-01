"""CPU unit tests for scripts.core.image_utils."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scripts.core.image_utils import (
    build_mask,
    enhance_input_contrast,
    is_image,
    largest_component,
    resize_long_side,
)


def test_resize_long_side_downscale():
    wide = np.random.RandomState(42).randint(0, 256, (3000, 2000), dtype=np.uint8)
    result = resize_long_side(wide, 2048)
    assert max(result.shape) == 2048


def test_resize_long_side_upscale_preserves_aspect():
    small = np.random.RandomState(42).randint(0, 256, (480, 768), dtype=np.uint8)
    result = resize_long_side(small, 2048)
    assert max(result.shape) == 2048
    assert abs((480 / 768) - (result.shape[0] / result.shape[1])) < 0.02


def test_resize_long_side_noop_and_zero():
    exact = np.random.RandomState(42).randint(0, 256, (1024, 2048), dtype=np.uint8)
    assert resize_long_side(exact, 2048).shape == exact.shape
    assert resize_long_side(exact, 0).shape == exact.shape


def test_is_image_extensions():
    assert is_image(Path("a.png"))
    assert is_image(Path("b.JPEG"))
    assert not is_image(Path("c.txt"))
    assert not is_image("not-a-path")


def test_largest_component_picks_biggest_blob():
    binary = np.zeros((64, 64), dtype=np.uint8)
    binary[2:6, 2:6] = 255
    binary[20:50, 20:50] = 255
    mask = largest_component(binary)
    assert int(mask.sum() // 255) == 30 * 30


def test_build_mask_on_synthetic_breast_blob():
    gray = np.zeros((128, 96), dtype=np.uint8)
    gray[20:110, 10:70] = 160
    mask = build_mask(gray)
    assert mask.dtype == np.uint8
    assert mask.shape == gray.shape
    assert mask.max() == 255
    assert mask.sum() > 0


def test_enhance_input_contrast_only_on_dark():
    bright = np.full((32, 32), 80, dtype=np.uint8)
    dark = np.full((32, 32), 20, dtype=np.uint8)
    assert enhance_input_contrast(bright) is bright or np.array_equal(
        enhance_input_contrast(bright), bright
    )
    out = enhance_input_contrast(dark)
    assert out.shape == dark.shape
    assert out.dtype == np.uint8
