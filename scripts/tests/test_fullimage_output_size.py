#!/usr/bin/env python3
"""验证 fullimage_output_long_side 对输出长边的控制（双向：放大 + 缩小）。

使用最小化假 pipe（直接返回输入潜变量对应的图像），不依赖真实 GPU 模型加载。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image
from unittest.mock import MagicMock


def test_resize_long_side_downscale():
    """Large image should be downscaled to target long side."""
    from scripts.core.image_utils import resize_long_side as resize_long_side_gray

    wide = np.random.RandomState(42).randint(0, 256, (3000, 2000), dtype=np.uint8)
    result = resize_long_side_gray(wide, 2048)
    assert max(result.shape) == 2048, f"Long side should be 2048, got {max(result.shape)}"


def test_resize_long_side_upscale():
    """Small image should be upscaled to target long side."""
    from scripts.core.image_utils import resize_long_side as resize_long_side_gray

    small = np.random.RandomState(42).randint(0, 256, (480, 768), dtype=np.uint8)
    result = resize_long_side_gray(small, 2048)
    assert max(result.shape) == 2048, f"Long side should be 2048, got {max(result.shape)}"
    aspect_orig = 480 / 768
    aspect_new = result.shape[0] / result.shape[1]
    assert abs(aspect_orig - aspect_new) < 0.02, f"Aspect ratio changed: {aspect_orig:.3f} -> {aspect_new:.3f}"


def test_resize_long_side_noop():
    """Image already at target should not be resized."""
    from scripts.core.image_utils import resize_long_side as resize_long_side_gray

    exact = np.random.RandomState(42).randint(0, 256, (1024, 2048), dtype=np.uint8)
    result = resize_long_side_gray(exact, 2048)
    assert result.shape == exact.shape, f"Should not resize when already at target, got {result.shape}"


def test_resize_long_side_zero_noop():
    """long_side=0 should keep native size."""
    from scripts.core.image_utils import resize_long_side as resize_long_side_gray

    native = np.random.RandomState(42).randint(0, 256, (800, 1200), dtype=np.uint8)
    result = resize_long_side_gray(native, 0)
    assert result.shape == native.shape, f"0 should keep native size, got {result.shape}"


def test_fullimage_generate_upscales_to_output_long_side():
    """fullimage_generate with output_long_side=2048 should produce 2048px long edge."""
    from scripts.generation.run_mammo_sd15 import fullimage_generate

    src = np.random.RandomState(42).randint(20, 200, (800, 600), dtype=np.uint8)

    class MockPipe:
        device = torch.device("cpu")
        def __call__(self, **kwargs):
            img = kwargs["image"]
            return MagicMock(images=[img])

    pipe = MockPipe()
    gen = torch.Generator(device="cpu").manual_seed(42)

    result = fullimage_generate(
        src_gray=src, pipe=pipe,
        prompt="test", negative_prompt="test",
        strength=0.5, guidance_scale=8.5,
        num_inference_steps=10, generator=gen,
        fullimage_long_side=768, fullimage_output_long_side=2048,
    )

    assert result.dtype == np.uint8
    assert result.ndim == 2
    assert max(result.shape) == 2048, f"Expected long side 2048, got {max(result.shape)}"


def test_fullimage_generate_native_when_zero():
    """fullimage_generate with output_long_side=0 should keep native resolution."""
    from scripts.generation.run_mammo_sd15 import fullimage_generate

    src = np.random.RandomState(42).randint(20, 200, (800, 600), dtype=np.uint8)

    class MockPipe:
        device = torch.device("cpu")
        def __call__(self, **kwargs):
            img = kwargs["image"]
            return MagicMock(images=[img])

    pipe = MockPipe()
    gen = torch.Generator(device="cpu").manual_seed(42)

    result = fullimage_generate(
        src_gray=src, pipe=pipe,
        prompt="test", negative_prompt="test",
        strength=0.5, guidance_scale=8.5,
        num_inference_steps=10, generator=gen,
        fullimage_long_side=768, fullimage_output_long_side=0,
    )

    assert result.dtype == np.uint8
    assert result.ndim == 2
    assert max(result.shape) <= 768, f"Native should be ≤768, got {max(result.shape)}"


if __name__ == "__main__":
    test_resize_long_side_downscale()
    test_resize_long_side_upscale()
    test_resize_long_side_noop()
    test_resize_long_side_zero_noop()
    test_fullimage_generate_upscales_to_output_long_side()
    test_fullimage_generate_native_when_zero()
    print("All tests passed.")
