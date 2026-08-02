"""Unit tests for GPU mem-profile resolution (no CUDA required)."""
from __future__ import annotations

import pytest

from scripts.core.device_profile import (
    resolve_mem_profile,
    should_empty_cache_between_images,
)

_GIB = 1024**3


@pytest.mark.parametrize(
    ("bytes_", "expected"),
    [
        (32 * _GIB, "cloud"),
        (20 * _GIB, "cloud"),
        (16 * _GIB, "local"),
        (12 * _GIB, "local"),
        (10 * _GIB, "local"),
        (8 * _GIB, "tight"),
        (0, "local"),
    ],
)
def test_resolve_mem_profile_auto_thresholds(bytes_: int, expected: str) -> None:
    total = None if bytes_ == 0 else bytes_
    assert resolve_mem_profile("auto", total_memory_bytes=total) == expected


@pytest.mark.parametrize("name", ["cloud", "local", "tight"])
def test_resolve_mem_profile_explicit(name: str) -> None:
    assert resolve_mem_profile(name, total_memory_bytes=8 * _GIB) == name


def test_resolve_mem_profile_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        resolve_mem_profile("ultra")


def test_should_empty_cache_between_images() -> None:
    assert should_empty_cache_between_images("local") is True
    assert should_empty_cache_between_images("tight") is True
    assert should_empty_cache_between_images("cloud") is False
