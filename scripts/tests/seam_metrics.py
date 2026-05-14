#!/usr/bin/env python3
"""接缝/网格能量快速量化（频域十字带尖峰尖锐度）。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def seam_score(img_gray: np.ndarray) -> float:
    """
    DFT 后对中心水平/垂直带取 log 幅值剖面，用剖面标准差之和表征「接缝尖峰」强度。
    越高往往越像规则网格/拼贴（与 new_full 低分方向一致时可用于对比实验）。
    """
    if img_gray.ndim != 2:
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
    g = img_gray.astype(np.float32)
    g = g - float(g.mean())
    f = np.fft.fftshift(np.fft.fft2(g))
    magnitude = np.log(np.abs(f) + 1.0)
    h, w = magnitude.shape
    band = max(3, min(h, w) // 128 + 2)
    cy, cx = h // 2, w // 2
    center_h = magnitude[cy - band : cy + band, :].mean(axis=0)
    center_v = magnitude[:, cx - band : cx + band].mean(axis=1)
    return float(np.std(center_h) + np.std(center_v))


def mean_seam_score_top_k(
    image_paths: list[Path],
    k: int = 5,
) -> tuple[float, list[tuple[str, float]]]:
    """对路径列表前 k 张（或全部）逐张算 seam_score，返回均值与明细。"""
    scores: list[tuple[str, float]] = []
    for p in image_paths[:k]:
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        scores.append((p.name, seam_score(g)))
    if not scores:
        return 0.0, []
    mean = float(np.mean([s for _, s in scores]))
    return mean, scores
