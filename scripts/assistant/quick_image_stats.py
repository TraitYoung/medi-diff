#!/usr/bin/env python3
"""
轻量「图像识别」摘要：对灰度医学图提取可解释统计，供顾问模型结合评审指标做建议。
不替代深度学习分类器；与 review_generated_images 的细指标互补。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def stats_one(path: Path) -> dict[str, Any]:
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if g is None:
        return {"file": path.name, "error": "read_failed"}
    g = g.astype(np.float64)
    lap = cv2.Laplacian(g.astype(np.uint8), cv2.CV_64F)
    lap_var = float(lap.var())
    edges = cv2.Canny(g.astype(np.uint8), 50, 150)
    edge_density = float(np.mean(edges > 0))
    # 简单「亮部占比」：高亮像素比例（粗看是否过曝）
    bright_ratio = float(np.mean(g > 240))
    dark_ratio = float(np.mean(g < 15))
    return {
        "file": path.name,
        "mean": round(float(np.mean(g)), 2),
        "std": round(float(np.std(g)), 2),
        "laplacian_var": round(lap_var, 2),
        "edge_density": round(edge_density, 4),
        "bright_ratio_gt240": round(bright_ratio, 4),
        "dark_ratio_lt15": round(dark_ratio, 4),
    }


def collect_paths(paths: list[Path]) -> list[dict[str, Any]]:
    """仅对给定路径做统计（顾问流水线优先：与 VL/排名抽样对齐，避免整目录重复读图）。"""
    return [stats_one(p) for p in paths if p.is_file()]


def collect_dir(
    images_dir: Path,
    *,
    exts: tuple[str, ...] = (".png", ".jpg", ".jpeg"),
    max_files: int | None = None,
) -> list[dict[str, Any]]:
    """遍历目录；若 max_files 则按文件名排序后至多处理这么多张（大图库时显著提速）。"""
    out: list[dict[str, Any]] = []
    for p in sorted(images_dir.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        out.append(stats_one(p))
        if max_files is not None and len(out) >= max_files:
            break
    return out
