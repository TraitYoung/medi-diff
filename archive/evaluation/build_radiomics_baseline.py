#!/usr/bin/env python3
"""
在真实 CBIS 钼靶图上统计 20 维简化影像组学特征，PCA 至 10 维后估计协方差，保存 Mahalanobis 基线。

输出 `radiomics_baseline.npz` 字段：
- pca_mean, pca_components, y_mean, inv_cov10

依赖：numpy, scipy, scikit-image, opencv-python（与主项目一致）。
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/evaluation"))

from review_semantics import extract_radiomics_vector  # noqa: E402
from review_generated_images import build_mask, is_image, resize_long_side  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="构建影像组学 Mahalanobis 基线（CBIS 真实图）")
    p.add_argument("--real-images-dir", type=Path, required=True, help="真实钼靶图根目录")
    p.add_argument("--output-npz", type=Path, default=ROOT / "outputs/eval/radiomics_baseline.npz")
    p.add_argument("--max-samples", type=int, default=500, help="最多抽样张数")
    p.add_argument("--resize-long-side", type=int, default=1024)
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(p for p in args.real_images_dir.rglob("*") if p.is_file() and is_image(p))
    if not paths:
        raise SystemExit(f"未找到图像: {args.real_images_dir}")

    rng = random.Random(args.seed)
    if args.max_samples > 0 and len(paths) > args.max_samples:
        paths = rng.sample(paths, args.max_samples)

    feats: list[np.ndarray] = []
    for p in tqdm(paths, desc="Extract radiomics"):
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        if args.resize_long_side > 0:
            gray = resize_long_side(gray, args.resize_long_side)
        mask = build_mask(gray)
        v = extract_radiomics_vector(gray, mask)
        if v is not None and v.size == 20:
            feats.append(v)

    if len(feats) < 30:
        raise SystemExit(f"有效特征过少 ({len(feats)})，请检查目录与掩膜。")

    X = np.stack(feats, axis=0).astype(np.float64)
    try:
        from sklearn.decomposition import PCA
    except Exception as e:
        raise SystemExit(f"需要 sklearn: {e}") from e

    pca = PCA(n_components=10, random_state=args.seed)
    Y = pca.fit_transform(X)
    y_mean = Y.mean(axis=0)
    Yc = Y - y_mean
    cov = np.cov(Yc, rowvar=False) + 1e-3 * np.eye(10)
    inv_cov = np.linalg.inv(cov)

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_npz,
        pca_mean=pca.mean_.astype(np.float64),
        pca_components=pca.components_.astype(np.float64),
        y_mean=y_mean.astype(np.float64),
        inv_cov10=inv_cov.astype(np.float64),
        n_samples=np.array([X.shape[0]], dtype=np.int32),
    )
    print(f"Saved {args.output_npz} (N={X.shape[0]}, explained_var_ratio_sum={pca.explained_variance_ratio_.sum():.4f})")


if __name__ == "__main__":
    main()
