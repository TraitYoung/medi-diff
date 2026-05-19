#!/usr/bin/env python3
"""
结构增强预处理 v2（Gabor 快速版）：用于 img2img 前的输入准备。

原理：
  腺体纤维具有明确方向性（放射状走向），扩散模型缺少这个先验时容易产生旋涡。
  本脚本用多角度 Gabor 滤波器检测所有方向的管状结构，将结果叠加到原图中，
  形成"结构强化输入"作为 img2img 的 init_image。

  vs Frangi：速度快 10-20x，适合批量处理；结构检测能力稍弱但够用。

用法：
  python3 scripts/preprocessing/enhance_structure.py \
      --input-dir  outputs/generated/... \
      --output-dir outputs/enhanced/...  \
      --alpha 0.35
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gabor 结构增强预处理（快速版）")
    p.add_argument("--input-dir",  type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--alpha", type=float, default=0.35,
                   help="增强叠加强度（0=原图，1=最大增强）")
    p.add_argument("--n-angles", type=int, default=8,
                   help="Gabor 检测角度数（默认 8，均匀覆盖 0~180°）")
    p.add_argument("--gabor-ksize", type=int, default=15, help="Gabor 核大小")
    p.add_argument("--gabor-sigma", type=float, default=3.0)
    p.add_argument("--gabor-lambda", type=float, default=8.0, help="Gabor 波长（控制检测尺度）")
    p.add_argument("--unsharp", action="store_true", default=True,
                   help="额外做一次非锐化掩蔽提升细节（默认开）")
    p.add_argument("--no-unsharp", dest="unsharp", action="store_false")
    p.add_argument("--unsharp-sigma", type=float, default=2.0)
    p.add_argument("--unsharp-amount", type=float, default=0.5)
    p.add_argument("--ext", type=str, default=".png")
    p.add_argument("--single", type=Path, default=None)
    return p.parse_args()


def gabor_structure_map(gray: np.ndarray, n_angles: int,
                         ksize: int, sigma: float, lam: float) -> np.ndarray:
    """
    多角度 Gabor 响应取最大值，得到无方向偏好的结构图。
    返回 float32，范围 [0,1]。
    """
    f32 = gray.astype(np.float32)
    response = np.zeros_like(f32)
    for i in range(n_angles):
        theta = np.pi * i / n_angles
        kern = cv2.getGaborKernel(
            (ksize, ksize), sigma=sigma, theta=theta,
            lambd=lam, gamma=0.5, psi=0, ktype=cv2.CV_32F
        )
        resp = cv2.filter2D(f32, cv2.CV_32F, kern)
        response = np.maximum(response, np.abs(resp))
    # 归一化到 [0,1]
    if response.max() > 0:
        response /= response.max()
    return response


def unsharp_mask(gray: np.ndarray, sigma: float, amount: float) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigma)
    sharpened = gray.astype(np.float32) + amount * (gray.astype(np.float32) - blurred)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def enhance_one(gray: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    # Step1：CLAHE 前处理（提升暗纤维可见度）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_input = clahe.apply(gray)

    # Step2：Gabor 结构图
    struct = gabor_structure_map(
        enhanced_input, args.n_angles, args.gabor_ksize,
        args.gabor_sigma, args.gabor_lambda
    )

    # Step3：叠加到原图（让纤维更亮）
    f32 = gray.astype(np.float32)
    result = np.clip(f32 + struct * args.alpha * 60.0, 0, 255)

    # 混合：保留大部分原始外观
    result = (1.0 - args.alpha) * f32 + args.alpha * result
    result = np.clip(result, 0, 255).astype(np.uint8)

    # Step4：可选非锐化掩蔽
    if args.unsharp:
        result = unsharp_mask(result, args.unsharp_sigma, args.unsharp_amount)

    return result


def main() -> None:
    args = parse_args()

    if args.single:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        gray = np.asarray(Image.open(args.single).convert("L"))
        result = enhance_one(gray, args)
        out = args.output_dir / (args.single.stem + args.ext)
        Image.fromarray(result, mode="L").save(out)
        print(f"完成: {out}")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    exts = {".png", ".jpg", ".jpeg"}
    images = [p for p in sorted(args.input_dir.iterdir()) if p.suffix.lower() in exts]
    if not images:
        print(f"[警告] 未找到图像: {args.input_dir}")
        return

    print(f"[enhance-structure-v2] {len(images)} 张  alpha={args.alpha}"
          f"  angles={args.n_angles}  unsharp={args.unsharp}")
    for i, path in enumerate(images):
        gray = np.asarray(Image.open(path).convert("L"))
        result = enhance_one(gray, args)
        Image.fromarray(result, mode="L").save(args.output_dir / (path.stem + args.ext))
        if (i + 1) % 10 == 0 or i == len(images) - 1:
            print(f"  [{i+1}/{len(images)}]")

    print(f"完成 → {args.output_dir}")


if __name__ == "__main__":
    main()
