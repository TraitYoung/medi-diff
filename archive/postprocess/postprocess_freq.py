#!/usr/bin/env python3
"""
增强后处理流水线 v2：
  1. Winsorize（百分位裁剪）     → 消灭高光溢出（死白斑块）
  2. 形态学空洞填充              → 缓解黑洞/空洞伪影
  3. 频域 β 斜率校正             → 修正功率谱，提升纹理自然度
  4. CLAHE 局部对比度增强        → 细节可见性（clip 默认 1.0，比旧版温和）
  5. 边缘羽化（可选）            → 平滑乳腺轮廓处的梳齿感

用法：
  python3 scripts/postprocess/postprocess_freq.py \
      --input-dir  outputs/generated/毕业论文_生成图像/<batch> \
      --output-dir outputs/generated/毕业论文_生成图像/<batch>_postv2 \
      --target-beta 2.8 --blend 0.8 --clahe --winsorize --fill-voids
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path for 'from scripts.core...' imports
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.core.postprocess_pipeline import (
    run_postprocess_on_dir,
    run_postprocess_on_image,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="增强后处理流水线 v2（Winsorize + 空洞填充 + 频域校正 + CLAHE）")
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    # Winsorize
    p.add_argument("--winsorize", action="store_true", help="百分位裁剪，消灭死白/死黑溢出")
    p.add_argument("--winsorize-low", type=float, default=0.5, help="下裁剪百分位（%%）")
    p.add_argument("--winsorize-high", type=float, default=99.5, help="上裁剪百分位（%%）")
    # 空洞填充
    p.add_argument("--fill-voids", action="store_true", help="形态学检测并填充孤立暗圆（黑洞伪影）")
    p.add_argument("--void-min-area", type=int, default=30, help="最小黑洞面积（像素），小于此忽略")
    p.add_argument("--void-max-area", type=int, default=3000, help="最大黑洞面积，超过此视为正常暗区")
    p.add_argument("--void-circularity", type=float, default=0.55, help="最低圆度阈值（越高越严格）")
    # 频域校正
    p.add_argument("--target-beta", type=float, default=2.8)
    p.add_argument("--blend", type=float, default=0.25,
                   help="频域滤波结果与原图混合比（1.0=纯滤波）")
    p.add_argument("--no-freq", action="store_true", help="跳过频域校正")
    # CLAHE
    p.add_argument("--clahe", action="store_true")
    p.add_argument("--clahe-clip", type=float, default=1.0, help="CLAHE clip（默认 1.0，比旧版 2.0 温和）")
    p.add_argument("--clahe-grid", type=int, default=8)
    # 边缘羽化
    p.add_argument("--edge-feather", action="store_true", help="对乳腺轮廓边缘做高斯羽化，平滑梳齿感")
    p.add_argument("--feather-ksize", type=int, default=5, help="羽化核大小（奇数）")
    # Unsharp masking 锐化
    p.add_argument("--sharpen", action="store_true", help="Unsharp-mask 锐化，增强组织纹理细节")
    p.add_argument("--sharpen-strength", type=float, default=0.15, help="锐化强度（0-1，越大越锐）")
    p.add_argument("--sharpen-radius", type=int, default=3, help="Unsharp Gaussian 模糊核半径")
    p.add_argument("--bilateral", action="store_true", help="边缘感知平滑，保留解剖边缘同时抑制纹理噪点")
    p.add_argument("--bilateral-d", type=int, default=5, help="Bilateral 滤波直径")
    p.add_argument("--bilateral-sigma-color", type=float, default=30, help="Bilateral 色彩 σ")
    p.add_argument("--bilateral-sigma-space", type=float, default=30, help="Bilateral 空间 σ")
    p.add_argument("--ext", type=str, default=".png")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# Thin CLI wrapper (delegates to scripts.core.postprocess_pipeline)
# ═══════════════════════════════════════════════════════════════════════════════


class _PostprocessArgsAdapter:
    """Adapter that exposes argparse attributes matching PostprocessParams fields."""

    def __init__(self, args: argparse.Namespace):
        self.winsorize = args.winsorize
        self.winsorize_low = args.winsorize_low
        self.winsorize_high = args.winsorize_high
        self.fill_voids = args.fill_voids
        self.void_min_area = args.void_min_area
        self.void_max_area = args.void_max_area
        self.void_circularity = args.void_circularity
        self.no_freq = args.no_freq
        self.target_beta = args.target_beta
        self.blend = args.blend
        self.clahe = args.clahe
        self.clahe_clip = args.clahe_clip
        self.clahe_grid = args.clahe_grid
        self.sharpen = args.sharpen
        self.sharpen_strength = args.sharpen_strength
        self.sharpen_radius = args.sharpen_radius
        self.bilateral = args.bilateral
        self.bilateral_d = args.bilateral_d
        self.bilateral_sigma_color = args.bilateral_sigma_color
        self.bilateral_sigma_space = args.bilateral_sigma_space
        self.edge_feather = args.edge_feather
        self.feather_ksize = args.feather_ksize
        self.ext = args.ext


def main() -> None:
    args = parse_args()
    run_postprocess_on_dir(args.input_dir, args.output_dir, _PostprocessArgsAdapter(args))


if __name__ == "__main__":
    main()
