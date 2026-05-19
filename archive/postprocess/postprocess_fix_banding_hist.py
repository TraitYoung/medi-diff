#!/usr/bin/env python3
"""
针对性修复 BANDING 和 HIST_OFF_REAL 的后处理脚本。

两步操作（均为纯后处理，不需要重新生成）：

Step 1 – 横纹频域压制（修 BANDING）
  对 banding_score < --banding-threshold 的图，
  在频域 FFT 中定向压制水平低频条纹分量，
  再混合回原图（blend 控制强度）。

Step 2 – 直方图匹配（修 HIST_OFF_REAL）
  对 hist_wass > --hist-threshold 的图，
  从真实钼靶集合随机采样一张参考图，
  用 skimage.exposure.match_histograms 对齐分布。
  仅在乳腺掩膜内做匹配，保留背景纯黑。

用法：
  python3 scripts/postprocess/postprocess_fix_banding_hist.py \
      --images-dir    outputs/generated/.../day5_official_100_*_000 \
      --report-csv    outputs/reviews/review_output/review_report.csv \
      --real-dir      datasets/jpeg \
      --output-dir    outputs/generated/.../day5_official_100_*_fixed
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.exposure import match_histograms

ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", type=Path, required=True)
    p.add_argument("--report-csv", type=Path,
                   default=ROOT / "outputs/reviews/review_output/review_report.csv")
    p.add_argument("--real-dir", type=Path, default=ROOT / "datasets/jpeg")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--banding-threshold", type=float, default=0.70,
                   help="banding_score 低于此值时触发频域横纹压制")
    p.add_argument("--hist-threshold", type=float, default=0.22,
                   help="hist_wass 高于此值时触发直方图匹配")
    p.add_argument("--banding-blend", type=float, default=0.60,
                   help="横纹压制混合比（0=不压制，1=完全替换）")
    p.add_argument("--banding-suppress-rows", type=int, default=8,
                   help="FFT 中压制水平频率的行数（以 DC=0 为中心向外）")
    p.add_argument("--hist-blend", type=float, default=0.70,
                   help="直方图匹配混合比（0=不匹配，1=完全匹配到参考图）")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ──────────────────────── 工具函数 ────────────────────────

def _breast_mask(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.count_nonzero(th) > th.size * 0.8:
        th = cv2.bitwise_not(th)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    if n <= 1:
        return np.zeros_like(gray, dtype=np.uint8)
    idx = int(np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1)
    return (labels == idx).astype(np.uint8) * 255


def suppress_horizontal_banding(gray: np.ndarray, suppress_rows: int,
                                  blend: float,
                                  mask: np.ndarray | None = None) -> np.ndarray:
    """
    局部横纹压制（掩膜内）：
    1. 计算每行在掩膜前景内的均值，构成一维"行均值序列"
    2. 用高通滤波去掉行均值中的低频趋势（保留正常的腺体梯度）
    3. 保留高频变化，将低频变化（即"横纹"）减掉
    4. 仅在乳腺掩膜内操作，不影响背景和轮廓

    相比 FFT 全局方法，此方法不会伤害轮廓形状。
    """
    f32 = gray.astype(np.float32)
    H, W = gray.shape

    if mask is None or np.count_nonzero(mask) < 500:
        return gray

    # 计算每行的掩膜内均值
    row_means = np.zeros(H, dtype=np.float32)
    for r in range(H):
        row_mask = mask[r] > 0
        if row_mask.sum() > 5:
            row_means[r] = float(f32[r][row_mask].mean())

    # 用大 sigma 高斯滤波提取"慢变化趋势"（正常解剖梯度）
    trend = cv2.GaussianBlur(row_means.reshape(-1, 1).astype(np.float32),
                              (1, 51), 0).flatten()
    # 行级"横纹偏差"= 实际均值 - 慢变化趋势
    banding_component = row_means - trend  # 正值表示该行偏亮，负值表示偏暗

    # 从每行减去横纹分量（仅在前景内）
    result = f32.copy()
    for r in range(H):
        row_mask = mask[r] > 0
        if row_mask.sum() > 5:
            result[r][row_mask] -= blend * banding_component[r]

    return np.clip(result, 0, 255).astype(np.uint8)


def histogram_match_masked(gen_gray: np.ndarray, ref_gray: np.ndarray,
                            mask: np.ndarray, blend: float) -> np.ndarray:
    """
    仅在乳腺掩膜内做直方图匹配，背景保持纯黑。
    """
    if np.count_nonzero(mask) < 500:
        return gen_gray

    # 提取掩膜内像素
    gen_fg = gen_gray[mask > 0]
    ref_fg = ref_gray[mask > 0]

    if len(gen_fg) < 100 or len(ref_fg) < 100:
        return gen_gray

    # 直方图匹配（仅前景）
    matched_fg = match_histograms(gen_fg.reshape(-1, 1),
                                   ref_fg.reshape(-1, 1)).flatten()
    matched_fg = np.clip(matched_fg, 0, 255)

    # 混合
    blended_fg = (1 - blend) * gen_fg.astype(np.float32) + blend * matched_fg
    blended_fg = np.clip(blended_fg, 0, 255).astype(np.uint8)

    result = gen_gray.copy()
    result[mask > 0] = blended_fg
    return result


def collect_real_fg_pixels(real_dir: Path, n_refs: int = 50,
                             seed: int = 42) -> list[np.ndarray]:
    """预采样若干真实图的前景灰度数组，供直方图匹配用。"""
    rng = random.Random(seed)
    jpgs: list[Path] = []
    for d in real_dir.iterdir():
        if d.is_dir():
            jpgs.extend(list(d.glob("*.jpg"))[:2])
        if len(jpgs) >= n_refs * 3:
            break
    rng.shuffle(jpgs)
    refs = []
    for p in jpgs[:n_refs]:
        try:
            gray = np.asarray(Image.open(p).convert("L"))
            if gray.size > 0:
                refs.append(gray)
        except Exception:
            pass
    return refs


# ──────────────────────── 主流程 ────────────────────────

def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 读评估报告
    report: dict[str, dict] = {}
    with open(args.report_csv) as f:
        for row in csv.DictReader(f):
            name = Path(row["image"]).name
            report[name] = row

    # 预加载真实图参考（直方图匹配用）
    print(f"[fix] 预加载真实图参考...")
    real_refs = collect_real_fg_pixels(args.real_dir, n_refs=80, seed=args.seed)
    print(f"[fix] {len(real_refs)} 张参考图加载完毕")

    exts = {".png", ".jpg", ".jpeg"}
    images = sorted([p for p in args.images_dir.iterdir() if p.suffix.lower() in exts])
    n_banding_fix = n_hist_fix = n_both = 0

    print(f"[fix] 处理 {len(images)} 张  banding阈值={args.banding_threshold}"
          f"  hist阈值={args.hist_threshold}")

    for img_path in images:
        name = img_path.name
        row = report.get(name, {})
        banding_score = float(row.get("banding_score", 1.0))
        hist_wass = float(row.get("hist_wass", 0.0))

        gray = np.asarray(Image.open(img_path).convert("L"))
        mask = _breast_mask(gray)
        result = gray.copy()
        actions = []

        # Step 1: 横纹压制（局部行级，不伤轮廓）
        if banding_score < args.banding_threshold:
            result = suppress_horizontal_banding(
                result, args.banding_suppress_rows, args.banding_blend, mask
            )
            actions.append(f"banding({banding_score:.3f})")
            n_banding_fix += 1

        # Step 2: 直方图匹配
        if hist_wass > args.hist_threshold and real_refs:
            ref_gray = rng.choice(real_refs)
            # ref_gray 可能尺寸不同，直接用像素池匹配
            if ref_gray.shape != gray.shape:
                ref_resized = cv2.resize(ref_gray, (gray.shape[1], gray.shape[0]),
                                          interpolation=cv2.INTER_AREA)
            else:
                ref_resized = ref_gray
            result = histogram_match_masked(result, ref_resized, mask, args.hist_blend)
            actions.append(f"hist({hist_wass:.4f})")
            n_hist_fix += 1

        if len(actions) == 2:
            n_both += 1

        out_path = args.output_dir / name
        Image.fromarray(result, mode="L").save(out_path)
        if actions:
            print(f"  {name}  [{' + '.join(actions)}]")

    # 没有需要修复的图直接复制
    fixed_names = {p.name for p in args.output_dir.iterdir() if p.suffix in exts}
    for img_path in images:
        if img_path.name not in fixed_names:
            import shutil
            shutil.copy2(img_path, args.output_dir / img_path.name)

    print(f"\n完成 → {args.output_dir}")
    print(f"  横纹压制: {n_banding_fix} 张  直方图匹配: {n_hist_fix} 张  两者均有: {n_both} 张")
    total_fixed = n_banding_fix + n_hist_fix - n_both
    print(f"  共修复 {total_fixed} 张（预期通过率提升 ~{total_fixed // 2}%）")


if __name__ == "__main__":
    main()
