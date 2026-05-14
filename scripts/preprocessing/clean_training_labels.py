#!/usr/bin/env python3
"""清除 CBIS-DDSM 训练图中的 DICOM burn-in 标签，生成 CBIS_CLEAN_V2 训练集。

流程：
  1. 读取 datasets/CBIS_CLEAN/metadata_clean.csv（已通过质量筛选的训练样本）
  2. 从 datasets/controls/ 找到对应原始 JPEG 图像
  3. 从 datasets/breast_masks/ 加载对应乳腺掩码（1024×1024）
  4. 把原图 resize 到 1024 长边，并对齐掩码
  5. **掩码区域以外**的亮连通域 = DICOM 标签 → 清零
  6. 转为 3 通道 RGB JPEG，保存到 datasets/CBIS_CLEAN_V2/ 同结构

输出：
  datasets/CBIS_CLEAN_V2/{CC,MLO}/{fatty,scattered,heterogeneous,dense}/*.jpg
  datasets/CBIS_CLEAN_V2/metadata_clean.csv  （file_name 更新为新路径）
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]

JPEG_ROOT_CANDIDATES = [
    ROOT / "datasets" / "controls",
    ROOT / "datasets" / "jpeg",
    Path("/root/autodl-tmp/datasets/jpeg"),
]
MASK_ROOT = ROOT / "datasets" / "breast_masks"


# ─────────────────────────────── 图像工具 ───────────────────────────────────

def resize_long_side(gray: np.ndarray, long_side: int = 1024) -> np.ndarray:
    h, w = gray.shape[:2]
    scale = long_side / max(h, w)
    if abs(scale - 1.0) < 0.01:
        return gray
    nh, nw = int(round(h * scale)), int(round(w * scale))
    return cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)


def align_mask(mask1024: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """把 1024×1024 的 letterbox mask 裁剪/缩放到 (target_h, target_w)。
    
    假设原图被 resize 到 1024 长边后居中填充到 1024×1024 正方形里。
    """
    mh, mw = mask1024.shape[:2]
    # 计算 letterbox padding
    if target_h >= target_w:          # 高 > 宽，上下无 padding，左右有 padding
        pad_x = (mw - int(target_w * mh / target_h)) // 2
        pad_y = 0
        crop = mask1024[pad_y:mh - pad_y if pad_y else mh, pad_x:mw - pad_x if pad_x else mw]
    else:                              # 宽 > 高，左右无 padding，上下有 padding
        pad_y = (mh - int(target_h * mw / target_w)) // 2
        pad_x = 0
        crop = mask1024[pad_y:mh - pad_y if pad_y else mh, pad_x:mw - pad_x if pad_x else mw]
    
    if crop.shape[0] != target_h or crop.shape[1] != target_w:
        crop = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return crop


def erase_background_labels(
    gray: np.ndarray,
    mask: np.ndarray,
    dilate_px: int = 5,
    label_thr_pct: float = 97.0,
) -> np.ndarray:
    """用乳腺掩码清除背景区域内的 DICOM 标签（直接清零背景中亮像素）。

    核心保证：mask 覆盖的组织区域（含极小边距）内的像素一律不动。
    策略：CBIS-DDSM 背景近乎纯黑（99% 像素 < 5），任何背景中灰度 > 阈值的像素
    一定是 DICOM 标签或伪影，直接置零即可，无需 CC 分析。
    """
    result = gray.copy()
    h, w = gray.shape[:2]

    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
        mask_safe = cv2.dilate((mask > 0).astype(np.uint8) * 255, k)
    else:
        mask_safe = (mask > 0).astype(np.uint8) * 255
    bg = mask_safe == 0  # 安全背景区

    bg_pixels = gray[bg]
    if bg_pixels.size < 100:
        return result

    # 背景 97th 分位 + 绝对下限 80（CBIS 背景近纯黑，标签字符像素 80-255 都需清）
    thr = max(float(np.percentile(bg_pixels, label_thr_pct)), 80.0)
    if thr >= 255:
        return result

    # 直接清零背景中 > thr 的像素（有掩码保护，无误伤风险）
    result[(gray >= int(thr)) & bg] = 0
    return result


def resolve_src(row_src: str) -> Optional[Path]:
    """把 metadata.csv 里的 src 字段映射到本地实际文件路径。"""
    for root in JPEG_ROOT_CANDIDATES:
        if "/jpeg/" in row_src:
            rel = row_src.split("/jpeg/")[-1]
            p = root / rel
        else:
            p = Path(row_src)
        if p.exists():
            return p
    return None


def resolve_mask(row_src: str) -> Optional[Path]:
    """src 的 UID/fname.jpg → MASK_ROOT/UID/fname.png。"""
    if "/jpeg/" in row_src:
        rel = row_src.split("/jpeg/")[-1]
    else:
        parts = Path(row_src).parts
        rel = str(Path(*parts[-2:]))
    uid, fname = rel.rsplit("/", 1)
    mask_p = MASK_ROOT / uid / fname.replace(".jpg", ".png").replace(".JPG", ".png")
    return mask_p if mask_p.exists() else None


# ─────────────────────────────── 主流程 ─────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="清除训练图 DICOM 标签，输出 CBIS_CLEAN_V2")
    p.add_argument("--csv", type=Path, default=ROOT / "datasets/CBIS_CLEAN/metadata_clean.csv")
    p.add_argument("--output-dir", type=Path, default=ROOT / "datasets/CBIS_CLEAN_V2")
    p.add_argument(
        "--long-side",
        type=int,
        default=1024,
        help="resize 长边（0=保持原图大小）。默认 1024 与推理 letterbox(768×1024) 不同域；重训若觉细节弱可试 0 或更大长边。",
    )
    p.add_argument("--view", choices=["CC", "MLO", "ALL"], default="MLO")
    p.add_argument("--quality", type=int, default=92, help="输出 JPEG 质量")
    p.add_argument("--dilate-px", type=int, default=5, help="掩码安全膨胀像素（训练数据用小值，组织边缘识别较准）")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def process_one(row: dict, args: argparse.Namespace) -> Optional[dict]:
    """处理单张图，返回新 metadata 行或 None（跳过）。"""
    src_path = resolve_src(row["src"])
    if src_path is None:
        return None

    mask_path = resolve_mask(row["src"])
    if mask_path is None:
        return None

    view = row.get("view", "")
    density = row.get("density", "")
    out_rel = Path(view) / density / Path(row["file_name"]).name
    out_path = args.output_dir / out_rel

    if args.dry_run:
        print(f"[dry-run] {src_path.name} → {out_rel}")
        return {**row, "file_name": str(out_rel)}

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 读原图（灰度）
    gray = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None

    # resize
    if args.long_side > 0:
        gray = resize_long_side(gray, args.long_side)

    # 读掩码并对齐
    mask1024 = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask1024 is None:
        return None
    h, w = gray.shape[:2]
    mask = align_mask(mask1024, h, w)

    # 清标签
    gray_clean = erase_background_labels(gray, mask, dilate_px=args.dilate_px)

    # 保存为灰度 JPEG（转 RGB 以便后续训练脚本统一读取）
    rgb = cv2.cvtColor(gray_clean, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(out_path), rgb, [cv2.IMWRITE_JPEG_QUALITY, args.quality])

    new_row = {**row, "file_name": str(out_rel)}
    return new_row


def main() -> None:
    args = parse_args()

    with open(args.csv, encoding="utf-8", newline="") as f:
        all_rows = list(csv.DictReader(f))

    if args.view != "ALL":
        rows = [r for r in all_rows if r.get("view", "") == args.view]
    else:
        rows = all_rows

    if args.limit > 0:
        rows = rows[: args.limit]

    print(f"待处理图像: {len(rows)} 张  输出目录: {args.output_dir}")

    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    kept: list[dict] = []
    skipped = 0

    if args.workers <= 1:
        for row in tqdm(rows, desc="清理标签"):
            result = process_one(row, args)
            if result is None:
                skipped += 1
            else:
                kept.append(result)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        futures = {}
        with ThreadPoolExecutor(max_workers=args.workers) as exe:
            for row in rows:
                f = exe.submit(process_one, row, args)
                futures[f] = row
            for f in tqdm(as_completed(futures), total=len(futures), desc="清理标签"):
                r = f.result()
                if r is None:
                    skipped += 1
                else:
                    kept.append(r)

    print(f"完成: 保留={len(kept)}  跳过={skipped}")

    if not args.dry_run and kept:
        out_csv = args.output_dir / "metadata_clean.csv"
        fields = list(kept[0].keys())
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(kept)
        print(f"metadata_clean.csv 已写入: {out_csv}")


if __name__ == "__main__":
    main()
