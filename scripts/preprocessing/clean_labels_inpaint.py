#!/usr/bin/env python3
"""清除 CBIS-DDSM 乳腺 mask 内的 DICOM burn-in 文字标注。

策略（参考 Kline et al. 2026 + mammo_label_heuristic.py 拓扑检测）：
  1. 连通域分析：在乳腺 mask 内检测小面积高亮连通域（文字标注特征）
  2. SD1.5 + LoRA v4 局部 img2img inpaint 修复文字区域
  3. 输出干净图像到 CBIS_CLEAN_V3

与 clean_training_labels.py 的关系：
  - clean_training_labels.py 清除 mask **外**的亮标签 → CBIS_CLEAN_V2
  - 本脚本清除 mask **内**的文字标注 → CBIS_CLEAN_V3

用法：
  # 扫描统计
  python3 scripts/preprocessing/clean_labels_inpaint.py --report-only --limit 100

  # 干燥运行
  python3 scripts/preprocessing/clean_labels_inpaint.py --dry-run --limit 10

  # 正式处理
  python3 scripts/preprocessing/clean_labels_inpaint.py --limit 50 --output-dir datasets/CBIS_CLEAN_V3_test
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


# ── SD1.5 + LoRA 加载 ──────────────────────────────────────────────────────
_pipe = None
_lora_path = None

def _load_pipe(lora_path: str):
    global _pipe, _lora_path
    if _pipe is not None and _lora_path == lora_path:
        return _pipe

    from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler
    from peft import PeftModel

    print(f"[Init] 加载 SD1.5 base: {ROOT / 'hf_cache/sd15'}")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        str(ROOT / "hf_cache/sd15"),
        torch_dtype=torch.float16,
        safety_checker=None,
        local_files_only=True,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    if lora_path:
        lora = PeftModel.from_pretrained(pipe.unet, lora_path)
        pipe.unet = lora.merge_and_unload()
        print(f"[Init] LoRA 合并: {lora_path}")

    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    _pipe = pipe
    _lora_path = lora_path
    return pipe


# ── 乳腺 mask 加载 ──────────────────────────────────────────────────────────
MASK_ROOT = ROOT / "datasets/breast_masks"
JPEG_ROOT = ROOT / "datasets/jpeg"


def _load_mask(src_rel: str) -> np.ndarray | None:
    parts = src_rel.rsplit("/", 1)
    if len(parts) != 2:
        return None
    uid, fname = parts
    mask_p = MASK_ROOT / uid / fname.replace(".jpg", ".png").replace(".JPG", ".png")
    return cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE) if mask_p.exists() else None


# ── 文字检测：连通域启发式 ──────────────────────────────────────────────────

def detect_text_cc(
    gray: np.ndarray,
    mask: np.ndarray,
    *,
    bright_thr_pct: float = 97.0,
    max_area_frac: float = 0.003,
    min_area: int = 12,
    margin_px: int = 6,
) -> list[tuple[int, int, int, int]]:
    """在乳腺 mask 内检测小面积高亮连通域（文字标注特征）。

    逻辑：
      - 真实乳腺组织是连续的灰度渐变，不会出现孤立的小亮块
      - DICOM 文字标注表现为 mask 内的小面积、高亮度连通域
      - 阈值自适应：用 mask 内像素的 bright_thr_pct 分位
    """
    mask_bin = (mask > 0).astype(np.uint8)
    tissue_pixels = gray[mask_bin > 0]
    if len(tissue_pixels) < 100:
        return []

    thr = np.percentile(tissue_pixels, bright_thr_pct)
    if thr < 180:  # 至少 180，避免在暗图上产生大量假阳性
        thr = max(thr, 180)

    # 二值化：mask 内的高亮像素
    bright = ((gray >= thr) & (mask_bin > 0)).astype(np.uint8)

    # 连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        bright, connectivity=8
    )

    # 最大连通域面积（乳腺主体）
    areas = stats[1:, cv2.CC_STAT_AREA]  # skip background
    max_area = areas.max() if len(areas) > 0 else 0
    abs_max = int(max_area * max_area_frac)

    h, w = gray.shape
    bboxes = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area or area > abs_max:
            continue

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]

        # 文本通常有较高的宽高比或较小的面积
        aspect = max(bw, bh) / max(min(bw, bh), 1)
        # 排除过于方正的大块（可能是钙化或噪声）
        if area > 200 and aspect < 2.0:
            continue

        x1 = max(0, x - margin_px)
        y1 = max(0, y - margin_px)
        x2 = min(w, x + bw + margin_px)
        y2 = min(h, y + bh + margin_px)
        bboxes.append((x1, y1, x2, y2))

    return bboxes


# ── SD Inpaint ───────────────────────────────────────────────────────────────

def _resize_to_512(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    scale = 512.0 / max(h, w)
    nh, nw = (int(h * scale) // 8) * 8, (int(w * scale) // 8) * 8
    return cv2.resize(gray, (max(nw, 8), max(nh, 8)), interpolation=cv2.INTER_LANCZOS4)


def inpaint_text_regions(
    gray: np.ndarray,
    bboxes: list[tuple[int, int, int, int]],
    pipe,
    *,
    strength: float = 0.30,
    guidance_scale: float = 5.0,
    num_steps: int = 25,
) -> np.ndarray:
    if not bboxes:
        return gray

    result = gray.copy().astype(np.float32)

    prompt = (
        "a mammography X-ray image, fibroglandular breast tissue, "
        "grayscale medical imaging, diagnostic quality, "
        "no text, no labels, no annotations, no markers"
    )
    negative = (
        "text, letters, watermark, barcode, patient label, "
        "ui, screenshot, annotation, marker, ruler"
    )

    for x1, y1, x2, y2 in bboxes:
        bw, bh = x2 - x1, y2 - y1
        if bw < 12 or bh < 12:
            continue

        patch = gray[y1:y2, x1:x2]
        patch_512 = _resize_to_512(patch)
        patch_rgb = np.stack([patch_512] * 3, axis=-1).astype(np.float32) / 255.0
        patch_rgb = patch_rgb * 2.0 - 1.0

        with torch.no_grad():
            out = pipe(
                prompt=prompt, negative_prompt=negative, image=patch_rgb,
                strength=strength, guidance_scale=guidance_scale,
                num_inference_steps=num_steps, output_type="np",
            ).images[0]

        out_gray = (out.mean(axis=-1) + 1.0) / 2.0 * 255.0
        out_resized = cv2.resize(out_gray.astype(np.float32), (bw, bh),
                                 interpolation=cv2.INTER_LANCZOS4)

        blend = np.ones((bh, bw), dtype=np.float32)
        f = max(3, min(bw, bh) // 10)
        if f > 1:
            blend[:f, :] = np.linspace(0, 1, f)[:, None]
            blend[-f:, :] = np.linspace(1, 0, f)[:, None]
            blend[:, :f] *= np.linspace(0, 1, f)[None, :]
            blend[:, -f:] *= np.linspace(1, 0, f)[None, :]

        result[y1:y2, x1:x2] = (
            blend * out_resized + (1.0 - blend) * result[y1:y2, x1:x2]
        )

    return np.clip(result, 0, 255).astype(np.uint8)


# ── 主流程 ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="清除乳腺 mask 内 DICOM 文字标注")
    p.add_argument("--csv", type=Path,
                   default=ROOT / "datasets/CBIS_CLEAN_V2/metadata_clean.csv")
    p.add_argument("--output-dir", type=Path,
                   default=ROOT / "datasets/CBIS_CLEAN_V3")
    p.add_argument("--lora-path", type=str,
                   default=str(ROOT / "outputs/lora/mammo_sd15_v4_clean/final_lora"))
    p.add_argument("--strength", type=float, default=0.30)
    p.add_argument("--bright-pct", type=float, default=97.0,
                   help="高亮阈值分位 (mask 内)")
    p.add_argument("--max-area-frac", type=float, default=0.003,
                   help="文字 CC 最大面积/乳腺最大 CC 面积")
    p.add_argument("--min-area", type=int, default=12)
    p.add_argument("--margin", type=int, default=6)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--report-only", action="store_true")
    p.add_argument("--save-debug", action="store_true",
                   help="保存检测可视化图到 output-dir/_debug/")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.csv.exists():
        print(f"[ERROR] metadata 不存在: {args.csv}", file=sys.stderr)
        return 1

    with open(args.csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if args.limit > 0:
        rows = rows[:args.limit]

    print(f"[Info] 共 {len(rows)} 张待扫描")

    # 扫描
    text_rows = []
    for row in tqdm(rows, desc="CC 扫描"):
        src_rel = row.get("src", row.get("file_name", ""))
        src_path = JPEG_ROOT / src_rel if src_rel else None
        if src_path is None or not src_path.exists():
            continue

        gray = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue

        mask = _load_mask(src_rel)
        if mask is None:
            continue

        # Resize gray to match mask (1024)
        h, w = gray.shape
        if max(h, w) != 1024:
            scale = 1024.0 / max(h, w)
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)),
                            interpolation=cv2.INTER_LANCZOS4)

        bboxes = detect_text_cc(
            gray, mask, bright_thr_pct=args.bright_pct,
            max_area_frac=args.max_area_frac, min_area=args.min_area,
            margin_px=args.margin,
        )
        if bboxes:
            text_rows.append((row, gray, mask, bboxes))

    pct = len(text_rows) / max(1, len(rows)) * 100
    print(f"[Info] mask 内检测到文字标注: {len(text_rows)}/{len(rows)} ({pct:.1f}%)")

    if args.report_only:
        for row, gray, mask, bboxes in text_rows:
            fn = row.get("file_name", row.get("src", "?"))
            for b in bboxes:
                print(f"  {fn}: bbox={b}")
        return 0

    if args.dry_run:
        for row, gray, mask, bboxes in text_rows:
            fn = row.get("file_name", row.get("src", "?"))
            print(f"[dry-run] {fn}: {len(bboxes)} 区域 → 将 inpaint")
        return 0

    if not text_rows:
        print("[Info] 无需处理")
        return 0

    # 加载 SD 模型
    pipe = _load_pipe(args.lora_path)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.save_debug:
        (out_dir / "_debug").mkdir(parents=True, exist_ok=True)

    stats = {"total": len(text_rows), "cleaned": 0, "skipped": 0, "total_bboxes": 0}
    for row, gray, mask, bboxes in tqdm(text_rows, desc="SD Inpaint"):
        fn = row.get("file_name", row.get("src", "?"))
        view = row.get("view", "MLO")
        density = row.get("density", "dense")
        stats["total_bboxes"] += len(bboxes)

        try:
            cleaned = inpaint_text_regions(
                gray, bboxes, pipe, strength=args.strength,
            )
        except Exception as e:
            print(f"\n[WARN] inpaint 失败 {fn}: {e}")
            stats["skipped"] += 1
            continue

        out_subdir = out_dir / view / density
        out_subdir.mkdir(parents=True, exist_ok=True)
        out_path = out_subdir / Path(fn).name
        rgb = np.stack([cleaned] * 3, axis=-1)
        Image.fromarray(rgb).save(str(out_path), quality=92)
        stats["cleaned"] += 1

        if args.save_debug:
            dbg = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            for x1, y1, x2, y2 in bboxes:
                cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.imwrite(str(out_dir / "_debug" / Path(fn).name.replace(".jpg", ".png")), dbg)

    print(f"\n[Done] 清洗: {stats['cleaned']} 张, 跳过 {stats['skipped']} 张, "
          f"共 {stats['total_bboxes']} 个文字区域")
    print(f"  输出: {out_dir.resolve()}")

    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
