#!/usr/bin/env python3
"""CRAFT + LaMa 文字擦除管线。

全自动擦除 CBIS_CLEAN_V2 训练数据中的 DICOM 文字标注，输出 CBIS_CLEAN_V3。

检测: EasyOCR CRAFT (字符几何特征, 不依赖亮度阈值)
擦除: LaMa traced model (FFT+GAN, Places2 训练, 无文字偏见)
回退: OpenCV INPAINT_TELEA (mask 面积 < 100px)

用法:
  # 扫描统计
  python3 scripts/preprocessing/clean_text_craft_lama.py --report-only --limit 50

  # 正式处理（全量）
  python3 scripts/preprocessing/clean_text_craft_lama.py

  # 指定输出 + debug 可视化
  python3 scripts/preprocessing/clean_text_craft_lama.py \
    --output-dir datasets/CBIS_CLEAN_V3 --save-debug

  # 断点续传
  python3 scripts/preprocessing/clean_text_craft_lama.py --start-idx 500
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ── 常量 ──────────────────────────────────────────────────────────────────

JPEG_ROOT_CANDIDATES = [
    ROOT / "datasets" / "jpeg",
    Path("/root/autodl-tmp/datasets/jpeg"),
]
MASK_ROOT = ROOT / "datasets" / "breast_masks"
DEFAULT_CSV = ROOT / "datasets" / "CBIS_CLEAN_V2" / "metadata_clean.csv"
DEFAULT_OUT = ROOT / "datasets" / "CBIS_CLEAN_V3"
LAMA_MODEL = ROOT / "hf_cache" / "lama" / "big-lama.pt"

# CRAFT 检测阈值 (低阈值 = 宁可多检不漏检)
CRAFT_TEXT_THRESHOLD = 0.3
CRAFT_LOW_TEXT = 0.2

# mask 膨胀像素
MASK_DILATE_PX = 4

# TELEA 回退阈值 (mask 区域面积 < 此值用 TELEA, 避免对大 mask 效果差)
TELEA_MAX_AREA = 100


# ── 图像加载 ──────────────────────────────────────────────────────────────

def _resolve_jpeg_root() -> Path | None:
    for p in JPEG_ROOT_CANDIDATES:
        if p.exists():
            return p
    return None


def _load_image(src_rel: str) -> np.ndarray | None:
    """从 CBIS_CLEAN_V2 的 src 字段加载原始 JPEG 灰度图。"""
    jpeg_root = _resolve_jpeg_root()
    if jpeg_root is None:
        return None
    # src 是完整路径，提取相对部分
    src = Path(src_rel)
    # 尝试直接路径
    if src.exists():
        return cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
    # 尝试从 jpeg_root + UID/filename 拼接
    parts = src_rel.split("/")
    for i, part in enumerate(parts):
        candidate = jpeg_root / Path(*parts[i:])
        if candidate.exists():
            return cv2.imread(str(candidate), cv2.IMREAD_GRAYSCALE)
    return None


def _resize_to_1024(gray: np.ndarray, long_side: int = 1024) -> np.ndarray:
    h, w = gray.shape[:2]
    m = max(h, w)
    if abs(m - long_side) < 2:
        return gray
    scale = long_side / m
    nh, nw = max(8, int(round(h * scale))), max(8, int(round(w * scale)))
    return cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_LANCZOS4)


# ── 文字检测: EasyOCR CRAFT ──────────────────────────────────────────────

_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(
            ["en"], gpu=True,
            model_storage_directory=str(ROOT / "hf_cache" / "easyocr"),
            download_enabled=True,
        )
    return _reader


def detect_text_craft(
    gray: np.ndarray,
) -> list[tuple[int, int, int, int]]:
    """CRAFT 检测图像中的文字区域。

    CRAFT 识别字符几何特征（笔画宽度、字符间距、空间排列），
    不依赖亮度阈值，因此能可靠区分文字标注和致密乳腺组织。

    Args:
        gray: 灰度图像 (H, W) uint8

    Returns:
        [(x1, y1, x2, y2), ...] 像素坐标 bbox 列表
    """
    reader = _get_reader()
    h_orig, w_orig = gray.shape[:2]

    # EasyOCR 需要 RGB
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    results = reader.detect(
        rgb,
        text_threshold=CRAFT_TEXT_THRESHOLD,
        low_text=CRAFT_LOW_TEXT,
    )

    if results is None:
        return []
    bboxes_raw = results[0] if isinstance(results, tuple) else results
    if bboxes_raw is None or len(bboxes_raw) == 0:
        return []

    bboxes = []
    for poly in bboxes_raw:
        if not poly or len(poly) < 4:
            continue
        poly_arr = np.array(poly).astype(np.int32)
        if poly_arr.ndim != 2 or poly_arr.shape[1] < 2:
            continue
        x1 = int(poly_arr[:, 0].min())
        y1 = int(poly_arr[:, 1].min())
        x2 = int(poly_arr[:, 0].max())
        y2 = int(poly_arr[:, 1].max())

        x1c = max(0, x1)
        y1c = max(0, y1)
        x2c = min(w_orig, x2)
        y2c = min(h_orig, y2)

        if x2c > x1c and y2c > y1c:
            bboxes.append((x1c, y1c, x2c, y2c))

    return bboxes


# ── Mask 生成 ────────────────────────────────────────────────────────────

def bboxes_to_mask(
    bboxes: list[tuple[int, int, int, int]],
    h: int,
    w: int,
    dilate_px: int = MASK_DILATE_PX,
) -> np.ndarray:
    """bbox 列表 → binary mask (0=保留原图, 255=擦除区域)。"""
    mask = np.zeros((h, w), dtype=np.uint8)
    for x1, y1, x2, y2 in bboxes:
        x1c = max(0, x1 - dilate_px)
        y1c = max(0, y1 - dilate_px)
        x2c = min(w, x2 + dilate_px)
        y2c = min(h, y2 + dilate_px)
        mask[y1c:y2c, x1c:x2c] = 255
    return mask


# ── LaMa 擦除 ────────────────────────────────────────────────────────────

_lama_model = None


def _get_lama():
    global _lama_model
    if _lama_model is None:
        _lama_model = torch.jit.load(str(LAMA_MODEL), map_location="cuda")
        _lama_model.eval()
    return _lama_model


def _resize_to_multiple(img: np.ndarray, divisor: int = 8) -> tuple[np.ndarray, int, int]:
    """Pad 使 H,W 均为 divisor 的倍数 (LaMa 要求)，保持原图内容不变。"""
    h, w = img.shape[:2]
    new_h = ((h + divisor - 1) // divisor) * divisor
    new_w = ((w + divisor - 1) // divisor) * divisor
    pad_bottom = new_h - h
    pad_right = new_w - w
    if pad_bottom > 0 or pad_right > 0:
        border = cv2.BORDER_REFLECT if img.ndim == 2 else cv2.BORDER_REFLECT
        img = cv2.copyMakeBorder(img, 0, pad_bottom, 0, pad_right, border)
    return img, h, w


def inpaint_lama(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """LaMa 擦除文字区域。

    Args:
        gray: 灰度图像 (H, W) uint8
        mask: binary mask (H, W) uint8, 255 = 擦除

    Returns:
        擦除后的图像 (H, W) uint8
    """
    h_orig, w_orig = gray.shape[:2]

    gray_padded, ph, pw = _resize_to_multiple(gray)
    mask_padded, _, _ = _resize_to_multiple(mask)

    # 转 RGB, [0,1] normalize
    rgb = np.stack([gray_padded] * 3, axis=-1).astype(np.float32) / 255.0
    m = mask_padded.astype(np.float32) / 255.0

    img_t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).cuda()  # [1,3,H,W]
    mask_t = torch.from_numpy(m).unsqueeze(0).unsqueeze(0).cuda()       # [1,1,H,W]

    lama = _get_lama()
    with torch.no_grad():
        result = lama(img_t, mask_t)

    result = result.squeeze(0).permute(1, 2, 0).cpu().numpy()
    result = np.clip(result * 255, 0, 255).astype(np.uint8)

    if ph != gray_padded.shape[0] or pw != gray_padded.shape[1]:
        result = result[:ph, :pw]

    result_gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)

    # 仅在 mask 区域使用 LaMa 结果，其他保留原图
    mask_bin = (mask > 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask_dilated = cv2.dilate(mask_bin, k)
    mask_eroded = cv2.erode(mask_bin, k)
    transition = (mask_dilated.astype(np.float32) - mask_eroded.astype(np.float32))
    core = mask_eroded.astype(np.float32)

    alpha = core + transition * 0.5
    alpha = np.clip(alpha, 0, 1)

    output = gray.astype(np.float32) * (1.0 - alpha) + result_gray.astype(np.float32) * alpha
    return np.clip(output, 0, 255).astype(np.uint8)


# ── TELEA 回退 ───────────────────────────────────────────────────────────

def inpaint_telea(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """OpenCV TELEA — 用于极小文字区域的快速回退。"""
    return cv2.inpaint(gray, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)


# ── 主流程 ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="CRAFT + LaMa 擦除 DICOM 文字 → CBIS_CLEAN_V3"
    )
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--limit", type=int, default=0,
                   help="限制处理数量 (0=全部)")
    p.add_argument("--report-only", action="store_true",
                   help="仅 CRAFT 扫描统计，不擦除")
    p.add_argument("--no-cuda", action="store_true",
                   help="禁用 GPU (LaMa 降级为纯 TELEA)")
    p.add_argument("--save-debug", action="store_true",
                   help="保存检测可视化到 _debug/")
    p.add_argument("--start-idx", type=int, default=0,
                   help="断点续传: 从第 N 张开始 (0-indexed)")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.csv.exists():
        print(f"[ERROR] metadata 不存在: {args.csv}", file=sys.stderr)
        return 1

    with open(args.csv, encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    if args.limit > 0:
        all_rows = all_rows[:args.limit]

    rows = all_rows[args.start_idx:]
    print(f"[Info] 总计 {len(all_rows)} 张, 从 #{args.start_idx} 开始, "
          f"本次 {len(rows)} 张")

    out_dir = args.output_dir
    if not args.report_only:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "_debug").mkdir(parents=True, exist_ok=True)
        (out_dir / "_logs").mkdir(parents=True, exist_ok=True)

    # 预加载 reader + LaMa
    if not args.report_only:
        print("[Init] 加载 EasyOCR CRAFT...")
        _get_reader()

    use_lama = not args.no_cuda and LAMA_MODEL.exists()
    if use_lama and not args.report_only:
        print("[Init] 加载 LaMa model...")
        _get_lama()
    elif not args.report_only:
        print("[Init] LaMa 不可用, 回退 TELEA")

    stats = {
        "total": len(rows),
        "detected": 0,
        "cleaned": 0,
        "skipped": 0,
        "total_bboxes": 0,
        "lama_used": 0,
        "telea_used": 0,
    }

    out_rows = []
    error_log = []

    pbar = tqdm(enumerate(rows), total=len(rows), desc="CRAFT+LaMa")
    for local_idx, row in pbar:
        idx = args.start_idx + local_idx
        src_rel = row.get("src", row.get("file_name", ""))
        file_name = row.get("file_name", src_rel)
        view = row.get("view", "MLO")
        density = row.get("density", "scattered")
        total_mask_area = 0

        # 1. 加载
        gray = _load_image(src_rel)
        if gray is None:
            stats["skipped"] += 1
            error_log.append({"file": file_name, "error": "image_load_failed", "idx": idx})
            continue

        # 2. Resize
        gray_1024 = _resize_to_1024(gray, 1024)

        # 3. CRAFT 检测
        bboxes = []
        try:
            bboxes = detect_text_craft(gray_1024)
        except Exception as e:
            error_log.append({"file": file_name, "error": f"craft: {e}", "idx": idx})

        # 4. 擦除
        if bboxes:
            stats["detected"] += 1
            stats["total_bboxes"] += len(bboxes)
            mask = bboxes_to_mask(bboxes, *gray_1024.shape)
            total_mask_area = int(np.sum(mask > 0))

            if use_lama and total_mask_area > TELEA_MAX_AREA:
                try:
                    cleaned = inpaint_lama(gray_1024, mask)
                    stats["lama_used"] += 1
                except Exception as e:
                    error_log.append({"file": file_name, "error": f"lama: {e}", "idx": idx})
                    cleaned = inpaint_telea(gray_1024, mask)
                    stats["telea_used"] += 1
            else:
                cleaned = inpaint_telea(gray_1024, mask)
                stats["telea_used"] += 1

            stats["cleaned"] += 1
        else:
            cleaned = gray_1024

        clean_method = (
            "lama" if (bboxes and use_lama and total_mask_area > TELEA_MAX_AREA)
            else "telea" if bboxes
            else "none"
        )

        # 5. 保存
        if not args.report_only:
            out_rel = Path(view) / density / Path(file_name).name
            out_path = out_dir / out_rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            rgb = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(str(out_path), rgb, [cv2.IMWRITE_JPEG_QUALITY, 92])

            if args.save_debug and bboxes:
                dbg = cv2.cvtColor(gray_1024, cv2.COLOR_GRAY2BGR)
                for x1, y1, x2, y2 in bboxes:
                    cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), thickness=1)
                dbg_name = Path(file_name).name.replace(".jpg", "_detect.png")
                cv2.imwrite(str(out_dir / "_debug" / dbg_name), dbg)

        new_row = dict(row)
        new_row["file_name"] = str(Path(view) / density / Path(file_name).name)
        new_row["text_bbox_count"] = len(bboxes)
        new_row["text_area_pct"] = (
            float(total_mask_area) / (gray_1024.shape[0] * gray_1024.shape[1]) * 100
            if bboxes else 0.0
        )
        new_row["clean_method"] = clean_method
        out_rows.append(new_row)

        pbar.set_postfix({
            "det": stats["detected"],
            "lama": stats["lama_used"],
            "telea": stats["telea_used"],
        })

    # ── 统计 ──────────────────────────────────────────────────────────
    n = max(stats["total"], 1)
    pct = stats["detected"] / n * 100
    print(f"\n[Stats] 检测到文字: {stats['detected']}/{stats['total']} ({pct:.1f}%)")
    print(f"[Stats] 总 bbox: {stats['total_bboxes']}, "
          f"LaMa: {stats['lama_used']}, TELEA: {stats['telea_used']}")
    print(f"[Stats] 跳过: {stats['skipped']}")

    # ── 写 metadata ───────────────────────────────────────────────────
    if not args.report_only and out_rows:
        out_csv = out_dir / "metadata_clean.csv"
        fields = list(out_rows[0].keys())
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(out_rows)
        print(f"[Done] metadata → {out_csv}")

        if error_log:
            log_path = out_dir / "_logs" / "error_log.jsonl"
            with log_path.open("w") as f:
                for e in error_log:
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")
            print(f"[Warn] {len(error_log)} 个错误 → {log_path}")

    if args.report_only:
        # Group report by view/density
        from collections import Counter
        det_counter = Counter()
        for r in out_rows:
            if int(r.get("text_bbox_count", 0)) > 0:
                key = f"{r.get('view','?')}/{r.get('density','?')}"
                det_counter[key] += 1
        if det_counter:
            print("\n[Report] 文字检出按 view/density:")
            for k, v in sorted(det_counter.items()):
                print(f"  {k}: {v}")

    global _lama_model, _reader
    _lama_model = None
    _reader = None
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n[Output] {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
