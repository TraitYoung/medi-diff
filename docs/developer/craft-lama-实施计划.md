# CRAFT + LaMa 文字擦除管线 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 全自动擦除 CBIS_CLEAN_V2 训练数据中的 DICOM 文字/标注，输出 CBIS_CLEAN_V3 并更新 captions。

**Architecture:** 新脚本 `clean_text_craft_lama.py` 作为主管线 — EasyOCR CRAFT 检测文字区域 → 生成 binary mask → LaMa (traced torch.jit model) 擦除 → 输出干净图。修改 `generate_captions.py` 注入 text-free 描述。环境：pip mirror 无 LaMa 包，直接从 GitHub 下载 traced model 用 torch.jit 加载。

**Tech Stack:** Python 3.12, PyTorch 2.11 (CUDA), EasyOCR 1.7.2 (CRAFT), LaMa big-lama.pt (traced model), OpenCV 4.13

---

### Task 1: 下载 LaMa traced model 并验证可用

**Files:**
- Create: `hf_cache/lama/big-lama.pt` (downloaded model, ~200MB)

- [ ] **Step 1: 下载 LaMa traced model**

```bash
mkdir -p /root/autodl-tmp/medi-diff/hf_cache/lama
wget -O /root/autodl-tmp/medi-diff/hf_cache/lama/big-lama.pt \
  "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt"
```

- [ ] **Step 2: 验证模型可加载且推理正确**

```python
# test_lama.py — run and check output
import torch
import cv2
import numpy as np

model = torch.jit.load("hf_cache/lama/big-lama.pt", map_location="cuda")
model.eval()

# Test with random image + mask
img = np.random.randint(0, 255, (3, 512, 512), dtype=np.uint8)
mask = np.zeros((1, 512, 512), dtype=np.uint8)
mask[:, 200:300, 200:300] = 255

img_t = torch.from_numpy(img).float().div(255).unsqueeze(0).cuda()
mask_t = torch.from_numpy(mask).float().div(255).unsqueeze(0).cuda()

with torch.no_grad():
    result = model(img_t, mask_t)

print(f"Input shape: {img_t.shape}, Output shape: {result.shape}")
print("LaMa model test OK")
```

Run: `python3 test_lama.py`
Expected: prints "LaMa model test OK" with output shape matching input

---

### Task 2: 创建主管线脚本 `clean_text_craft_lama.py`

**Files:**
- Create: `scripts/preprocessing/clean_text_craft_lama.py`

- [ ] **Step 1: 编写脚本头部和参数解析**

```python
#!/usr/bin/env python3
"""CRAFT 文字检测 + LaMa 擦除管线。

擦除 CBIS_CLEAN_V2 训练数据中的 DICOM 文字标注，输出 CBIS_CLEAN_V3。
检测: EasyOCR CRAFT (字符几何特征, 不依赖亮度)
擦除: LaMa traced model (FFT+GAN, Places2 训练, 无文字偏见)
回退: OpenCV INPAINT_TELEA (mask 面积 < 100px)

用法:
  # 扫描统计
  python3 scripts/preprocessing/clean_text_craft_lama.py --report-only --limit 50

  # 正式处理
  python3 scripts/preprocessing/clean_text_craft_lama.py

  # 指定输出
  python3 scripts/preprocessing/clean_text_craft_lama.py --output-dir datasets/CBIS_CLEAN_V3
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

JPEG_ROOT = ROOT / "datasets" / "jpeg"
MASK_ROOT = ROOT / "datasets" / "breast_masks"
DEFAULT_CSV = ROOT / "datasets" / "CBIS_CLEAN_V2" / "metadata_clean.csv"
DEFAULT_OUT = ROOT / "datasets" / "CBIS_CLEAN_V3"
LAMA_MODEL = ROOT / "hf_cache" / "lama" / "big-lama.pt"

# CRAFT 检测阈值 (低阈值 = 宁可多检)
CRAFT_TEXT_THRESHOLD = 0.3
CRAFT_LOW_TEXT = 0.2

# mask 膨胀像素
MASK_DILATE_PX = 4

# TELEA 回退阈值 (mask 区域面积 < 此值用 TELEA)
TELEA_MAX_AREA = 100
```

- [ ] **Step 2: 编写 EasyOCR CRAFT 检测函数**

```python
# ── 文字检测: EasyOCR CRAFT ──────────────────────────────────────────────

_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(
            ["en"], gpu=True,
            # Only load detector, skip recognizer for speed
            model_storage_directory=str(ROOT / "hf_cache" / "easyocr"),
            download_enabled=True,
        )
    return _reader


def detect_text_craft(
    gray: np.ndarray,
    mask: np.ndarray | None = None,
) -> list[tuple[int, int, int, int]]:
    """CRAFT 检测图像中的文字区域。

    CRAFT 检测的是字符几何特征 (笔画宽度、间距、排列)，
    不依赖亮度阈值，因此能区分文字和致密乳腺组织。

    Args:
        gray: 灰度图像 (uint8)
        mask: 可选的乳腺 mask，仅检测 mask 内区域

    Returns:
        [(x1, y1, x2, y2), ...] bbox 列表
    """
    reader = _get_reader()

    # EasyOCR 需要 RGB 输入
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    # CRAFT 检测 (只跑 detector，跳过 recognizer 以加速)
    h_orig, w_orig = gray.shape[:2]
    results = reader.detect(
        rgb,
        text_threshold=CRAFT_TEXT_THRESHOLD,
        low_text=CRAFT_LOW_TEXT,
    )

    # EasyOCR 返回 (bboxes, None) 或只返回 bboxes
    if results is None or (isinstance(results, tuple) and results[0] is None):
        return []

    bboxes_raw = results[0] if isinstance(results, tuple) else results
    if bboxes_raw is None or len(bboxes_raw) == 0:
        return []

    bboxes = []
    for poly in bboxes_raw:
        # poly: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        poly = np.array(poly).astype(np.int32)
        x1, y1 = poly[:, 0].min(), poly[:, 1].min()
        x2, y2 = poly[:, 0].max(), poly[:, 1].max()

        # 裁剪到图像边界
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_orig, x2), min(h_orig, y2)

        if x2 > x1 and y2 > y1:
            bboxes.append((int(x1), int(y1), int(x2), int(y2)))

    return bboxes
```

- [ ] **Step 3: 编写 mask 生成函数**

```python
# ── Mask 生成 ────────────────────────────────────────────────────────────

def bboxes_to_mask(
    bboxes: list[tuple[int, int, int, int]],
    h: int,
    w: int,
    dilate_px: int = MASK_DILATE_PX,
) -> np.ndarray:
    """bbox 列表 → binary mask (0=保留, 255=擦除)。"""
    mask = np.zeros((h, w), dtype=np.uint8)
    for x1, y1, x2, y2 in bboxes:
        x1c = max(0, x1 - dilate_px)
        y1c = max(0, y1 - dilate_px)
        x2c = min(w, x2 + dilate_px)
        y2c = min(h, y2 + dilate_px)
        mask[y1c:y2c, x1c:x2c] = 255
    return mask
```

- [ ] **Step 4: 编写 LaMa 擦除函数 (含 TELEA 回退)**

```python
# ── LaMa 擦除 ────────────────────────────────────────────────────────────

_lama_model = None


def _get_lama():
    global _lama_model
    if _lama_model is None:
        if not LAMA_MODEL.exists():
            raise FileNotFoundError(
                f"LaMa model not found at {LAMA_MODEL}. Run Task 1 first."
            )
        _lama_model = torch.jit.load(str(LAMA_MODEL), map_location="cuda")
        _lama_model.eval()
    return _lama_model


def _resize_to_multiple(img: np.ndarray, divisor: int = 8) -> tuple[np.ndarray, int, int]:
    """Resize 图像使 H,W 均为 divisor 的倍数 (LaMa 要求)。"""
    h, w = img.shape[:2]
    new_h = ((h + divisor - 1) // divisor) * divisor
    new_w = ((w + divisor - 1) // divisor) * divisor
    if new_h != h or new_w != w:
        if img.ndim == 2:
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        else:
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    return img, h, w  # 返回原尺寸用于裁剪


def inpaint_lama(
    gray: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """LaMa 擦除文字区域。

    Args:
        gray: 灰度图像 (H, W) uint8
        mask: binary mask (H, W) uint8, 255=擦除区域

    Returns:
        擦除后的图像 (H, W) uint8
    """
    h_orig, w_orig = gray.shape[:2]

    # Resize 到 8 的倍数
    gray_padded, ph, pw = _resize_to_multiple(gray)
    mask_padded, _, _ = _resize_to_multiple(mask)

    # 转 RGB, normalize
    rgb = cv2.cvtColor(gray_padded, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
    m = mask_padded.astype(np.float32) / 255.0

    img_t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).cuda()  # [1,3,H,W]
    mask_t = torch.from_numpy(m).unsqueeze(0).unsqueeze(0).cuda()       # [1,1,H,W]

    lama = _get_lama()
    with torch.no_grad():
        result = lama(img_t, mask_t)

    result = result.squeeze(0).permute(1, 2, 0).cpu().numpy()  # [H,W,3]
    result = np.clip(result * 255, 0, 255).astype(np.uint8)

    # Crop back to original size
    if ph != gray_padded.shape[0] or pw != gray_padded.shape[1]:
        result = result[:ph, :pw]

    # 转灰度
    result_gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)

    # 仅在 mask 区域使用 LaMa 结果，其他区域保留原图
    full_mask_3d = (mask > 0).astype(np.uint8)
    # 边缘羽化融合 (8px 过渡带)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask_dilated = cv2.dilate(full_mask_3d, k)
    mask_eroded = cv2.erode(full_mask_3d, k)
    transition = mask_dilated.astype(np.float32) - mask_eroded.astype(np.float32)
    core = mask_eroded.astype(np.float32)

    alpha = core + transition * 0.5
    alpha = np.clip(alpha, 0, 1)

    output = gray.astype(np.float32) * (1 - alpha) + result_gray.astype(np.float32) * alpha
    return np.clip(output, 0, 255).astype(np.uint8)


def inpaint_telea(
    gray: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """OpenCV TELEA inpainting — 用于极小文字区域的快速回退。"""
    return cv2.inpaint(gray, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
```

- [ ] **Step 5: 编写乳腺 mask 加载和对齐 (复用现有逻辑)**

```python
# ── 乳腺 mask 加载 ──────────────────────────────────────────────────────


def _load_mask(src_rel: str) -> np.ndarray | None:
    """从 CBIS_CLEAN_V2 的 src 字段加载对应乳腺 mask。"""
    parts = src_rel.rsplit("/", 1)
    if len(parts) != 2:
        return None
    uid, fname = parts
    mask_p = MASK_ROOT / uid / fname.replace(".jpg", ".png").replace(".JPG", ".png")
    return cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE) if mask_p.exists() else None


def _load_image(src_rel: str) -> np.ndarray | None:
    """从 CBIS_CLEAN_V2 的 src 字段加载原始 JPEG。"""
    for root_candidate in [ROOT / "datasets" / "jpeg", Path("/root/autodl-tmp/datasets/jpeg")]:
        p = root_candidate / src_rel
        if p.exists():
            return cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return None


def _resize_to_1024(gray: np.ndarray, long_side: int = 1024) -> np.ndarray:
    """Resize 到指定长边。"""
    h, w = gray.shape[:2]
    scale = long_side / max(h, w)
    if abs(scale - 1.0) < 0.01:
        return gray
    nh, nw = int(round(h * scale)), int(round(w * scale))
    return cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
```

- [ ] **Step 6: 编写主处理循环**

```python
# ── 主流程 ───────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="CRAFT + LaMa 擦除 DICOM 文字，输出 CBIS_CLEAN_V3"
    )
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--limit", type=int, default=0,
                   help="限制处理数量 (0=全部)")
    p.add_argument("--report-only", action="store_true",
                   help="仅扫描统计，不擦除")
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
          f"本次处理 {len(rows)} 张")

    # 输出目录
    out_dir = args.output_dir
    if not args.report_only:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "_debug").mkdir(parents=True, exist_ok=True)

    # 初始化 EasyOCR (延迟加载)
    if not args.report_only:
        _get_reader()

    # 预加载 LaMa
    use_lama = not args.no_cuda and LAMA_MODEL.exists()
    if use_lama and not args.report_only:
        print("[Init] 加载 LaMa model...")
        _get_lama()
        print("[Init] LaMa 就绪")
    elif not args.report_only:
        print("[Init] LaMa 不可用, 回退到 TELEA")

    stats = {
        "total": len(rows),
        "detected": 0,       # 检测到文字
        "cleaned": 0,        # 成功擦除
        "skipped": 0,        # 读取失败跳过
        "total_bboxes": 0,   # 总检测框数
        "lama_used": 0,      # LaMa 擦除次数
        "telea_used": 0,     # TELEA 回退次数
    }

    out_rows = []
    error_log = []

    for idx, row in enumerate(
        tqdm(rows, desc="CRAFT + LaMa"), start=args.start_idx
    ):
        src_rel = row.get("src", row.get("file_name", ""))
        file_name = row.get("file_name", src_rel)
        view = row.get("view", "MLO")
        density = row.get("density", "scattered")

        # 1. 加载图像
        gray = _load_image(src_rel)
        if gray is None:
            stats["skipped"] += 1
            error_log.append({
                "file": file_name, "error": "image_load_failed", "idx": idx
            })
            continue

        # 2. Resize 到 1024
        gray_1024 = _resize_to_1024(gray, 1024)

        # 3. CRAFT 检测
        try:
            bboxes = detect_text_craft(gray_1024)
        except Exception as e:
            error_log.append({
                "file": file_name, "error": f"craft_detect: {e}", "idx": idx
            })
            bboxes = []

        # 4. 擦除
        if bboxes:
            stats["detected"] += 1
            stats["total_bboxes"] += len(bboxes)
            mask = bboxes_to_mask(bboxes, *gray_1024.shape)

            # 决定用 LaMa 还是 TELEA
            total_mask_area = np.sum(mask > 0)
            if use_lama and total_mask_area > TELEA_MAX_AREA:
                cleaned = inpaint_lama(gray_1024, mask)
                stats["lama_used"] += 1
            else:
                cleaned = inpaint_telea(gray_1024, mask)
                stats["telea_used"] += 1

            stats["cleaned"] += 1
        else:
            cleaned = gray_1024

        # 5. 保存
        if not args.report_only:
            out_rel = Path(view) / density / Path(file_name).name
            out_path = out_dir / out_rel
            out_path.parent.mkdir(parents=True, exist_ok=True)

            rgb = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(str(out_path), rgb, [cv2.IMWRITE_JPEG_QUALITY, 92])

            # Debug 可视化
            if args.save_debug and bboxes:
                dbg = cv2.cvtColor(gray_1024, cv2.COLOR_GRAY2BGR)
                for x1, y1, x2, y2 in bboxes:
                    cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), 1)
                dbg_name = Path(file_name).name.replace(".jpg", "_detect.png")
                cv2.imwrite(str(out_dir / "_debug" / dbg_name), dbg)

        new_row = dict(row)
        new_row["file_name"] = str(Path(view) / density / Path(file_name).name)
        new_row["text_bbox_count"] = len(bboxes)
        new_row["text_area_pct"] = (
            float(np.sum(mask > 0)) / (gray_1024.shape[0] * gray_1024.shape[1]) * 100
            if bboxes else 0.0
        )
        new_row["clean_method"] = (
            "lama" if (bboxes and use_lama and total_mask_area > TELEA_MAX_AREA)
            else "telea" if bboxes
            else "none"
        )
        out_rows.append(new_row)

    # ── 输出统计 ────────────────────────────────────────────────────────
    pct = stats["detected"] / max(1, stats["total"]) * 100
    print(f"\n[Stats] 检测到文字: {stats['detected']}/{stats['total']} ({pct:.1f}%)")
    print(f"[Stats] 总 bbox: {stats['total_bboxes']}, "
          f"LaMa: {stats['lama_used']}, TELEA: {stats['telea_used']}")
    print(f"[Stats] 跳过: {stats['skipped']}")

    # ── 写 metadata ─────────────────────────────────────────────────────
    if not args.report_only and out_rows:
        out_csv = out_dir / "metadata_clean.csv"
        fields = list(out_rows[0].keys())
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(out_rows)
        print(f"[Done] metadata → {out_csv}")

        # 错误日志
        if error_log:
            log_path = out_dir / "_logs" / "error_log.jsonl"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w") as f:
                for e in error_log:
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")
            print(f"[Warn] {len(error_log)} 个错误, 日志 → {log_path}")

    # ── 报告模式 ────────────────────────────────────────────────────────
    if args.report_only:
        # 按 view/density 分组统计
        from collections import Counter
        det_by_vd = Counter()
        for row, bboxes in [(r, []) for r in rows]:
            pass  # report-only re-runs detection below
        print("\n[Report] 详情见上方统计")

    print(f"\n[Output] {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 7: 验证脚本语法**

```bash
python3 -c "import py_compile; py_compile.compile('scripts/preprocessing/clean_text_craft_lama.py', doraise=True); print('Syntax OK')"
```

---

### Task 3: 小规模测试 + 效果验证

**Files:**
- (no new files)

- [ ] **Step 1: 报告模式 — 扫描 50 张图**

```bash
python3 scripts/preprocessing/clean_text_craft_lama.py --report-only --limit 50
```

预期: 输出检测到文字标注的图像数量和 bbox 数

- [ ] **Step 2: 干燥运行 — 实际擦除 20 张图**

```bash
python3 scripts/preprocessing/clean_text_craft_lama.py \
  --limit 20 --save-debug \
  --output-dir datasets/CBIS_CLEAN_V3_test
```

- [ ] **Step 3: 检查结果**

```bash
# 检查输出的影像
ls datasets/CBIS_CLEAN_V3_test/MLO/*/
ls datasets/CBIS_CLEAN_V3_test/_debug/
# 查看 metadata
python3 -c "
import csv
with open('datasets/CBIS_CLEAN_V3_test/metadata_clean.csv') as f:
    rows = list(csv.DictReader(f))
    det = [r for r in rows if int(r['text_bbox_count']) > 0]
    print(f'检测到文字: {len(det)}/{len(rows)}')
    for r in det[:5]:
        print(f\"  {r['file_name']}: {r['text_bbox_count']} bboxes, area={r['text_area_pct']:.2f}%, method={r['clean_method']}\")
"
```

- [ ] **Step 4: 目视检查若干 debug 图像**

读取 2-3 张 `_debug/` 下的检测可视化图，确认 bbox 是否准确覆盖文字区域而非乳腺组织。

---

### Task 4: 全量处理 1296 张图

**Files:**
- (no new files, writes to `datasets/CBIS_CLEAN_V3/`)

- [ ] **Step 1: 全量处理**

```bash
python3 scripts/preprocessing/clean_text_craft_lama.py \
  --save-debug --output-dir datasets/CBIS_CLEAN_V3
```

预期耗时: ~30-60 分钟 (1296 张 × 1-2s/张，含 CRAFT + LaMa)

- [ ] **Step 2: 验证输出完整性**

```bash
python3 -c "
import csv
with open('datasets/CBIS_CLEAN_V3/metadata_clean.csv') as f:
    rows = list(csv.DictReader(f))
print(f'Total output: {len(rows)}')
det = [r for r in rows if int(r['text_bbox_count']) > 0]
print(f'With text detected: {len(det)}')
print(f'  LaMa: {sum(1 for r in det if r[\"clean_method\"]==\"lama\")}')
print(f'  TELEA: {sum(1 for r in det if r[\"clean_method\"]==\"telea\")}')
# Check marker_score correlation
from collections import Counter
high_marker = [r for r in rows if int(r.get('marker_score', 0)) >= 6]
print(f'high marker_score≥6 images: {len(high_marker)}')
det_in_high = [r for r in high_marker if int(r['text_bbox_count']) > 0]
print(f'  detected text in high marker: {len(det_in_high)}/{len(high_marker)}')
"
```

- [ ] **Step 3: 检查处理日志**

```bash
cat datasets/CBIS_CLEAN_V3/_logs/error_log.jsonl 2>/dev/null || echo "No errors"
```

---

### Task 5: 更新 Captions — 注入 text-free 描述

**Files:**
- Modify: `scripts/preprocessing/generate_captions.py`

- [ ] **Step 1: 修改 `make_caption` 函数，注入 text-free 关键词**

在 `generate_captions.py` 的 `make_caption` 函数中，于 `ANATOMY_KEYWORDS` 之后插入 text-free 描述。

```python
# 在 ANATOMY_KEYWORDS 后新增
TEXT_FREE_KEYWORDS = (
    "no text",
    "no labels",
    "no annotations",
    "no DICOM markers",
    "no alphanumeric overlay",
    "clean diagnostic image",
)
```

And update `make_caption`:

```python
def make_caption(view: str, laterality: str, density: str) -> str:
    pieces = [
        "mammography",
        VIEW_TEXT.get(view.upper(), "mammogram view"),
        LATERALITY_TEXT.get(laterality.upper(), "breast"),
        DENSITY_TEXT.get(density, "fibroglandular density"),
        *ANATOMY_KEYWORDS,
        *TEXT_FREE_KEYWORDS,  # ← 新增
        "grayscale",
        "high contrast",
        "medical imaging",
        "radiograph",
        "diagnostic quality",
    ]
    if view.upper() == "MLO":
        pieces.insert(5, "pectoral muscle")
    else:
        pieces.insert(5, "compressed breast tissue")
    return ", ".join(pieces)
```

- [ ] **Step 2: 为 CBIS_CLEAN_V3 重新生成 captions**

```bash
python3 scripts/preprocessing/generate_captions.py \
  --clean-dir datasets/CBIS_CLEAN_V3 \
  --write-sidecar --include-rejected --balanced-metadata
```

预期: 在 `CBIS_CLEAN_V3/` 下生成 `metadata.csv`、`metadata_balanced.csv`、每张图的 `.txt` sidecar

- [ ] **Step 3: 验证 caption 内容**

```bash
head -5 datasets/CBIS_CLEAN_V3/metadata.csv
cat "$(head -1 datasets/CBIS_CLEAN_V3/metadata.csv | cut -d, -f1 | xargs dirname)/$(head -1 datasets/CBIS_CLEAN_V3/metadata.csv | cut -d, -f1 | xargs basename | sed 's/.jpg/.txt/')" 2>/dev/null
# Or simpler:
python3 -c "
lines = open('datasets/CBIS_CLEAN_V3/metadata.csv').readlines()
print(lines[1][:200])
# Check text-free keywords present
assert 'no text' in lines[1].lower() or 'no labels' in lines[1].lower(), 'Missing text-free keywords!'
print('Caption check PASSED')
"
```

---

### Task 6: 准备 LoRA v5 训练数据集 (可选 — 取决于是否需要立即重训)

**Files:**
- Modify: `scripts/training/prepare_lora_dataset.py`

- [ ] **Step 1: 修改 `prepare_lora_dataset.py` 的 `build_caption` 函数**

在 `build_caption` 返回的 caption 末尾追加 text-free 描述:

```python
def build_caption(meta: dict) -> str:
    # ... existing code ...
    caption = (
        f"a mammography X-ray image of the {side_str} breast, "
        f"{view_str} view, {density_str}, {finding_str}, "
        f"grayscale medical imaging, diagnostic quality, "
        f"no text, no labels, no annotations, no DICOM markers"
    )
    return caption
```

- [ ] **Step 2: 用 CBIS_CLEAN_V3 源图生成训练集**

```bash
python3 scripts/training/prepare_lora_dataset.py \
  --jpeg-dir datasets/CBIS_CLEAN_V3 \
  --meta datasets/CBIS_CLEAN_V3/metadata.csv \
  --out outputs/lora_dataset_v5 \
  --resolution 512
```

- [ ] **Step 3: 验证训练集**

```bash
ls outputs/lora_dataset_v5/ | head -20
echo "---"
wc -l outputs/lora_dataset_v5/metadata.csv
echo "---"
cat outputs/lora_dataset_v5/00000.txt
```

---

### Task 7: 训练 LoRA v5 (取决于用户是否要此刻重训)

**Files:**
- Modify: (none, use existing `train_lora_quick.py`)

- [ ] **Step 1: 启动 LoRA v5 训练**

```bash
python3 scripts/training/train_lora_quick.py \
  --dataset-dir outputs/lora_dataset_v5 \
  --output-dir outputs/lora/mammo_sd15_v5_clean \
  --base-model hf_cache/sd15 \
  --num-epochs 10 \
  --batch-size 2 \
  --learning-rate 1e-4
```

- [ ] **Step 2: 用 LoRA v5 跑一次生成测试**

```bash
python3 scripts/generation/run_mammo_sd15.py \
  --base-model-local hf_cache/sd15 \
  --lora-path outputs/lora/mammo_sd15_v5_clean/final_lora \
  --metadata-csv datasets/CBIS_CLEAN_V2/metadata_clean.csv \
  --filter-view MLO --filter-density dense \
  --num-images 4 --seed 2026 \
  --output-base outputs/generated/test_v5
```

- [ ] **Step 3: 肉眼检查生成图有无文字伪影**

---

## 依赖与前置条件

- EasyOCR 1.7.2 (已安装)
- PyTorch 2.11 + CUDA (已安装)
- OpenCV 4.13 (已安装)
- LaMa `big-lama.pt` (Task 1 下载, ~200MB)
- CBIS_CLEAN_V2 + metadata_clean.csv (已存在)
- 无需 pip 安装新包 (LaMa 通过 torch.jit 加载)

## 断点续传

主管线脚本支持 `--start-idx N` 参数，若中途中断可从此恢复:
```bash
python3 scripts/preprocessing/clean_text_craft_lama.py --start-idx 500
```
