"""批量为真实钼靶图提取乳腺二值 mask，给 T2I-Adapter 训练/推理用。

- 输入：`datasets/jpeg/<series>/<file>.jpg` 10237 张
- 输出：`datasets/breast_masks/<series>/<file>.png`，灰度 L 模式，mask=0/255
- 复用 review_generated_images.py 里的 build_mask（Otsu + 形态学 + 最大连通域）

为了 T2I-Adapter 训练加速，输出 mask 直接按训练分辨率重采样（默认 1024x1024），
同时保留一个 `meta.jsonl` 记录原始 size / letterbox bbox 方便后续对齐。
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.core.image_utils import build_mask


def extract_breast_mask(gray: np.ndarray) -> np.ndarray:
    """乳腺区域二值掩膜（0/255）。与 `review_generated_images.build_mask` 同源，供评审漏斗显式复用。"""
    return build_mask(gray)


def list_jpegs(root: Path, pattern: str = "1-*.jpg") -> list[Path]:
    """CBIS-DDSM 命名：1-*.jpg 为原始钼靶，2-*.jpg 为病灶 ROI mask。
    默认只取 1-*.jpg。
    """
    return sorted(p for p in root.rglob(pattern) if p.is_file())


def letterbox_to_square(
    img: np.ndarray, target: int, interp: int = cv2.INTER_AREA
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """等比缩放后居中填充到 target x target。返回 (img, bbox(x,y,w,h))。"""
    h, w = img.shape[:2]
    scale = target / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    out = np.zeros((target, target), dtype=img.dtype)
    x = (target - nw) // 2
    y = (target - nh) // 2
    out[y : y + nh, x : x + nw] = resized
    return out, (x, y, nw, nh)


def process_one(
    args: tuple[Path, Path, Path, int, float, float],
) -> tuple[str, int, int, int, int, float]:
    """返回 (rel_str, x, y, w, h, mask_ratio)；不合格样本 w/h=0 或 mask_ratio<0 表示跳过。"""
    src, jpeg_root, mask_root, target, min_ratio, max_ratio = args
    rel = src.relative_to(jpeg_root)
    out_path = mask_root / rel.with_suffix(".png")
    if out_path.exists():
        return (str(rel), -1, -1, -1, -1, -1.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gray = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return (str(rel), 0, 0, 0, 0, 0.0)

    mask = extract_breast_mask(gray)
    mask_ratio = float(np.count_nonzero(mask)) / float(mask.size)

    # 过滤 build_mask 失败或几乎没乳腺的样本
    if mask_ratio < min_ratio or mask_ratio > max_ratio:
        return (str(rel), 0, 0, 0, 0, mask_ratio)

    if target > 0:
        # mask 用 NEAREST 防止插值把 0/255 打成灰阶
        mask_sq, (x, y, nw, nh) = letterbox_to_square(mask, target, interp=cv2.INTER_NEAREST)
        # 额外再做一道二值化保险
        mask_sq = np.where(mask_sq >= 128, 255, 0).astype(np.uint8)
        Image.fromarray(mask_sq, mode="L").save(out_path)
        return (str(rel), x, y, nw, nh, mask_ratio)
    # 原尺寸直接写（已是 0/255）
    Image.fromarray(mask, mode="L").save(out_path)
    return (str(rel), 0, 0, gray.shape[1], gray.shape[0], mask_ratio)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jpeg-root", type=Path, default=ROOT / "datasets/jpeg")
    ap.add_argument("--mask-root", type=Path, default=ROOT / "datasets/breast_masks")
    ap.add_argument("--pattern", default="1-*.jpg", help="glob 模式（默认只取 CBIS-DDSM 的 1-*.jpg 原图）")
    ap.add_argument("--target-size", type=int, default=1024, help="输出正方形尺寸，0=原图大小不 letterbox")
    ap.add_argument("--min-mask-ratio", type=float, default=0.03)
    ap.add_argument("--max-mask-ratio", type=float, default=0.80)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 张（调试用，0=全部）")
    args = ap.parse_args()

    args.mask_root.mkdir(parents=True, exist_ok=True)
    paths = list_jpegs(args.jpeg_root, args.pattern)
    if args.limit > 0:
        paths = paths[: args.limit]
    print(f"Total {len(paths)} images (pattern={args.pattern}) → {args.mask_root}")

    payloads = [
        (p, args.jpeg_root, args.mask_root, args.target_size, args.min_mask_ratio, args.max_mask_ratio)
        for p in paths
    ]
    results: list[tuple[str, int, int, int, int, float]] = []

    if args.workers <= 1:
        for pl in payloads:
            results.append(process_one(pl))
            if len(results) % 500 == 0:
                print(f"  processed {len(results)}/{len(payloads)}")
    else:
        with mp.Pool(args.workers) as pool:
            for i, r in enumerate(pool.imap_unordered(process_one, payloads, chunksize=16), 1):
                results.append(r)
                if i % 500 == 0:
                    print(f"  processed {i}/{len(payloads)}")

    # 写 meta：只保留合格样本
    meta_path = args.mask_root / "meta.jsonl"
    skip_path = args.mask_root / "skipped.jsonl"
    ok = skip = pre = fail = 0
    with meta_path.open("w", encoding="utf-8") as f, skip_path.open("w", encoding="utf-8") as fs:
        for rel, x, y, w, h, mr in results:
            if x == -1:
                pre += 1
                continue  # 已存在
            if w == 0 and mr == 0.0:
                fail += 1
                fs.write(json.dumps({"rel": rel, "reason": "read_failed"}) + "\n")
                continue
            if w == 0:
                skip += 1
                fs.write(
                    json.dumps({"rel": rel, "reason": "mask_ratio_out_of_range", "mask_ratio": mr}) + "\n"
                )
                continue
            f.write(
                json.dumps({"rel": rel, "x": x, "y": y, "w": w, "h": h, "mask_ratio": mr}) + "\n"
            )
            ok += 1
    print(f"Meta → {meta_path}")
    print(f"Skip log → {skip_path}")
    print(
        f"Done: ok={ok}, skipped_by_ratio={skip}, failed_read={fail}, pre_existing={pre} "
        f"(total input {len(results)})"
    )


if __name__ == "__main__":
    main()
