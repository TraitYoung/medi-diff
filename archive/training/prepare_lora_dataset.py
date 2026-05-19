"""
prepare_lora_dataset.py
=======================
从 CBIS-DDSM JPEG 图像 + metadata.jsonl 生成 LoRA 训练数据集。

输出目录结构（HuggingFace datasets 格式）：
  outputs/lora_dataset/
    ├── 00000.png  ← 归一化后的 1024×768 钼靶图（长边裁至 768）
    ├── 00000.txt  ← 文本描述
    ├── ...
    └── metadata.csv

文本描述策略（从 CBIS-DDSM metadata 提取）：
  - BI-RADS 密度（1/2/3/4）→ "almost entirely fatty / scattered fibroglandular /
                               heterogeneously dense / extremely dense"
  - 投照方向（MLO/CC）
  - 左/右侧
  - 是否有 mass / calcification → 加入描述
  - 无 annotation → 健康样本描述

格式统一为：
  "a mammography X-ray image of the {side} breast, {view} view,
   {density}, {finding}"

Usage:
  python3 scripts/training/prepare_lora_dataset.py \
      --jpeg-dir datasets/jpeg \
      --meta    datasets/metadata.jsonl \
      --out     outputs/lora_dataset \
      --max-images 5000 \
      --long-side 768
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


DENSITY_MAP = {
    "1": "almost entirely fatty tissue",
    "2": "scattered fibroglandular density",
    "3": "heterogeneously dense breast",
    "4": "extremely dense breast",
    "A": "almost entirely fatty tissue",
    "B": "scattered fibroglandular density",
    "C": "heterogeneously dense breast",
    "D": "extremely dense breast",
}

VIEW_MAP = {
    "MLO": "mediolateral oblique",
    "CC": "craniocaudal",
    "LM": "lateromedial",
    "ML": "mediolateral",
}


def build_caption(meta: dict) -> str:
    side = meta.get("side", "").upper()
    side_str = "left" if side == "L" or side == "LEFT" else "right" if side == "R" or side == "RIGHT" else "bilateral"

    view = str(meta.get("view", "")).upper().split("_")[0]
    view_str = VIEW_MAP.get(view, "mammogram")

    density = str(meta.get("breast_density", meta.get("density", "")))
    density_str = DENSITY_MAP.get(density, "fibroglandular density")

    finding_parts = []
    pathology = str(meta.get("pathology", "")).lower()
    if "malignant" in pathology:
        finding_parts.append("malignant lesion")
    elif "benign" in pathology:
        finding_parts.append("benign finding")

    abn_type = str(meta.get("abnormality_type", "")).lower()
    if "mass" in abn_type:
        finding_parts.append("breast mass")
    elif "calcification" in abn_type or "calc" in abn_type:
        finding_parts.append("microcalcifications")

    if not finding_parts:
        finding_str = "no significant abnormality"
    else:
        finding_str = ", ".join(finding_parts)

    caption = (
        f"a mammography X-ray image of the {side_str} breast, "
        f"{view_str} view, {density_str}, {finding_str}, "
        f"grayscale medical imaging, diagnostic quality"
    )
    return caption


def resize_to_long_side(img: np.ndarray, long_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= long_side:
        return img
    if h > w:
        new_h = long_side
        new_w = int(w * long_side / h)
    else:
        new_w = long_side
        new_h = int(h * long_side / w)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)


def center_crop_square(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    min_dim = min(h, w)
    start_y = (h - min_dim) // 2
    start_x = (w - min_dim) // 2
    cropped = img[start_y:start_y + min_dim, start_x:start_x + min_dim]
    return cv2.resize(cropped, (size, size), interpolation=cv2.INTER_LANCZOS4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jpeg-dir", default="datasets/jpeg")
    parser.add_argument("--meta", default="datasets/metadata.jsonl")
    parser.add_argument("--out", default="outputs/lora_dataset")
    parser.add_argument("--max-images", type=int, default=5000)
    parser.add_argument("--resolution", type=int, default=512,
                        help="输出正方形分辨率（SD 1.5 默认 512）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--symlink", action="store_true", default=False,
                        help="使用软链接而不是复制图像（节省磁盘）")
    args = parser.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    jpeg_dir = Path(args.jpeg_dir)
    meta_path = Path(args.meta)

    # 构建 filename → metadata 映射
    meta_by_file = {}
    if meta_path.exists():
        with open(meta_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    fname = obj.get("file_name") or obj.get("filename") or obj.get("image")
                    if fname:
                        meta_by_file[Path(fname).name] = obj
                        meta_by_file[Path(fname).stem] = obj
                except Exception:
                    pass

    # 收集所有 JPEG
    all_imgs = sorted(jpeg_dir.rglob("*.jpg")) + sorted(jpeg_dir.rglob("*.jpeg"))
    print(f"发现 {len(all_imgs)} 张 JPEG 图像")

    if len(all_imgs) > args.max_images:
        all_imgs = random.sample(all_imgs, args.max_images)
        print(f"随机抽样 {args.max_images} 张")

    csv_rows = ["file_name,text"]
    processed = 0
    skipped = 0

    for img_path in tqdm(all_imgs, desc="处理图像"):
        try:
            # 仅验证图像可读性（不解码全图以节省时间）
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                skipped += 1
                continue

            # 跳过太小的图（<256px）
            if min(img.shape) < 256:
                skipped += 1
                continue

            # 获取 caption
            meta = meta_by_file.get(img_path.name, meta_by_file.get(img_path.stem, {}))
            caption = build_caption(meta)

            # 输出文件名
            out_name = f"{processed:05d}.jpg"
            out_path = out_dir / out_name

            if args.symlink:
                # 软链接（0字节额外磁盘）
                if out_path.exists() or out_path.is_symlink():
                    out_path.unlink()
                out_path.symlink_to(img_path.resolve())
            else:
                # 直接复制原始 JPEG（避免解码再编码，节省时间）
                shutil.copy2(str(img_path), str(out_path))

            # 写 caption 文件
            (out_dir / f"{processed:05d}.txt").write_text(caption)
            csv_rows.append(f"{out_name},{caption}")
            processed += 1

        except Exception as e:
            print(f"  跳过 {img_path.name}: {e}")
            skipped += 1

    # 写 metadata.csv（HuggingFace datasets 格式）
    (out_dir / "metadata.csv").write_text("\n".join(csv_rows))

    print(f"\n完成！处理: {processed} 张，跳过: {skipped} 张")
    print(f"输出目录: {out_dir}")
    print(f"示例 caption: {csv_rows[1] if len(csv_rows) > 1 else 'N/A'}")


if __name__ == "__main__":
    main()
