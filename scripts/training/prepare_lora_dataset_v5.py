#!/usr/bin/env python3
"""为 v5 训练准备数据集。

当前使用 CBIS_CLEAN_V2 MLO（1296 张清洗版，与 v4 相同源），
配合 train_mammo_lora.py 的数据增强（RandomHorizontalFlip + Rotation + ColorJitter）。
CC 视图因 CBIS_CLEAN 中 symlink 断链暂未包含，后续可从 datasets/jpeg/ + metadata.jsonl 重建。
"""

from __future__ import annotations

import csv
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

V2_MLO_DIR = ROOT / "datasets/CBIS_CLEAN_V2/MLO"
V2_CLEAN_CSV = ROOT / "datasets/CBIS_CLEAN_V2/metadata_clean.csv"

OUT = ROOT / "datasets/CBIS_CLEAN_V3"
PER_BUCKET_TARGET = 600  # 4 buckets × 600 = 2400

DENSITY_MAP = {
    "fatty": "predominantly fatty breast tissue",
    "scattered": "scattered fibroglandular density",
    "heterogeneous": "heterogeneously dense breast tissue",
    "dense": "extremely dense breast tissue",
}


def build_caption(fn: str, view: str, density: str, laterality: str) -> str:
    lat_text = {"LEFT": "left breast", "RIGHT": "right breast"}.get(laterality.upper(), "breast")
    density_text = DENSITY_MAP.get(density, "breast tissue")
    view_text = "mediolateral oblique view"
    return (
        f"mammography, {view_text}, {lat_text}, {density_text}, "
        f"skin line, pectoral muscle, nipple in profile, breast contour, "
        f"fibroglandular tissue, radiographic texture, "
        f"no text, no labels, no annotations, no DICOM markers, "
        f"no alphanumeric overlay, clean diagnostic image, "
        f"grayscale, high contrast, medical imaging, radiograph, diagnostic quality"
    )


def parse_v2_clean(csv_path: Path) -> list[dict]:
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            fn = r["file_name"].strip()  # MLO/density/P_xxx.jpg
            view = "MLO"
            density = r.get("density", "unknown")
            laterality = r.get("laterality", "")
            subpath = fn.split("/", 1)[1] if "/" in fn else fn
            src = V2_MLO_DIR / subpath
            if not src.is_file():
                continue
            caption = build_caption(fn, view, density, laterality)
            rows.append({
                "file_name": fn, "text": caption,
                "view": view, "density": density,
                "src": src,
            })
    return rows


def main():
    print("[v5 prepare] 扫描 CBIS_CLEAN_V2 MLO...")
    rows = parse_v2_clean(V2_CLEAN_CSV)
    print(f"  有效: {len(rows)} 张")

    # 分桶
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        buckets[r["density"]].append(r)

    print(f"  桶分布 ({len(buckets)} 桶):")
    for density in sorted(buckets):
        print(f"    MLO/{density:<15s}: {len(buckets[density]):>4d} 张")

    # Balanced oversampling
    balanced: list[dict] = []
    for density in sorted(buckets):
        bucket_rows = buckets[density]
        if len(bucket_rows) >= PER_BUCKET_TARGET:
            balanced.extend(bucket_rows[:PER_BUCKET_TARGET])
        else:
            factor = PER_BUCKET_TARGET // len(bucket_rows) + 1
            balanced.extend((bucket_rows * factor)[:PER_BUCKET_TARGET])

    print(f"  Balanced: {len(balanced)} 张 ({len(buckets)} × {PER_BUCKET_TARGET})")

    # 写出
    OUT.mkdir(parents=True, exist_ok=True)
    for density in sorted(buckets):
        (OUT / "MLO" / density).mkdir(parents=True, exist_ok=True)

    with open(OUT / "metadata.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file_name", "text"])
        w.writeheader()
        for r in balanced:
            w.writerow({"file_name": r["file_name"], "text": r["text"]})

    # 清理旧 V3 残留（CC 目录中的断链等）
    for old_cc in OUT.glob("CC"):
        if old_cc.is_dir():
            shutil.rmtree(old_cc)
            print(f"  清理旧目录: {old_cc}")

    # 拷贝
    copied, skipped, missing = 0, 0, 0
    for r in balanced:
        dst = OUT / r["file_name"]
        if dst.exists():
            skipped += 1
            continue
        src = r.get("src")
        if src is None or not src.is_file():
            missing += 1
            if missing <= 3:
                print(f"    [missing] {r['file_name']}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1

    print(f"  拷贝完成: {copied} copied, {skipped} skipped, {missing} missing")
    print(f"\n[Done] v5 训练集就绪: {OUT}")
    print(f"  训练:")
    print(f"    bash scripts/training/train_lora_v5.sh")


if __name__ == "__main__":
    main()
