#!/usr/bin/env python3
"""为 CBIS_CLEAN 生成结构化 caption。

正样本 caption 强绑定 mammography + 视图 + 左右 + 密度 + 解剖关键词；
rejected 目录写负样本 caption，供模态分类器或后续对比学习使用。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

DENSITY_TEXT = {
    "fatty": "fatty breast tissue",
    "scattered": "scattered fibroglandular density",
    "heterogeneous": "heterogeneously dense breast tissue",
    "dense": "extremely dense breast tissue",
}

VIEW_TEXT = {
    "CC": "craniocaudal view",
    "MLO": "mediolateral oblique view",
}

LATERALITY_TEXT = {
    "LEFT": "left breast",
    "RIGHT": "right breast",
}

ANATOMY_KEYWORDS = (
    "skin line",
    "nipple in profile",
    "breast contour",
    "fibroglandular tissue",
    "radiographic texture",
)

TEXT_FREE_KEYWORDS = (
    "no text",
    "no labels",
    "no annotations",
    "no DICOM markers",
    "no alphanumeric overlay",
    "clean diagnostic image",
)

NEGATIVE_CAPTION = (
    "chest x-ray, lung, rib, ribcage, heart, spine, bone, skeletal, "
    "ultrasound, sonography, ct scan, mri, natural image, color image, "
    "collage, tiled, grid, seams, patchwork, not mammography"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="为 CBIS_CLEAN 写结构化 caption")
    p.add_argument("--clean-dir", type=Path, default=Path("datasets/CBIS_CLEAN"))
    p.add_argument(
        "--write-sidecar",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="为每张图写同名 .txt（kohya/sd-scripts 友好）",
    )
    p.add_argument(
        "--include-rejected",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="为 rejected 写负样本 caption 并输出 metadata_rejected.csv",
    )
    p.add_argument(
        "--balanced-metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="额外输出按 view/density 过采样均衡的 metadata_balanced.csv（不复制图）。",
    )
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def make_caption(view: str, laterality: str, density: str) -> str:
    pieces = [
        "mammography",
        VIEW_TEXT.get(view.upper(), "mammogram view"),
        LATERALITY_TEXT.get(laterality.upper(), "breast"),
        DENSITY_TEXT.get(density, "fibroglandular density"),
        *ANATOMY_KEYWORDS,
        *TEXT_FREE_KEYWORDS,
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


def read_clean_metadata(clean_dir: Path) -> list[dict]:
    path = clean_dir / "metadata_clean.csv"
    if not path.is_file():
        raise FileNotFoundError(f"缺少清洗元数据: {path}")
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    args = parse_args()
    clean_dir = args.clean_dir.resolve()
    rows = read_clean_metadata(clean_dir)

    out_rows: list[dict[str, str]] = []
    for r in rows:
        rel = Path(r["file_name"])
        img_path = clean_dir / rel
        cap = make_caption(r.get("view", ""), r.get("laterality", ""), r.get("density", ""))
        if args.write_sidecar:
            txt_path = img_path.with_suffix(".txt")
            txt_path.write_text(cap + "\n", encoding="utf-8")
        out_rows.append({"file_name": str(rel), "text": cap})

    with (clean_dir / "metadata.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file_name", "text"])
        w.writeheader()
        w.writerows(out_rows)

    if args.balanced_metadata:
        rng = random.Random(args.seed)
        buckets: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for meta, cap_row in zip(rows, out_rows):
            buckets[(meta.get("view", ""), meta.get("density", ""))].append(cap_row)
        target = max((len(v) for v in buckets.values()), default=0)
        balanced: list[dict[str, str]] = []
        for key, vals in sorted(buckets.items()):
            if not vals:
                continue
            balanced.extend(vals)
            need = target - len(vals)
            if need > 0:
                balanced.extend(rng.choice(vals) for _ in range(need))
        rng.shuffle(balanced)
        with (clean_dir / "metadata_balanced.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["file_name", "text"])
            w.writeheader()
            w.writerows(balanced)
        counts = Counter()
        for r in balanced:
            rel = Path(r["file_name"])
            counts[f"{rel.parts[0]}/{rel.parts[1]}"] += 1
        (clean_dir / "metadata_balanced_summary.json").write_text(
            json.dumps({k: int(v) for k, v in sorted(counts.items())}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    neg_rows: list[dict[str, str]] = []
    if args.include_rejected:
        rej_path = clean_dir / "rejected.jsonl"
        if rej_path.is_file():
            with rej_path.open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    rel = rec.get("file_name")
                    if not rel:
                        continue
                    img_path = clean_dir / rel
                    if img_path.exists() or img_path.is_symlink():
                        if args.write_sidecar:
                            img_path.with_suffix(".txt").write_text(NEGATIVE_CAPTION + "\n", encoding="utf-8")
                        neg_rows.append({"file_name": rel, "text": NEGATIVE_CAPTION})
        with (clean_dir / "metadata_rejected.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["file_name", "text"])
            w.writeheader()
            w.writerows(neg_rows)

    print(f"正样本 captions: {len(out_rows)} -> {clean_dir / 'metadata.csv'}")
    if args.balanced_metadata:
        print(f"均衡 captions: {len(balanced)} -> {clean_dir / 'metadata_balanced.csv'}")
    if args.include_rejected:
        print(f"负样本 captions: {len(neg_rows)} -> {clean_dir / 'metadata_rejected.csv'}")

    # 抽查前 5 条，方便终端验收。
    for r in out_rows[:5]:
        print(f"[sample] {r['file_name']}: {r['text']}")


if __name__ == "__main__":
    main()
