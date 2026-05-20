#!/usr/bin/env python3
"""构建 CBIS-DDSM 分层清洗集。

输出目录结构：

datasets/CBIS_CLEAN/
├── CC/{fatty,scattered,heterogeneous,dense}/
├── MLO/{fatty,scattered,heterogeneous,dense}/
└── rejected/

设计目标是给 LoRA v3 提供“够干净、够均衡、可追溯”的训练地基。脚本默认使用
symlink，避免复制数千张全幅钼靶造成额外磁盘占用；如需实体文件可加 --copy。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.core.image_utils import build_mask, is_image, resize_long_side

DENSITY_NAMES = {
    "1": "fatty",
    "2": "scattered",
    "3": "heterogeneous",
    "4": "dense",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="清洗并分层 CBIS-DDSM 全幅钼靶图")
    p.add_argument("--jpeg-root", type=Path, default=ROOT / "datasets/jpeg")
    p.add_argument("--csv-root", type=Path, default=ROOT / "datasets/csv")
    p.add_argument("--output-dir", type=Path, default=ROOT / "datasets/CBIS_CLEAN")
    p.add_argument("--copy", action="store_true", help="默认 symlink；打开后复制文件")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="调试用；0=全量")
    p.add_argument("--resize-long-side", type=int, default=1024, help="质量检测缩放长边，0=原图")
    p.add_argument("--min-mask-ratio", type=float, default=0.03)
    p.add_argument("--max-mask-ratio", type=float, default=0.90)
    p.add_argument("--max-extreme-ratio", type=float, default=0.05, help="乳腺区过曝/欠曝占比上限")
    p.add_argument(
        "--require-skin-line",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="边缘皮肤线缺失则 reject（默认开；若保留量不足可 --no-require-skin-line）",
    )
    p.add_argument(
        "--min-long-side",
        type=int,
        default=1200,
        help="原图长边下限；过滤 cropped/local magnification（默认 1200）",
    )
    p.add_argument("--min-short-side", type=int, default=700)
    return p.parse_args()


def read_case_density(csv_root: Path) -> dict[tuple[str, str, str], str]:
    """(patient_id, laterality, view) -> density_name."""
    out: dict[tuple[str, str, str], str] = {}
    for name in (
        "calc_case_description_train_set.csv",
        "calc_case_description_test_set.csv",
        "mass_case_description_train_set.csv",
        "mass_case_description_test_set.csv",
    ):
        path = csv_root / name
        if not path.is_file():
            continue
        with path.open(encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                pid = (row.get("patient_id") or "").strip()
                lat = (row.get("left or right breast") or "").strip().upper()
                view = (row.get("image view") or "").strip().upper()
                density_raw = (
                    row.get("breast density")
                    or row.get("breast_density")
                    or row.get("density")
                    or ""
                )
                density = DENSITY_NAMES.get(str(density_raw).strip(), "")
                if pid and lat and view in {"CC", "MLO"} and density:
                    out[(pid, lat, view)] = density
    return out


def iter_full_mammo_rows(csv_root: Path) -> Iterable[dict]:
    dicom_info = csv_root / "dicom_info.csv"
    with dicom_info.open(encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if (row.get("Modality") or "").strip() != "MG":
                continue
            desc = (row.get("SeriesDescription") or "").strip().lower()
            if "full mammogram" not in desc:
                continue
            patient_id = (row.get("PatientID") or "").strip()
            # PatientID 例：Mass-Training_P_01754_RIGHT_CC
            m = re.search(r"(P_\d+)_(LEFT|RIGHT)_(CC|MLO)", patient_id, flags=re.IGNORECASE)
            if not m:
                continue
            row["_patient"] = m.group(1)
            row["_laterality"] = m.group(2).upper()
            row["_view"] = m.group(3).upper()
            yield row


def resolve_jpeg_path(jpeg_root: Path, image_path: str) -> Path:
    """dicom_info.image_path 形如 CBIS-DDSM/jpeg/<uid>/1-xxx.jpg。"""
    marker = "CBIS-DDSM/jpeg/"
    if marker in image_path:
        rel = image_path.split(marker, 1)[1]
        return jpeg_root / rel
    return jpeg_root / Path(image_path).name


def skin_line_present(gray: np.ndarray, band_px: int = 20) -> bool:
    """检测边缘 20px 内连续亮线。全幅图上只作为清洗规则，阈值保持保守。"""
    h, w = gray.shape[:2]
    band = max(5, min(band_px, h // 10, w // 10))
    edge_roi = np.zeros_like(gray, dtype=np.uint8)
    edge_roi[:band, :] = 255
    edge_roi[-band:, :] = 255
    edge_roi[:, :band] = 255
    edge_roi[:, -band:] = 255

    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.bitwise_and(edges, edge_roi)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    longest = max((float(cv2.arcLength(c, False)) for c in cnts), default=0.0)
    return longest >= max(80.0, 0.10 * max(h, w))


def marker_score(gray: np.ndarray, mask: np.ndarray) -> int:
    """粗略文字/标尺检测：背景/边缘里的小亮连通域数量。过多说明有 marker/ruler。"""
    h, w = gray.shape[:2]
    edge = np.zeros_like(mask, dtype=np.uint8)
    margin = max(20, min(h, w) // 20)
    edge[:margin, :] = 255
    edge[-margin:, :] = 255
    edge[:, :margin] = 255
    edge[:, -margin:] = 255
    bg_edge = (edge > 0) & (mask == 0)
    if np.count_nonzero(bg_edge) < 100:
        return 0
    vals = gray[bg_edge]
    thr = max(180.0, float(np.percentile(vals, 99.0)))
    bright = ((gray >= thr) & bg_edge).astype(np.uint8) * 255
    n, _, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    cnt = 0
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if 8 <= area <= 800:
            cnt += 1
    return cnt


def quality_check(path: Path, args: argparse.Namespace) -> tuple[bool, list[str], dict]:
    gray0 = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray0 is None:
        return False, ["read_failed"], {}
    h0, w0 = gray0.shape[:2]
    reasons: list[str] = []
    if max(h0, w0) < args.min_long_side or min(h0, w0) < args.min_short_side:
        reasons.append("too_small_or_cropped")

    gray = resize_long_side(gray0, args.resize_long_side, only_downscale=True, min_side=16)
    mask = build_mask(gray)
    mask_ratio = float(np.count_nonzero(mask)) / float(mask.size)
    if mask_ratio < args.min_mask_ratio or mask_ratio > args.max_mask_ratio:
        reasons.append("mask_ratio_out_of_range")

    vals = gray[mask > 0]
    if vals.size < 1000:
        reasons.append("empty_breast_mask")
        bright_ratio = dark_ratio = 1.0
    else:
        v = vals.astype(np.float32) / 255.0
        bright_ratio = float(np.mean(v >= 0.98))
        dark_ratio = float(np.mean(v <= 0.02))
        # 只看乳腺区两端直方图，避免黑背景把欠曝比例拉满。
        if bright_ratio > args.max_extreme_ratio:
            reasons.append("overexposed")
        if dark_ratio > args.max_extreme_ratio:
            reasons.append("underexposed")

    if args.require_skin_line and not skin_line_present(gray):
        reasons.append("skin_line_missing")

    markers = marker_score(gray, mask)
    if markers > 40:
        reasons.append("marker_or_ruler_heavy")

    meta = {
        "height": int(h0),
        "width": int(w0),
        "mask_ratio": round(mask_ratio, 6),
        "bright_ratio": round(bright_ratio, 6),
        "dark_ratio": round(dark_ratio, 6),
        "marker_score": int(markers),
    }
    return not reasons, reasons, meta


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src, dst)


def main() -> None:
    args = parse_args()
    density_map = read_case_density(args.csv_root)
    rows = list(iter_full_mammo_rows(args.csv_root))
    if args.limit > 0:
        rows = rows[: args.limit]

    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for view in ("CC", "MLO"):
            for d in DENSITY_NAMES.values():
                (args.output_dir / view / d).mkdir(parents=True, exist_ok=True)
        (args.output_dir / "rejected").mkdir(parents=True, exist_ok=True)

    kept_rows: list[dict] = []
    rejected_rows: list[dict] = []
    counters = Counter()
    strat = Counter()

    for row in tqdm(rows, desc="Clean CBIS"):
        view = row["_view"]
        lat = row["_laterality"]
        pid = row["_patient"]
        density = density_map.get((pid, lat, view), "")
        src = resolve_jpeg_path(args.jpeg_root, row.get("image_path", ""))
        base = f"{pid}_{lat}_{view}_{Path(src).stem}.jpg"

        if not density:
            counters["missing_density"] += 1
            rejected_rows.append({"src": str(src), "reason": "missing_density", "patient_id": pid, "view": view})
            continue
        if not src.is_file():
            counters["missing_file"] += 1
            rejected_rows.append({"src": str(src), "reason": "missing_file", "patient_id": pid, "view": view})
            continue

        ok, reasons, qmeta = quality_check(src, args)
        if ok:
            rel = Path(view) / density / base
            if not args.dry_run:
                link_or_copy(src, args.output_dir / rel, args.copy)
            rec = {
                "file_name": str(rel),
                "src": str(src),
                "patient_id": pid,
                "laterality": lat,
                "view": view,
                "density": density,
                **qmeta,
            }
            kept_rows.append(rec)
            counters["kept"] += 1
            strat[(view, density)] += 1
        else:
            reason = "|".join(reasons)
            for r in reasons:
                counters[f"reject_{r}"] += 1
            rel = Path("rejected") / base
            if not args.dry_run:
                link_or_copy(src, args.output_dir / rel, args.copy)
            rejected_rows.append(
                {
                    "file_name": str(rel),
                    "src": str(src),
                    "patient_id": pid,
                    "laterality": lat,
                    "view": view,
                    "density": density,
                    "reason": reason,
                    **qmeta,
                }
            )

    summary = {
        "total_candidates": len(rows),
        "kept": len(kept_rows),
        "rejected": len(rejected_rows),
        "copy_mode": bool(args.copy),
        "strata": {f"{k[0]}/{k[1]}": int(v) for k, v in sorted(strat.items())},
        "counters": {k: int(v) for k, v in counters.most_common()},
    }
    counts = list(strat.values())
    if counts:
        summary["min_stratum"] = int(min(counts))
        summary["max_stratum"] = int(max(counts))
        summary["stratum_imbalance_ratio"] = round(float(max(counts)) / max(1.0, float(min(counts))), 4)

    if not args.dry_run:
        with (args.output_dir / "metadata_clean.csv").open("w", encoding="utf-8", newline="") as f:
            fields = [
                "file_name",
                "src",
                "patient_id",
                "laterality",
                "view",
                "density",
                "height",
                "width",
                "mask_ratio",
                "bright_ratio",
                "dark_ratio",
                "marker_score",
            ]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(kept_rows)
        with (args.output_dir / "rejected.jsonl").open("w", encoding="utf-8") as f:
            for r in rejected_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if len(kept_rows) < 2000:
        print("[WARN] 保留数 < 2000；可尝试 --no-require-skin-line 或放宽曝光/尺寸阈值。")
    if counts and min(counts) > 0:
        min_c, max_c = min(counts), max(counts)
        if (max_c - min_c) / max_c > 0.20:
            print("[WARN] 四类/视图分层仍不均衡；训练时建议使用 weighted sampler 或过采样。")


if __name__ == "__main__":
    main()
