#!/usr/bin/env python3
"""带标签置信度的钼靶「抽取—重试—保守裁条带」预处理。

策略（每条输出槽位）：
1. 随机抽一张图，用语义无关的边角纹理启发式判断三态：明确干净 / 明确像有标签 / 不确定。
2. 若「明确干净」→ 原图进入下一步（可选复制或软链）。
3. 若「明确像有标签」→ 直接保守裁条带后输出（不再重抽）。
4. 若「不确定」→ 再随机抽一张，最多共抽 --max-picks 张（默认 5）；若仍从未遇到前两种确定态，
   则对**最后一次**抽到的图做保守裁条带。

依赖：OpenCV、numpy；与 prepare_lora_dataset 相同方式写 caption（可选 metadata.jsonl）。
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "training"))
from prepare_lora_dataset import build_caption  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts" / "preprocessing"))
from mammo_label_heuristic import LabelVerdict, compute_label_heuristic  # noqa: E402


def conservative_strip_crop(
    gray: np.ndarray,
    *,
    top: float,
    bottom: float,
    left: float,
    right: float,
) -> np.ndarray:
    """按比例从四边裁掉条带；比例 ∈ [0, 0.49)。"""
    h, w = gray.shape[:2]
    yt = int(round(h * top))
    yb = int(round(h * bottom))
    xl = int(round(w * left))
    xr = int(round(w * right))
    y1, y2 = yt, h - yb
    x1, x2 = xl, w - xr
    if y2 <= y1 + 16 or x2 <= x1 + 16:
        return gray
    return gray[y1:y2, x1:x2]


def collect_jpegs(jpeg_dir: Path) -> list[Path]:
    return sorted(jpeg_dir.rglob("*.jpg")) + sorted(jpeg_dir.rglob("*.jpeg"))


def load_meta_map(meta_path: Path) -> dict[str, dict]:
    meta_by_file: dict[str, dict] = {}
    if not meta_path.exists():
        return meta_by_file
    with meta_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            fname = obj.get("file_name") or obj.get("filename") or obj.get("image")
            if fname:
                p = Path(fname)
                meta_by_file[p.name] = obj
                meta_by_file[p.stem] = obj
    return meta_by_file


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="标签置信度守护下的钼靶抽样与裁条带")
    p.add_argument("--jpeg-dir", type=Path, default=ROOT / "datasets" / "jpeg")
    p.add_argument("--meta", type=Path, default=ROOT / "datasets" / "metadata.jsonl")
    p.add_argument("--out", type=Path, default=ROOT / "outputs" / "lora_dataset_label_guard")
    p.add_argument("--max-images", type=int, default=200, help="输出样本条数（每个槽位一条）")
    p.add_argument("--max-picks", type=int, default=5, help="不确定时最多抽样次数（含首次）")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--symlink-clean", action="store_true", help="明确干净的样本用软链省空间")
    p.add_argument(
        "--jpeg-quality",
        type=int,
        default=92,
        help="写盘 JPEG 质量（裁切后的图必写文件）",
    )
    # 启发式阈值（略像「置信度分界」）
    p.add_argument("--clean-below", type=float, default=0.22, help="label_score 低于此视为明确干净")
    p.add_argument("--labeled-above", type=float, default=0.48, help="label_score 高于此视为明确有标签")
    p.add_argument("--analysis-max-side", type=int, default=640)
    # 保守裁条带（占边比例）
    p.add_argument("--strip-top", type=float, default=0.04)
    p.add_argument("--strip-bottom", type=float, default=0.04)
    p.add_argument("--strip-left", type=float, default=0.03)
    p.add_argument("--strip-right", type=float, default=0.03)
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="写入 JSONL 记录每条条目的路径、判别与是否裁切；默认 out/label_guard_manifest.jsonl",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    jpeg_dir: Path = args.jpeg_dir
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or (out_dir / "label_guard_manifest.jsonl")

    pool = collect_jpegs(jpeg_dir)
    if not pool:
        raise SystemExit(f"未在 {jpeg_dir} 找到 JPEG")

    meta_by_file = load_meta_map(args.meta)

    csv_rows = ["file_name,text"]
    stats = {
        "certain_clean": 0,
        "certain_labeled_strip": 0,
        "fallback_strip_uncertain": 0,
        "skipped_unreadable": 0,
    }

    manifest_lines: list[str] = []

    for idx in tqdm(range(args.max_images), desc="label_guard_pick"):
        tried: set[str] = set()
        last_gray: np.ndarray | None = None
        last_path: Path | None = None
        settled = False
        action_kind = "fallback_strip_uncertain"

        for pick_i in range(args.max_picks):
            # 尽量不在同一槽位重复使用已试路径（池用尽则放行重复）
            choices = [p for p in pool if str(p) not in tried]
            if not choices:
                choices = pool
                tried.clear()
            path = rng.choice(choices)
            tried.add(str(path))

            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                stats["skipped_unreadable"] += 1
                continue
            last_gray = gray
            last_path = path

            hr = compute_label_heuristic(
                gray,
                analysis_max_side=args.analysis_max_side,
                clean_below=args.clean_below,
                labeled_above=args.labeled_above,
            )

            if hr.verdict == LabelVerdict.CERTAIN_CLEAN:
                action_kind = "certain_clean"
                settled = True
                break

            if hr.verdict == LabelVerdict.CERTAIN_LABELED:
                last_gray = conservative_strip_crop(
                    gray,
                    top=args.strip_top,
                    bottom=args.strip_bottom,
                    left=args.strip_left,
                    right=args.strip_right,
                )
                action_kind = "certain_labeled_strip"
                settled = True
                stats[action_kind] += 1
                break

            # UNCERTAIN：继续抽；保留 last_gray 以备最后一刀

        if last_gray is None or last_path is None:
            continue

        if not settled:
            # 连续 max_picks 次均不确定 → 保守裁最后一次
            last_gray = conservative_strip_crop(
                last_gray,
                top=args.strip_top,
                bottom=args.strip_bottom,
                left=args.strip_left,
                right=args.strip_right,
            )
            action_kind = "fallback_strip_uncertain"
            stats[action_kind] += 1
        elif action_kind == "certain_clean":
            stats["certain_clean"] += 1

        out_name = f"{idx:05d}.jpg"
        out_path = out_dir / out_name

        meta = meta_by_file.get(last_path.name, meta_by_file.get(last_path.stem, {}))
        caption = build_caption(meta)
        (out_dir / f"{idx:05d}.txt").write_text(caption, encoding="utf-8")
        csv_rows.append(f"{out_name},{caption}")

        if action_kind == "certain_clean" and args.symlink_clean:
            if out_path.exists() or out_path.is_symlink():
                out_path.unlink()
            out_path.symlink_to(last_path.resolve())
        else:
            ok, buf = cv2.imencode(
                ".jpg",
                last_gray,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(args.jpeg_quality, 50, 100))],
            )
            if not ok:
                stats["skipped_unreadable"] += 1
                continue
            out_path.write_bytes(buf.tobytes())

        rec = {
            "index": idx,
            "source": str(last_path),
            "action": action_kind,
            "output": str(out_path),
        }
        manifest_lines.append(json.dumps(rec, ensure_ascii=False))

    (out_dir / "metadata.csv").write_text("\n".join(csv_rows), encoding="utf-8")
    manifest_path.write_text("\n".join(manifest_lines) + ("\n" if manifest_lines else ""), encoding="utf-8")

    print("完成。统计:", stats)
    print("输出目录:", out_dir)
    print("manifest:", manifest_path)


if __name__ == "__main__":
    main()
