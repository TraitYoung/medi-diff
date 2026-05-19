#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Make top5 compare image: original vs generated")
    p.add_argument("--review-csv", type=Path, required=True)
    p.add_argument("--mapping-json", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--tile-w", type=int, default=512)
    p.add_argument("--tile-h", type=int, default=512)
    return p.parse_args()


def fit_gray(path: Path, size: tuple[int, int]) -> Image.Image:
    img = Image.open(path).convert("L")
    return img.resize(size, Image.BILINEAR)


def main() -> None:
    args = parse_args()
    mapping_data = json.loads(args.mapping_json.read_text(encoding="utf-8"))
    gen_to_src = {str(Path(x["generated"]).resolve()): str(Path(x["source"]).resolve()) for x in mapping_data}

    rows: list[dict] = []
    with args.review_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    rows.sort(key=lambda x: float(x.get("total_score", 0.0)), reverse=True)

    picked: list[tuple[Path, Path, float]] = []
    for r in rows:
        gen = Path(r["image"]).resolve()
        src = gen_to_src.get(str(gen))
        if not src:
            continue
        src_path = Path(src)
        if gen.exists() and src_path.exists():
            picked.append((src_path, gen, float(r["total_score"])))
        if len(picked) >= args.top_k:
            break
    if not picked:
        raise RuntimeError("No matched top images found between review csv and source mapping.")

    pad = 14
    label_h = 30
    row_h = args.tile_h + label_h
    canvas_w = pad * 3 + args.tile_w * 2
    canvas_h = pad + len(picked) * (row_h + pad)
    canvas = Image.new("L", (canvas_w, canvas_h), color=0)
    draw = ImageDraw.Draw(canvas)

    for i, (src, gen, score) in enumerate(picked):
        y = pad + i * (row_h + pad)
        src_img = fit_gray(src, (args.tile_w, args.tile_h))
        gen_img = fit_gray(gen, (args.tile_w, args.tile_h))
        canvas.paste(src_img, (pad, y))
        canvas.paste(gen_img, (pad * 2 + args.tile_w, y))
        draw.text((pad, y + args.tile_h + 6), f"Original #{i+1}", fill=255)
        draw.text((pad * 2 + args.tile_w, y + args.tile_h + 6), f"LoRA Inpaint  score={score:.2f}", fill=255)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(f"[Done] compare image saved: {args.output}")


if __name__ == "__main__":
    main()
