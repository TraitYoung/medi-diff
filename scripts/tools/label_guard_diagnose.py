#!/usr/bin/env python3
"""对已保存的钼靶 PNG 诊断「VL 框 + 片缘涂黑」效果（无需完整 SD 生成脚本）。

用于阶梯式对照：同一张图上看 raw bbox、涂黑后结果、以及统计（多少框因面积/片缘被丢弃）。

示例：
  LABEL_GUARD_DEBUG=1 python3 scripts/tools/label_guard_diagnose.py \\
    --image outputs/generated/某批次/sd15_xxx_000.png --out /tmp/blacked.png

  # 已有 bbox JSON（跳过 API）：
  python3 scripts/tools/label_guard_diagnose.py --image a.png --bboxes-json '[{"x1":2,"y1":5,"x2":18,"y2":12}]'
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "preprocessing"))
from mammo_label_heuristic import (  # noqa: E402
    ask_vl_label_bbox,
    blackout_label_regions,
)


def main() -> int:
    p = argparse.ArgumentParser(description="Label guard VL bbox + blackout 诊断")
    p.add_argument("--image", type=Path, required=True, help="输入 PNG/JPG")
    p.add_argument("--out", type=Path, default=None, help="涂黑后保存路径（可选）")
    p.add_argument(
        "--bboxes-json",
        type=str,
        default="",
        help='若提供则跳过 VL：如 \'[{"x1":0,"y1":0,"x2":10,"y2":10}]\' 像素或百分比与 VL 一致',
    )
    p.add_argument("--no-vl", action="store_true", help="等价于只涂黑空框列表（对照原图）")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    img = cv2.imread(str(args.image), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"无法读取: {args.image}", file=sys.stderr)
        return 1
    if img.ndim == 3 and img.shape[2] >= 3:
        gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    else:
        gray = np.asarray(img)

    bboxes: list[tuple[int, int, int, int]] = []
    if args.no_vl:
        bboxes = []
    elif args.bboxes_json.strip():
        try:
            raw = json.loads(args.bboxes_json)
        except json.JSONDecodeError as e:
            print(f"JSON 解析失败: {e}", file=sys.stderr)
            return 1
        if not isinstance(raw, list):
            print("--bboxes-json 须为 JSON 数组", file=sys.stderr)
            return 1
        h, w = gray.shape[:2]
        for b in raw:
            if not isinstance(b, dict):
                continue
            try:
                px1 = float(b["x1"])
                py1 = float(b["y1"])
                px2 = float(b["x2"])
                py2 = float(b["y2"])
            except (KeyError, TypeError, ValueError):
                continue
            mx = max(abs(px1), abs(py1), abs(px2), abs(py2))
            if mx <= 1.000001:
                ax1 = max(0, int(round(px1 * w)))
                ax2 = max(0, int(round(px2 * w)))
                ay1 = max(0, int(round(py1 * h)))
                ay2 = max(0, int(round(py2 * h)))
            elif mx > 100.0:
                ax1 = int(np.clip(px1, 0, w - 1))
                ax2 = int(np.clip(px2, 0, w))
                ay1 = int(np.clip(py1, 0, h - 1))
                ay2 = int(np.clip(py2, 0, h))
            else:
                ax1 = max(0, int(px1 / 100.0 * w))
                ay1 = max(0, int(py1 / 100.0 * h))
                ax2 = min(w, int(px2 / 100.0 * w))
                ay2 = min(h, int(py2 / 100.0 * h))
            if ax2 < ax1:
                ax1, ax2 = ax2, ax1
            if ay2 < ay1:
                ay1, ay2 = ay2, ay1
            if ax2 > ax1 and ay2 > ay1:
                bboxes.append((ax1, ay1, ax2, ay2))
    else:
        bboxes = ask_vl_label_bbox(gray)
        logging.info("VL 返回 %d 个框", len(bboxes))

    out_arr, stats = blackout_label_regions(gray, bboxes, return_stats=True)
    print(json.dumps({"bboxes": [list(t) for t in bboxes], "stats": stats}, ensure_ascii=False, indent=2))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.out), out_arr)
        logging.info("已写入 %s", args.out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
