#!/usr/bin/env python3
"""
模拟专家主观评分工具（任务书第 5 条）。

按「放射科医生主观评审」场景，对已生成批次中的代表性图像逐项打分，
输出结构化 CSV 和 Markdown 报告。适合答辩时展示「定性评估」结果。

评分维度（参考 ACR/MQSA + Sickles 2013）：
  1. 解剖轮廓合理性   (0–5)  乳腺边缘轮廓、胸肌可见性（MLO）
  2. 腺体纹理真实性   (0–5)  纤维腺体密度梯度、小叶结构
  3. 亮度与对比度     (0–5)  背景纯黑、腺体灰阶范围
  4. 伪影/异常       (0–5)  无格线/条纹/环形/拼接痕
  5. 总体钼靶相似度   (0–5)  整体上是否像真实 FFDM

用法（交互模式，逐图评分）：
  python3 scripts/evaluation/simulated_expert_score.py \\
    --images-dir outputs/generated/毕业论文_生成图像/sd15_demo_xxx_000 \\
    --evaluator "评审人A"

用法（批量填入预设分数，适合脚本化复现）：
  python3 scripts/evaluation/simulated_expert_score.py \\
    --images-dir outputs/eval/trial_eff_20260503/top_k_images \\
    --scores-csv docs/expert_scores_preset.csv \\
    --output outputs/eval/trial_eff_20260503/expert_report.md
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DIMENSIONS = [
    ("anatomy",   "解剖轮廓合理性"),
    ("texture",   "腺体纹理真实性"),
    ("contrast",  "亮度与对比度"),
    ("artifact",  "伪影/异常缺失"),
    ("overall",   "总体钼靶相似度"),
]
MAX_SCORE = 5
WEIGHTS = [0.25, 0.25, 0.20, 0.20, 0.10]  # 加权总分（满分 5.0）


def weighted_total(scores: list[float]) -> float:
    return round(sum(s * w for s, w in zip(scores, WEIGHTS)), 2)


def load_preset_csv(path: Path) -> dict[str, list[float]]:
    """从预设 CSV 加载分数，格式：filename,anatomy,texture,contrast,artifact,overall"""
    result: dict[str, list[float]] = {}
    with path.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            fn = row.get("filename", "").strip()
            if not fn:
                continue
            scores = []
            for dim, _ in DIMENSIONS:
                try:
                    scores.append(float(row[dim]))
                except (KeyError, ValueError):
                    scores.append(3.0)
            result[fn] = scores
    return result


def interactive_score(img_path: Path) -> tuple[list[float], str]:
    print(f"\n{'─'*60}")
    print(f"图像：{img_path.name}")
    scores: list[float] = []
    for dim_key, dim_name in DIMENSIONS:
        while True:
            raw = input(f"  {dim_name} (0–{MAX_SCORE})：").strip()
            try:
                v = float(raw)
                if 0 <= v <= MAX_SCORE:
                    scores.append(v)
                    break
            except ValueError:
                pass
            print(f"    请输入 0–{MAX_SCORE} 的数字")
    comment = input("  备注（可为空）：").strip()
    return scores, comment


def score_images(
    paths: list[Path],
    preset: dict[str, list[float]] | None,
    evaluator: str,
) -> list[dict]:
    rows: list[dict] = []
    for img in paths:
        if preset is not None:
            scores = preset.get(img.name, [3.0] * len(DIMENSIONS))
            comment = "（预设分数）"
        else:
            scores, comment = interactive_score(img)
        total = weighted_total(scores)
        row = {
            "filename": img.name,
            "evaluator": evaluator,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        for (dim_key, _), s in zip(DIMENSIONS, scores):
            row[dim_key] = s
        row["weighted_total"] = total
        row["comment"] = comment
        rows.append(row)
    return rows


def build_report(rows: list[dict], evaluator: str, images_dir: Path) -> str:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    n = len(rows)
    avg_total = round(sum(r["weighted_total"] for r in rows) / n, 2) if n else 0

    dim_avgs = {}
    for dim_key, dim_name in DIMENSIONS:
        dim_avgs[dim_name] = round(sum(r[dim_key] for r in rows) / n, 2) if n else 0

    lines = [
        f"# 模拟专家主观评分报告",
        "",
        f"> **评审人**：{evaluator}  |  **时间**：{ts}  |  **图像目录**：`{images_dir}`",
        "",
        "## 总体摘要",
        "",
        f"| 指标 | 值 |",
        f"|------|-----|",
        f"| 评审张数 | {n} |",
        f"| 加权总分均值 | **{avg_total} / 5.0** |",
    ]
    for dim_name, avg in dim_avgs.items():
        lines.append(f"| {dim_name} 均分 | {avg} / {MAX_SCORE} |")

    lines += [
        "",
        "## 评分详情",
        "",
        "| 文件名 | " + " | ".join(n for _, n in DIMENSIONS) + " | 加权总分 | 备注 |",
        "|------|" + "|".join("---:" for _ in DIMENSIONS) + "|---:|:---|",
    ]
    for r in rows:
        cols = [r["filename"]]
        for dim_key, _ in DIMENSIONS:
            cols.append(str(r[dim_key]))
        cols.append(str(r["weighted_total"]))
        cols.append(r.get("comment", ""))
        lines.append("| " + " | ".join(cols) + " |")

    lines += [
        "",
        "## 评分说明",
        "",
        "评分维度参考 ACR MQSA 乳腺摄影质量标准与 Sickles 2013 评审规范：",
        "- **解剖轮廓合理性**：边缘轮廓、MLO 体位胸肌可见性；",
        "- **腺体纹理真实性**：纤维腺体密度梯度、小叶纹理细节；",
        "- **亮度与对比度**：背景纯黑、腺体灰阶范围合理；",
        "- **伪影/异常缺失**：无格线/条纹/环形/拼接可见痕；",
        "- **总体钼靶相似度**：整体上是否接近真实 FFDM 影像。",
        "",
        "> 本评分为模拟专家评审（非真实放射科医生），仅供毕业设计定性展示使用。",
    ]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="模拟专家主观评分（任务书第 5 条）")
    p.add_argument("--images-dir", type=Path, required=True, help="待评分图像目录")
    p.add_argument("--evaluator", type=str, default="模拟评审人", help="评审人姓名/编号")
    p.add_argument("--max-images", type=int, default=10, help="最多评审张数（默认 10）")
    p.add_argument(
        "--scores-csv",
        type=Path,
        default=None,
        help="预设分数 CSV（含列 filename,anatomy,texture,contrast,artifact,overall）；"
             "不提供则进入交互逐图评分模式",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Markdown 报告输出路径（默认 <images-dir>/expert_report.md）",
    )
    p.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="同时导出评分 CSV（默认 <images-dir>/expert_scores.csv）",
    )
    args = p.parse_args()

    img_dir = args.images_dir.resolve()
    if not img_dir.is_dir():
        print(f"[ERROR] 目录不存在: {img_dir}", file=sys.stderr)
        sys.exit(1)

    paths = sorted([*img_dir.glob("*.png"), *img_dir.glob("*.jpg")])[:args.max_images]
    if not paths:
        print(f"[ERROR] 未在 {img_dir} 找到图像", file=sys.stderr)
        sys.exit(1)

    preset = None
    if args.scores_csv is not None:
        preset = load_preset_csv(args.scores_csv.resolve())
        print(f"[INFO] 使用预设分数 CSV: {args.scores_csv}（{len(preset)} 条）")
    else:
        print(f"[INFO] 进入交互评分模式，共 {len(paths)} 张。")
        print(f"[INFO] 每张图依次输入 {len(DIMENSIONS)} 个维度的分数（0–{MAX_SCORE}）。")

    rows = score_images(paths, preset, args.evaluator)

    out_md = args.output or (img_dir / "expert_report.md")
    out_csv = args.csv_out or (img_dir / "expert_scores.csv")

    md = build_report(rows, args.evaluator, img_dir)
    out_md.write_text(md, encoding="utf-8")
    print(f"[OK] 专家评分报告 → {out_md}")

    csv_rows = [["filename"] + [k for k, _ in DIMENSIONS] + ["weighted_total", "evaluator", "comment"]]
    for r in rows:
        csv_rows.append(
            [r["filename"]] + [r[k] for k, _ in DIMENSIONS] + [r["weighted_total"], r["evaluator"], r.get("comment", "")]
        )
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows(csv_rows)
    print(f"[OK] 评分 CSV      → {out_csv}")

    # 终端摘要
    avg = round(sum(r["weighted_total"] for r in rows) / len(rows), 2)
    print(f"\n评审 {len(rows)} 张 · 加权总分均值: {avg} / 5.0")


if __name__ == "__main__":
    main()
