#!/usr/bin/env python3
"""
多次 review 结果对比器。

用途：
- 把多个 review_output_*/summary.json 合并成一张对比表；
- 输出 compare_table.csv / compare_table.md，并在终端打印 leaderboard；
- 目的是：调参后只看数字，不用再肉眼对图。

约定：
- 每次跑 review_generated_images.py 前，把 --output-dir 命名带上参数特征，
  例如 review_output_s018_cfg4_step30；
- 本脚本会把目录名作为 run_name 放入第一列。

典型用法：
    # 自动扫描当前目录下所有 review_output_*
    python compare_runs.py

    # 或显式指定几个目录
    python compare_runs.py --runs review_output_s018_cfg4 review_output_s035_cfg7

    # 只关心 D 组伪影，按伪影组分排序
    python compare_runs.py --sort-by group_D --sort-desc
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# 与 review_generated_images.py 新框架里硬性 tag 保持一致
HARD_TAGS = [
    "AREA_BAD",
    "SHAPE_ODD",
    "BLOWOUT",
    "DEAD_DARK",
    "ARTIFACT_BUBBLES",
    "EDGE_VOIDS",
    "BANDING",
    "NO_PECTORAL",
    "LOW_CONTRAST",
    "OVEREXPOSED",
    "LOW_DR",
    "OVER_STRETCHED",
    "TEXTURE_UNNATURAL",
    "TOO_UNIFORM",
    "TOO_NOISY",
    "HIST_OFF_REAL",
    "DENSITY_MISMATCH",
    "AREA_BORDER",
    "read_failed",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="合并多份 review summary.json 生成参数对比表")
    p.add_argument(
        "--runs-dir",
        type=Path,
        default=ROOT / "outputs/reviews",
        help=(
            "扫描哪个目录下的 review_output_* 子目录（默认：outputs/reviews）。"
            "历史 SDXL 阶段报告在 outputs/_legacy/sdxl/reviews/。"
        ),
    )
    p.add_argument(
        "--runs",
        type=Path,
        nargs="*",
        default=None,
        help="显式指定一组 review_output_* 目录；给了就不再自动扫描。",
    )
    p.add_argument(
        "--pattern",
        type=str,
        default="review_output_*",
        help="自动扫描时使用的 glob 模式（默认 review_output_*）",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/comparisons/compare_runs_output",
        help="输出目录",
    )
    p.add_argument(
        "--sort-by",
        type=str,
        default="mean_total_score",
        help="排序字段：mean_total_score / pass_rate / group_A / group_B / group_C / group_D / group_E / run_name",
    )
    p.add_argument(
        "--sort-desc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="默认降序。用 --no-sort-desc 切换升序。",
    )
    p.add_argument(
        "--top-violations",
        type=int,
        default=6,
        help="Markdown 表里包含的违规 tag 列数（取所有 run 合集中最常见的前 K 个）",
    )
    return p.parse_args()


def discover_runs(runs_dir: Path, pattern: str) -> list[Path]:
    dirs: list[Path] = []
    for p in sorted(runs_dir.glob(pattern)):
        if not p.is_dir():
            continue
        if ".ipynb_checkpoints" in p.parts:
            continue
        if (p / "summary.json").exists():
            dirs.append(p)
    return dirs


def load_summary(run_dir: Path) -> dict | None:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return None
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[跳过] {summary_path} 读取失败: {e}")
        return None

    row: dict = {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "total_images": int(data.get("total_images", 0)),
        "pass_count": int(data.get("pass_count", 0)),
        "pass_rate": float(data.get("pass_rate", 0.0)),
        "mean_total_score": float(data.get("mean_total_score", 0.0)),
        "has_real_baseline": bool(data.get("has_real_baseline", False)),
    }
    group = data.get("group_mean_scores", {}) or {}
    for g in "ABCDEF":
        row[f"group_{g}"] = float(group.get(g, 0.0))
    row["_violations"] = {k: float(v) for k, v in (data.get("violation_rates", {}) or {}).items()}
    row["_thresholds"] = data.get("thresholds", {}) or {}
    row["_weights"] = data.get("weights", {}) or {}
    return row


def pick_top_violation_tags(rows: list[dict], k: int) -> list[str]:
    acc: dict[str, float] = {}
    for r in rows:
        for tag, rate in r["_violations"].items():
            acc[tag] = acc.get(tag, 0.0) + rate
    known_order = {t: i for i, t in enumerate(HARD_TAGS)}
    items = sorted(
        acc.items(),
        key=lambda kv: (-kv[1], known_order.get(kv[0], 999), kv[0]),
    )
    return [t for t, _ in items[: max(1, k)]]


def to_csv(rows: list[dict], violation_tags: list[str], out_path: Path) -> None:
    base_fields = [
        "run_name",
        "run_dir",
        "total_images",
        "pass_count",
        "pass_rate",
        "mean_total_score",
        "group_A",
        "group_B",
        "group_C",
        "group_D",
        "group_E",
        "group_F",
        "has_real_baseline",
    ]
    fields = base_fields + [f"rate_{t}" for t in violation_tags]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            out = {k: r.get(k, "") for k in base_fields}
            for t in violation_tags:
                out[f"rate_{t}"] = round(float(r["_violations"].get(t, 0.0)), 4)
            w.writerow(out)


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def to_markdown(rows: list[dict], violation_tags: list[str], out_path: Path) -> None:
    headers = [
        "run_name",
        "N",
        "pass_rate",
        "mean_score",
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
    ] + violation_tags

    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        row_cells = [
            r["run_name"],
            str(r["total_images"]),
            _fmt_pct(r["pass_rate"]),
            f"{r['mean_total_score']:.1f}",
            f"{r['group_A']:.2f}",
            f"{r['group_B']:.2f}",
            f"{r['group_C']:.2f}",
            f"{r['group_D']:.2f}",
            f"{r['group_E']:.2f}",
            f"{r.get('group_F', 0.0):.2f}",
        ]
        for t in violation_tags:
            row_cells.append(_fmt_pct(float(r["_violations"].get(t, 0.0))))
        lines.append("| " + " | ".join(row_cells) + " |")

    header_note = [
        "# Review Runs Comparison",
        "",
        "- N = 该批次图像总数",
        "- pass_rate = 通过率（总分≥50 且无硬性 tag）",
        "- mean_score = 批次平均总分 (0-100)",
        "- A/B/C/D/E/F = 构图/灰阶/纹理/伪影/分布/无参考质量 维度分组均分 (0-1)",
        "- 其余列 = 违规标签出现率",
        "",
    ]
    out_path.write_text("\n".join(header_note + lines) + "\n", encoding="utf-8")


def print_leaderboard(rows: list[dict], violation_tags: list[str]) -> None:
    if not rows:
        print("（没有可用的 summary.json）")
        return
    name_w = max(10, max(len(r["run_name"]) for r in rows))
    header = f"{'run_name':<{name_w}}  N     pass%   score  A     B     C     D     E     F"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['run_name']:<{name_w}}  "
            f"{r['total_images']:<4}  "
            f"{r['pass_rate']*100:5.1f}%  "
            f"{r['mean_total_score']:5.1f}  "
            f"{r['group_A']:.2f}  {r['group_B']:.2f}  "
            f"{r['group_C']:.2f}  {r['group_D']:.2f}  {r['group_E']:.2f}  {r.get('group_F', 0.0):.2f}"
        )

    if violation_tags:
        print()
        top = violation_tags[: min(5, len(violation_tags))]
        sub_header = f"{'run_name':<{name_w}}  " + "  ".join(f"{t[:14]:>14}" for t in top)
        print(sub_header)
        print("-" * len(sub_header))
        for r in rows:
            cells = [f"{r['run_name']:<{name_w}}"]
            for t in top:
                cells.append(f"{r['_violations'].get(t, 0.0)*100:13.1f}%")
            print("  ".join(cells))


def sort_rows(rows: list[dict], key: str, desc: bool) -> list[dict]:
    def _get(r: dict):
        v = r.get(key)
        if v is None:
            return (1, 0.0)  # 缺字段的排最后
        try:
            return (0, float(v)) if not isinstance(v, str) else (0, v)
        except Exception:
            return (0, v)

    return sorted(rows, key=_get, reverse=desc)


def main() -> None:
    args = parse_args()
    if args.runs:
        run_dirs = [p.resolve() for p in args.runs if p.exists()]
    else:
        run_dirs = discover_runs(args.runs_dir.resolve(), args.pattern)

    if not run_dirs:
        raise SystemExit(
            f"未找到任何含 summary.json 的 review_output 目录。"
            f"请检查 --runs-dir={args.runs_dir} 或显式传入 --runs。"
        )

    rows: list[dict] = []
    for d in run_dirs:
        row = load_summary(d)
        if row is not None:
            rows.append(row)

    if not rows:
        raise SystemExit("所有目录都缺少 summary.json 或解析失败。")

    violation_tags = pick_top_violation_tags(rows, args.top_violations)
    rows_sorted = sort_rows(rows, args.sort_by, args.sort_desc)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "compare_table.csv"
    md_path = args.output_dir / "compare_table.md"
    to_csv(rows_sorted, violation_tags, csv_path)
    to_markdown(rows_sorted, violation_tags, md_path)

    print_leaderboard(rows_sorted, violation_tags)
    print()
    print(f"CSV:      {csv_path}")
    print(f"Markdown: {md_path}")
    print(f"对比 run 数: {len(rows_sorted)}")


if __name__ == "__main__":
    main()
