#!/usr/bin/env python3
"""
消融实验对比表生成器（任务书第 5 条）。

从多个 eval 目录的 summary.json 中提取关键指标，合并为 Markdown / CSV 消融表。
支持为每个实验组手动注标「消融因子」说明（--labels），或自动从目录名推断。

用法：
  # 自动扫描 outputs/eval/，生成 outputs/reports/ablation_table.md
  python3 scripts/evaluation/make_ablation_table.py

  # 指定目录并分别标注实验说明
  python3 scripts/evaluation/make_ablation_table.py \\
    --runs outputs/eval/baseline outputs/eval/no_lora outputs/eval/sdxl_compare \\
    --labels "基线(SD1.5+LoRA)" "消融:无LoRA" "对比:SDXL" \\
    --output outputs/reports/ablation_table.md

  # 按 pass_rate 降序，只取前 10 组
  python3 scripts/evaluation/make_ablation_table.py --sort-by pass_rate --top 10
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

METRIC_COLS = [
    ("pass_rate",        "通过率",        lambda s: f"{float(s.get('pass_rate',0))*100:.1f}%"),
    ("mean_total_score", "均分",          lambda s: f"{float(s.get('mean_total_score',0)):.2f}"),
    ("fid",              "FID↓",          lambda s: _fmt_academic(s, "fid")),
    ("mean_brisque",     "BRISQUE↓",      lambda s: _fmt_academic(s, "mean_brisque")),
    ("mean_ps_slope",    "PS-β",          lambda s: _fmt_academic(s, "mean_ps_slope_beta")),
    ("group_A",          "A-构图",        lambda s: f"{float((s.get('group_mean_scores') or {}).get('A', 0)):.3f}"),
    ("group_B",          "B-灰阶",        lambda s: f"{float((s.get('group_mean_scores') or {}).get('B', 0)):.3f}"),
    ("group_C",          "C-纹理",        lambda s: f"{float((s.get('group_mean_scores') or {}).get('C', 0)):.3f}"),
    ("group_D",          "D-伪影",        lambda s: f"{float((s.get('group_mean_scores') or {}).get('D', 0)):.3f}"),
    ("group_E",          "E-分布",        lambda s: f"{float((s.get('group_mean_scores') or {}).get('E', 0)):.3f}"),
    ("total_images",     "总张数",        lambda s: str(s.get("total_images", 0))),
    ("top_violation",    "主要违规",       _top_violation),
]

SORT_MAP = {
    "pass_rate":   lambda s: float(s.get("pass_rate", 0)),
    "mean_total_score": lambda s: float(s.get("mean_total_score", 0)),
    "fid":         lambda s: float((s.get("academic_metrics") or {}).get("fid") or 9999),
    "brisque":     lambda s: float((s.get("academic_metrics") or {}).get("mean_brisque") or 9999),
}


def _fmt_academic(s: dict, key: str) -> str:
    v = (s.get("academic_metrics") or {}).get(key)
    if v is None:
        return "—"
    return f"{float(v):.2f}"


def _top_violation(s: dict) -> str:
    vr = s.get("violation_rates") or {}
    if not vr:
        return "—"
    top = sorted(vr.items(), key=lambda x: -float(x[1]))[:2]
    return " / ".join(f"{k}({float(v)*100:.0f}%)" for k, v in top)


def load_summary(path: Path) -> dict:
    p = path if path.name == "summary.json" else path / "summary.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def collect_runs(
    paths: list[Path],
    labels: list[str] | None,
) -> list[tuple[str, dict]]:
    rows: list[tuple[str, dict]] = []
    for i, p in enumerate(paths):
        s = load_summary(p)
        if not s:
            print(f"[WARN] 跳过（无 summary.json）: {p}")
            continue
        if labels and i < len(labels):
            name = labels[i]
        else:
            name = p.name
        rows.append((name, s))
    return rows


def auto_scan(base: Path, limit: int) -> list[Path]:
    dirs = sorted(
        (d for d in base.iterdir() if d.is_dir() and (d / "summary.json").is_file()),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    return dirs[:limit]


def build_md_table(rows: list[tuple[str, dict]]) -> str:
    header_keys = ["实验组"] + [col[1] for col in METRIC_COLS]
    sep = [":---"] + ["---:"] * len(METRIC_COLS)
    lines = ["| " + " | ".join(header_keys) + " |",
             "| " + " | ".join(sep) + " |"]
    for name, s in rows:
        vals = [name] + [col[2](s) for col in METRIC_COLS]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def build_csv_rows(rows: list[tuple[str, dict]]) -> list[list[str]]:
    header = ["实验组"] + [col[1] for col in METRIC_COLS]
    data = []
    for name, s in rows:
        data.append([name] + [col[2](s) for col in METRIC_COLS])
    return [header] + data


def main() -> None:
    p = argparse.ArgumentParser(description="生成消融实验对比表（任务书第 5 条）")
    p.add_argument(
        "--runs",
        nargs="*",
        type=Path,
        default=None,
        help="eval 目录列表（含 summary.json）；不指定则自动扫描 outputs/eval/",
    )
    p.add_argument("--labels", nargs="*", default=None,
                   help="与 --runs 对应的实验组名称，顺序一致")
    p.add_argument("--runs-dir", type=Path, default=ROOT / "outputs/eval",
                   help="自动扫描的根目录（默认 outputs/eval/）")
    p.add_argument("--top", type=int, default=30,
                   help="自动扫描时最多取最近 N 个目录（默认 30）")
    p.add_argument("--sort-by",
                   choices=["pass_rate", "mean_total_score", "fid", "brisque", "name"],
                   default="pass_rate",
                   help="排序字段（默认 pass_rate 降序）")
    p.add_argument("--sort-asc", action="store_true",
                   help="升序排列（默认降序；fid/brisque 用时建议加此项）")
    p.add_argument("--output", type=Path,
                   default=ROOT / "outputs/reports/ablation_table.md",
                   help="Markdown 输出路径")
    p.add_argument("--csv", type=Path, default=None,
                   help="同时导出 CSV（不指定则不导出）")
    p.add_argument("--title", type=str, default="消融实验对比表",
                   help="Markdown 文件标题")
    args = p.parse_args()

    # 收集目录
    if args.runs:
        paths = [p.resolve() for p in args.runs]
    else:
        paths = auto_scan(args.runs_dir, args.top)
    if not paths:
        print(f"[ERROR] 未找到任何含 summary.json 的目录（扫描路径: {args.runs_dir}）")
        return

    rows = collect_runs(paths, args.labels)
    if not rows:
        print("[ERROR] 所有指定路径均无有效 summary.json。")
        return

    # 排序
    if args.sort_by == "name":
        rows.sort(key=lambda x: x[0], reverse=not args.sort_asc)
    else:
        key_fn = SORT_MAP.get(args.sort_by, lambda s: float(s.get("pass_rate", 0)))
        rows.sort(key=lambda x: key_fn(x[1]), reverse=not args.sort_asc)

    # 输出 Markdown
    args.output.parent.mkdir(parents=True, exist_ok=True)
    md = f"# {args.title}\n\n"
    md += f"> 共 {len(rows)} 组 · 排序字段: `{args.sort_by}` · "
    md += ("升序" if args.sort_asc else "降序") + "\n\n"
    md += "> 指标说明：通过率=自动评审通过（满足全部硬规则）；FID↓ 越低越好（<30 良好）；"
    md += "BRISQUE↓ 越低越好（<25 优秀）；PS-β 真实钼靶典型值 2.0～3.5。\n\n"
    md += build_md_table(rows)
    md += "\n"
    args.output.write_text(md, encoding="utf-8")
    print(f"[OK] Markdown 消融表 → {args.output}")

    # 可选 CSV
    csv_path = args.csv or args.output.with_suffix(".csv")
    csv_rows = build_csv_rows(rows)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows(csv_rows)
    print(f"[OK] CSV 消融表     → {csv_path}")

    # 终端预览（前 5 行）
    print("\n─── 预览（前 5 组）───")
    for name, s in rows[:5]:
        pr = f"{float(s.get('pass_rate',0))*100:.1f}%"
        sc = f"{float(s.get('mean_total_score',0)):.1f}"
        fid = _fmt_academic(s, "fid")
        br  = _fmt_academic(s, "mean_brisque")
        print(f"  {name:<40} 通过率={pr}  均分={sc}  FID={fid}  BRISQUE={br}")


if __name__ == "__main__":
    main()
