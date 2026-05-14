#!/usr/bin/env python3
"""
端到端测试编排：评审管线 GT 指标 +（可选）两批次对比 + Top5 seam_score + test_report.md

示例：

  # 仅验证 test_set_60（先自行放好图并按 good_/bad_ 命名）
  python3 scripts/tests/end_to_end_test.py \\
      --test-images-dir test_set_60 \\
      --run-review \\
      --review-output-dir outputs/eval/test_validation

  # 已有 review_report.csv
  python3 scripts/tests/end_to_end_test.py \\
      --review-csv outputs/eval/test_validation/review_report.csv

  # 加上两批次评审目录对比（各自目录下须有 review_report.csv）
  python3 scripts/tests/end_to_end_test.py \\
      --review-csv outputs/eval/test_validation/review_report.csv \\
      --baseline-review-dir outputs/eval/baseline_review \\
      --new-review-dir outputs/eval/new_full_review \\
      --baseline-images-dir outputs/generated/xxx/baseline_000 \\
      --new-images-dir outputs/generated/yyy/new_full_000
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT / "scripts/tests"))
from review_metrics import (  # noqa: E402
    compute_validation_metrics,
    format_report_md,
    rows_from_csv,
    rows_from_summary,
)
from seam_metrics import mean_seam_score_top_k  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="评审/生成端到端验证报告")
    p.add_argument("--test-images-dir", type=Path, default=None, help="含 good_/bad_ 命名图的目录")
    p.add_argument("--run-review", action="store_true", help="调用 review_generated_images.py")
    p.add_argument(
        "--review-output-dir",
        type=Path,
        default=ROOT / "outputs/eval/test_validation",
        help="--run-review 时 --output-dir",
    )
    p.add_argument(
        "--review-extra-args",
        type=str,
        default="",
        help="附加传给评审脚本的参数（引号包裹），如 '--real-baseline-json path.json'",
    )
    p.add_argument("--review-csv", type=Path, default=None, help="直接指定 review_report.csv")
    p.add_argument("--review-summary", type=Path, default=None, help="或用 summary.json（per_image）")

    p.add_argument("--baseline-review-dir", type=Path, default=None, help="含 review_report.csv")
    p.add_argument("--new-review-dir", type=Path, default=None)
    p.add_argument(
        "--baseline-images-dir",
        type=Path,
        default=None,
        help="与 baseline 评审对应的生成图目录（取 Top5 算 seam）",
    )
    p.add_argument("--new-images-dir", type=Path, default=None)

    p.add_argument("--out-report", type=Path, default=ROOT / "outputs/eval/test_report.md")
    p.add_argument("--top-k-seam", type=int, default=5)
    return p.parse_args()


def run_review_pipeline(
    images_dir: Path,
    output_dir: Path,
    extra: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "scripts/evaluation/review_generated_images.py"),
        "--images-dir",
        str(images_dir),
        "--output-dir",
        str(output_dir),
        "--no-recursive",
    ]
    if extra.strip():
        cmd.extend(extra.split())
    print("[e2e] 运行:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def load_rows(csv_path: Path | None, summary_path: Path | None) -> list[dict]:
    if csv_path and csv_path.is_file():
        return rows_from_csv(csv_path)
    if summary_path and summary_path.is_file():
        return rows_from_summary(summary_path)
    raise FileNotFoundError("需要有效的 --review-csv 或 --review-summary")


def top_n_image_paths_from_csv(
    review_dir: Path,
    images_root: Path | None,
    n: int,
) -> list[Path]:
    csv_path = review_dir / "review_report.csv"
    rows = rows_from_csv(csv_path)
    rows.sort(
        key=lambda r: float(r.get("final_rank_score") or 0.0),
        reverse=True,
    )
    out: list[Path] = []
    for r in rows[:n]:
        img = r.get("image") or ""
        p = Path(img)
        if p.is_file():
            out.append(p)
            continue
        if images_root and (images_root / p.name).is_file():
            out.append(images_root / p.name)
            continue
        if images_root:
            hits = list(images_root.glob(p.name))
            if hits:
                out.append(hits[0])
    return out


def compare_two_runs(
    name_a: str,
    dir_a: Path,
    img_a: Path | None,
    name_b: str,
    dir_b: Path,
    img_b: Path | None,
    top_k: int,
) -> str:
    lines = [
        f"## 批次对比：{name_a} vs {name_b}",
        "",
    ]
    for label, d, img_root in (
        (name_a, dir_a, img_a),
        (name_b, dir_b, img_b),
    ):
        summ = d / "summary.json"
        csv_p = d / "review_report.csv"
        if summ.is_file():
            data = json.loads(summ.read_text(encoding="utf-8"))
            th = data.get("tier_histogram") or {}
            lines.append(f"### {label}")
            lines.append("")
            lines.append(f"- tier 分布: `{th}`")
            lines.append(f"- pass_rate: {data.get('pass_rate')}")
            lines.append(f"- mean_total_score: {data.get('mean_total_score')}")
            lines.append("")
        if csv_p.is_file() and img_root and img_root.is_dir():
            paths = top_n_image_paths_from_csv(d, img_root, top_k)
            mean_s, detail = mean_seam_score_top_k(paths, k=top_k)
            lines.append(f"- Top{top_k} 平均 seam_score: **{mean_s:.4f}**")
            if detail:
                lines.append("- 明细:")
                for nm, sc in detail:
                    lines.append(f"  - {nm}: {sc:.4f}")
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    if args.run_review:
        if not args.test_images_dir or not args.test_images_dir.is_dir():
            raise SystemExit("--run-review 需要有效的 --test-images-dir")
        run_review_pipeline(args.test_images_dir, args.review_output_dir, args.review_extra_args)
        csv_path = args.review_output_dir / "review_report.csv"
    else:
        csv_path = args.review_csv
        if csv_path is None and args.review_output_dir:
            cand = args.review_output_dir / "review_report.csv"
            if cand.is_file():
                csv_path = cand
        if csv_path is None and args.review_summary is None:
            raise SystemExit("请指定 --review-csv、--review-summary，或使用 --run-review")

    rows = load_rows(csv_path, args.review_summary)
    res = compute_validation_metrics(rows)
    md = format_report_md(res, title="评审管线验证（Ground Truth）")

    extra_sections: list[str] = []

    # semantic 分布提示
    good_sem = [
        float(r.get("semantic_score") or 0)
        for r in rows
        if Path(str(r.get("image", ""))).name.lower().startswith("good_")
    ]
    bad_sem = [
        float(r.get("semantic_score") or 0)
        for r in rows
        if Path(str(r.get("image", ""))).name.lower().startswith("bad_")
    ]
    if good_sem and bad_sem:
        import statistics

        extra_sections.append("## Tier / Semantic 分布提示")
        extra_sections.append("")
        extra_sections.append(
            f"- good semantic_score: median={statistics.median(good_sem):.3f} "
            f"(n={len(good_sem)})"
        )
        extra_sections.append(
            f"- bad semantic_score: median={statistics.median(bad_sem):.3f} "
            f"(n={len(bad_sem)})"
        )
        extra_sections.append("")

    if args.baseline_review_dir and args.new_review_dir:
        extra_sections.append(
            compare_two_runs(
                "baseline",
                args.baseline_review_dir,
                args.baseline_images_dir,
                "new_full",
                args.new_review_dir,
                args.new_images_dir,
                args.top_k_seam,
            )
        )

    # 验收判定
    ok_recall = res.veto_recall >= 0.90
    ok_fk = res.good_false_kill_rate <= 0.10
    ok_top5 = res.top5_purity
    extra_sections.append("## 验收判定（相对目标）")
    extra_sections.append("")
    extra_sections.append(f"| Veto 召回 ≥90% | {'✓' if ok_recall else '✗'} ({res.veto_recall:.2%}) |")
    extra_sections.append(f"| 正常误杀 ≤10% | {'✓' if ok_fk else '✗'} ({res.good_false_kill_rate:.2%}) |")
    extra_sections.append(f"| Top5 纯度 100% | {'✓' if ok_top5 else '✗'} |")
    extra_sections.append("")

    full = md + "\n" + "\n".join(extra_sections)
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(full, encoding="utf-8")
    print(full)
    print(f"\n[OK] 报告已写入: {args.out_report}")


if __name__ == "__main__":
    main()
