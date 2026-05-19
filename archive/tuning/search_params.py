#!/usr/bin/env python3
"""
生成参数网格搜索：对每组参数调用 run_mammo_sd15 + review_generated_images，
记录 tier1 比例、平均 semantic / quality，目标函数 tier1_ratio * avg_semantic_score。

使用前请根据本机环境修改：input_dir、lora_path、基线 JSON 等常量。
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SD15 生成参数网格搜索（慢，慎用）")
    p.add_argument("--input-dir", type=Path, default=ROOT / "datasets/jpeg")
    p.add_argument("--num-images", type=int, default=10, help="每组只生成少量图以提速")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--output-base", type=Path, default=ROOT / "outputs/generated/tuning_runs")
    p.add_argument("--eval-base", type=Path, default=ROOT / "outputs/eval/tuning")
    p.add_argument("--dry-run", action="store_true", help="只打印命令不执行")
    p.add_argument(
        "--review-extra",
        type=str,
        default="",
        help="传给 review 的额外参数片段，如 --modality-classifier-path ...",
    )
    return p.parse_args()


def run_cmd(cmd: list[str], dry: bool) -> int:
    print("[cmd]", " ".join(cmd))
    if dry:
        return 0
    r = subprocess.run(cmd, cwd=str(ROOT))
    return r.returncode


def summarize_review(summary_path: Path) -> dict:
    if not summary_path.is_file():
        return {}
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    per = data.get("per_image") or []
    n = len(per)
    if n == 0:
        return {"n": 0, "tier1_ratio": 0.0, "avg_semantic": 0.0, "avg_quality": 0.0}
    t1 = sum(1 for x in per if int(x.get("tier", 3)) == 1)
    sem = [float(x.get("semantic_score") or 0) for x in per]
    qual = [float(x.get("quality_score") or x.get("total_score") or 0) for x in per]
    import statistics

    return {
        "n": n,
        "tier1_ratio": t1 / n,
        "avg_semantic": float(statistics.mean(sem)) if sem else 0.0,
        "avg_quality": float(statistics.mean(qual)) if qual else 0.0,
    }


def main() -> None:
    args = parse_args()

    strengths = [0.25, 0.30, 0.35, 0.40, 0.45]
    overlap_ratios = [0.5, 0.625, 0.75]
    guide_strengths = [0.0, 0.15, 0.25, 0.35]

    rows_out: list[dict] = []
    args.eval_base.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    for s in strengths:
        for o in overlap_ratios:
            for gs in guide_strengths:
                tag = f"s{s:.2f}_o{o:.3f}_g{gs:.2f}".replace(".", "p")
                gen_sub = f"grid_{ts}_{tag}"
                eval_dir = args.eval_base / gen_sub

                gen_cmd = [
                    sys.executable,
                    str(ROOT / "scripts/generation/run_mammo_sd15.py"),
                    "--input-dir",
                    str(args.input_dir),
                    "--num-images",
                    str(args.num_images),
                    "--seed",
                    str(args.seed),
                    "--strength",
                    str(s),
                    "--overlap-ratio",
                    str(o),
                    "--blend-mode",
                    "gaussian",
                    "--output-base",
                    str(args.output_base),
                    "--output-subdir-prefix",
                    gen_sub,
                    "--no-postprocess",
                ]
                if gs > 0:
                    gen_cmd.extend(
                        [
                            "--global-guide",
                            "--global-guide-strength",
                            str(gs),
                        ]
                    )
                else:
                    gen_cmd.append("--no-global-guide")

                if run_cmd(gen_cmd, args.dry_run) != 0:
                    print("[warn] 生成失败，跳过该组")
                    continue

                if args.dry_run:
                    continue

                # run_mammo_sd15 会再拼一层时间戳目录：{prefix}_{ts}_000
                cands = sorted(
                    [p for p in args.output_base.iterdir() if p.is_dir() and gen_sub in p.name],
                    key=lambda p: p.stat().st_mtime,
                )
                if not cands:
                    print("[warn] 未找到生成目录，跳过")
                    continue
                out_dir = cands[-1]

                rev_cmd = [
                    sys.executable,
                    str(ROOT / "scripts/evaluation/review_generated_images.py"),
                    "--images-dir",
                    str(out_dir),
                    "--output-dir",
                    str(eval_dir),
                    "--no-recursive",
                ]
                if args.review_extra.strip():
                    rev_cmd.extend(args.review_extra.split())

                if run_cmd(rev_cmd, False) != 0:
                    print("[warn] 评审失败，跳过该组")
                    continue

                stats = summarize_review(eval_dir / "summary.json")
                obj = stats["tier1_ratio"] * stats["avg_semantic"]
                rows_out.append(
                    {
                        "strength": s,
                        "overlap_ratio": o,
                        "guide_strength": gs,
                        **stats,
                        "objective": obj,
                        "gen_dir": str(out_dir),
                        "eval_dir": str(eval_dir),
                    }
                )

    if not rows_out:
        print("无有效结果（检查 dry-run 或路径）")
        return

    rows_out.sort(key=lambda r: r["objective"], reverse=True)
    csv_path = args.eval_base / f"grid_search_{ts}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)

    best = rows_out[0]
    print("\n[Best] objective=tier1_ratio×avg_semantic:", best)
    print(f"[OK] 表格: {csv_path}")


if __name__ == "__main__":
    main()
