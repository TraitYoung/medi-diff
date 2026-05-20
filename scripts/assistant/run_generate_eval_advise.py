#!/usr/bin/env python3
"""Generate → Evaluate → Advise 全自动流水线编排器。

示例:
    python3 scripts/assistant/run_generate_eval_advise.py \
        --metadata-csv datasets/CBIS_CLEAN_V2/metadata_clean.csv \
        --filter-view MLO --filter-density dense \
        --num-images 6 --tag-prefix my_run

    python3 scripts/assistant/run_generate_eval_advise.py \
        --metadata-csv datasets/CBIS_CLEAN_V2/metadata_clean.csv \
        --filter-view MLO --filter-density dense \
        --num-images 6 --tag-prefix auto \
        --mode full-image --fullimage-output-long-side 2048 \
        --enable-seam-check
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

GEN_SCRIPT = ROOT / "scripts/generation/run_mammo_sd15.py"
EVAL_SCRIPT = ROOT / "scripts/evaluation/review_generated_images.py"
ADVISOR_SCRIPT = ROOT / "scripts/assistant/ask_advisor.py"

OUTPUT_BASE = ROOT / "outputs"
GEN_BASE = OUTPUT_BASE / "generated"
EVAL_BASE = OUTPUT_BASE / "eval"
REPORT_BASE = OUTPUT_BASE / "reports"


def _run(cmd: list[str], label: str = "") -> subprocess.CompletedProcess:
    print(f"[{label}] $ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=str(ROOT), text=True,
                          capture_output=False, check=False)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate → Evaluate → Advise")
    # Source
    p.add_argument("--metadata-csv", type=str,
                   default="datasets/CBIS_CLEAN_V2/metadata_clean.csv")
    p.add_argument("--filter-view", type=str, default="MLO")
    p.add_argument("--filter-density", type=str, default="dense")
    p.add_argument("--num-images", type=int, default=6)
    p.add_argument("--tag-prefix", type=str, default="auto")

    # Model
    p.add_argument("--base-model-local", type=str, default="hf_cache/sd15")
    p.add_argument("--lora-path", type=str,
                   default="outputs/lora/mammo_sd15_v6_allMLO/final_lora")

    # Generation
    p.add_argument("--mode", type=str, default="full-image",
                   choices=["full-image"])
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--source-seed", type=int, default=None)
    p.add_argument("--strength", type=float, default=0.44)
    p.add_argument("--guidance-scale", type=float, default=7.5)
    p.add_argument("--num-steps", type=int, default=40)
    # --overlap-ratio removed (full-image only)
    p.add_argument("--scheduler", type=str, default="dpm")
    p.add_argument("--fullimage-long-side", type=int, default=768)
    p.add_argument("--fullimage-output-long-side", type=int, default=2048)
    # Postprocess flags archived — see archive/postprocess/
    p.add_argument("--source-quality-sort", action="store_true", default=False)
    p.add_argument("--no-source-quality-sort", action="store_false",
                   dest="source_quality_sort")

    # Evaluation
    p.add_argument("--enable-seam-check", action="store_true", default=False)
    p.add_argument("--no-auto-calibrate", action="store_true", default=False)
    p.add_argument("--real-images-dir", type=str, default="")
    p.add_argument("--output-base", type=str, default=str(GEN_BASE),
                   help="Base directory for generated image batches")

    # Skip stages
    p.add_argument("--skip-generate", action="store_true", default=False)
    p.add_argument("--skip-eval", action="store_true", default=False)
    p.add_argument("--skip-advise", action="store_true", default=False)

    args = p.parse_args()

    tag_prefix = args.tag_prefix or "auto"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    gen_dir = None
    gen_base = Path(args.output_base)
    if not gen_base.is_absolute():
        gen_base = ROOT / gen_base

    # ── Stage 1: Generate ──────────────────────────────────────────────
    if not args.skip_generate:
        gen_cmd = [
            sys.executable, str(GEN_SCRIPT),
            "--base-model-local", str(args.base_model_local),
            "--lora-path", str(args.lora_path),
            "--metadata-csv", str(args.metadata_csv),
            "--filter-view", args.filter_view,
            "--filter-density", args.filter_density,
            "--num-images", str(args.num_images),
            "--seed", str(args.seed),
            "--num-steps", str(args.num_steps),
            "--strength", str(args.strength),
            "--guidance-scale", str(args.guidance_scale),
            "--scheduler", args.scheduler,
            "--mode", args.mode,
            "--fullimage-long-side", str(args.fullimage_long_side),
            "--fullimage-output-long-side", str(args.fullimage_output_long_side),
            "--output-base", str(gen_base),
            "--output-subdir-prefix", tag_prefix,
        ]
        if args.source_seed is not None:
            gen_cmd.extend(["--source-seed", str(args.source_seed)])
        # Postprocess and bg-clean flags archived

        t0 = time.time()
        r = _run(gen_cmd, "GENERATE")
        if r.returncode != 0:
            print(f"Generation failed with code {r.returncode}")
            sys.exit(1)
        print(f"Generation done in {time.time() - t0:.0f}s")

        # Find output dir
        candidates = sorted(gen_base.glob(f"{tag_prefix}_*"), key=lambda p: p.stat().st_mtime)
        if candidates:
            gen_dir = candidates[-1]
            print(f"Generated dir: {gen_dir}")

    # ── Stage 2: Evaluate ─────────────────────────────────────────────
    if not args.skip_eval and gen_dir and gen_dir.is_dir():
        eval_name = f"{tag_prefix}_{ts}"
        eval_dir = EVAL_BASE / eval_name
        eval_cmd = [
            sys.executable, str(EVAL_SCRIPT),
            "--images-dir", str(gen_dir),
            "--output-dir", str(eval_dir),
            "--no-recursive",
        ]
        if args.enable_seam_check:
            eval_cmd.append("--enable-seam-check")
        if args.no_auto_calibrate:
            eval_cmd.append("--no-auto-calibrate")
        if args.real_images_dir:
            eval_cmd.extend(["--real-images-dir", str(args.real_images_dir)])

        r = _run(eval_cmd, "EVALUATE")
        if r.returncode != 0:
            print(f"Evaluation failed with code {r.returncode}")
            sys.exit(1)

        # Print summary
        summary_path = eval_dir / "summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            print(f"  pass_rate: {summary.get('pass_rate')}")
            print(f"  mean_total_score: {summary.get('mean_total_score')}")
            print(f"  strict_pass_rate: {summary.get('strict_pass_rate')}")

            report_dir = REPORT_BASE / eval_name
            report_dir.mkdir(parents=True, exist_ok=True)
            report_md = report_dir / "FINAL_REPORT.md"
            report_md.write_text(
                f"# 全自动流水线报告 — {eval_name}\n\n"
                f"- 生成目录: {gen_dir}\n"
                f"- 评估目录: {eval_dir}\n"
                f"- 通过率: {summary.get('pass_rate')}\n"
                f"- 严格通过率: {summary.get('strict_pass_rate')}\n"
                f"- 平均总分: {summary.get('mean_total_score')}\n\n"
                f"评估模式: {'auto-calibrated' if summary.get('auto_calibrated') else 'strict'}\n",
                encoding="utf-8")
            print(f"Report written: {report_md}")

            # ── Stage 3: Advise ──────────────────────────────────────
            if not args.skip_advise:
                # Build advisor prompt from evaluation summary
                prompt_lines = [
                    "你是乳腺钼靶图像质量评估专家。以下是生成图像的评估结果，请给出调参建议。",
                    "",
                    f"生成模式: {args.mode}",
                    f"评估名称: {eval_name}",
                    f"通过率 (calibrated): {summary.get('pass_rate')}",
                    f"严格通过率 (strict): {summary.get('strict_pass_rate')}",
                    f"平均总分: {summary.get('mean_total_score')}",
                    f"BRISQUE: {summary.get('academic_metrics', {}).get('mean_brisque', 'N/A')}",
                ]
                violations = summary.get("violation_rates", {})
                if violations:
                    prompt_lines.append("违规项:")
                    for k, v in sorted(violations.items(), key=lambda x: -x[1]):
                        prompt_lines.append(f"  - {k}: {v*100:.1f}%")
                prompt_lines.append("")
                if args.mode == 'full-image':
                    prompt_lines.append("请给出具体的参数调整建议（strength, guidance_scale, num_steps 等），")
                else:
                    prompt_lines.append("请给出具体的参数调整建议（strength, guidance_scale, num_steps 等），")
                prompt_lines.append("以 JSON 格式输出 parameters 字段，并附中文解释。")

                advisor_prompt = "\n".join(prompt_lines)

                advisor_cmd = [
                    sys.executable, str(ADVISOR_SCRIPT),
                    advisor_prompt,
                    "--system", "你是乳腺钼靶图像质量评估和参数调优专家，回答用中文。",
                ]

                r = _run(advisor_cmd, "ADVISE")
                # Advisor output goes to stdout; capture would require changes

                # Build report
                report_md = report_dir / "FINAL_REPORT.md"
                report_md.write_text(
                    f"# 全自动流水线报告 — {eval_name}\n\n"
                    f"- 生成目录: {gen_dir}\n"
                    f"- 评估目录: {eval_dir}\n"
                    f"- 通过率: {summary.get('pass_rate')}\n"
                    f"- 严格通过率: {summary.get('strict_pass_rate')}\n"
                    f"- 平均总分: {summary.get('mean_total_score')}\n\n"
                    f"评估模式: {'auto-calibrated' if summary.get('auto_calibrated') else 'strict'}\n",
                    encoding="utf-8")

                # Sync tuning state
                from scripts.assistant.tuning_state import sync_latest_tuning_state
                sync_latest_tuning_state(
                    eval_dir=eval_dir, report_base=REPORT_BASE,
                    source_tag=tag_prefix,
                    source_seed=args.source_seed,
                    source_quality_sort=args.source_quality_sort,
                    gen_dir=gen_dir,
                )

    print("\nDone.")


if __name__ == "__main__":
    main()
