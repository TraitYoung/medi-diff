#!/usr/bin/env python3
"""一键全流程报告：调用 run_generate_eval_advise.py 并生成完整报告。

作为 CLI 便利入口，所有参数转发给 run_generate_eval_advise.py。
支持 --from-latest-tuning 从 LATEST_NEXT_RUN.json 加载参数。

用法:
    python3 scripts/assistant/run_full_report.py \
        --metadata-csv datasets/CBIS_CLEAN_V2/metadata_clean.csv \
        --filter-view MLO --filter-density dense \
        --num-images 6 --tag-prefix my_run

    python3 scripts/assistant/run_full_report.py --from-latest-tuning
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PIPELINE_SCRIPT = ROOT / "scripts/assistant/run_generate_eval_advise.py"
REPORT_BASE = ROOT / "outputs/reports"
LATEST_FILE = REPORT_BASE / "LATEST_NEXT_RUN.json"


def main() -> None:
    p = argparse.ArgumentParser(
        description="一键全流程报告 (Generate → Evaluate → Advise)")
    p.add_argument("--from-latest-tuning", action="store_true", default=False,
                   help="从 LATEST_NEXT_RUN.json 加载参数")
    p.add_argument("--tag-prefix", type=str, default="report")

    # Forwarded args (used as defaults when not --from-latest-tuning)
    p.add_argument("--metadata-csv", type=str,
                   default="datasets/CBIS_CLEAN_V2/metadata_clean.csv")
    p.add_argument("--filter-view", type=str, default="MLO")
    p.add_argument("--filter-density", type=str, default="dense")
    p.add_argument("--num-images", type=int, default=6)
    p.add_argument("--mode", type=str, default="full-image")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--strength", type=float, default=0.42)
    p.add_argument("--guidance-scale", type=float, default=7.9)
    p.add_argument("--num-steps", type=int, default=40)
    p.add_argument("--overlap-ratio", type=float, default=0.85)
    p.add_argument("--fullimage-output-long-side", type=int, default=2048)
    p.add_argument("--postprocess", action="store_true", default=False)
    p.add_argument("--eval-profile", type=str, default="full")
    p.add_argument("--real-images-dir", type=str, default="")
    p.add_argument("--output-base", type=str,
                   default="outputs/generated/毕业论文_生成图像")

    args, unknown = p.parse_known_args()

    cmd = [sys.executable, str(PIPELINE_SCRIPT)]

    if args.from_latest_tuning:
        if not LATEST_FILE.is_file():
            print(f"LATEST_NEXT_RUN.json not found at {LATEST_FILE}")
            sys.exit(1)
        data = json.loads(LATEST_FILE.read_text(encoding="utf-8"))
        params = data.get("parameters", {})
        if not params:
            print("No parameters found in LATEST_NEXT_RUN.json")
            sys.exit(1)
        print(f"Loading parameters from {LATEST_FILE}")
        print(f"Parameters: {json.dumps(params, indent=2, ensure_ascii=False)}")

        cmd.extend([
            "--tag-prefix", args.tag_prefix,
            "--num-images", str(args.num_images),
            "--mode", args.mode,
            "--eval-profile", args.eval_profile,
            "--output-base", args.output_base,
        ])
        # Apply parameter overrides from JSON
        for key in ("strength", "guidance_scale", "num_steps", "overlap_ratio",
                     "seed", "fullimage_output_long_side"):
            if key in params:
                k = key.replace("_", "-")
                cmd.extend([f"--{k}", str(params[key])])
        if params.get("postprocess"):
            cmd.append("--postprocess")
        if args.real_images_dir:
            cmd.extend(["--real-images-dir", args.real_images_dir])
    else:
        # Forward all known args
        cmd.extend([
            "--tag-prefix", args.tag_prefix,
            "--metadata-csv", args.metadata_csv,
            "--filter-view", args.filter_view,
            "--filter-density", args.filter_density,
            "--num-images", str(args.num_images),
            "--mode", args.mode,
            "--seed", str(args.seed),
            "--strength", str(args.strength),
            "--guidance-scale", str(args.guidance_scale),
            "--num-steps", str(args.num_steps),
            "--overlap-ratio", str(args.overlap_ratio),
            "--fullimage-output-long-side", str(args.fullimage_output_long_side),
            "--eval-profile", args.eval_profile,
            "--output-base", args.output_base,
        ])
        if args.postprocess:
            cmd.append("--postprocess")
        if args.real_images_dir:
            cmd.extend(["--real-images-dir", args.real_images_dir])

    # Pass through unknown args
    if unknown:
        cmd.extend(unknown)

    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(ROOT), check=False)


if __name__ == "__main__":
    main()
