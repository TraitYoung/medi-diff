#!/usr/bin/env python3
"""自动评审 LoRA checkpoint，选择 tier1_ratio 最高的权重。

流程：
1. 扫描 outputs/lora/mammo_sd15_v3/checkpoint-* 与 final_lora
2. 每个 checkpoint 固定 seed 生成 N 张
3. 使用 review_generated_images.py --eval-profile full 评审
4. 汇总 tier1_ratio / mean semantic / pass_rate，输出 checkpoint_selection.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="选择 LoRA v3 最佳 checkpoint")
    p.add_argument("--lora-dir", type=Path, default=ROOT / "outputs/lora/mammo_sd15_v3")
    p.add_argument("--input-dir", type=Path, default=ROOT / "datasets/jpeg")
    p.add_argument("--output-base", type=Path, default=ROOT / "outputs/generated/checkpoint_eval")
    p.add_argument("--eval-base", type=Path, default=ROOT / "outputs/eval/checkpoint_eval")
    p.add_argument("--num-images", type=int, default=10)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--strength", type=float, default=0.30)
    p.add_argument("--overlap-ratio", type=float, default=0.75)
    p.add_argument("--global-guide-strength", type=float, default=0.25)
    p.add_argument("--max-checkpoints", type=int, default=0, help="0=全部；调试可限制数量")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def checkpoints(lora_dir: Path) -> list[Path]:
    out = sorted(lora_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    final = lora_dir / "final_lora"
    if final.exists():
        out.append(final)
    usable: list[Path] = []
    for p in out:
        if not p.is_dir():
            continue
        if (p / "adapter_config.json").is_file() or (p / "pytorch_lora_weights.bin").is_file() or (p / "pytorch_lora_weights.safetensors").is_file():
            usable.append(p)
        else:
            print(f"[skip] {p} 缺少可加载 LoRA 配置/权重（旧 checkpoint 仅有裸 lora_weights.safetensors）")
    return usable


def run(cmd: list[str], dry: bool = False) -> int:
    print("[cmd]", " ".join(cmd))
    if dry:
        return 0
    return subprocess.run(cmd, cwd=str(ROOT)).returncode


def summarize(eval_dir: Path) -> dict:
    path = eval_dir / "summary.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    per = data.get("per_image") or []
    n = max(1, len(per))
    tier1 = sum(1 for x in per if int(x.get("tier", 3)) == 1)
    sem = [float(x.get("semantic_score") or 0.0) for x in per]
    qual = [float(x.get("quality_score") or 0.0) for x in per]
    return {
        "total_images": len(per),
        "tier1_ratio": tier1 / n,
        "pass_rate": float(data.get("pass_rate", 0.0)),
        "mean_semantic": sum(sem) / n,
        "mean_quality": sum(qual) / n,
    }


def main() -> None:
    args = parse_args()
    cks = checkpoints(args.lora_dir)
    if args.max_checkpoints > 0:
        cks = cks[: args.max_checkpoints]
    if not cks:
        raise SystemExit(f"未找到 checkpoint: {args.lora_dir}")

    args.output_base.mkdir(parents=True, exist_ok=True)
    args.eval_base.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for ck in cks:
        tag = ck.name
        gen_prefix = f"eval_{tag}"
        gen_cmd = [
            sys.executable,
            str(ROOT / "scripts/generation/run_mammo_sd15.py"),
            "--input-dir",
            str(args.input_dir),
            "--output-base",
            str(args.output_base),
            "--output-subdir-prefix",
            gen_prefix,
            "--num-images",
            str(args.num_images),
            "--seed",
            str(args.seed),
            "--strength",
            str(args.strength),
            "--blend-mode",
            "gaussian",
            "--overlap-ratio",
            str(args.overlap_ratio),
            "--global-guide",
            "--global-guide-strength",
            str(args.global_guide_strength),
            "--lora-path",
            str(ck),
            "--no-postprocess",
        ]
        if run(gen_cmd, args.dry_run) != 0:
            print(f"[WARN] 生成失败: {ck}")
            continue
        if args.dry_run:
            continue

        cands = sorted(
            [p for p in args.output_base.iterdir() if p.is_dir() and gen_prefix in p.name],
            key=lambda p: p.stat().st_mtime,
        )
        if not cands:
            print(f"[WARN] 找不到生成目录: {gen_prefix}")
            continue
        gen_dir = cands[-1]
        eval_dir = args.eval_base / tag
        rev_cmd = [
            sys.executable,
            str(ROOT / "scripts/evaluation/review_generated_images.py"),
            "--images-dir",
            str(gen_dir),
            "--output-dir",
            str(eval_dir),
            "--no-recursive",
            "--eval-profile",
            "full",
            "--enable-seam-check",
        ]
        if run(rev_cmd, False) != 0:
            print(f"[WARN] 评审失败: {ck}")
            continue
        row = {
            "checkpoint": str(ck),
            "gen_dir": str(gen_dir),
            "eval_dir": str(eval_dir),
            **summarize(eval_dir),
        }
        rows.append(row)

    if not rows:
        print("无结果。")
        return
    rows.sort(key=lambda r: (float(r["tier1_ratio"]), float(r["mean_semantic"])), reverse=True)
    csv_path = args.eval_base / "checkpoint_selection.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[Best] {rows[0]}")
    print(f"[CSV] {csv_path}")


if __name__ == "__main__":
    main()
