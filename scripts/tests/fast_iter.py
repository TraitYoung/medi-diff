#!/usr/bin/env python3
"""快速迭代：生成 6 张（轻量参数）→ 评估 → 顾问点评 → 参数建议。

目标：每轮 ~1 分钟，让文本+视觉顾问驱动收敛。

用法：
  python3 scripts/tests/fast_iter.py                          # 默认参数跑一轮
  python3 scripts/tests/fast_iter.py --rounds 5 --auto-apply   # 连跑 5 轮，自动应用建议
  python3 scripts/tests/fast_iter.py --with-vl                 # 附加视觉顾问看图

快速参数（可在命令行覆盖）：
  --overlap-ratio 0.50  --num-steps 8  --no-postprocess  --no-global-guide
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable
GEN = ROOT / "scripts/generation/run_mammo_sd15.py"
EVAL = ROOT / "scripts/evaluation/review_generated_images.py"

LATEST_PARAMS = ROOT / "outputs/eval/fast_iter_latest_params.json"
BASE_MODEL = ROOT / "hf_cache/sd15"
LORA = ROOT / "outputs/lora/mammo_sd15_v4_clean/final_lora"
METADATA = ROOT / "datasets/CBIS_CLEAN_V2/metadata_clean.csv"


def run(cmd: list, timeout: int = 300) -> subprocess.CompletedProcess:
    print(f"  RUN: {' '.join(str(x) for x in cmd[:8])}...")
    return subprocess.run([str(x) for x in cmd], cwd=str(ROOT),
                          capture_output=True, text=True, timeout=timeout)


def generate(params: dict) -> Path | None:
    """快速生成 6 张图，返回输出目录。"""
    t0 = time.time()
    tag = f"fast_{params.get('tag', int(t0))}"
    gen_cmd = [
        PYTHON, str(GEN),
        "--base-model-local", str(BASE_MODEL),
        "--lora-path", str(LORA),
        "--metadata-csv", str(METADATA),
        "--filter-view", "MLO", "--filter-density", "dense",
        "--num-images", "6",
        "--seed", str(params.get("seed", 2026)),
        "--strength", str(params.get("strength", 0.42)),
        "--overlap-ratio", str(params.get("overlap_ratio", 0.50)),
        "--guidance-scale", str(params.get("guidance_scale", 7.9)),
        "--num-steps", str(params.get("num_steps", 8)),
        "--gabor-alpha", str(params.get("gabor_alpha", 0.45)),
        "--target-w", str(params.get("target_w", 1024)),
        "--target-h", str(params.get("target_h", 768)),
        "--output-base", str(ROOT / "outputs/generated/fast_iter"),
        "--output-subdir-prefix", tag,
        "--no-global-guide",
        "--no-legacy-label-guard",
        "--no-postprocess",
        "--no-input-clahe",
    ]
    r = run(gen_cmd, timeout=600)
    elapsed = time.time() - t0
    print(f"  Generate: {elapsed:.0f}s")
    if r.returncode != 0:
        print(f"  GEN FAILED:\n{r.stderr[-500:]}")
        return None
    # find output dir
    out_base = ROOT / "outputs/generated/fast_iter"
    dirs = sorted(out_base.glob(f"{tag}_*"))
    return dirs[-1] if dirs else None


def evaluate(images_dir: Path) -> dict | None:
    """评估生成图，返回 summary dict。"""
    t0 = time.time()
    eval_dir = ROOT / "outputs/eval/fast_iter"
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_cmd = [
        PYTHON, str(EVAL),
        "--images-dir", str(images_dir),
        "--output-dir", str(eval_dir),
        "--no-recursive",
        "--review-workers", "4",
        "--real-baseline-json", str(ROOT / "outputs/eval/sd15_lora_v2_final/real_baseline_stats.json"),
    ]
    r = run(eval_cmd, timeout=120)
    elapsed = time.time() - t0
    print(f"  Evaluate: {elapsed:.0f}s")
    if r.returncode != 0:
        print(f"  EVAL FAILED:\n{r.stderr[-500:]}")
        return None
    summary_path = eval_dir / "summary.json"
    if summary_path.is_file():
        with open(summary_path) as f:
            return json.load(f)
    return None


def build_summary_text(params: dict, s: dict) -> str:
    """从 eval summary 构建给文本顾问的摘要。"""
    lines = [
        f"## 当前参数",
        f"overlap_ratio={params.get('overlap_ratio', 0.50)}, "
        f"strength={params.get('strength', 0.42)}, "
        f"guidance_scale={params.get('guidance_scale', 7.9)}, "
        f"num_steps={params.get('num_steps', 8)}, "
        f"gabor_alpha={params.get('gabor_alpha', 0.45)}",
        f"",
        f"## 评估结果",
        f"平均总分: {s.get('mean_total_score', '?'):.1f}" if isinstance(s.get('mean_total_score'), (int, float)) else f"平均总分: {s.get('mean_total_score', '?')}",
        f"通过率: {s.get('pass_rate', 0):.0%}",
    ]
    if isinstance(s.get('mean_total_score'), (int, float)):
        lines.append(f"平均总分: {s['mean_total_score']:.1f}")
        lines.append(f"通过率: {s['pass_rate']:.0%}")

    groups = s.get("group_mean_scores", {})
    if groups:
        gs = ", ".join(f"{k}={v:.2f}" for k, v in sorted(groups.items()))
        lines.append(f"分组: {gs}")

    am = s.get("academic_metrics", {})
    if am:
        lines.append(f"BRISQUE: {am.get('mean_brisque', '?'):.1f}" if isinstance(am.get('mean_brisque'), (int, float)) else f"BRISQUE: {am.get('mean_brisque', '?')}")
        lines.append(f"ps_slope_beta: {am.get('mean_ps_slope_beta', '?'):.2f}" if isinstance(am.get('mean_ps_slope_beta'), (int, float)) else f"ps_slope_beta: {am.get('mean_ps_slope_beta', '?')}")

    viols = {k: v for k, v in s.get("violation_rates", {}).items() if v > 0}
    if viols:
        vlist = ", ".join(f"{k}={v*100:.0f}%" for k, v in sorted(viols.items(), key=lambda x: -x[1]))
        lines.append(f"违规: {vlist}")

    # edge_density
    import csv
    csv_path = ROOT / "outputs/eval/fast_iter/review_report.csv"
    if csv_path.is_file():
        with open(csv_path) as f:
            eds = []
            for row in csv.DictReader(f):
                try:
                    eds.append(float(row.get("edge_density", -1)))
                except (ValueError, TypeError):
                    pass
            if eds:
                ok = sum(1 for e in eds if e >= 0.008)
                lines.append(f"edge_density OK: {ok}/{len(eds)}")
                lines.append(f"edge_density values: {[f'{e:.5f}' for e in eds]}")

    lines.append("")
    lines.append("请给出下一轮参数建议（JSON 格式，只包含需要调整的键）：")
    return "\n".join(lines)


def ask_text_advisor(summary: str) -> str:
    """调用文本顾问。"""
    sys.path.insert(0, str(ROOT / "scripts/assistant"))
    from ask_advisor import ask_advisor
    system = (
        "你是乳腺钼靶扩散生成系统的参数优化顾问。根据评估指标给出下一轮参数建议。"
        "重点关注：降低 BRISQUE（目标<30）、消除 BANDING/接缝伪影、"
        "改善 edge_density（目标>=0.008）、提升 C 组纹理评分（目标>0.85）。"
        "视觉质量优先于统计指标——即使灰度分布匹配良好，过平滑/接缝可见的图像也是不合格的。"
        "可调参数：overlap_ratio(0.3-0.9)、strength(0.2-0.7)、guidance_scale(5-12)、"
        "num_steps(4-60)、gabor_alpha(0.1-0.6)、sharpen_strength(0.0-0.5)。"
        "输出格式：先给简短分析（2-3句），然后给出 JSON 参数建议。"
    )
    return ask_advisor(summary, system=system, timeout=180)


def _can_vl() -> bool:
    """检查是否配置了 VL API。"""
    import os
    return bool(os.environ.get("QWEN_VL_MODEL") or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY"))


def ask_vl_advisor(images_dir: Path, summary_text: str, prev_vl_score: float | None = None) -> tuple[str, float | None]:
    """调用视觉顾问看 best+worst 图，返回 (建议文本, 平均视觉评分)。"""
    sys.path.insert(0, str(ROOT / "scripts/assistant"))
    from ask_advisor import ask_advisor_vl

    eval_dir = ROOT / "outputs/eval/fast_iter"
    with open(eval_dir / "summary.json") as f:
        s = json.load(f)
    per_img = sorted(s.get("per_image", []), key=lambda x: x.get("final_rank_score", 0), reverse=True)

    image_paths = []
    for img in per_img[:3] + per_img[-2:]:
        p = images_dir / img["image"]
        if p.is_file():
            image_paths.append(p)

    if not image_paths:
        return "(VL: no images found)", None

    trend_line = ""
    if prev_vl_score is not None:
        trend_line = f"上轮 VL 均分: {prev_vl_score:.2f}/5。请判断本轮是否改善。\n"

    prompt = (
        f"{summary_text}\n\n"
        f"{trend_line}"
        f"以上是这批生成图的评估指标。以下是 best-3 和 worst-2 图像。"
        f"请逐张点评：接缝可见性（最优先）、纹理自然度（过平滑/塑料感）、伪影（banding/环形/蜂窝）、"
        f"解剖合理性（轮廓/胸肌/皮肤线），每张给出 1-5 分评分（1=灾难 3=可用 5=优秀）。"
        f"最后输出一行 'VL_AVG_SCORE: X.X' 给出平均分。"
    )
    result = ask_advisor_vl(prompt, system="你是乳腺钼靶医学影像评审专家。视觉质量优先于统计指标。用中文点评。",
                          image_paths=image_paths, timeout=300)

    # 尝试从 VL 回复中提取平均分
    vl_score = None
    import re
    m = re.search(r"VL_AVG_SCORE:\s*([\d.]+)", result)
    if m:
        try:
            vl_score = float(m.group(1))
        except ValueError:
            pass
    return result, vl_score


def main():
    p = argparse.ArgumentParser(description="快速迭代：生成→评估→顾问（视觉优先）")
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--no-vl", action="store_true", help="禁用视觉顾问（默认自动启用）")
    p.add_argument("--vl-weight", type=float, default=0.6,
                   help="视觉影响参数的 VL 权重（0.5-1.0，默认 0.6）")
    p.add_argument("--overlap-ratio", type=float, default=0.50)
    p.add_argument("--strength", type=float, default=0.42)
    p.add_argument("--guidance-scale", type=float, default=7.9)
    p.add_argument("--num-steps", type=int, default=8)
    p.add_argument("--gabor-alpha", type=float, default=0.45)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()

    use_vl = (not args.no_vl) and _can_vl()
    if args.no_vl:
        print("[info] VL 顾问已禁用 (--no-vl)")
    elif not _can_vl():
        print("[info] VL 顾问未配置 API key，仅文本模式")

    params = {
        "overlap_ratio": args.overlap_ratio,
        "strength": args.strength,
        "guidance_scale": args.guidance_scale,
        "num_steps": args.num_steps,
        "gabor_alpha": args.gabor_alpha,
        "seed": args.seed,
    }

    round_history: list[dict] = []
    prev_vl_score = None

    for rnd in range(1, args.rounds + 1):
        print(f"\n{'='*60}")
        print(f"Round {rnd}/{args.rounds}")
        print(f"{'='*60}")
        params["tag"] = f"r{rnd}"
        params["seed"] = args.seed + rnd

        # 1) Generate
        images_dir = generate(params)
        if images_dir is None:
            print("Generation failed, stopping.")
            break
        print(f"  Images: {images_dir}")

        # 2) Evaluate
        s = evaluate(images_dir)
        if s is None:
            print("Evaluation failed, stopping.")
            break

        # 3) Build summary
        summary = build_summary_text(params, s)
        print(f"\n--- Summary ---\n{summary}\n---------------")

        # 4) Text advisor
        print("\n[Text Advisor]")
        text_advice = ""
        try:
            text_advice = ask_text_advisor(summary)
            print(text_advice)
        except Exception as e:
            print(f"  Text advisor error: {e}")

        # 5) VL advisor (auto-enabled)
        vl_advice = ""
        vl_score = None
        if use_vl:
            print("\n[VL Advisor]")
            try:
                vl_advice, vl_score = ask_vl_advisor(images_dir, summary, prev_vl_score)
                print(vl_advice)
                if vl_score is not None:
                    print(f"  VL_AVG_SCORE: {vl_score:.2f}/5")
                    prev_vl_score = vl_score
            except Exception as e:
                print(f"  VL advisor error: {e}")

        # 6) Track round
        round_info = {
            "round": rnd,
            "params": dict(params),
            "auto_score": s.get("mean_total_score"),
            "brisque": s.get("academic_metrics", {}).get("mean_brisque"),
            "vl_score": vl_score,
        }
        round_history.append(round_info)

        # 7) Save all
        with open(LATEST_PARAMS, "w") as f:
            json.dump({"params": params, "history": round_history}, f, indent=2)
        print(f"\nParams + history saved to {LATEST_PARAMS}")

    # 最终汇总
    if len(round_history) > 1:
        print(f"\n{'='*60}")
        print("Round History:")
        for rh in round_history:
            vl_str = f"VL={rh['vl_score']:.2f}" if rh["vl_score"] else "VL=N/A"
            print(f"  R{rh['round']}: auto={rh['auto_score']:.1f} BRISQUE={rh['brisque']:.1f} {vl_str}")


if __name__ == "__main__":
    main()
