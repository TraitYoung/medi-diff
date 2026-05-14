#!/usr/bin/env python3
"""消融实验：单变量对照，定位 BRISQUE/edge_density 瓶颈。

矩阵：
  E0: baseline — v4 LoRA + postprocess, overlap=0.86
  E1: no postprocess — v4 LoRA + no-pp, overlap=0.86
  E2: high overlap — v4 LoRA + postprocess, overlap=0.92
  E3: no postprocess + high overlap — v4 LoRA + no-pp, overlap=0.92
  E4: v3 LoRA — v3 LoRA + postprocess, overlap=0.86

每条件 2 张，固定 seed=2026，全部 MLO dense。
"""

import subprocess, sys, json, shutil, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable
GEN_SCRIPT = ROOT / "scripts/generation/run_mammo_sd15.py"
EVAL_SCRIPT = ROOT / "scripts/evaluation/review_generated_images.py"

BASE_MODEL = ROOT / "hf_cache/sd15"
METADATA = ROOT / "datasets/CBIS_CLEAN_V2/metadata_clean.csv"
V4_LORA = ROOT / "outputs/lora/mammo_sd15_v4_clean/final_lora"
V3_LORA = ROOT / "outputs/lora/mammo_sd15_v3_retrain_20260502_1723/checkpoint-3000"

EXPERIMENTS = [
    {"id": "E0_baseline",     "lora": V4_LORA, "postprocess": True,  "overlap": 0.86, "desc": "v4+pp+ov0.86"},
    {"id": "E1_no_pp",       "lora": V4_LORA, "postprocess": False, "overlap": 0.86, "desc": "v4+no_pp+ov0.86"},
    {"id": "E2_hi_overlap",  "lora": V4_LORA, "postprocess": True,  "overlap": 0.92, "desc": "v4+pp+ov0.92"},
    {"id": "E3_no_pp_hi_ov", "lora": V4_LORA, "postprocess": False, "overlap": 0.92, "desc": "v4+no_pp+ov0.92"},
    {"id": "E4_v3_lora",     "lora": V3_LORA, "postprocess": True,  "overlap": 0.86, "desc": "v3+pp+ov0.86"},
]

GEN_OUT = ROOT / "outputs/generated/毕业论文_生成图像/ablation"
EVAL_OUT = ROOT / "outputs/eval/ablation"


def run(cmd, cwd=ROOT):
    print(f"  RUN: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  FAIL: {result.stderr[-500:]}")
    return result


def main():
    results = []
    for exp in EXPERIMENTS:
        print(f"\n{'='*60}")
        print(f"[{exp['id']}] {exp['desc']}")
        print(f"{'='*60}")

        tag = f"abl_{exp['id']}"

        # 1) Generate
        gen_cmd = [
            PYTHON, str(GEN_SCRIPT),
            "--base-model-local", str(BASE_MODEL),
            "--lora-path", str(exp["lora"]),
            "--metadata-csv", str(METADATA),
            "--filter-view", "MLO", "--filter-density", "dense",
            "--num-images", "2", "--seed", "2026",
            "--strength", "0.42", "--overlap-ratio", str(exp["overlap"]),
            "--guidance-scale", "7.9", "--num-steps", "52",
            "--output-base", str(GEN_OUT),
            "--output-subdir-prefix", tag,
        ]
        if not exp["postprocess"]:
            gen_cmd.append("--no-postprocess")

        t0 = time.time()
        print("  Generating...")
        r = run(gen_cmd)
        gen_elapsed = time.time() - t0
        print(f"  Generate done ({gen_elapsed:.0f}s)")

        # Find generated dir
        gen_dirs = sorted(GEN_OUT.glob(f"{tag}_*"))
        if not gen_dirs:
            print(f"  SKIP: no output dir found for {tag}")
            continue
        gen_dir = gen_dirs[-1]
        n_images = len(list(gen_dir.glob("sd15_*.png")))
        print(f"  Output: {gen_dir} ({n_images} images)")

        # 2) Evaluate
        eval_dir = EVAL_OUT / exp["id"]
        eval_cmd = [
            PYTHON, str(EVAL_SCRIPT),
            "--images-dir", str(gen_dir),
            "--output-dir", str(eval_dir),
            "--no-recursive", "--eval-profile", "full",
            "--enable-seam-check",
        ]
        print("  Evaluating...")
        r = run(eval_cmd)
        print(f"  Eval done")

        # 3) Read summary
        summary_path = eval_dir / "summary.json"
        if summary_path.is_file():
            with open(summary_path) as f:
                s = json.load(f)
            r = {
                "id": exp["id"],
                "desc": exp["desc"],
                "n_images": n_images,
                "gen_elapsed_s": int(gen_elapsed),
                "pass_rate": s.get("pass_rate"),
                "mean_score": s.get("mean_total_score"),
                "brisque": s["academic_metrics"].get("mean_brisque"),
                "ps_slope": s["academic_metrics"].get("mean_ps_slope_beta"),
                "group_A": s["group_mean_scores"].get("A"),
                "group_C": s["group_mean_scores"].get("C"),
                "group_F": s["group_mean_scores"].get("F"),
                "violations": {k: v for k, v in s.get("violation_rates", {}).items() if v > 0},
            }
            results.append(r)
        else:
            results.append({"id": exp["id"], "desc": exp["desc"], "error": "no summary"})

    # 4) Print comparison table
    print("\n\n" + "="*80)
    print("ABLATION RESULTS")
    print("="*80)
    header = f"{'ID':<18} {'Pass':>5} {'Score':>7} {'BRISQUE':>8} {'ps_β':>6} {'A':>6} {'C':>6} {'F':>6}  Top violations"
    print(header)
    print("-" * len(header))
    for r in results:
        if "error" in r:
            print(f"{r['id']:<18}  ERROR: {r['error']}")
            continue
        viols = ", ".join(f"{k}={v*100:.0f}%" for k, v in sorted(r["violations"].items(), key=lambda x: -x[1])[:3])
        print(f"{r['id']:<18} {r['pass_rate']:>5.0%} {r['mean_score']:>7.2f} {r['brisque']:>8.1f} {r['ps_slope']:>6.2f} {r['group_A']:>6.2f} {r['group_C']:>6.2f} {r['group_F']:>6.2f}  {viols}")

    # Per-image edge_density
    print("\nPer-image edge_density:")
    for exp_id in [r["id"] for r in results if "error" not in r]:
        csv_path = EVAL_OUT / exp_id / "review_report.csv"
        if csv_path.is_file():
            with open(csv_path) as f:
                lines = f.readlines()
                if len(lines) > 1:
                    for line in lines[1:]:
                        fn = line.split(",")[0].split("/")[-1]
                        ed = line.split(",")[-3]  # edge_density column
                        try:
                            ed = float(ed)
                            status = "OK" if ed >= 0.008 else "LOW"
                            print(f"  [{exp_id}] {fn:35s} edge_density={ed:.6f} [{status}]")
                        except ValueError:
                            pass

    print(f"\nResults saved to {EVAL_OUT}")


if __name__ == "__main__":
    main()
