#!/usr/bin/env python3
"""
Gradio / 单机入口「接线」自检（不导入 gradio/torch）。

在项目根执行：
  python3 scripts/tools/verify_ui_wiring.py

用法：答辩或部署前快速确认脚本路径、默认目录与 METADATA/LORA 存在性；
      不涉及真实 GPU 推理。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _ok(msg: str) -> None:
    print(f"[OK] {msg}")


def _fail(msg: str) -> int:
    print(f"[FAIL] {msg}", file=sys.stderr)
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Gradio/CLI wiring smoke check")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="将「推荐存在」的检查项视作失败条件",
    )
    args = ap.parse_args()

    errs = 0
    if not (ROOT / "requirements.txt").is_file() or not (ROOT / "apps/app_gradio.py").is_file():
        errs += _fail(f"推导的项目根无效或不是本仓库: {ROOT}")

    need_files = [
        ROOT / "apps/app_gradio.py",
        ROOT / "apps/start.sh",
        ROOT / "scripts/generation/run_mammo_sd15.py",
        ROOT / "scripts/evaluation/review_generated_images.py",
        ROOT / "scripts/assistant/run_full_report.py",
        ROOT / "scripts/assistant/run_generate_eval_advise.py",
    ]
    for p in need_files:
        if not p.is_file():
            errs += _fail(f"缺失: {p.relative_to(ROOT)}")
        else:
            _ok(str(p.relative_to(ROOT)))

    thesis = ROOT / "outputs/generated/毕业论文_生成图像"
    thesis.mkdir(parents=True, exist_ok=True)
    _ok(f"毕业论文输出根可写（或已存在）: {thesis.relative_to(ROOT)}")

    meta_v2 = ROOT / "datasets/CBIS_CLEAN_V2/metadata_clean.csv"
    meta_v1 = ROOT / "datasets/CBIS_CLEAN/metadata_clean.csv"
    if meta_v2.is_file():
        _ok(f"元数据(V2): {meta_v2.relative_to(ROOT)}")
    elif meta_v1.is_file():
        print(f"[WARN] 仅用 V1 元数据: {meta_v1.relative_to(ROOT)}")
        if args.strict:
            errs += _fail("strict：缺少 CBIS_CLEAN_V2/metadata_clean.csv")
    else:
        errs += _fail("缺少 datasets/CBIS_CLEAN[_V2]/metadata_clean.csv")

    v4 = ROOT / "outputs/lora/mammo_sd15_v4_clean/final_lora/adapter_model.safetensors"
    if v4.is_file():
        _ok("LoRA v4 final: adapter_model.safetensors")
    elif args.strict:
        errs += _fail("strict：未发现 v4 final LoRA 权重")

    exe = sys.executable
    for rel in (
        "scripts/generation/run_mammo_sd15.py",
        "scripts/evaluation/review_generated_images.py",
        "scripts/assistant/run_generate_eval_advise.py",
        "scripts/assistant/run_full_report.py",
    ):
        sp = ROOT / rel
        r = subprocess.run(
            [exe, str(sp), "--help"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=60,
        )
        if r.returncode != 0:
            errs += _fail(f"--help 失败: {rel} (exit {r.returncode})")
        else:
            _ok(f"--help: {rel}")

    if errs:
        print(f"\n共检测到 {errs} 个硬性错误。", file=sys.stderr)
        return 1
    print("\n接线自检完成：路径与 argparse 可读；训练/演示资源请按需补充。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
