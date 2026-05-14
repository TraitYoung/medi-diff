#!/usr/bin/env python3
"""
将「非新架构」或散落的生成结果挪到 outputs/_legacy/，减轻 Gradio 画廊扫描压力。

默认行为（与 app_gradio 一致）：
  - 保留 outputs/generated/毕业论文_生成图像/ 作为当前主线输出根。
  - 其余 outputs/generated/ 顶层条目（目录或文件）整体移到归档目录。

可选：按前缀把毕业论文目录下的旧批次 *_000 一并移走。

用法：
  python3 scripts/tools/archive_legacy_generated_for_gallery.py --dry-run
  python3 scripts/tools/archive_legacy_generated_for_gallery.py
  python3 scripts/tools/archive_legacy_generated_for_gallery.py \\
      --thesis-deny-prefixes img2img_,adapter_,controlnet_,sdxl_,twopass_
"""
from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GENERATED = ROOT / "outputs/generated"
THESIS = GENERATED / "毕业论文_生成图像"
LEGACY_PARENT = ROOT / "outputs/_legacy"

KEEP_ROOT_NAMES = frozenset({"毕业论文_生成图像", "README.md", ".gitkeep"})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true", help="只打印将移动的路径，不写入")
    p.add_argument(
        "--skip-non-thesis-root",
        action="store_true",
        help="不移动 outputs/generated 下除毕业论文_生成图像外的顶层条目",
    )
    p.add_argument(
        "--thesis-deny-prefixes",
        type=str,
        default="",
        help="逗号分隔前缀：毕业论文_生成图像 下以此前缀开头的 *_000 批次目录会移到归档",
    )
    args = p.parse_args()

    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive = LEGACY_PARENT / f"generated_archived_{tag}"
    from_root = archive / "from_generated_root"
    from_thesis = archive / "from_thesis"

    deny = tuple(x.strip() for x in args.thesis_deny_prefixes.split(",") if x.strip())

    moves: list[tuple[Path, Path]] = []

    if not args.skip_non_thesis_root and GENERATED.is_dir():
        for item in sorted(GENERATED.iterdir(), key=lambda x: x.name):
            if item.name in KEEP_ROOT_NAMES:
                continue
            moves.append((item, from_root / item.name))

    if deny and THESIS.is_dir():
        for item in sorted(THESIS.iterdir(), key=lambda x: x.name):
            if not item.is_dir() or not item.name.endswith("_000"):
                continue
            if any(item.name.startswith(pref) for pref in deny):
                moves.append((item, from_thesis / item.name))

    if not moves:
        print("没有需要归档的路径。")
        return

    print(f"归档目标: {archive} （共 {len(moves)} 项）")
    for src, dst in moves:
        print(f"  {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")

    if args.dry_run:
        print("--dry-run：未移动。")
        return

    LEGACY_PARENT.mkdir(parents=True, exist_ok=True)

    for src, dst in moves:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            raise SystemExit(f"目标已存在，中止: {dst}")
        shutil.move(str(src), str(dst))
    print(f"完成。已移动到 {archive.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
