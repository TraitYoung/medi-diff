"""Gradio 界面：乳腺钼靶扩散生成系统答辩/日常主入口。

通过 subprocess 调用 scripts/ 下的管线脚本，与 FastAPI 服务逻辑独立。
"""

import argparse
import json
from typing import List, Dict, Tuple, Optional
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import gradio as gr

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
GENERATED_DIR = ROOT / "outputs/generated/毕业论文_生成图像"
EVAL_DIR = ROOT / "outputs/eval"
REPORT_DIR = ROOT / "outputs/reports"
DEFAULT_MODEL = ROOT / "hf_cache/sd15"
DEFAULT_LORA = ROOT / "outputs/lora/mammo_sd15_v6_allMLO/final_lora"
METADATA_CSV = ROOT / "datasets/CBIS_CLEAN_V2/metadata_clean.csv"
DEFAULT_OUTPUT_LONG_SIDE = 2048

# ── 工具函数 ─────────────────────────────────────────────────────────────────

def _run(command: list[str]) -> str:
    """Run subprocess, capture output with timing."""
    t0 = time.time()
    cmd_head = " ".join(command[:8])
    logger.info("subprocess start: %s ...", cmd_head)
    proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    dt = time.time() - t0
    logger.info("subprocess done in %.1fs (rc=%d): %s ...", dt, proc.returncode, cmd_head)
    output = ["$ " + " ".join(command), f"# elapsed={dt:.1f}s rc={proc.returncode}", ""]
    if proc.stdout:
        output.append(proc.stdout[-8000:])
    if proc.stderr:
        output.append("\n[stderr]\n" + proc.stderr[-8000:])
    output.append(f"\nreturn_code={proc.returncode}")
    return "\n".join(output)


def _batch_dirs() -> list[Path]:
    """返回毕业论文生成目录下所有批次，按修改时间倒序。"""
    if not GENERATED_DIR.exists():
        return []
    image_exts = {".png", ".jpg", ".jpeg"}

    def has_images(path: Path) -> bool:
        return any(
            child.is_file() and child.suffix.lower() in image_exts
            for child in path.iterdir()
        )

    def latest_activity(path: Path) -> float:
        times = [path.stat().st_mtime]
        for child in path.iterdir():
            if child.is_file() and child.suffix.lower() in image_exts:
                times.append(child.stat().st_mtime)
        return max(times)

    return sorted(
        [p for p in GENERATED_DIR.iterdir() if p.is_dir() and has_images(p)],
        key=latest_activity,
        reverse=True,
    )


def _batch_choices() -> list[str]:
    return [p.name for p in _batch_dirs()]


def _gallery_images(batch_name: str) -> list[str]:
    if not batch_name:
        return []
    batch = GENERATED_DIR / batch_name
    if not batch.exists():
        return []
    images = sorted(
        list(batch.glob("*.png"))
        + list(batch.glob("*.jpg"))
        + list(batch.glob("*.jpeg"))
    )
    return [str(p) for p in images[:40]]


def _read_source_map(batch_name: str) -> dict[str, str]:
    """读取批次目录下的 source_map.json，返回 {gen_filename: source_path}。"""
    if not batch_name:
        return {}
    sm = GENERATED_DIR / batch_name / "source_map.json"
    if not sm.is_file():
        return {}
    try:
        return json.loads(sm.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _find_eval_summary(batch_name: str) -> dict | None:
    """在 outputs/eval/ 下查找对应批次的 summary.json。"""
    if not batch_name:
        return None
    eval_root = EVAL_DIR
    if not eval_root.is_dir():
        return None
    for d in sorted(eval_root.iterdir(), reverse=True):
        if d.is_dir() and batch_name in d.name:
            sp = d / "summary.json"
            if sp.is_file():
                try:
                    return json.loads(sp.read_text(encoding="utf-8"))
                except Exception:
                    continue
    return None


_source_cache: dict[str, list[tuple[str, str]]] = {}

def _gallery_with_sources(batch_name: str) -> list[tuple[str, str]]:
    """返回 (生成图, 源图) 对列表，用于源图对比模式。结果缓存避免重复 I/O。"""
    if batch_name in _source_cache:
        return _source_cache[batch_name]
    gen_images = _gallery_images(batch_name)
    source_map = _read_source_map(batch_name)
    pairs: list[tuple[str, str]] = []
    for gen_path in gen_images:
        gen_name = Path(gen_path).name
        src_path = source_map.get(gen_name, "")
        pairs.append((gen_path, src_path if src_path else gen_path))
    _source_cache[batch_name] = pairs
    return pairs



# ── 精选导出辅助 ──────────────────────────────────────────────────────────


def _on_curated_batch_change(batch_name: str):
    """批次变化 → 纯预览 + 待选列表（按评估排序）。"""
    if not batch_name:
        return [], gr.update(choices=[], value=[])
    gen_images = _gallery_images(batch_name)
    summary = _find_eval_summary(batch_name)
    scored: dict[str, tuple[int, float]] = {}
    if summary:
        for item in summary.get("per_image", []) or []:
            name = item.get("image", "")
            tier = int(item.get("tier", 4) or 4)
            score = float(item.get("final_rank_score", 0) or 0)
            scored[name] = (tier, -score)
    gen_images.sort(key=lambda p: scored.get(Path(p).name, (4, 0.0)))
    preview: list[tuple[str, str]] = []
    choices: list[tuple[str, str]] = []
    for p in gen_images[:40]:
        fname = Path(p).name
        s = scored.get(fname)
        label = f"T{s[0]} | {fname}" if s else fname
        preview.append((p, label))
        choices.append((fname, label))
    return preview, gr.update(choices=choices, value=[])


def _export_curated(selected_fnames: list[str], batch_name: str,
                    export_label: str) -> str:
    """导出选中图片到 outputs/curated/<label>/."""
    if not selected_fnames:
        return "未选择任何图片。"
    if not batch_name:
        return "未选择批次。"
    # 解析文件名（兼容 CheckboxGroup 返回 label 前缀）
    wanted: set[str] = set()
    for raw in selected_fnames:
        wanted.add(raw.split(" | ")[-1].strip() if " | " in raw else raw)
    if not export_label.strip():
        from datetime import datetime
        export_label = datetime.now().strftime("curated_%Y%m%d_%H%M%S")
    export_label = export_label.strip().replace(" ", "_")
    dest = ROOT / "outputs/curated" / export_label
    dest.mkdir(parents=True, exist_ok=True)

    import shutil
    source_map = _read_source_map(batch_name)
    copied = 0
    manifest: list[dict] = []
    # 从批次目录直接枚举图片，匹配选中文件名
    for p in _gallery_images(batch_name):
        fname = Path(p).name
        if fname not in wanted:
            continue
        dest_name = fname
        if (dest / dest_name).exists():
            dest_name = f"{batch_name}_{fname}"
        shutil.copy2(p, str(dest / dest_name))
        manifest.append({
            "image": dest_name,
            "source_batch": batch_name,
            "source_image": source_map.get(fname, ""),
        })
        copied += 1

    manifest_path = dest / "export_manifest.json"
    manifest_path.write_text(
        json.dumps({"exported_at": export_label, "count": copied,
                    "images": manifest}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    return f"已导出 {copied} 张图片到 {dest}"


def _pct(value: float) -> str:
    return f"{float(value) * 100:.1f}%"


def _fmt(value, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "-"


def _top_violations(summary: dict, limit: int = 4) -> str:
    rates = summary.get("violation_rates", {}) or {}
    items = sorted(rates.items(), key=lambda kv: float(kv[1]), reverse=True)
    if not items:
        return "无明显高频扣分项"
    return "；".join(f"{k} {_pct(v)}" for k, v in items[:limit])


def _eval_summary_markdown(output_dir: Path) -> str:
    summary_path = output_dir / "summary.json"
    if not summary_path.is_file():
        return "评估尚未生成摘要。"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return f"摘要读取失败：{exc}"

    count = int(summary.get("count", summary.get("total_images", 0)) or 0)
    ok_count = int(summary.get("ok_count", 0) or 0)
    groups = summary.get("group_means") or summary.get("group_mean_scores") or {}
    tier_hist = summary.get("tier_hist", {}) or {}
    tier_text = " / ".join(f"T{k}:{v}" for k, v in sorted(tier_hist.items())) or "-"
    brisque = float(summary.get("mean_brisque", -1) or -1)
    brisque_text = _fmt(brisque, 2) if brisque >= 0 else "未计算"
    group_text = (
        f"A构图 {_fmt(groups.get('A', 0))} · "
        f"B灰度 {_fmt(groups.get('B', 0))} · "
        f"C纹理 {_fmt(groups.get('C', 0))} · "
        f"D伪影 {_fmt(groups.get('D', 0))} · "
        f"E分布 {_fmt(groups.get('E', 0))} · "
        f"F解剖 {_fmt(groups.get('F', 0))}"
    )
    return (
        f"**评估摘要**\n\n"
        f"- 图像数：{count}，通过：{ok_count}，通过率：**{_pct(summary.get('pass_rate', 0))}**\n"
        f"- 平均总分：**{_fmt(summary.get('mean_total_score', 0), 2)}**，平均 BRISQUE：**{brisque_text}**，功率谱 β：**{_fmt(summary.get('mean_ps_slope', 0), 3)}**\n"
        f"- 分组得分：{group_text}\n"
        f"- Tier 分布：{tier_text}\n"
        f"- 主要扣分项：{_top_violations(summary)}\n"
        f"- 输出目录：`{output_dir}`"
    )


def _eval_preview_images(batch_name: str, output_dir: Path, limit: int = 12) -> list[str]:
    batch = GENERATED_DIR / batch_name
    summary_path = output_dir / "summary.json"
    if not batch.exists() or not summary_path.is_file():
        return []
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        per_image = summary.get("per_image", []) or []
    except Exception:
        return _gallery_images(batch_name)[:limit]

    ranked = sorted(
        per_image,
        key=lambda item: (
            0 if item.get("ok") else 1,
            int(item.get("tier", 4) or 4),
            -float(item.get("total_score", 0) or 0),
        ),
    )
    images: list[str] = []
    for item in ranked:
        path = batch / str(item.get("image", ""))
        if path.is_file():
            images.append(str(path))
        if len(images) >= limit:
            break
    return images or _gallery_images(batch_name)[:limit]


def _review_rows() -> list[list]:
    """扫描 outputs/eval/ 生成评估汇总表行。"""
    rows = []
    if not EVAL_DIR.exists():
        return rows
    for path in sorted(EVAL_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        summary_path = path / "summary.json"
        if not path.is_dir() or not summary_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            groups = summary.get("group_means") or summary.get("group_mean_scores") or {}
            tier_hist = summary.get("tier_hist", {}) or {}
            tier_text = " / ".join(f"T{k}:{v}" for k, v in sorted(tier_hist.items()))
            rows.append([
                path.name,
                int(summary.get("count", summary.get("total_images", 0)) or 0),
                _pct(summary.get("pass_rate", 0)),
                round(float(summary.get("mean_total_score", 0)), 3),
                round(float(summary.get("mean_brisque", -1)), 2),
                round(float(groups.get("A", 0)), 3),
                round(float(groups.get("B", 0)), 3),
                round(float(groups.get("C", 0)), 3),
                round(float(groups.get("D", 0)), 3),
                round(float(groups.get("E", 0)), 3),
                round(float(groups.get("F", 0)), 3),
                int(summary.get("veto_count", 0) or 0),
                tier_text,
                _top_violations(summary, limit=2),
            ])
        except Exception:
            continue
    return rows[:30]


def _param_history_rows() -> list[list]:
    """读取 PARAM_HISTORY.json，返回 Dataframe 行。"""
    hist_path = REPORT_DIR / "PARAM_HISTORY.json"
    if not hist_path.is_file():
        return []
    try:
        history = json.loads(hist_path.read_text(encoding="utf-8"))
        if not isinstance(history, list):
            return []
        rows = []
        for entry in history:
            params = entry.get("parameters", {})
            metrics = entry.get("metrics", {})
            rows.append([
                entry.get("index", ""),
                entry.get("recorded_at", "")[:16],
                entry.get("source_tag", ""),
                round(float(metrics.get("pass_rate", 0)), 3),
                round(float(metrics.get("strict_pass_rate", 0)), 3),
                round(float(metrics.get("mean_total_score", 0)), 2),
                round(float(metrics.get("mean_brisque", 0)), 2),
                round(float(params.get("strength", 0)), 3),
                round(float(params.get("guidance_scale", 0)), 2),
                int(params.get("num_steps", 0)),
                str(params.get("notes_zh", ""))[:60],
            ])
        return rows
    except Exception:
        return []


# ── 后端逻辑 ─────────────────────────────────────────────────────────────────

def _with_labels(paths: list[str]) -> list[tuple[str, str]]:
    """给图片路径加上文件名标签。"""
    return [(p, Path(p).name) for p in paths]


def switch_gallery_mode(mode: str, batch_name: str):
    """根据模式返回 gr.update(value=images, columns=n)。"""
    if not batch_name:
        return gr.update(value=[], columns=4)
    if mode == "源图对比":
        pairs = _gallery_with_sources(batch_name)
        result: list = []
        for gen, src in pairs:
            result.append((gen, f"生成: {Path(gen).name}"))
            result.append((src, f"源图: {Path(src).name}"))
        return gr.update(value=result, columns=2)
    else:
        return gr.update(value=_with_labels(_gallery_images(batch_name)), columns=4)


def refresh_gallery(batch_name: str | None = None, mode: str = "浏览全部"):
    _source_cache.clear()
    choices = _batch_choices()
    value = batch_name or (choices[0] if choices else None)
    if mode == "源图对比":
        pairs = _gallery_with_sources(value or "")
        result: list = []
        for gen, src in pairs:
            result.append((gen, f"生成: {Path(gen).name}"))
            result.append((src, f"源图: {Path(src).name}"))
        return gr.update(choices=choices, value=value), gr.update(value=result, columns=2)
    else:
        return gr.update(choices=choices, value=value), gr.update(
            value=_with_labels(_gallery_images(value or "")), columns=4)


def refresh_eval_batch():
    choices = _batch_choices()
    value = choices[0] if choices else None
    return gr.update(choices=choices, value=value)


def run_generation(
    num_images: int,
    steps: int,
    strength: float,
    guidance_scale: float,
    seed: int,
    source_seed: int,
    filter_view: str,
    filter_density: str,
    prefix: str,
) -> str:
    command = [
        sys.executable,
        str(ROOT / "scripts/generation/run_mammo_sd15.py"),
        "--base-model-local", str(DEFAULT_MODEL),
        "--lora-path", str(DEFAULT_LORA),
        "--metadata-csv", str(METADATA_CSV),
        "--num-images", str(int(num_images)),
        "--num-steps", str(int(steps)),
        "--strength", str(float(strength)),
        "--guidance-scale", str(float(guidance_scale)),
        "--seed", str(int(seed)),
        "--output-subdir-prefix", prefix.strip() or "gradio_sd15",
        "--output-base", str(GENERATED_DIR),
        "--mode", "full-image",
        "--fullimage-output-long-side", str(DEFAULT_OUTPUT_LONG_SIDE),
    ]
    if filter_view and filter_view != "不限":
        command.extend(["--filter-view", filter_view])
    if filter_density and filter_density != "不限":
        command.extend(["--filter-density", filter_density])
    if source_seed and int(source_seed) > 0:
        command.extend(["--source-seed", str(int(source_seed))])
    return _run(command)


def run_pipeline(
    tag_prefix: str,
    num_images: int,
    steps: int,
    strength: float,
    guidance_scale: float,
    seed: int,
    filter_view: str,
    filter_density: str,
    eval_profile: str,
    from_latest_tuning: bool,
) -> str:
    command = [
        sys.executable,
        str(ROOT / "scripts/assistant/run_full_report.py"),
        "--tag-prefix", tag_prefix.strip() or "gradio_pipeline",
        "--base-model-local", str(DEFAULT_MODEL),
        "--lora-path", str(DEFAULT_LORA),
        "--metadata-csv", str(METADATA_CSV),
        "--num-images", str(int(num_images)),
        "--mode", "full-image",
        "--num-steps", str(int(steps)),
        "--strength", str(float(strength)),
        "--guidance-scale", str(float(guidance_scale)),
        "--seed", str(int(seed)),
        "--eval-profile", eval_profile,
        "--output-base", str(GENERATED_DIR),
        "--fullimage-output-long-side", str(DEFAULT_OUTPUT_LONG_SIDE),
    ]
    if filter_view and filter_view != "不限":
        command.extend(["--filter-view", filter_view])
    if filter_density and filter_density != "不限":
        command.extend(["--filter-density", filter_density])
    if from_latest_tuning:
        command.append("--from-latest-tuning")
    return _run(command)


def run_review(batch_name: str, output_name: str, top_k: int, eval_profile: str):
    if not batch_name:
        return "请先选择一个生成批次。", "未选择批次。", []
    batch = GENERATED_DIR / batch_name
    if not batch.exists():
        return f"批次目录不存在：{batch}", "批次目录不存在。", []
    safe_batch = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in batch_name)
    default_name = f"review_{safe_batch}_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir = EVAL_DIR / (output_name.strip() or default_name)
    command = [
        sys.executable,
        str(ROOT / "scripts/evaluation/review_generated_images.py"),
        "--images-dir", str(batch),
        "--no-recursive",
        "--output-dir", str(output_dir),
        "--top-k", str(int(top_k)),
        "--eval-profile", eval_profile,
        "--enable-seam-check",
    ]
    log = _run(command)
    return log, _eval_summary_markdown(output_dir), _eval_preview_images(batch_name, output_dir)


def refresh_param_history():
    return _param_history_rows()


def load_latest_next_run_into_tuning():
    """读取 LATEST_NEXT_RUN.json，返回各控件更新值（供调参历史 Tab 使用）。"""
    path = REPORT_DIR / "LATEST_NEXT_RUN.json"
    if not path.is_file():
        return "未找到 LATEST_NEXT_RUN.json", {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        params = data.get("parameters", {})
        summary = (
            f"来源：{data.get('source_tag', '未知')}\n"
            f"更新时间：{data.get('updated_at', '未知')}\n"
            f"评估模式：{data.get('eval_mode', '未知')}\n"
            f"源图种子：{data.get('source_seed', '未知')}\n\n"
            f"参数建议：\n{json.dumps(params, ensure_ascii=False, indent=2)}"
        )
        return summary
    except Exception as exc:
        return f"读取失败：{exc}"


# ── UI 配置 ──────────────────────────────────────────────────────────────────

_VIEW_CHOICES = ["不限", "MLO", "CC"]
_DENSITY_CHOICES = ["不限", "fatty", "scattered", "heterogeneous", "dense"]
_EVAL_PROFILES = ["full"]

# ── 莫兰蒂医用蓝主题 ─────────────────────────────────────────────────────────
_THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    font_mono=["JetBrains Mono", "Consolas", "monospace"],
).set(
    # 背景
    body_background_fill="#EDF2F7",
    block_background_fill="#FFFFFF",
    # 边框
    border_color_primary="#C2D6E8",
    block_border_width="1px",
    # 主按钮
    button_primary_background_fill="#4A7FA5",
    button_primary_background_fill_hover="#2E5F82",
    button_primary_text_color="#FFFFFF",
    button_primary_border_color="#4A7FA5",
    # 次按钮
    button_secondary_background_fill="#FFFFFF",
    button_secondary_background_fill_hover="#E8F1F8",
    button_secondary_border_color="#C2D6E8",
    button_secondary_text_color="#4A7FA5",
    # 输入框
    input_background_fill="#FAFCFE",
    input_border_color="#C2D6E8",
    input_border_color_focus="#4A7FA5",
    input_shadow_focus="0 0 0 3px rgba(74,127,165,0.15)",
    # 滑条
    slider_color="#4A7FA5",
    # 标签文字
    block_label_text_color="#3A5470",
    block_label_text_weight="500",
    block_title_text_color="#2C4260",
    block_title_text_weight="600",
    # Accordion
    accordion_text_color="#3A5470",
    # 阴影
    block_shadow="0 1px 4px rgba(74,127,165,0.08)",
)

_CSS = """
/* === Morandie Medical Blue · Custom Overrides === */

/* ─ 页面背景 ─ */
body { background: #EDF2F7 !important; }
.gradio-container { max-width: 1180px !important; margin: 0 auto !important; }

/* ─ 顶栏 ─ */
.app-header {
    background: linear-gradient(135deg, #2C4260 0%, #4A7FA5 60%, #6BA3B8 100%);
    border-radius: 12px;
    padding: 22px 28px 20px;
    margin-bottom: 4px;
    box-shadow: 0 4px 16px rgba(44,66,96,0.22);
}
.app-header h1 {
    margin: 0; color: #fff !important;
    font-size: 1.45rem; font-weight: 600; letter-spacing: 0.01em;
}
.app-header .sub {
    color: rgba(255,255,255,0.72) !important;
    font-size: 0.83rem; margin-top: 5px; letter-spacing: 0.02em;
}
.app-header .source {
    color: rgba(255,255,255,0.78) !important;
    font-size: 0.78rem; margin-top: 6px;
}
.app-header .source a {
    color: #FFFFFF !important; text-decoration: underline;
    text-underline-offset: 2px;
}
.tag-pill {
    display: inline-block; background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.30); color: rgba(255,255,255,0.88) !important;
    border-radius: 20px; padding: 2px 10px; font-size: 0.75rem;
    margin-right: 6px; margin-top: 8px; letter-spacing: 0.02em;
}

/* ─ Tab 导航 ─ */
.tab-nav { border-bottom: 1px solid #C2D6E8 !important; gap: 2px !important; }
button[role="tab"] {
    color: #7A9BB5 !important; font-weight: 500 !important;
    font-size: 0.86rem !important; padding: 9px 16px !important;
    border-radius: 6px 6px 0 0 !important;
    border: none !important; background: transparent !important;
    transition: color .18s, background .18s !important;
}
button[role="tab"]:hover { color: #4A7FA5 !important; background: #E8F1F8 !important; }
button[role="tab"][aria-selected="true"] {
    color: #2C4260 !important; font-weight: 600 !important;
    background: #FFFFFF !important;
    box-shadow: inset 0 -2px 0 #4A7FA5 !important;
}

/* ─ 卡片块 ─ */
.block { border-radius: 8px !important; }

/* ─ 主按钮悬停动画 ─ */
button.primary {
    border-radius: 7px !important; font-weight: 500 !important;
    letter-spacing: 0.01em !important;
    transition: all .18s ease !important;
}
button.primary:hover { transform: translateY(-1px) !important; }
button.primary:active { transform: translateY(0) !important; }

/* ─ 次按钮 ─ */
button.secondary {
    border-radius: 7px !important; font-weight: 500 !important;
    transition: all .18s ease !important;
}

/* ─ 日志/Textbox ─ */
.log-box textarea {
    font-family: 'JetBrains Mono', 'Fira Code', Consolas, monospace !important;
    font-size: 0.795rem !important; line-height: 1.55 !important;
    background: #F5F9FD !important; color: #2C4260 !important;
    overflow-y: auto !important;
}

/* ─ Dataframe 表头 ─ */
table thead tr { background: #D8EAF4 !important; }
table thead th {
    color: #2C4260 !important; font-weight: 600 !important;
    font-size: 0.78rem !important; padding: 8px 10px !important;
}
table tbody tr:nth-child(even) { background: #F2F8FC !important; }
table tbody tr:hover { background: #E4F0F8 !important; }

/* ─ 区块标题 markdown ─ */
.prose h3 {
    color: #2C4260 !important; font-weight: 600 !important;
    font-size: 1rem !important;
    border-left: 3px solid #4A7FA5; padding-left: 10px;
    margin-bottom: 14px !important;
}
.prose h4 { color: #4A7FA5 !important; font-size: 0.9rem !important; }
.prose blockquote {
    border-left: 3px solid #A8C8DC !important;
    background: #EBF4FA !important;
    border-radius: 0 6px 6px 0 !important;
    padding: 7px 14px !important; margin: 8px 0 !important;
    color: #5A7A90 !important; font-size: 0.85rem !important;
}

/* ─ Accordion ─ */
.accordion > .label-wrap {
    background: #F0F6FA !important;
    border-bottom: 1px solid #C2D6E8 !important;
    border-radius: 6px !important;
}
.accordion.open > .label-wrap { border-radius: 6px 6px 0 0 !important; }

/* ─ 滚动条 ─ */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #EDF2F7; }
::-webkit-scrollbar-thumb { background: #A8C4DA; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #4A7FA5; }

/* ─ 画廊 ─ */
.gallery-item img { border-radius: 6px !important; }
/* 源图对比模式（2列）：列间距小（同组紧凑），行间距大（组间分隔） */
.gallery-pairs .grid-container {
    column-gap: 6px !important;
    row-gap: 18px !important;
}

/* ─ Radio/Checkbox 选中色 ─ */
input[type="radio"], input[type="checkbox"] { accent-color: #4A7FA5 !important; }
"""

# ── UI 构建 ──────────────────────────────────────────────────────────────────

with gr.Blocks(title="乳腺钼靶扩散生成系统") as demo:
    gr.HTML("""
    <div class="app-header">
      <h1>乳腺钼靶扩散生成系统</h1>
      <div class="sub">Mammography Synthesis · SD&nbsp;1.5&nbsp;+&nbsp;LoRA&nbsp;Fine-tuning</div>
      <div class="source">训练数据集来源：<a href="https://github.com/sposso/CBIS-DDSM-DATASET" target="_blank" rel="noopener noreferrer">CBIS-DDSM-DATASET</a></div>
      <div>
        <span class="tag-pill">CBIS-DDSM</span>
        <span class="tag-pill">LoRA</span>
        <span class="tag-pill">视觉（Qwen-VL）+ 指标（DeepSeek）评估体系</span>
      </div>
    </div>
    """)

    # ── Tab 1: 生成 ──────────────────────────────────────────────────────────
    with gr.Tab("生成"):
        with gr.Row():
            g_filter_view = gr.Dropdown(
                _VIEW_CHOICES, value="MLO", label="体位筛选"
            )
            g_filter_density = gr.Dropdown(
                _DENSITY_CHOICES, value="scattered", label="密度筛选"
            )
            g_num_images = gr.Slider(1, 20, value=6, step=1, label="生成数量")

        with gr.Row():
            g_steps = gr.Slider(5, 100, value=40, step=1, label="采样步数", visible=False)
            g_strength = gr.Slider(0.05, 0.95, value=0.44, step=0.01, label="strength", visible=False)
            g_guidance = gr.Slider(1.0, 15.0, value=7.5, step=0.1, label="guidance scale", visible=False)

        with gr.Accordion("高级采样参数", open=False):
            with gr.Row():
                g_steps_v = gr.Slider(5, 100, value=40, step=1, label="采样步数")
                g_strength_v = gr.Slider(0.05, 0.95, value=0.44, step=0.01, label="strength")
                g_guidance_v = gr.Slider(1.0, 15.0, value=7.5, step=0.1, label="guidance scale")
            with gr.Row():
                g_seed = gr.Number(value=2026, precision=0, label="生成种子 seed")
                g_source_seed = gr.Number(value=0, precision=0, label="源图种子 source-seed（0=随机）")
        with gr.Row():
            g_prefix = gr.Textbox(value="gradio_sd15", label="输出目录前缀")
        g_btn = gr.Button("开始生成", variant="primary")
        g_log = gr.Textbox(label="生成日志", lines=16, elem_classes=["log-box"])

        # Accordion 滑条同步回隐藏滑条
        g_steps_v.change(fn=lambda v: v, inputs=[g_steps_v], outputs=[g_steps])
        g_strength_v.change(fn=lambda v: v, inputs=[g_strength_v], outputs=[g_strength])
        g_guidance_v.change(fn=lambda v: v, inputs=[g_guidance_v], outputs=[g_guidance])

        g_btn.click(
            fn=run_generation,
            inputs=[
                g_num_images, g_steps, g_strength, g_guidance,
                g_seed, g_source_seed,
                g_filter_view, g_filter_density,
                g_prefix,
            ],
            outputs=[g_log],
        )

    # ── Tab 2: 一键流水线 ────────────────────────────────────────────────────
    with gr.Tab("一键流水线"):
        gr.Markdown(
            "### 生成 → 评估 → 顾问报告\n"
            "> 整个流程在后台同步运行，完成后在日志中查看结果路径。",
            elem_classes=["prose"],
        )
        with gr.Row():
            p_filter_view = gr.Dropdown(_VIEW_CHOICES, value="MLO", label="体位")
            p_filter_density = gr.Dropdown(_DENSITY_CHOICES, value="scattered", label="密度")
            p_num_images = gr.Slider(1, 20, value=6, step=1, label="生成数量")
        with gr.Row():
            p_steps = gr.Slider(5, 100, value=40, step=1, label="采样步数", visible=False)
            p_strength = gr.Slider(0.05, 0.95, value=0.44, step=0.01, label="strength", visible=False)
            p_guidance = gr.Slider(1.0, 15.0, value=7.5, step=0.1, label="guidance", visible=False)
        with gr.Accordion("采样参数", open=False):
            with gr.Row():
                p_steps_v = gr.Slider(5, 100, value=40, step=1, label="采样步数")
                p_strength_v = gr.Slider(0.05, 0.95, value=0.44, step=0.01, label="strength")
                p_guidance_v = gr.Slider(1.0, 15.0, value=7.5, step=0.1, label="guidance scale")
            p_seed = gr.Number(value=2026, precision=0, label="seed")
        with gr.Row():
            p_tag_prefix = gr.Textbox(value="gradio_pipeline", label="报告标签前缀")
            p_eval_profile = gr.Dropdown(_EVAL_PROFILES, value="full", label="评估 Profile")
        with gr.Row():
            p_from_latest = gr.Checkbox(value=False, label="从 LATEST_NEXT_RUN.json 加载参数")
        p_btn = gr.Button("启动一键流水线", variant="primary")
        p_log = gr.Textbox(label="流水线日志", lines=18, elem_classes=["log-box"])

        p_steps_v.change(fn=lambda v: v, inputs=[p_steps_v], outputs=[p_steps])
        p_strength_v.change(fn=lambda v: v, inputs=[p_strength_v], outputs=[p_strength])
        p_guidance_v.change(fn=lambda v: v, inputs=[p_guidance_v], outputs=[p_guidance])

        p_btn.click(
            fn=run_pipeline,
            inputs=[
                p_tag_prefix, p_num_images, p_steps,
                p_strength, p_guidance, p_seed,
                p_filter_view, p_filter_density,
                p_eval_profile, p_from_latest,
            ],
            outputs=[p_log],
        )

    # ── Tab 3: 评估 ──────────────────────────────────────────────────────────
    with gr.Tab("评估"):
        with gr.Row():
            r_batch = gr.Dropdown(
                label="待评估批次",
                choices=_batch_choices(),
                value=_batch_choices()[0] if _batch_choices() else None,
            )
            r_refresh_btn = gr.Button("刷新批次列表")
        with gr.Row():
            r_output = gr.Textbox(value="", label="评估输出目录名（留空自动）")
            r_eval_profile = gr.Dropdown(_EVAL_PROFILES, value="full", label="评估模式")
            r_top_k = gr.Slider(1, 200, value=30, step=1, label="推荐图数量")
        r_btn = gr.Button("开始评估", variant="primary")
        r_summary = gr.Markdown("选择批次后点击开始评估。", elem_classes=["prose"])
        r_preview = gr.Gallery(label="评估排序预览", columns=4, height=420)
        with gr.Accordion("运行日志", open=False):
            r_log = gr.Textbox(label="评估日志", lines=14, elem_classes=["log-box"])
        r_refresh_btn.click(fn=refresh_eval_batch, inputs=[], outputs=[r_batch])
        r_btn.click(
            fn=run_review,
            inputs=[r_batch, r_output, r_top_k, r_eval_profile],
            outputs=[r_log, r_summary, r_preview],
        )

    # ── Tab 4: 画廊 ──────────────────────────────────────────────────────────
    with gr.Tab("画廊"):
        gr.Markdown("### 浏览生成批次图像", elem_classes=["prose"])
        _init_choices = _batch_choices()
        _init_value = _init_choices[0] if _init_choices else None
        with gr.Row():
            gal_batch = gr.Dropdown(
                label="生成批次",
                choices=_init_choices,
                value=_init_value,
            )
            gal_mode = gr.Radio(
                ["浏览全部", "源图对比"],
                value="浏览全部",
                label="画廊模式",
            )
            gal_refresh_btn = gr.Button("刷新")
        gal_gallery = gr.Gallery(
            label="批次图像",
            value=_with_labels(_gallery_images(_init_value or "")),
            columns=4,
            height=600,
            object_fit="contain",
            allow_preview=True,
            elem_classes=["gallery-pairs"],
        )
        gal_refresh_btn.click(
            fn=refresh_gallery,
            inputs=[gal_batch, gal_mode],
            outputs=[gal_batch, gal_gallery],
        )
        gal_batch.change(
            fn=switch_gallery_mode,
            inputs=[gal_mode, gal_batch],
            outputs=[gal_gallery],
        )
        gal_mode.change(
            fn=switch_gallery_mode,
            inputs=[gal_mode, gal_batch],
            outputs=[gal_gallery],
        )

        # 精选导出
        with gr.Accordion("精选导出", open=False):
            gr.Markdown(
                "选择批次 → 勾选图片 → 一键导出到 `outputs/curated/`",
                elem_classes=["prose"],
            )
            curated_batch = gr.Dropdown(
                choices=_init_choices,
                value=_init_value,
                label="选择批次",
            )
            with gr.Row():
                curated_preview = gr.Gallery(
                    label="预览（勾号 = 已选）",
                    columns=4,
                    height=400,
                    object_fit="contain",
                    allow_preview=False,
                )
                curated_checklist = gr.CheckboxGroup(
                    choices=[],
                    value=[],
                    label="选择图片",
                )
            with gr.Row():
                curated_label = gr.Textbox(
                    value="",
                    placeholder="留空使用时间戳",
                    label="导出标签",
                )
                curated_export_btn = gr.Button("导出选中图片", variant="primary")
            curated_export_status = gr.Markdown("")

            curated_batch.change(
                fn=_on_curated_batch_change,
                inputs=[curated_batch],
                outputs=[curated_preview, curated_checklist],
            )
            curated_export_btn.click(
                fn=_export_curated,
                inputs=[curated_checklist, curated_batch, curated_label],
                outputs=[curated_export_status],
            )

    # ── Tab 5: 调参历史 ──────────────────────────────────────────────────────
    with gr.Tab("调参历史"):
        gr.Markdown(
            "### 顾问调参历史（`PARAM_HISTORY.json`）\n"
            "> 来自 `run_full_report.py` 的历次推荐参数与对应评估指标。",
            elem_classes=["prose"],
        )
        hist_refresh_btn = gr.Button("刷新历史")
        hist_table = gr.Dataframe(
            headers=[
                "轮次", "时间", "来源标签",
                "通过率", "严格通过率", "均分", "BRISQUE",
                "strength", "guidance", "steps",
                "备注（截60字）",
            ],
            value=_param_history_rows(),
            interactive=False,
        )
        gr.Markdown("#### 最新顾问建议参数（LATEST_NEXT_RUN.json）", elem_classes=["prose"])
        hist_load_btn = gr.Button("加载最新建议")
        hist_latest_out = gr.Textbox(label="最新建议内容", lines=14, elem_classes=["log-box"])
        hist_refresh_btn.click(fn=refresh_param_history, inputs=[], outputs=[hist_table])
        hist_load_btn.click(fn=load_latest_next_run_into_tuning, inputs=[], outputs=[hist_latest_out])

    # ── Tab 6: 评估汇总 ──────────────────────────────────────────────────────
    with gr.Tab("评估汇总"):
        gr.Markdown("### 历次评估结果汇总", elem_classes=["prose"])
        ev_refresh_btn = gr.Button("刷新")
        ev_table = gr.Dataframe(
            headers=[
                "评估目录", "图像数", "通过率", "均分", "BRISQUE",
                "A构图", "B灰度", "C纹理", "D伪影", "E分布", "F解剖",
                "否决数", "Tier分布", "主要扣分项",
            ],
            value=_review_rows(),
            interactive=False,
        )
        ev_refresh_btn.click(fn=_review_rows, inputs=[], outputs=[ev_table])

    # 精选导出默认加载首个批次
    _init_batch = _init_value or ""
    demo.load(
        fn=lambda: _on_curated_batch_change(_init_batch),
        inputs=[],
        outputs=[curated_preview, curated_checklist],
    )
    # 画廊默认加载最新批次
    demo.load(
        fn=lambda: refresh_gallery(None, "浏览全部"),
        inputs=[],
        outputs=[gal_batch, gal_gallery],
    )


# ── 入口 ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    )
    ap = argparse.ArgumentParser(description="乳腺钼靶 Gradio UI")
    ap.add_argument("--host", default="0.0.0.0", help="监听地址")
    ap.add_argument("--port", type=int, default=7860, help="监听端口")
    ap.add_argument("--share", action="store_true", help="创建公开 Gradio 链接")
    args = ap.parse_args()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share,
                theme=_THEME, css=_CSS)
