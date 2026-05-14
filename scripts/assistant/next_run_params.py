#!/usr/bin/env python3
"""
从顾问正文中解析「下一轮可调参数」JSON，并做范围裁剪，保证闭环优化时方向可控。

顾问须在回答最后输出唯一 ```json 代码块（见 run_generate_eval_advise 中 ADVISOR_SYSTEM_TEXT / ADVISOR_SYSTEM_VISION）。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# Import from shared core library (single source of truth)
from scripts.core.pipeline_config import ALLOWED_KEYS, INT_KEYS, PARAM_BOUNDS as BOUNDS


def _clamp(name: str, v: float) -> float:
    if name in BOUNDS:
        lo, hi = BOUNDS[name]
        return max(lo, min(hi, float(v)))
    return float(v)


def normalize_parameters(raw: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in raw.items():
        if k not in ALLOWED_KEYS:
            continue
        if v is None:
            continue
        if k == "notes_zh":
            out[k] = str(v)[:2000]
            continue
        if k == "lora_path":
            out[k] = str(v).strip()
            continue
        if k == "eval_profile":
            s = str(v).strip().lower()
            if s in ("full", "patch"):
                out[k] = s
            continue
        if k == "no_qwen_vl":
            out[k] = bool(v)
            continue
        if k in INT_KEYS:
            try:
                iv = int(round(float(v)))
            except (TypeError, ValueError):
                continue
            if k == "num_steps":
                iv = max(15, min(60, iv))
            if k == "vl_max_side":
                iv = max(256, min(1536, iv))
            out[k] = iv
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if k in BOUNDS:
            out[k] = round(_clamp(k, fv), 4)
        else:
            out[k] = fv
    return out


def parse_advisor_response(text: str) -> dict[str, Any] | None:
    """从顾问全文提取最后一个可解析的 JSON 对象（通常为 ```json 块）。"""
    if not text or not text.strip():
        return None
    candidates: list[dict[str, Any]] = []
    for m in re.finditer(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE):
        chunk = m.group(1).strip()
        if not chunk.startswith("{"):
            continue
        try:
            obj = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            candidates.append(obj)
    if not candidates:
        # 兜底：全文里第一个 { ... } 行块
        m2 = re.search(r"\{[\s\S]*\}\s*$", text.strip())
        if m2:
            try:
                obj = json.loads(m2.group(0))
                if isinstance(obj, dict):
                    candidates.append(obj)
            except json.JSONDecodeError:
                pass
    if not candidates:
        return None
    merged: dict[str, Any] = {}
    for c in candidates:
        merged.update(c)
    norm = normalize_parameters(merged)
    return norm if norm else None


def merge_lora_path_if_invalid(
    params: dict[str, Any],
    root: Path,
    fallback: Path | None,
) -> dict[str, Any]:
    """顾问 JSON 常编造占位 lora 路径；若解析出的路径在仓库内不存在，则回退为本次生成实际使用的路径。"""
    out = dict(params)
    if fallback is None or not (fallback.is_file() or fallback.is_dir()):
        return out
    fb = str(fallback.resolve())
    lp = out.get("lora_path")
    if not lp or not str(lp).strip():
        out["lora_path"] = fb
        return out
    p = Path(str(lp).strip())
    if not p.is_absolute():
        p = (root / p).resolve()
    else:
        p = p.resolve()
    if not (p.is_file() or p.is_dir()):
        out["lora_path"] = fb
    else:
        # 若检测到旧版 LoRA，自动升级到推荐版本
        _v4 = root / "outputs/lora/mammo_sd15_v4_clean/final_lora"
        _v4_safe = _v4 / "adapter_model.safetensors"
        _name = str(p)
        if _v4_safe.is_file() and ("v3" in _name or "v2" in _name or "v1" in _name):
            out["lora_path"] = str(_v4.resolve())
    return out


def merge_with_vl_priority(
    text_params: dict[str, Any],
    vl_params: dict[str, Any],
    vl_weight: float = 0.6,
) -> dict[str, Any]:
    """当 both 模式下文本顾问与 VL 顾问均有有效 JSON 时，视觉影响参数以 VL 为准。

    视觉敏感参数（VL 高权重，vl_weight 默认 0.6）：
      overlap_ratio, gabor_alpha, sharpen_strength, blend_sigma_divisor
    中性参数（50/50 平均）：
      strength, guidance_scale, num_steps, global_guide_strength,
      global_guide_blend, target_beta, blend
    """
    visual_keys = {"overlap_ratio", "gabor_alpha", "sharpen_strength", "blend_sigma_divisor"}
    out: dict[str, Any] = {}

    all_keys = set(text_params.keys()) | set(vl_params.keys())
    for k in all_keys:
        if k not in ALLOWED_KEYS:
            continue
        tv = text_params.get(k)
        vv = vl_params.get(k)
        if tv is None and vv is None:
            continue
        if tv is None:
            out[k] = vv
        elif vv is None:
            out[k] = tv
        elif k in visual_keys:
            # VL 加权混合：偏向 VL 建议
            if k in BOUNDS:
                lo, hi = BOUNDS[k]
                blended = float(vv) * vl_weight + float(tv) * (1.0 - vl_weight)
                out[k] = round(max(lo, min(hi, blended)), 4)
            else:
                out[k] = float(vv) * vl_weight + float(tv) * (1.0 - vl_weight)
        else:
            # 中性参数取平均
            try:
                out[k] = round((float(tv) + float(vv)) / 2.0, 4)
            except (TypeError, ValueError):
                out[k] = vv if vv is not None else tv
    return out


def write_next_run_file(eval_dir: Path, params: dict[str, Any]) -> Path:
    path = eval_dir / "next_run_parameters.json"
    payload = {
        "_meta": {
            "description": "由顾问输出解析；供 run_full_report.py --from-latest-tuning 使用",
            "schema": sorted(ALLOWED_KEYS),
        },
        "parameters": params,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
