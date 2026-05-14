#!/usr/bin/env python3
"""将 next_run_parameters.json 中的 parameters 合并到 argparse Namespace（用于 run_full_report）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.core.pipeline_config import load_parameters_blob


def apply_to_namespace(root: Path, ns: Any, params: dict[str, Any]) -> list[str]:
    """把非 null 字段写入 ns（仅当 hasattr）。返回人类可读的应用列表。"""
    applied: list[str] = []
    for key, val in params.items():
        if val is None or key == "notes_zh":
            continue
        if key == "lora_path":
            p = Path(str(val))
            if not p.is_absolute():
                p = (root / p).resolve()
            if p.is_file() or p.is_dir():
                setattr(ns, "lora_path", p)
                applied.append(f"lora_path={p}")
            else:
                applied.append(f"lora_path(跳过无效路径，保留命令行/默认)={p}")
            continue
        if key == "no_qwen_vl":
            if val is True:
                setattr(ns, "no_qwen_vl", True)
                applied.append("no_qwen_vl=True")
            continue
        if not hasattr(ns, key):
            continue
        setattr(ns, key, val)
        applied.append(f"{key}={val}")
    return applied
