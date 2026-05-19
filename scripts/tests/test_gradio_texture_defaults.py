#!/usr/bin/env python3
"""Tests for Gradio defaults that avoid smeared low-texture generations."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_fine_preset_uses_less_conservative_img2img_strength():
    source = (ROOT / "apps/app_gradio.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "_SPEED_PRESETS":
                    module = ast.Module(body=[node], type_ignores=[])
                    ast.fix_missing_locations(module)
                    ns: dict[str, object] = {}
                    exec(compile(module, "<gradio-presets>", "exec"), ns)
                    presets = ns["_SPEED_PRESETS"]
                    assert presets["精细 (50步)"]["strength"] >= 0.42
                    assert presets["快速 (20步)"]["strength"] >= 0.38
                    return
    raise AssertionError("_SPEED_PRESETS not found")


def test_core_generation_defaults_match_texture_safe_preset():
    source = (ROOT / "scripts/core/pipeline_config.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "GenParams":
            values = {}
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    if stmt.target.id in {"strength", "guidance_scale"}:
                        values[stmt.target.id] = ast.literal_eval(stmt.value)
            assert values["strength"] >= 0.42
            assert values["guidance_scale"] >= 6.8
            return
    raise AssertionError("GenParams not found")


if __name__ == "__main__":
    test_fine_preset_uses_less_conservative_img2img_strength()
    test_core_generation_defaults_match_texture_safe_preset()
    print("All tests passed.")
