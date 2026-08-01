#!/usr/bin/env python3
"""Tests for Gradio / GenParams defaults that keep texture-safe img2img strength."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_gradio_default_strength_matches_mainline():
    """UI sliders should default to GenParams-aligned strength (not overly conservative)."""
    source = (ROOT / "apps/app_gradio.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    strengths: list[float] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "Slider":
                for kw in node.keywords:
                    if kw.arg == "label" and isinstance(kw.value, ast.Constant) and kw.value.value == "strength":
                        for kw2 in node.keywords:
                            if kw2.arg == "value" and isinstance(kw2.value, ast.Constant):
                                strengths.append(float(kw2.value.value))
    assert strengths, "No strength Slider defaults found in app_gradio.py"
    assert all(s >= 0.42 for s in strengths), strengths


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
    test_gradio_default_strength_matches_mainline()
    test_core_generation_defaults_match_texture_safe_preset()
    print("All tests passed.")
