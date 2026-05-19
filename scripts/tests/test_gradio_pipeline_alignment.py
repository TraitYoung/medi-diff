#!/usr/bin/env python3
"""Regression tests for Gradio generation/pipeline parameter alignment."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _run_pipeline_string_literals() -> set[str]:
    tree = ast.parse((ROOT / "apps/app_gradio.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "run_pipeline":
            return {n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    raise AssertionError("run_pipeline function not found")


def _function_string_literals(function_name: str) -> set[str]:
    tree = ast.parse((ROOT / "apps/app_gradio.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return {n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    raise AssertionError(f"{function_name} function not found")


def test_pipeline_uses_same_model_inputs_as_direct_generation():
    """Pipeline launch should not silently fall back to older model defaults."""
    literals = _run_pipeline_string_literals()
    assert "--base-model-local" in literals
    assert "--lora-path" in literals
    assert "--metadata-csv" in literals


def test_full_report_default_lora_matches_current_gradio_default():
    """CLI pipeline defaults should use the current v6 LoRA, not the older v4 path."""
    app_tree = ast.parse((ROOT / "apps/app_gradio.py").read_text(encoding="utf-8"))
    report_text = (ROOT / "scripts/assistant/run_full_report.py").read_text(encoding="utf-8")
    app_literals = {
        n.value
        for n in ast.walk(app_tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    }
    current_lora = "outputs/lora/mammo_sd15_v6_allMLO/final_lora"
    assert current_lora in app_literals
    assert current_lora in report_text
    assert "outputs/lora/mammo_sd15_v4_clean/final_lora" not in report_text


def test_source_quality_sort_archived_from_gradio():
    """Gradio generation should no longer pass source-quality-sort flags."""
    literals = _function_string_literals("run_generation")
    assert "--source-quality-sort" not in literals
    assert "--no-source-quality-sort" not in literals


def test_source_quality_sort_archived_from_pipeline():
    """Gradio pipeline should no longer pass source-quality-sort flags."""
    literals = _function_string_literals("run_pipeline")
    assert "--source-quality-sort" not in literals
    assert "--no-source-quality-sort" not in literals


def test_postprocess_archived_from_gradio_generation():
    """Gradio generation should no longer pass postprocess flags."""
    literals = _function_string_literals("run_generation")
    assert "--postprocess" not in literals
    assert "--no-postprocess" not in literals


def test_postprocess_archived_from_gradio_pipeline():
    """Gradio pipeline should no longer pass postprocess flags."""
    literals = _function_string_literals("run_pipeline")
    assert "--postprocess" not in literals
    assert "--no-postprocess" not in literals


if __name__ == "__main__":
    test_pipeline_uses_same_model_inputs_as_direct_generation()
    test_full_report_default_lora_matches_current_gradio_default()
    test_source_quality_sort_archived_from_gradio()
    test_source_quality_sort_archived_from_pipeline()
    test_postprocess_archived_from_gradio_generation()
    test_postprocess_archived_from_gradio_pipeline()
    print("All tests passed.")
