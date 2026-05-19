#!/usr/bin/env python3
"""Verify postprocess has been archived from main pipeline."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_gradio_postprocess_checkboxes_removed():
    """Postprocess checkboxes should no longer exist in Gradio UI."""
    tree = ast.parse((ROOT / "apps/app_gradio.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            for t in targets:
                assert t not in {"g_postprocess", "p_postprocess"}, \
                    f"Postprocess checkbox {t} still present in Gradio"


def test_generation_postprocess_block_removed():
    """Postprocess block should be removed from run_mammo_sd15.py."""
    tree = ast.parse((ROOT / "scripts/generation/run_mammo_sd15.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func_id = getattr(node.func, "id", "")
            assert func_id != "PostprocessParams", \
                "PostprocessParams still used in run_mammo_sd15.py"


def test_postprocess_params_enabled_defaults_false():
    """PostprocessParams.enabled should default to False."""
    from scripts.core.pipeline_config import PostprocessParams
    assert PostprocessParams().enabled is False


if __name__ == "__main__":
    test_gradio_postprocess_checkboxes_removed()
    test_generation_postprocess_block_removed()
    test_postprocess_params_enabled_defaults_false()
    print("All tests passed.")
