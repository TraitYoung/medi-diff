#!/usr/bin/env python3
"""Tests for source artifact pre-filtering without importing CV dependencies."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_helper():
    tree = ast.parse((ROOT / "scripts/generation/run_mammo_sd15.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_has_source_artifact_burden":
            module = ast.Module(body=[node], type_ignores=[])
            ast.fix_missing_locations(module)
            ns: dict[str, object] = {}
            exec(compile(module, "<artifact-helper>", "exec"), ns)
            return ns[node.name]
    raise AssertionError("_has_source_artifact_burden helper not found")


def test_clean_source_is_kept():
    helper = _load_helper()
    assert not helper(
        marker_score=1,
        bg_marker_count=3,
        bg_marker_frac=0.001,
        fg_dot_count=3,
        texture_lap_p75=24,
        texture_grad_p75=22,
        circumscribed_mass_count=0,
        calc_cluster_count=0,
        calc_dot_count=0,
    )


def test_high_metadata_marker_score_is_rejected():
    helper = _load_helper()
    assert helper(
        marker_score=8,
        bg_marker_count=2,
        bg_marker_frac=0.001,
        fg_dot_count=2,
        texture_lap_p75=24,
        texture_grad_p75=22,
        circumscribed_mass_count=0,
        calc_cluster_count=0,
        calc_dot_count=0,
    )


def test_many_background_specks_are_rejected():
    helper = _load_helper()
    assert helper(
        marker_score=1,
        bg_marker_count=20,
        bg_marker_frac=0.004,
        fg_dot_count=2,
        texture_lap_p75=24,
        texture_grad_p75=22,
        circumscribed_mass_count=0,
        calc_cluster_count=0,
        calc_dot_count=0,
    )


def test_many_tissue_bright_dots_are_rejected():
    helper = _load_helper()
    assert helper(
        marker_score=1,
        bg_marker_count=2,
        bg_marker_frac=0.001,
        fg_dot_count=10,
        texture_lap_p75=24,
        texture_grad_p75=22,
        circumscribed_mass_count=0,
        calc_cluster_count=0,
        calc_dot_count=0,
    )


def test_low_texture_source_is_rejected():
    helper = _load_helper()
    assert helper(
        marker_score=1,
        bg_marker_count=2,
        bg_marker_frac=0.001,
        fg_dot_count=2,
        texture_lap_p75=10,
        texture_grad_p75=10,
        circumscribed_mass_count=0,
        calc_cluster_count=0,
        calc_dot_count=0,
    )


def test_low_laplacian_texture_source_is_rejected():
    helper = _load_helper()
    assert helper(
        marker_score=1,
        bg_marker_count=2,
        bg_marker_frac=0.001,
        fg_dot_count=2,
        texture_lap_p75=12,
        texture_grad_p75=16,
        circumscribed_mass_count=0,
        calc_cluster_count=0,
        calc_dot_count=0,
    )


def test_circumscribed_mass_like_source_is_rejected():
    helper = _load_helper()
    assert helper(
        marker_score=1,
        bg_marker_count=2,
        bg_marker_frac=0.001,
        fg_dot_count=2,
        texture_lap_p75=24,
        texture_grad_p75=22,
        circumscribed_mass_count=1,
        calc_cluster_count=0,
        calc_dot_count=0,
    )


def test_clustered_calcification_like_source_is_rejected():
    helper = _load_helper()
    assert helper(
        marker_score=1,
        bg_marker_count=2,
        bg_marker_frac=0.001,
        fg_dot_count=2,
        texture_lap_p75=24,
        texture_grad_p75=22,
        circumscribed_mass_count=0,
        calc_cluster_count=1,
        calc_dot_count=0,
    )


def test_many_calcification_like_dots_are_rejected():
    helper = _load_helper()
    assert helper(
        marker_score=1,
        bg_marker_count=2,
        bg_marker_frac=0.001,
        fg_dot_count=2,
        texture_lap_p75=24,
        texture_grad_p75=22,
        circumscribed_mass_count=0,
        calc_cluster_count=0,
        calc_dot_count=6,
    )


if __name__ == "__main__":
    test_clean_source_is_kept()
    test_high_metadata_marker_score_is_rejected()
    test_many_background_specks_are_rejected()
    test_many_tissue_bright_dots_are_rejected()
    test_low_texture_source_is_rejected()
    test_low_laplacian_texture_source_is_rejected()
    test_circumscribed_mass_like_source_is_rejected()
    test_clustered_calcification_like_source_is_rejected()
    test_many_calcification_like_dots_are_rejected()
    print("All tests passed.")
