"""Unit tests for GenParams defaults."""
from __future__ import annotations

from scripts.core.pipeline_config import GenParams


def test_genparams_defaults_match_mainline():
    p = GenParams()
    assert p.strength == 0.44
    assert p.guidance_scale == 7.5
    assert p.num_steps == 40
    assert p.scheduler == "dpm"
    assert p.fullimage_long_side == 768
    assert p.fullimage_output_long_side == 2048


def test_genparams_prompts_discourage_text_and_lesions():
    p = GenParams()
    assert "no text" in p.prompt.lower() or "no labels" in p.prompt.lower()
    assert "text" in p.negative_prompt.lower()
    assert "lesion" in p.negative_prompt.lower() or "mass" in p.negative_prompt.lower()
