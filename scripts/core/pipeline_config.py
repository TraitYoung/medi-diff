"""GenParams: single source of truth for all generation parameters.

All parameter defaults live here. CLI argparse definitions in run_mammo_sd15.py
reference these defaults rather than duplicating them.

Removed (2026-05-19): LabelGuardParams, PostprocessParams, PipelineConfig,
PARAM_BOUNDS, ALLOWED_KEYS, INT_KEYS, to_cli_args, from_argparse, _apply_flat,
load_parameters_blob. These were only used by pipeline_orchestrator.py which has
also been removed — the active pipeline uses run_mammo_sd15.py directly, not
through the GenerationPipeline abstraction.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GenParams:
    """SD1.5 + LoRA generation parameters (full-image single-pass img2img)."""
    strength: float = 0.44
    guidance_scale: float = 7.5
    num_steps: int = 50
    scheduler: str = "dpm"
    prompt: str = (
        "medical grayscale FFDM mammogram, sharp MLO anatomy, triangular "
        "pectoralis muscle, clear inframammary fold, directional "
        "fibroglandular strands toward nipple, fine ligament texture, "
        "no text, no labels"
    )
    negative_prompt: str = (
        "face, color, text, letters, numbers, labels, watermark, marker, "
        "metal marker, round white dots, circular blobs, random calcifications, "
        "mass, tumor, lesion, nodule, fibroadenoma, circumscribed mass, "
        "microcalcification cluster, clustered calcifications, suspicious opacity, "
        "watercolor smearing, cloudy texture, radial fan blur, cone blur, "
        "jagged edge, broken border, overexposed, mosaic, artifacts"
    )
    fullimage_long_side: int = 768
    fullimage_output_long_side: int = 2048
