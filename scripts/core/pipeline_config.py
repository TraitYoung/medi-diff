"""PipelineConfig: single source of truth for all generation parameters.

All parameter defaults live here. CLI argparse definitions in other modules should
reference these defaults rather than duplicating them.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "overlap_ratio": (0.4, 0.95),
    "strength": (0.05, 0.8),
    "global_guide_strength": (0, 0.6),
    "global_guide_blend": (0, 0.7),
    "blend_sigma_divisor": (1.05, 3),
    "guidance_scale": (1, 12),
    "gabor_alpha": (0.1, 0.6),
    "target_beta": (2, 4),
    "blend": (0.3, 1),
    "sharpen_strength": (0, 0.8),
}

ALLOWED_KEYS = frozenset({
    "seed", "blend", "notes_zh", "strength", "lora_path", "num_steps",
    "scheduler", "vl_n_best", "blend_mode", "no_qwen_vl", "num_images",
    "vl_n_worst", "gabor_alpha", "target_beta", "vl_max_side", "eval_profile",
    "overlap_ratio", "vl_max_images", "guidance_scale", "pyramid_levels",
    "sharpen_strength", "global_guide_blend", "blend_sigma_divisor",
    "global_guide_strength", "global_guide_cfg_scale",
})

INT_KEYS = frozenset({
    "seed", "num_steps", "vl_n_best", "num_images", "vl_n_worst",
    "vl_max_side", "vl_max_images", "pyramid_levels",
})


@dataclass
class GenParams:
    """SD1.5 + LoRA + patch-overlap generation parameters."""
    overlap_ratio: float = 0.85
    strength: float = 0.36
    guidance_scale: float = 6.5
    num_steps: int = 50
    scheduler: str = "dpm"
    patch_size: int = 640
    blend_mode: str = "hann"
    blend_sigma_divisor: float = 1.48
    gabor_alpha: float = 0.55
    global_guide: bool = True
    global_guide_strength: float = 0.25
    global_guide_scale: float = 0.3
    global_guide_blend: float = 0.44
    global_guide_cfg_scale: float = 7.5
    pyramid_levels: int = 4
    stride: int | None = None
    target_h: int = 768
    target_w: int = 1024
    tight_letterbox: bool = True
    tight_letterbox_min_frac: float = 0.06
    input_clahe: bool = True
    min_input_signal_frac: float = 0.015
    bg_min_signal_frac: float = 0.012
    bg_pixel_min: int = 4
    prompt: str = (
        "medical grayscale FFDM mammogram, sharp MLO anatomy, triangular "
        "pectoralis muscle, clear inframammary fold, directional "
        "fibroglandular strands toward nipple, fine ligament texture, "
        "no text, no labels"
    )
    negative_prompt: str = (
        "face, color, text, letters, numbers, labels, watermark, marker, "
        "metal marker, round white dots, circular blobs, random calcifications, "
        "watercolor smearing, cloudy texture, radial fan blur, cone blur, "
        "jagged edge, broken border, overexposed, mosaic, artifacts"
    )
    fullimage_long_side: int = 768
    fullimage_output_long_side: int = 2048


@dataclass
class LabelGuardParams:
    """DICOM label/annotation detection and erasure parameters."""
    enabled: bool = True
    preclean_border_labels: bool = False
    bright_border_mode: str = "cc"
    bright_border_frac: float = 0.028
    bright_border_pct: float = 99
    post_vl_all: bool = False
    canvas_edge_feather: int = 3
    bg_clean: bool = False


@dataclass
class PostprocessParams:
    """Frequency-domain and spatial postprocessing parameters."""
    enabled: bool = True
    winsorize: bool = True
    winsorize_low: float = 0.5
    winsorize_high: float = 99.5
    fill_voids: bool = True
    void_min_area: int = 30
    void_max_area: int = 3000
    void_circularity: float = 0.55
    no_freq: bool = False
    target_beta: float = 2.8
    blend: float = 0.7
    clahe: bool = True
    clahe_clip: float = 0.5
    clahe_grid: int = 8
    sharpen: bool = True
    sharpen_strength: float = 0.4
    sharpen_radius: int = 3
    bilateral: bool = False
    bilateral_d: int = 5
    bilateral_sigma_color: float = 30
    bilateral_sigma_space: float = 30
    edge_feather: bool = False
    feather_ksize: int = 5


@dataclass
class PipelineConfig:
    """Top-level configuration composed of sub-configs."""
    gen: GenParams = field(default_factory=GenParams)
    label_guard: LabelGuardParams = field(default_factory=LabelGuardParams)
    postprocess: PostprocessParams = field(default_factory=PostprocessParams)
    seed: int | None = None
    num_images: int = 8
    lora_path: str | None = None
    base_model: str = "runwayml/stable-diffusion-v1-5"
    base_model_local: str | None = None
    hf_endpoint: str | None = None
    output_base: str | None = None

    def to_cli_args(self) -> list[str]:
        """Convert to CLI argument list for run_mammo_sd15.py."""
        args: list[str] = []
        g = self.gen
        args += ["--overlap-ratio", str(g.overlap_ratio)]
        args += ["--strength", str(g.strength)]
        args += ["--guidance-scale", str(g.guidance_scale)]
        args += ["--num-steps", str(g.num_steps)]
        args += ["--scheduler", g.scheduler]
        args += ["--patch-size", str(g.patch_size)]
        args += ["--blend-mode", g.blend_mode]
        args += ["--blend-sigma-divisor", str(g.blend_sigma_divisor)]
        args += ["--gabor-alpha", str(g.gabor_alpha)]
        if g.global_guide:
            args += ["--global-guide-strength", str(g.global_guide_strength)]
            args += ["--global-guide-blend", str(g.global_guide_blend)]
            args += ["--global-guide-cfg-scale", str(g.global_guide_cfg_scale)]
        args += ["--pyramid-levels", str(g.pyramid_levels)]
        if g.stride is not None:
            args += ["--stride", str(g.stride)]
        args += ["--fullimage-long-side", str(g.fullimage_long_side)]
        args += ["--fullimage-output-long-side", str(g.fullimage_output_long_side)]
        if self.seed is not None:
            args += ["--seed", str(self.seed)]
        args += ["--num-images", str(self.num_images)]
        if self.lora_path:
            args += ["--lora-path", self.lora_path]
        if self.base_model_local:
            args += ["--base-model-local", self.base_model_local]
        elif self.base_model:
            args += ["--base-model", self.base_model]
        if self.output_base:
            args += ["--output-base", self.output_base]
        lg = self.label_guard
        if not lg.enabled:
            args.append("--no-legacy-label-guard")
        if lg.preclean_border_labels:
            args.append("--preclean-border-labels")
        if lg.bg_clean:
            args.append("--bg-clean")
        pp = self.postprocess
        if pp.enabled:
            args.append("--postprocess")
        return args

    @classmethod
    def from_argparse(cls, ns: argparse.Namespace) -> "PipelineConfig":
        """Build from argparse namespace (partial — only fields present in ns)."""
        g = GenParams()
        lg = LabelGuardParams()
        pp = PostprocessParams()
        cfg = cls(gen=g, label_guard=lg, postprocess=pp)
        for key, val in vars(ns).items():
            if val is None:
                continue
            k_under = key.replace("-", "_")
            if hasattr(g, k_under):
                setattr(g, k_under, val)
            elif hasattr(lg, k_under):
                setattr(lg, k_under, val)
            elif hasattr(pp, k_under):
                setattr(pp, k_under, val)
            elif hasattr(cfg, k_under):
                setattr(cfg, k_under, val)
        return cfg


def _apply_flat(config: PipelineConfig, d: dict[str, Any]) -> None:
    """Apply flat key-value pairs to a PipelineConfig."""
    gen_attrs = {f.name for f in GenParams.__dataclass_fields__.values()}
    lg_attrs = {f.name for f in LabelGuardParams.__dataclass_fields__.values()}
    pp_attrs = {f.name for f in PostprocessParams.__dataclass_fields__.values()}
    for key, val in d.items():
        if val is None:
            continue
        if key in gen_attrs:
            setattr(config.gen, key, val)
        elif key in lg_attrs:
            setattr(config.label_guard, key, val)
        elif key in pp_attrs:
            setattr(config.postprocess, key, val)
        elif hasattr(config, key):
            setattr(config, key, val)


def load_parameters_blob(path: Path | None = None) -> dict[str, Any]:
    """Read a next_run_parameters.json or LATEST_NEXT_RUN.json and extract the 'parameters' dict."""
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "parameters" in data:
        return dict(data["parameters"])
    if isinstance(data, dict):
        return {k: v for k, v in data.items() if k in ALLOWED_KEYS}
    return {}
