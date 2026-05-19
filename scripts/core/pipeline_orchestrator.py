'''GenerationPipeline: composable orchestrator for the mammography generation pipeline.

Assembles SDPipeline, LabelGuard, and postprocessing into a single interface where
each stage can be individually enabled/disabled or tested in isolation.

Usage:
    from scripts.core.pipeline_config import PipelineConfig
    from scripts.core.pipeline_orchestrator import GenerationPipeline

    config = PipelineConfig()
    pipeline = GenerationPipeline(config)
    pipeline.load_model()

    # Full generation
    result = pipeline.generate_image(src_gray)

    # Batch
    for result in pipeline.generate_batch(sources):
        result.save(...)

    # Verify labels on already-generated images (no GPU needed)
    verdicts = pipeline.verify_labels(list_of_image_paths)

    # Apply postprocess only (no GPU needed)
    pipeline.apply_postprocess(input_dir, output_dir)
'''
from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from scripts.core.generation_pipeline import SDPipeline
from scripts.core.label_guard import clean_background, erase_background_labels, erase_bright_border_labels, feather_canvas_edge
from scripts.core.pipeline_config import PipelineConfig

class GenerationPipeline:
    '''Composable mammography generation pipeline.

    Each stage is independently toggled via PipelineConfig flags. You can run:
    - pipeline.generate_image(src)              full flow
    - pipeline.generate_only(src)                skip label guard + postprocess
    - pipeline.verify_labels(image_paths)        label check only, no GPU
    - pipeline.apply_postprocess(input_dir, ...) postprocess only, no GPU
    '''
    
    def __init__(self, config = None):
        self.config = config
        self._sd = None
        self._generator_created = False

    
    def load_model(self):
        '''Load SD1.5 + LoRA model (~10s on GPU).'''
        self._sd = SDPipeline(base_model_local = self.config.base_model_local, base_model = self.config.base_model, lora_path = self.config.lora_path, hf_endpoint = self.config.hf_endpoint)
        self._sd._ensure_model_loaded()
        self._generator_created = True

    @property
    def sd(self):
        return self._sd
    
    def _pre_filter(self, gray = None):
        '''Label guard: pre-generation source filtering.'''
        if not self.config.label_guard.enabled:
            return gray
        lg = self.config.label_guard
        if lg.preclean_border_labels:
            gray = erase_bright_border_labels(gray, border_frac = lg.bright_border_frac, bright_pct = lg.bright_border_pct, mode = lg.bright_border_mode)
        return gray

    
    def _post_filter(self, gray = None):
        '''Label guard: post-generation output filtering.'''
        if not self.config.label_guard.enabled:
            return gray
        lg = self.config.label_guard
        gray = erase_background_labels(gray)
        gray = erase_bright_border_labels(gray, border_frac = lg.bright_border_frac, bright_pct = lg.bright_border_pct, mode = lg.bright_border_mode)
        if lg.bg_clean:
            gray = clean_background(gray)
        if lg.canvas_edge_feather > 0:
            gray = feather_canvas_edge(gray, feather_px = lg.canvas_edge_feather)
        return gray

    
    def _run_sd(self, src_gray=None, prompt=None, negative_prompt: str = ""):
        '''Dispatch to full-image or patch-overlap mode based on config.'''
        g = self.config.gen
        seed = self.config.seed or 0
        mode = getattr(g, 'mode', 'full-image')
        if mode == 'full-image':
            result = self.sd.generate(
                src_gray=src_gray, prompt=prompt, negative_prompt=negative_prompt,
                strength=g.strength, guidance_scale=g.guidance_scale,
                num_inference_steps=g.num_steps, mode='full-image',
                fullimage_long_side=getattr(g, 'fullimage_long_side', 768),
                fullimage_output_long_side=getattr(g, 'fullimage_output_long_side', 2048),
                gabor_alpha=g.gabor_alpha, base_seed=seed,
            )
            from scripts.generation.run_mammo_sd15 import _apply_upscale
            result = _apply_upscale(result, getattr(g, 'upscale', 'none'), getattr(g, 'upscale_factor', 2))
            output_long_side = int(getattr(g, 'fullimage_output_long_side', 2048))
            if output_long_side > 0:
                from scripts.generation.run_mammo_sd15 import resize_long_side_gray
                result = resize_long_side_gray(result, output_long_side)
            return result
        global_guide_gray = None
        if g.global_guide:
            global_guide_gray = self.sd.run_global_guide(
                src_gray, prompt=prompt, negative_prompt=negative_prompt,
                strength=g.global_guide_strength, guide_scale=g.global_guide_scale,
                num_steps=g.num_steps, seed=seed + 17,
            )
        H, W = src_gray.shape
        latent_field = self.sd.build_latent_smooth_field(H, W, seed=seed + 91)
        return self.sd.generate(
            src_gray=src_gray, prompt=prompt, negative_prompt=negative_prompt,
            strength=g.strength, guidance_scale=g.guidance_scale,
            num_inference_steps=g.num_steps, mode='patch',
            patch_size=g.patch_size, stride=self._compute_stride(),
            gabor_alpha=g.gabor_alpha, blend_mode=g.blend_mode,
            blend_sigma_divisor=g.blend_sigma_divisor,
            global_guide_gray=global_guide_gray,
            global_guide_blend=g.global_guide_blend,
            latent_smooth_field=latent_field, base_seed=seed,
            bg_min_signal_frac=g.bg_min_signal_frac,
            bg_pixel_min=g.bg_pixel_min, pyramid_levels=g.pyramid_levels,
        )

    
    def generate_image(self, src_gray = None, *, prompt, negative_prompt):
        '''Full pipeline on a single source image: pre-filter → SD → post-filter → postprocess.

        Args:
            src_gray: (H, W) uint8 grayscale source (already letterboxed to target size)

        Returns:
            (H, W) uint8 grayscale result
        '''
        src = self._pre_filter(src_gray)
        if not prompt:
            prompt = self.config.gen.prompt
        if not negative_prompt:
            negative_prompt = self.config.gen.negative_prompt
        result = self._run_sd(src, prompt, negative_prompt)
        result = self._post_filter(result)
        if self.config.postprocess.enabled:
            from archive.core.postprocess_pipeline import run_postprocess_on_image
            result = run_postprocess_on_image(result, self.config.postprocess)
        return result

    
    def generate_only(self, src_gray = None):
        '''Generation only: no label guard, no postprocess.'''
        g = self.config.gen
        return self._run_sd(src_gray, g.prompt, g.negative_prompt)

    
    def verify_labels(self, image_paths=None):
        '''Run label guard heuristic on already-generated images. No GPU needed.

        Returns list of {path, verdict, score, confidence} dicts.
        '''
        from scripts.preprocessing.mammo_label_heuristic import compute_label_heuristic
        results = []
        if not image_paths:
            return results
        for path in image_paths:
            heuristic = compute_label_heuristic(path)
            results.append({
                'path': str(path),
                'verdict': heuristic.get('verdict', 'unknown'),
                'score': heuristic.get('score', 0.0),
                'confidence': heuristic.get('confidence', 0.0),
            })
        return results

    
    def apply_postprocess(self, input_dir = None, output_dir = None):
        '''Apply postprocessing to an existing image directory. No GPU needed. (archived hook)'''
        from archive.core.postprocess_pipeline import run_postprocess_on_dir
        run_postprocess_on_dir(input_dir, output_dir, self.config.postprocess)

    
    def apply_postprocess_image(self, gray = None):
        '''Apply postprocessing to a single image array. No GPU needed. (archived hook)'''
        from archive.core.postprocess_pipeline import run_postprocess_on_image
        return run_postprocess_on_image(gray, self.config.postprocess)

    
    def apply_label_filters(self, gray = None):
        '''Apply label guard post-filters to a single image. No GPU needed.'''
        return self._post_filter(gray)

    
    def _compute_stride(self):
        g = self.config.gen
        return int(g.patch_size * (1 - g.overlap_ratio))


