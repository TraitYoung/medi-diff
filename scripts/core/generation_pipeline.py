'''SDPipeline: importable SD1.5+LoRA generation engine.

Wraps model loading + patch-overlap generation in a reusable class so callers
(run_mammo_sd15.py, run_generate_eval_advise.py, fast_iter.py, api_server.py, etc.)
can generate images via direct import instead of subprocess+CLI serialization.

Core generation functions (patch_generate, run_global_guide, etc.) live in
scripts/generation/run_mammo_sd15.py as the authoritative implementation.
This module imports them and provides the orchestration layer.
'''
from __future__ import annotations
import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
ROOT = Path(__file__).resolve().parents[2]

class SDPipeline:
    '''SD1.5 + LoRA generation pipeline.

    Usage:
        pipe = SDPipeline(
            base_model_local="hf_cache/sd15",
            lora_path="outputs/lora/mammo_sd15_v4_clean/final_lora",
        )
        result = pipe.generate(
            src_gray=letterboxed_source,      # (H,W) uint8
            prompt="...",
            negative_prompt="...",
        strength=0.42,
        guidance_scale=8.5,
        num_inference_steps=50,
            patch_size=640,
            stride=77,
            gabor_alpha=0.50,
            blend_mode="hann",
            blend_sigma_divisor=1.48,
            global_guide_gray=None,
            global_guide_blend=0.44,
            latent_smooth_field=None,
            base_seed=2026,
            bg_min_signal_frac=0.012,
            bg_pixel_min=4,
        )
    '''
    
    def __init__(self, base_model_local = None, base_model = None, lora_path = None, hf_endpoint=None, *, scheduler='dpm', load_dotenv=True):
        self.base_model = base_model
        self.base_model_local = base_model_local
        self.lora_path = lora_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.scheduler = scheduler
        if load_dotenv:
            self._ensure_dotenv()
        if hf_endpoint:
            os.environ.setdefault('HF_ENDPOINT', hf_endpoint)
        self.pipe = None
        self._torch_generator = None

    
    def _ensure_dotenv(self):
        env_path = ROOT / '.env'
        if env_path.is_file():
            
            try:
                from dotenv import load_dotenv
                load_dotenv(env_path, override = False)
                return None
            except ImportError:
                return None


    
    def _ensure_model_loaded(self):
        if self.pipe is not None:
            return
        from diffusers import StableDiffusionImg2ImgPipeline
        model_path = self.base_model_local or self.base_model
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        pipe.to(self.device)
        self.pipe = pipe
        self._apply_scheduler()
        if self.lora_path:
            lora_path = Path(self.lora_path)
            if (lora_path / "adapter_model.safetensors").is_file():
                from peft import PeftModel

                self.pipe.unet = PeftModel.from_pretrained(self.pipe.unet, str(lora_path))
            else:
                self.pipe.load_lora_weights(self.lora_path)

    
    def _apply_scheduler(self):
        if self.scheduler == 'pndm':
            return None
        if self.scheduler == 'dpm':
            from diffusers import DPMSolverMultistepScheduler
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
            return None
        if self.scheduler == 'ddim':
            from diffusers import DDIMScheduler
            self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
            return None

    @property
    def torch_generator(self):
        return self._torch_generator
    
    def set_seed(self, seed = None):
        self._torch_generator = torch.Generator(device = self.device).manual_seed(int(seed))

    
    def generate(self, src_gray = None, *, prompt, negative_prompt, strength, guidance_scale, num_inference_steps, mode, fullimage_long_side=768, fullimage_output_long_side=2048, patch_size=640, stride=96, gabor_alpha=0.5, blend_mode="hann", blend_sigma_divisor=1.48, global_guide_gray=None, global_guide_blend=0.44, latent_smooth_field=None, base_seed=2026, bg_min_signal_frac=0.012, bg_pixel_min=4, pyramid_levels=4):
        '''Generate a single mammography image.

        Args:
            src_gray: (H, W) uint8 grayscale source (letterboxed to target size)
            mode: "full-image" (single-pass, no seams) or "patch" (legacy patch-overlap)

        Returns:
            (H, W) uint8 grayscale result image
        '''
        self._ensure_model_loaded()
        if mode == 'full-image':
            from scripts.generation.run_mammo_sd15 import fullimage_generate
            return fullimage_generate(src_gray = src_gray, pipe = self.pipe, prompt = prompt, negative_prompt = negative_prompt, strength = strength, guidance_scale = guidance_scale, num_inference_steps = num_inference_steps, generator = self.torch_generator, gabor_alpha = gabor_alpha, fullimage_long_side = fullimage_long_side, fullimage_output_long_side = fullimage_output_long_side)
        from scripts.generation.run_mammo_sd15 import patch_generate
        return patch_generate(
            src_gray=src_gray,
            pipe=self.pipe,
            prompt=prompt,
            negative_prompt=negative_prompt,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            patch_size=patch_size,
            stride=stride,
            gabor_alpha=gabor_alpha,
            blend_mode=blend_mode,
            blend_sigma_divisor=blend_sigma_divisor,
            global_guide_gray=global_guide_gray,
            global_guide_blend=global_guide_blend,
            latent_smooth_field=latent_smooth_field,
            base_seed=base_seed,
            bg_min_signal_frac=bg_min_signal_frac,
            bg_pixel_min=bg_pixel_min,
            pyramid_levels=pyramid_levels,
        )

    
    def run_global_guide(self, src_gray = None, *, prompt, negative_prompt, strength, guide_scale, num_steps, seed):
        '''Low-resolution global anatomic guide pass.'''
        self._ensure_model_loaded()
        from scripts.generation.run_mammo_sd15 import run_global_guide
        g_gen = torch.Generator(device = self.device).manual_seed(seed)
        return run_global_guide(self.pipe, src_gray, self.device, g_gen, guide_scale = guide_scale, strength = strength, prompt = prompt, negative_prompt = negative_prompt, num_steps = num_steps)

    
    def build_latent_smooth_field(self, H=None, W=None, seed=None):
        '''Low-frequency latent noise smooth field for cross-patch consistency.'''
        import torch.nn.functional as F
        if seed is None:
            seed = 0
        g = torch.Generator(device='cpu').manual_seed(seed)
        small_h, small_w = max(1, H // 32), max(1, W // 32)
        small = torch.randn(1, 1, small_h, small_w, generator=g, dtype=torch.float32)
        field = F.interpolate(small, size=(H, W), mode='bilinear', align_corners=False)
        return field.squeeze().numpy()

