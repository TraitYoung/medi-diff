'''SDPipeline: importable SD1.5+LoRA generation engine (archived).

Production uses scripts/generation/run_mammo_sd15.py (Gradio/API/CLI subprocess).
Import via archive.core.generation_pipeline for notebook / GenerationPipeline only.
'''
from __future__ import annotations

import os
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


class SDPipeline:
    '''SD1.5 + LoRA generation pipeline (full-image single-pass img2img).'''

    def __init__(
        self,
        base_model_local=None,
        base_model=None,
        lora_path=None,
        hf_endpoint=None,
        *,
        scheduler='dpm',
        load_dotenv=True,
    ):
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

    def _ensure_dotenv(self) -> None:
        env_path = ROOT / '.env'
        if not env_path.is_file():
            return
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path, override=False)
        except ImportError:
            return

    def _ensure_model_loaded(self) -> None:
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

    def _apply_scheduler(self) -> None:
        if self.scheduler == 'pndm':
            return
        if self.scheduler == 'dpm':
            from diffusers import DPMSolverMultistepScheduler
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config
            )
            return
        if self.scheduler == 'ddim':
            from diffusers import DDIMScheduler
            self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)

    @property
    def torch_generator(self):
        return self._torch_generator

    def set_seed(self, seed=None) -> None:
        self._torch_generator = torch.Generator(device=self.device).manual_seed(int(seed))

    def generate(
        self,
        src_gray=None,
        *,
        prompt,
        negative_prompt,
        strength,
        guidance_scale,
        num_inference_steps,
        fullimage_long_side=768,
        fullimage_output_long_side=2048,
        base_seed=2026,
    ):
        self._ensure_model_loaded()
        if self._torch_generator is None and base_seed is not None:
            self.set_seed(base_seed)
        from scripts.generation.run_mammo_sd15 import fullimage_generate

        return fullimage_generate(
            src_gray=src_gray,
            pipe=self.pipe,
            prompt=prompt,
            negative_prompt=negative_prompt,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=self.torch_generator,
            fullimage_long_side=fullimage_long_side,
            fullimage_output_long_side=fullimage_output_long_side,
        )
