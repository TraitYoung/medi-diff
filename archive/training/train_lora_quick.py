#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]


class MammogramDataset(Dataset):
    def __init__(self, jsonl_path: Path, max_samples: int = 800) -> None:
        lines = jsonl_path.read_text(encoding="utf-8").splitlines()
        items = [json.loads(x) for x in lines if x.strip()]
        self.items = items[:max_samples] if max_samples > 0 else items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict | None:
        item = self.items[i]
        img_path = Path(item["file_name"])
        if not img_path.exists():
            return None
        try:
            image = Image.open(img_path).convert("RGB").resize((1024, 1024))
            arr = np.asarray(image, dtype=np.float32)
            img_tensor = torch.from_numpy(arr).permute(2, 0, 1) / 127.5 - 1.0
            return {"image": img_tensor}
        except Exception:
            return None


def collate_fn(examples: list[dict | None]) -> dict | None:
    valid = [e for e in examples if e is not None]
    if not valid:
        return None
    return {"image": torch.stack([v["image"] for v in valid])}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quick SDXL LoRA finetune for mammogram texture")
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--metadata-jsonl", type=Path, default=ROOT / "datasets/metadata.jsonl")
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs/lora/sdxl_lora_output_quick")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-samples", type=int, default=800)
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    accelerator = Accelerator(mixed_precision="fp16")

    vae = AutoencoderKL.from_pretrained(
        str(args.model_path),
        subfolder="vae",
        variant="fp16",
        use_safetensors=True,
        local_files_only=True,
    ).to(accelerator.device, dtype=torch.float32)
    vae.requires_grad_(False)

    unet = UNet2DConditionModel.from_pretrained(
        str(args.model_path),
        subfolder="unet",
        variant="fp16",
        use_safetensors=True,
        local_files_only=True,
    )
    noise_scheduler = DDPMScheduler.from_pretrained(str(args.model_path), subfolder="scheduler")
    lora_config = LoraConfig(r=16, lora_alpha=16, target_modules=["to_k", "to_q", "to_v", "to_out.0"])
    unet = get_peft_model(unet, lora_config)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.lr)

    dataset = MammogramDataset(args.metadata_jsonl, max_samples=args.max_samples)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)

    unet.train()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress = tqdm(range(args.steps), desc="LoRA training")
    step = 0
    while step < args.steps:
        for batch in dataloader:
            if batch is None:
                continue
            pixel_values = batch["image"].to(accelerator.device, dtype=torch.float32)
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                latents = latents.to(dtype=torch.float16)

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=accelerator.device)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            encoder_hidden_states = torch.zeros((bsz, 77, 2048), device=accelerator.device, dtype=torch.float16)
            added_cond_kwargs = {
                "text_embeds": torch.zeros((bsz, 1280), device=accelerator.device, dtype=torch.float16),
                "time_ids": torch.zeros((bsz, 6), device=accelerator.device, dtype=torch.float16),
            }
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, added_cond_kwargs=added_cond_kwargs).sample
            loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            step += 1
            progress.update(1)
            progress.set_postfix({"loss": f"{loss.item():.4f}"})
            if step >= args.steps:
                break

    unet.save_pretrained(str(args.output_dir))
    print(f"[Done] LoRA saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
