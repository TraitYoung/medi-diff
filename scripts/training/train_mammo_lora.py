# [ARCHIVED] 本脚本为早期实验路线归档，不参与当前主线流程（SD1.5 Patch-Overlap）。
# 主线入口：scripts/generation/run_mammo_sd15.py
# 保留仅供历史实验复现，请勿直接调用。
"""
train_mammo_lora.py  v2
========================
使用 PEFT + diffusers 对 SD 1.5 进行乳腺钼靶 LoRA 微调。

支持 diffusers >= 0.20 的 PEFT 集成（兼容新旧 LoRA API）。

Usage:
  python3 scripts/training/train_mammo_lora.py \
      --dataset-dir   outputs/lora_dataset \
      --base-model-local hf_cache/sd15 \
      --output-dir    outputs/lora/mammo_sd15_v1 \
      --max-train-steps 2000 \
      --learning-rate 1e-4 \
      --rank          16 \
      --batch-size    4 \
      --seed          42
"""

import argparse
import math
import os
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer


def save_peft_adapter_only(model, save_dir: Path) -> None:
    """只保存 PEFT LoRA adapter，避免误写 3.3GB 全量 UNet。

    某些 diffusers/peft 组合下 `model.save_pretrained()` 会走 UNet 的保存逻辑，
    写出 `diffusion_pytorch_model.safetensors`。这里显式调用 PEFT state_dict 与 config。
    """
    from peft import get_peft_model_state_dict
    import safetensors.torch as st

    save_dir.mkdir(parents=True, exist_ok=True)
    state = get_peft_model_state_dict(model)
    if not state:
        raise RuntimeError("PEFT adapter state_dict is empty; refusing to save full UNet.")
    # PEFT 的 PeftModel.from_pretrained(UNet, path) 期望文件内 key 形如：
    # base_model.model.xxx.lora_A.weight
    # 注意不要把 adapter 名 default 写进 tensor key；PEFT 会在加载时自行注入 adapter 名。
    fixed_state = {}
    for k, v in state.items():
        nk = k
        if not nk.startswith("base_model.model."):
            nk = "base_model.model." + nk
        fixed_state[nk] = v
    st.save_file(fixed_state, str(save_dir / "adapter_model.safetensors"))

    peft_config = getattr(model, "peft_config", None)
    if peft_config:
        cfg = peft_config.get("default") if isinstance(peft_config, dict) else peft_config
        if hasattr(cfg, "save_pretrained"):
            cfg.save_pretrained(str(save_dir))
        else:
            (save_dir / "adapter_config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    if not (save_dir / "adapter_config.json").is_file():
        raise RuntimeError("PEFT adapter_config.json was not written; checkpoint would be unloadable.")


# ── Dataset ────────────────────────────────────────────────────────────────

class RandomSharpening:
    """以 p 概率对训练图做 mild unsharp-mask，引导模型生成更锐利的纹理。

    仅对 50% 图像做锐化（strength 0.1-0.25），避免模型过度依赖人工锐化信号。
    """

    def __init__(self, p: float = 0.5, strength_range: tuple = (0.1, 0.25), radius: int = 2):
        self.p = p
        self.lo, self.hi = strength_range
        self.radius = radius

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        arr = np.array(img.convert("L")).astype(np.float32)
        blurred = cv2.GaussianBlur(arr, (0, 0), sigmaX=self.radius)
        strength = random.uniform(self.lo, self.hi)
        sharpened = cv2.addWeighted(arr, 1.0 + strength, blurred, -strength, 0)
        return Image.fromarray(np.clip(sharpened, 0, 255).astype(np.uint8)).convert("RGB")


class MammoLoRADataset(Dataset):
    def __init__(self, dataset_dir: str, resolution: int, tokenizer):
        self.dir = Path(dataset_dir)
        self.resolution = resolution
        self.tokenizer = tokenizer
        self.pairs = []

        meta_csv = self.dir / "metadata.csv"
        if meta_csv.exists():
            lines = meta_csv.read_text().strip().split("\n")[1:]
            for line in lines:
                parts = line.split(",", 1)
                if len(parts) == 2:
                    fname, caption = parts
                    img_p = self.dir / fname.strip()
                    if img_p.exists():
                        self.pairs.append((str(img_p), caption.strip()))
        else:
            # 支持 JPEG + PNG
            exts = ["*.png", "*.jpg", "*.jpeg"]
            for ext in exts:
                for img_p in sorted(self.dir.glob(ext)):
                    txt_p = img_p.with_suffix(".txt")
                    cap = txt_p.read_text().strip() if txt_p.exists() else "a mammography X-ray image"
                    self.pairs.append((str(img_p), cap))

        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=8, fill=0, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ColorJitter(brightness=0.05, contrast=0.05),
            RandomSharpening(p=0.5, strength_range=(0.1, 0.25), radius=2),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        print(f"[Dataset] {len(self.pairs)} 张训练样本")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, caption = self.pairs[idx]
        img = Image.open(img_path).convert("RGB")
        pixel_values = self.transform(img)
        input_ids = self.tokenizer(
            caption, truncation=True, padding="max_length",
            max_length=self.tokenizer.model_max_length, return_tensors="pt",
        ).input_ids[0]
        return {"pixel_values": pixel_values, "input_ids": input_ids}


# ── LoRA 安装（兼容新旧 API）────────────────────────────────────────────────

def install_lora(unet, rank: int):
    """
    优先尝试 PEFT 方式（diffusers >= 0.21），
    失败则退回旧版 LoRAAttnProcessor。
    """
    try:
        from peft import LoraConfig, get_peft_model
        # PEFT 方式：更稳定
        target_modules = []
        for name, mod in unet.named_modules():
            if hasattr(mod, "weight") and ("to_q" in name or "to_k" in name
                    or "to_v" in name or "to_out.0" in name):
                target_modules.append(name)

        lora_config = LoraConfig(
            r=rank,
            lora_alpha=rank,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            lora_dropout=0.0,
            bias="none",
        )
        unet = get_peft_model(unet, lora_config)
        unet.print_trainable_parameters()
        print("[LoRA] 使用 PEFT API")
        return unet, "peft"
    except ImportError:
        pass

    # 旧版 diffusers LoRAAttnProcessor
    try:
        from diffusers.models.attention_processor import LoRAAttnProcessor
        from diffusers.loaders import AttnProcsLayers
    except ImportError:
        from diffusers.models.attention_processor import LoRAAttnProcessor2_0 as LoRAAttnProcessor
        from diffusers.models.lora import AttnProcsLayers

    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        else:
            hidden_size = unet.config.block_out_channels[0]

        lora_attn_procs[name] = LoRAAttnProcessor(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=rank,
        )

    unet.set_attn_processor(lora_attn_procs)
    trainable_layers = AttnProcsLayers(unet.attn_processors)
    total = sum(p.numel() for p in trainable_layers.parameters())
    print(f"[LoRA] 使用旧版 API | 可训参数: {total:,}")
    return trainable_layers, "old"


def get_lora_params(unet, api_type):
    if api_type == "peft":
        return [p for p in unet.parameters() if p.requires_grad]
    else:
        # trainable_layers is AttnProcsLayers
        return list(unet.parameters())


# ── Main ────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="outputs/lora_dataset")
    parser.add_argument("--base-model", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--base-model-local", default=None)
    parser.add_argument("--output-dir", default="outputs/lora/mammo_sd15_v1")
    parser.add_argument("--max-train-steps", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--lr-scheduler", default="cosine")
    parser.add_argument("--lr-warmup-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed-precision", default="fp16")
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("HF_ENDPOINT", args.hf_endpoint)
    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation,
        mixed_precision=args.mixed_precision,
        project_config=ProjectConfiguration(project_dir=str(out_dir)),
    )

    model_id = args.base_model_local or args.base_model
    print(f"[Train] 加载基座: {model_id}")

    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    # 冻结 VAE + text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # 安装 LoRA
    lora_obj, api_type = install_lora(unet, rank=args.rank)

    # 数据集
    dataset = MammoLoRADataset(args.dataset_dir, args.resolution, tokenizer)
    train_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )

    # 仅优化 LoRA 参数
    if api_type == "peft":
        trainable_params = [p for p in unet.parameters() if p.requires_grad]
    else:
        # lora_obj is AttnProcsLayers
        trainable_params = list(lora_obj.parameters())

    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=1e-2)

    from transformers import get_scheduler as get_lr_scheduler
    lr_scheduler = get_lr_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation,
        num_training_steps=args.max_train_steps * args.gradient_accumulation,
    )

    if api_type == "peft":
        unet, optimizer, train_loader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_loader, lr_scheduler
        )
    else:
        lora_obj, optimizer, train_loader, lr_scheduler = accelerator.prepare(
            lora_obj, optimizer, train_loader, lr_scheduler
        )

    vae = vae.to(accelerator.device, dtype=torch.float16)
    text_encoder = text_encoder.to(accelerator.device)
    if api_type == "old":
        unet = unet.to(accelerator.device)

    global_step = 0
    progress = tqdm(total=args.max_train_steps, desc="LoRA Training")
    loss_ema = None

    unet.train()
    train_iter = iter(train_loader)

    while global_step < args.max_train_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        accum_ctx = accelerator.accumulate(unet if api_type == "peft" else lora_obj)
        with accum_ctx:
            # Encode images → latents
            pv = batch["pixel_values"].to(dtype=torch.float16)
            with torch.no_grad():
                latents = vae.encode(pv).latent_dist.sample() * vae.config.scaling_factor

            # Add noise
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (bsz,), device=latents.device, dtype=torch.long,
            )
            noisy = noise_scheduler.add_noise(latents, noise, timesteps)

            # Text encode
            with torch.no_grad():
                enc_hs = text_encoder(batch["input_ids"].to(accelerator.device))[0]

            # Predict noise
            pred = unet(noisy, timesteps, enc_hs).sample
            loss = F.mse_loss(pred.float(), noise.float())
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                params_to_clip = trainable_params if api_type == "peft" else list(lora_obj.parameters())
                accelerator.clip_grad_norm_(params_to_clip, 1.0)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        if accelerator.sync_gradients:
            global_step += 1
            loss_ema = loss.item() if loss_ema is None else 0.98 * loss_ema + 0.02 * loss.item()
            progress.update(1)
            progress.set_postfix({
                "loss": f"{loss_ema:.4f}",
                "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}",
            })

            # 保存 checkpoint。PEFT 模式必须保存 adapter_config.json，否则推理脚本无法加载中间权重。
            if global_step % args.save_steps == 0 or global_step == args.max_train_steps:
                ckpt_dir = out_dir / f"checkpoint-{global_step}"
                ckpt_dir.mkdir(exist_ok=True)

                unwrapped = accelerator.unwrap_model(unet)
                if api_type == "peft":
                    save_peft_adapter_only(unwrapped, ckpt_dir)
                    print(f"\n[Save] Step {global_step} → {ckpt_dir} (PEFT adapter)")
                else:
                    unwrapped.save_attn_procs(str(ckpt_dir))
                    print(f"\n[Save] Step {global_step} → {ckpt_dir} (attn_procs)")

    # 最终保存
    progress.close()
    unwrapped = accelerator.unwrap_model(unet)
    final_dir = out_dir / "final_lora"
    final_dir.mkdir(exist_ok=True)

    if api_type == "peft":
        save_peft_adapter_only(unwrapped, final_dir)
        print(f"[Done] PEFT LoRA 保存至: {final_dir}/")
    else:
        unwrapped.save_attn_procs(str(final_dir))
        print(f"[Done] LoRA attn_procs 保存至: {final_dir}/")

    # 打印最终 loss
    print(f"[Done] 最终 EMA loss: {loss_ema:.4f}")
    print(f"[Done] 输出目录: {out_dir}/")


if __name__ == "__main__":
    main()
