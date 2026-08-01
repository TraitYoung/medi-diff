"""Static MammoGen gallery Space (CPU only — no live diffusion)."""
from __future__ import annotations

from pathlib import Path

import gradio as gr
from PIL import Image

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"

CAPTIONS = {
    "图片 1.png": "Generate — full-image presets (MLO / density filters)",
    "图片 2.png": "Gallery — browse generated batches",
    "图片 3.png": "Gallery — source vs generated compare layout",
    "图片 4.png": "Gallery — curated export by evaluation rank",
    "图片 5.png": "One-click pipeline — generate → eval → advise",
    "图片 6.png": "Evaluation — pass rate / BRISQUE / A–F groups",
    "图片 7.png": "Tuning history — rollback last runs",
}


def _load_gallery() -> list[tuple[Image.Image, str]]:
    items: list[tuple[Image.Image, str]] = []
    if not ASSETS.is_dir():
        return items
    for path in sorted(ASSETS.glob("*.png")):
        caption = CAPTIONS.get(path.name, path.stem)
        items.append((Image.open(path), caption))
    return items


INTRO = """
# MammoGen Gallery

**Synthetic mammography generation with automated quality control.**

This Space is a **static demo** (no GPU inference).  
For full pipeline / LoRA weights:

- GitHub: https://github.com/TraitYoung/medi-diff
- LoRA (when published): https://huggingface.co/TraitYoung/mammo-sd15-lora-v6
- Metadata card: https://huggingface.co/datasets/TraitYoung/cbis-clean-v2

> **Not for clinical diagnosis.** Research and education only.
"""


def build() -> gr.Blocks:
    gallery_data = _load_gallery()
    with gr.Blocks(title="MammoGen Gallery") as demo:
        gr.Markdown(INTRO)
        gr.Markdown("## UI walkthrough")
        if gallery_data:
            gr.Gallery(
                value=gallery_data,
                columns=2,
                height=640,
                object_fit="contain",
                label="Local Gradio screenshots",
            )
        else:
            gr.Markdown("_No assets/ PNGs found — re-run publish script with screenshots._")
        gr.Markdown(
            """
## Honest quality note

Strict in-domain `pass_rate` is often around **~50%**.  
Expect failure modes such as banding / shape oddities; see the GitHub evaluation docs.
"""
        )
    return demo


demo = build()

if __name__ == "__main__":
    demo.launch()
