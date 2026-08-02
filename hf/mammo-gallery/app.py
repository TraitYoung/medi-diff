"""Static MammoGen gallery Space (CPU only — no live diffusion)."""
from __future__ import annotations

from pathlib import Path

import gradio as gr
from PIL import Image

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"

CAPTIONS = {
    "ui-generate.png": "Generate — full-image presets (MLO / density filters)",
    "ui-gallery.png": "Gallery — browse generated batches",
    "ui-compare.png": "Gallery — source vs generated compare layout",
    "ui-curated.png": "Gallery — curated export by evaluation rank",
    "ui-pipeline.png": "One-click pipeline — generate → eval → advise",
    "ui-eval.png": "Evaluation — pass rate / BRISQUE / A–F groups",
    "ui-tuning.png": "Tuning history — rollback last runs",
}


def _load_gallery() -> list[tuple[Image.Image, str]]:
    items: list[tuple[Image.Image, str]] = []
    if not ASSETS.is_dir():
        return items
    for name in CAPTIONS:
        path = ASSETS / name
        if path.is_file():
            items.append((Image.open(path), CAPTIONS[name]))
    for path in sorted(ASSETS.glob("*.png")):
        if path.name not in CAPTIONS:
            items.append((Image.open(path), path.stem))
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
