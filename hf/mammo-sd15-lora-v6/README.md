---
license: creativeml-openrail-m
base_model: runwayml/stable-diffusion-v1-5
tags:
  - stable-diffusion
  - lora
  - mammography
  - medical-imaging
  - diffusers
library_name: diffusers
pipeline_tag: image-to-image
---

# MammoGen LoRA v6 (SD1.5, all MLO)

LoRA adapter for **Stable Diffusion 1.5** trained for full-field digital mammography (FFDM) style synthesis (MLO views).

> **Not for clinical diagnosis.** Research / education / synthetic-data tooling only.

## Model details

| Item | Value |
|------|--------|
| Base | Stable Diffusion 1.5 |
| Adapter | LoRA (`peft`), rank **r=32** |
| Views | All **MLO** (training mix) |
| Intended use | img2img full-image generation via [MammoGen](https://github.com/TraitYoung/medi-diff) |
| Companion code | `scripts/generation/run_mammo_sd15.py` |

## Files

Expect a Diffusers / PEFT LoRA directory, typically:

- `adapter_config.json`
- `adapter_model.safetensors` (or `.bin`)

## Quick start

```bash
# from the MammoGen repo
huggingface-cli download TraitYoung/mammo-sd15-lora-v6 --local-dir outputs/lora/mammo_sd15_v6_allMLO/final_lora

python3 scripts/generation/run_mammo_sd15.py \
  --base-model-local hf_cache/sd15 \
  --lora-path outputs/lora/mammo_sd15_v6_allMLO/final_lora \
  --metadata-csv datasets/CBIS_CLEAN_V2/metadata_clean.csv \
  --filter-view MLO --filter-density scattered \
  --num-images 4 --mode full-image
```

Recommended defaults: `strength=0.44`, `guidance_scale=7.5`, `num_steps=40`, scheduler `dpm`.

## Training data

Derived from [CBIS-DDSM](https://www.cancerimagingarchive.net/collection/cbis-ddsm/) (TCIA) after the MammoGen cleaning pipeline (`CBIS_CLEAN_V2`).  
See also dataset card: [`TraitYoung/cbis-clean-v2`](https://huggingface.co/datasets/TraitYoung/cbis-clean-v2).

## Evaluation (honest)

In-domain 16-dim rule evaluation often yields strict `pass_rate` around **~50%** depending on source pool and calibration.  
Common failure tags: `BANDING`, `SHAPE_ODD`, `ARTIFACT_BUBBLES`. Prefer reporting both strict and calibrated metrics — auto-calibration can over-relax hard gates.

## License & attribution

- Adapter weights: follow **CreativeML OpenRAIL-M** obligations inherited from SD1.5 unless a more specific license file is attached in this repo.
- Training images remain under CBIS-DDSM / TCIA terms — this card does **not** relicense source mammograms.
- Project code: MIT ([GitHub](https://github.com/TraitYoung/medi-diff)).
