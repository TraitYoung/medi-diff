# Hugging Face packaging (M3)

Target namespaces (personal account):

| Asset | Repo id | Local pack folder |
|-------|---------|-------------------|
| LoRA weights | `TraitYoung/mammo-sd15-lora-v6` | `hf/mammo-sd15-lora-v6/` |
| Clean metadata | `TraitYoung/cbis-clean-v2` | `hf/cbis-clean-v2/` |
| Static gallery Space | `TraitYoung/mammo-gallery` | `hf/mammo-gallery/` |

## Prerequisites

1. [Hugging Face](https://huggingface.co/) account `TraitYoung` (or edit IDs).
2. Token with **write** access: `export HF_TOKEN=hf_...`
3. Local assets (gitignored):
   - LoRA dir: `outputs/lora/mammo_sd15_v6_allMLO/final_lora/`
   - Metadata: `datasets/CBIS_CLEAN_V2/metadata_clean.csv`
   - Optional sample images for the gallery under `outputs/generated/samples/`

## Publish

```bash
pip install "huggingface_hub>=0.23" gradio
python3 scripts/tools/publish_hf_assets.py --all
# or selectively:
python3 scripts/tools/publish_hf_assets.py --model --dataset --space
```

The script uploads Model/Dataset **cards** always; binary weights / CSV upload only when local paths exist.  
Space is CPU-only (no live diffusion).

## License reminders

- **Code** in this GitHub repo: MIT.
- **CBIS-DDSM images**: follow [TCIA / CBIS-DDSM](https://www.cancerimagingarchive.net/collection/cbis-ddsm/) terms (attribution required). Prefer publishing **cleaned metadata + scripts** if image redistribution is unclear.
- **SD1.5 / LoRA**: follow the base model license when redistributing adapters.
