# Repository layout

Chinese detail: [`../developer/项目结构说明.md`](../developer/项目结构说明.md).

Root is written as `<repo>/` (wherever you cloned `medi-diff`).

```text
<repo>/
├─ apps/                 # Gradio + FastAPI + start/stop
├─ scripts/
│  ├─ core/              # GenParams, label_guard, image_utils, device_profile
│  ├─ preprocessing/     # CBIS clean → masks → CBIS_CLEAN_V2
│  ├─ training/          # LoRA training
│  ├─ generation/        # run_mammo_sd15.py (mainline)
│  ├─ evaluation/        # review_generated_images.py, compare_runs
│  ├─ assistant/         # generate → eval → advise
│  ├─ tools/             # publish_hf_assets, check_local_gpu, verify_ui_wiring
│  └─ tests/             # pytest (CI: not heavy)
├─ docs/
│  ├─ en/                # English public guides (this tree)
│  ├─ developer/         # ZH operator docs + maintainer notes
│  └─ private/           # Author-only notes (unsupported)
├─ hf/                   # Hugging Face cards / Space packs
├─ archive/              # Retired experiments (unsupported)
├─ datasets/             # Local data (usually gitignored)
├─ outputs/              # Generated images, eval, reports, LoRA
├─ hf_cache/             # Local SD1.5 snapshot
├─ README.md             # English front door
└─ README.zh-CN.md       # Chinese front door
```

## Important output paths

| Path | Role |
|------|------|
| `outputs/generated/samples/` | Default Gradio / report batch root |
| `outputs/eval/` | Evaluation runs |
| `outputs/reports/` | Advisor markdown, `LATEST_NEXT_RUN.json`, `PARAM_HISTORY.json` |
| `outputs/lora/mammo_sd15_v6_allMLO/final_lora/` | Recommended LoRA weights |

## Mainline vs archive

Use `scripts/{core,preprocessing,training,generation,evaluation,assistant}` for production.  
Treat `archive/` as historical — not CI-tested, not API-stable.
