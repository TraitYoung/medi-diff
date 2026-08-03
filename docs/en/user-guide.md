# User guide

**Audience:** researchers and engineers running MammoGen for FFDM synthesis, QC, API integration, or demos.

Chinese counterpart (more detail): [`../developer/用户操作手册.md`](../developer/用户操作手册.md).

---

## 1. Requirements

| Item | Requirement |
|------|-------------|
| OS | Linux (recommended) / Windows WSL2 |
| Python | 3.10+ |
| CUDA | 12.x recommended; **RTX 50-series needs PyTorch cu128+** |
| VRAM | ≥8 GB practical; 12–16 GB comfortable for defaults |
| Disk | ≥40 GB (models, data, outputs) |

---

## 2. Install & start

```bash
cd /path/to/medi-diff
pip install -r requirements.txt
# or: pip install -e ".[full]"

# Blackwell / RTX 50-series:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python3 scripts/tools/check_local_gpu.py

cp .env.example .env   # optional advisor keys
bash apps/start.sh
```

| Service | Default URL |
|---------|-------------|
| Gradio | http://127.0.0.1:7860 |
| FastAPI / Swagger | http://127.0.0.1:8000/docs |

Stop with `Ctrl+C` or `bash apps/stop.sh`.  
See `bash apps/start.sh --help` for `--ui-only`, `--api-only`, `--api-reload`, `--gradio-share`.

### Optional advisor APIs

Generation and rule-based eval work **without** keys. Keys are only for LLM reports (`run_generate_eval_advise` / `run_full_report`).

With `ADVISOR_TEXT_BACKEND=auto`, priority is DeepSeek → GLM → DashScope Qwen. Vision uses Qwen-VL via DashScope when `--advisor-mode both`.

---

## 3. Gradio tabs (summary)

UI labels are currently Chinese; mapping:

| Tab (ZH) | Role |
|----------|------|
| 生成 | Generate — density/view filters, speed preset, full-image img2img |
| 一键流水线 | One-click: generate → eval → advise |
| 评估 | Run rule eval on a batch |
| 画廊 | Browse batches / compare to sources |
| 调参历史 | Parameter history / rollback |
| 评估汇总 | Eval summary |

**Speed presets (VRAM):**

| Preset (UI) | Meaning |
|-------------|---------|
| 本地流畅 (default) | **Local speed** — 28 steps, `--mem-profile local` |
| 云端质量 | **Cloud quality** — 40 steps, `--mem-profile auto` |

Sampling defaults otherwise match `GenParams`: strength `0.44`, guidance `7.5`, output long side `2048`.

---

## 4. CLI essentials

```bash
# Generate
python3 scripts/generation/run_mammo_sd15.py \
  --base-model-local hf_cache/sd15 \
  --lora-path outputs/lora/mammo_sd15_v6_allMLO/final_lora \
  --metadata-csv datasets/CBIS_CLEAN_V2/metadata_clean.csv \
  --filter-view MLO --filter-density scattered \
  --num-images 6 --seed 2026 --source-seed 20260513 \
  --mem-profile local --num-steps 28

# Evaluate
python3 scripts/evaluation/review_generated_images.py \
  --images-dir outputs/generated/samples/<batch> \
  --output-dir outputs/eval/<name> \
  --no-recursive --eval-profile full --enable-seam-check

# Full pipeline
python3 scripts/assistant/run_generate_eval_advise.py \
  --metadata-csv datasets/CBIS_CLEAN_V2/metadata_clean.csv \
  --filter-view MLO --filter-density dense \
  --num-images 6 --tag-prefix my_run
```

Default batch root: `outputs/generated/samples/`.

---

## 5. Outputs

| Path | Role |
|------|------|
| `outputs/generated/samples/<prefix>_<ts>/` | PNGs + `source_map.json` + `run_params.json` |
| `outputs/eval/<name>/` | `summary.json`, `review_report.csv`, top-k lists |
| `outputs/reports/` | Advisor reports, `LATEST_NEXT_RUN.json`, `PARAM_HISTORY.json` |

---

## 6. FAQ

**Do I need an LLM key?**  
No for generate/eval.

**Gallery empty?**  
Ensure batches live under `outputs/generated/samples/`.

**OOM on 12–16 GB?**  
Use `--mem-profile local` (or Gradio Local speed). Confirm cu128 on RTX 50-series: `python3 scripts/tools/check_local_gpu.py`.

**Patch-overlap mode?**  
Removed from mainline; see `archive/` / git history.

---

## 7. Pre-flight checks

```bash
python3 scripts/tools/verify_ui_wiring.py
python3 scripts/tools/check_local_gpu.py
```

More: [API](api.md) · [Evaluation](evaluation.md) · [SECURITY](../../SECURITY.md).
