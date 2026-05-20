# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

A mammography (FFDM) image generation and evaluation system using **Stable Diffusion 1.5 + LoRA** fine-tuned on CBIS-DDSM. **Default generation is full-image single-pass img2img** (no patch grid; fast, no seam banding). Bachelor's capstone project. Pipeline: data preprocessing → LoRA training → conditional generation → automated quality evaluation → optional LLM advisor reports.

## Common commands

```bash
# Start Gradio UI + FastAPI backend (default ports: 7860, 8000)
bash apps/start.sh

# Gradio only / API only / with hot-reload for dev
bash apps/start.sh --ui-only
bash apps/start.sh --api-only
bash apps/start.sh --api-reload

# Stop services
bash apps/stop.sh

# Generate images (main entry point; default mode is full-image, all defaults match good batches)
python3 scripts/generation/run_mammo_sd15.py \
  --base-model-local hf_cache/sd15 \
  --lora-path outputs/lora/mammo_sd15_v6_allMLO/final_lora \
  --metadata-csv datasets/CBIS_CLEAN_V2/metadata_clean.csv \
  --filter-view MLO --filter-density scattered \
  --num-images 6 --seed 2026 --source-seed 20260513 \
  --mode full-image \
  --fullimage-long-side 768 \
  --fullimage-min-short-side 384 \
  --fullimage-output-long-side 2048 \
  --scheduler dpm --num-steps 40 \
  --strength 0.44 --guidance-scale 7.5

# Evaluate generated images
python3 scripts/evaluation/review_generated_images.py \
  --images-dir outputs/generated/<batch_dir> \
  --output-dir outputs/eval/<eval_name> \
  --no-recursive --eval-profile full --enable-seam-check

# Full automated pipeline (generate → evaluate → advise; primary entry point)
python3 scripts/assistant/run_generate_eval_advise.py \
  --metadata-csv datasets/CBIS_CLEAN_V2/metadata_clean.csv \
  --filter-view MLO --filter-density dense \
  --num-images 6 --tag-prefix my_run

# Convenience wrapper (calls run_generate_eval_advise via subprocess)
python3 scripts/assistant/run_full_report.py \
  --metadata-csv datasets/CBIS_CLEAN_V2/metadata_clean.csv \
  --filter-view MLO --filter-density dense \
  --num-images 6 --tag-prefix my_run

# Multi-run comparison table
python3 scripts/evaluation/compare_runs.py --runs-dir outputs/eval

# UI wiring smoke test (no GPU needed)
python3 scripts/tools/verify_ui_wiring.py
```

## Architecture

### Pipeline layers

0. **Shared core library** (`scripts/core/`): Reusable modules imported by generation, evaluation, and preprocessing scripts. Contains:
   - `pipeline_config.py` — `GenParams` dataclass: single source of truth for all generation parameters (strength, guidance_scale, num_steps, scheduler, prompt, negative_prompt, fullimage sizes).
   - `label_guard.py` — Stateless image filter functions (label erasure, edge feathering) used post-generation by `run_mammo_sd15.py`.
   - `image_utils.py` — Shared image utilities (2026-05-20): unified `resize_long_side`, `build_mask`, `is_image`, `largest_component`, `enhance_input_contrast`.  Eliminates 4 duplicate resize variants and 2 sys.path hacks.
   - `pipeline_orchestrator.py` removed (2026-05-19) — production uses `run_mammo_sd15.py` directly.
   - `generation_pipeline.py` archived to `archive/core/generation_pipeline.py` — `SDPipeline` class no longer used by active code.
   - Postprocess archived to `archive/core/postprocess_pipeline.py` and `archive/postprocess/`.

1. **Preprocessing** (`scripts/preprocessing/`): CBIS-DDSM cleaning, breast mask extraction, DICOM burn-in removal, caption generation. Pipeline: `clean_cbis.py` (raw→CBIS_CLEAN) → `build_breast_masks.py` (Otsu+形态学) → `clean_training_labels.py` (掩膜外标签清零→CBIS_CLEAN_V2) → `generate_captions.py` (训练描述文本). Current canonical training data: `datasets/CBIS_CLEAN_V2/metadata_clean.csv`. CRAFT+LaMa / SD-Inpaint mask-internal text removal archived to `archive/preprocessing/`.

2. **Training** (`scripts/training/`): LoRA fine-tuning on SD1.5 UNet via `peft`. Current recommended weights: `outputs/lora/mammo_sd15_v6_allMLO/final_lora` (r=32, trained on all MLO views). `prepare_lora_dataset_v5.py` 构建训练 JSONL；`train_mammo_lora.py` 执行 LoRA 训练。辅助脚本（`train_lora_quick.py`、`prepare_lora_dataset.py`、`select_lora_checkpoint.py`、`train_lora_v5.sh`）已归档至 `archive/training/`。

3. **Generation** (`scripts/generation/`): `run_mammo_sd15.py` is the **sole active** generation script. Full-image single-pass img2img via `fullimage_generate()`: scale so long side ≤ `--fullimage-long-side` (default **768**), short side ≥ `--fullimage-min-short-side` (default **384**) when feasible, hard cap 1024 px; **output then LANCZOS4-upscaled to long side == `--fullimage-output-long-side`** (default **2048**; **0** = keep native). **Default LoRA: `mammo_sd15_v6_allMLO/final_lora`** (r=32). **Default generation parameters (2026-05-18):** strength=**0.44**, guidance_scale=**7.5**, num_steps=**40**, scheduler=dpm. **Source pool:** `filter_source_pool()` drops extreme aspect ratios (default **2.2**) and applies **shape quality pre-filter** (circularity<0.30 or convex_defect>0.45 → skip). **Post-filter sequence (legacy_label_guard=True):** `erase_background_labels` → `erase_bright_border_labels` → `feather_canvas_edge(feather_px=3)`. **Metal BB markers:** `_final_metal_sweep()` (3-round Hough+TELEA, always runs). **Isolated bright spots:** `_soften_isolated_bright_spots()` run unconditionally. **Qwen-VL POST verification is NOT wired into main generation path** (`--no-qwen-vl` is parsed but unused in `main()`). **Patch-overlap generation has been removed** (2026-05-19); reference the commit history for the archived implementation. Each batch writes `source_map.json` + `run_params.json` (full parameters). **Postprocess has been archived** (`--postprocess` flag removed from main pipeline; see `archive/postprocess/`).

4. **Postprocessing** (`archive/postprocess/`, 已归档): `postprocess_freq.py` 已归档至 `archive/postprocess/`。核心函数在 `archive/core/postprocess_pipeline.py`（默认禁用，`PostprocessParams.enabled=False`）。如需恢复，从归档目录运行 CLI 或手动启用。

5. **Evaluation** (`scripts/evaluation/`): `review_generated_images.py` is the main scorer. 16-dim rule-based scores across groups A–F. **Current default thresholds (2026-05-18):** min_circularity=**0.32**, max_contour_concavity=**0.45**, max_isolated_round=**6**, BRISQUE trigger=**60** (真实钼靶 ~50-70), max_bright_spots=**20**. **`hard_tags`** (fatal for `ok`) include BANDING (`--min-banding-score` default **0.62**), SHAPE_ODD, ARTIFACT_BUBBLES, CONTOUR_FRACTURED, EDGE_VOIDS, etc. **`HIGH_BRISQUE` is NOT a hard tag** — it only reduces F-group score. **`SKIN_LINE_MISSING`** and **`GRID_SEAM`** funnel into **`soft_reasons`** in full-image mode (lower semantic_score but don't directly block `ok`). This mirrors the anatomy_structure_check pattern where eval_profile='full' demotes hard vetoes to soft penalties. Use **`--real-images-dir`** for density-matched real baselines. Outputs `summary.json` + `review_report.csv`. Current strict pass_rate (no-auto-calibrate): ~50%.

6. **Assistant/reporting** (`scripts/assistant/`): `run_generate_eval_advise.py` is the primary orchestrator (generate → evaluate → advisor). When **`--mode full-image`**, both **`run_generate_eval_advise.py`** and **`run_full_report.py`** forward **`--fullimage-long-side`** / **`--fullimage-output-long-side`** (**not** **`--fullimage-min-short-side`**, which stays at **`run_mammo_sd15.py`** defaults unless you invoke that script directly with overrides). **`run_full_report.py`** is a subprocess wrapper around **`run_generate_eval_advise`**. **`tuning_state.py`** reads/writes **`LATEST_NEXT_RUN.json`** for CLI + assistant jobs. **`ask_advisor.py`** calls external LLMs (DeepSeek, GLM, or DashScope Qwen for text; Qwen‑VL for vision). API keys live in **`.env`**.

7. **UI & API** (`apps/`): Gradio (`app_gradio.py`) calls scripts via `subprocess` — does NOT depend on the FastAPI server. Both are started together by `start.sh` but are logically independent. The API server (`api_server.py`) wraps the same scripts with async job queues. The **生成** tab is **a single streamlined panel**: speed preset + filters; sampling sliders live in **`visible=False`** and are driven by presets (aligned with **`GenParams`**). Default density **`scattered`**. Full-image runs pass **`--mode full-image`** and **`--fullimage-output-long-side 2048`**. Separate **一键流水线 / 评估 / 画廊 / 调参历史** tabs remain; **`load_latest_next_run_into_tuning`** exists but is **not wired** in the simplified UI — apply advisor JSON primarily via **`run_full_report.py --from-latest-tuning`** or copy fields from **`LATEST_NEXT_RUN.json`** manually into CLI/UI.

### Unit tests (`scripts/tests/`)

- **`test_fullimage_output_size.py`**: asserts **`fullimage_output_long_side`** caps the saved long edge (uses a minimal fake img2img **`pipe`** that echoes the latent image).

### Key state files

| Path | Role |
|------|------|
| `datasets/CBIS_CLEAN_V2/metadata_clean.csv` | Canonical training/generation metadata (burn-in cleaned) |
| `outputs/lora/mammo_sd15_v6_allMLO/final_lora/` | Current recommended LoRA weights (r=32, all MLO views) |
| `outputs/reports/LATEST_NEXT_RUN.json` | Most recent advisor-suggested **`parameters`** and tuning context (`eval_mode`, **`source_seed`**, etc.). Apply with **`python3 scripts/assistant/run_full_report.py --from-latest-tuning`** (CLI); **`PARAM_HISTORY`** + **「调参历史」** Tab in Gradio support rollback reruns |
| `outputs/reports/PARAM_HISTORY.json` | Last 5 tuning rounds with metrics |
| `hf_cache/sd15/` | Local SD1.5 snapshot for offline inference |
| `<batch_dir>/source_map.json` | Per-batch output→source image path mapping |

### Gradio/Batch directory convention

Generated images go to `outputs/generated/毕业论文_生成图像/<prefix>_<timestamp>_<seq>/`. The UI gallery only scans directories ending in `_000` within `毕业论文_生成图像/` and the top level of `generated/`. The `--output-base` flag in `run_full_report.py` explicitly targets `毕业论文_生成图像/` to keep UI-visible batches consistent.

### Label guard system

Label guard functions live in `scripts/core/label_guard.py` (decoupled from the generation script). `run_mammo_sd15.py` and `GenerationPipeline` both import from this shared module. The guard (`--legacy-label-guard`, enabled by default) detects and suppresses DICOM burn-in text artifacts (e.g., "R MLO") via heuristic + optional Qwen-VL verification. The training data in `CBIS_CLEAN_V2` has already been cleaned of mask-internal annotations, but the base SD1.5 model may still inject text patterns from its pre-training. **POST VL** runs for full-image when guard + key: **60 s** HTTP timeout **per image** (`_post_timeout`); there is **no longer** a skip path that disabled POST VL solely because the batch was full-image.

### .env and LLM advisor

Copy `.env.example` to `.env` and configure at least one text API backend. The advisor system (`scripts/assistant/ask_advisor.py`) uses this priority order when `ADVISOR_TEXT_BACKEND=auto`: DeepSeek > GLM > DashScope Qwen. Vision analysis uses Qwen-VL via DashScope. The `.env` file is git-ignored.

### Archive directory

`archive/` 按子系统组织已归档脚本，不在主线 pipeline 中使用：
- `archive/preprocessing/` — CRAFT+LaMa / SD-Inpaint 掩膜内文字擦除实验
- `archive/training/` — LoRA 训练辅助脚本（prepare_lora_dataset、train_lora_quick、select_lora_checkpoint 等，已归档）
- `archive/tuning/` — 超参数搜索实验
- `archive/evaluation/` — 独立评估分析工具（消融表、Top-5 对比、影像组学基线、模拟专家评分）
- `archive/assistant/` — 独立 CLI 工具（apply_next_run、next_run_params、quick_image_stats）
- `archive/postprocess/` — 实验性后处理修复
- `archive/core/` — `SDPipeline` / 后处理纯函数、`source_quality_vl`（未接入主线；生产生成见 `run_mammo_sd15.py`）

Do not modify or reference these in active development. Historical experiment conclusions are documented in `docs/developer/开发日志.md`.

## Current tuning bottleneck remediation plan

This section is an execution brief for Claude Code CLI when asked to fix the current image-quality tuning bottleneck. Do not continue blind parameter search before stabilizing the evaluation loop.

### Diagnosis to preserve

- The main bottleneck is the feedback loop, not a single generation parameter. Recent uncalibrated evaluations fail mainly on `SHAPE_ODD`, `BANDING`, `HIGH_BRISQUE`, `CONTOUR_FRACTURED`, and `OVEREXPOSED`; auto-calibrated runs can report `pass_rate=1.0` while Qwen-VL/visual review still rejects visible seams, plastic texture, weak skin lines, and abnormal images.
- Auto calibration currently over-relaxes some hard gates when a real baseline is passed. Observed examples: `min_banding_score` can become `0.0`, `min_circularity` about `0.097`, and `max_contour_concavity` about `1.284`. That makes pass/fail unsuitable as the only tuning target.
- Source image selection is a confounder. `run_mammo_sd15.py` has `--source-seed None` defaulting to a timestamp; Gradio leaves **「源图种子」** blank for variation, while tuning/A-B runs should fix `--source-seed`.
- Postprocess has been archived (2026-05-18). Frequency-domain correction was found to amplify high-frequency noise as scattered gray spots, making images look worse.

### Execution order

1. **Stabilize evaluation before changing generation.**
   - In `scripts/evaluation/review_generated_images.py`, separate "diagnostic strict thresholds" from "real-baseline calibrated scoring". Calibration may adjust soft scores, but it must not silently erase hard defect tags such as `BANDING`, `SHAPE_ODD`, and `CONTOUR_FRACTURED`.
   - Add summary fields that expose both views in one run: calibrated score/pass and strict defect tags/pass. Keep existing output keys backward compatible where possible.
   - If changing thresholds, document the rationale in `docs/developer/开发日志.md`.

2. **Make source selection reproducible.**
   - In `scripts/generation/run_mammo_sd15.py` and assistant wrappers, add a standard validation path that uses a fixed `--source-seed`, writes `source_map.json`, and optionally reuses the same source list across A/B runs.
   - Prefer adding a small "golden source set" or source whitelist for 4-8 representative MLO dense images before evaluating parameter changes.

3. **Run controlled ablations, one variable at a time.**
   - Baseline command should fix `--seed`, `--source-seed`, `--metadata-csv`, `--filter-view MLO`, `--filter-density dense`, `--num-images`, `--eval-profile full`, and the same source set.
   - Compare only one of these at a time: `strength`, `guidance_scale`, `num_steps`.
   - Record each run's command, source map, `summary.json`, and a short visual verdict. Do not accept `pass_rate` alone as success.

4. **Attack the image defects in dependency order.**
   - Control exposure: adjust strength and guidance_scale after source selection is fixed. Re-check `OVEREXPOSED`, `mean_brisque`, and visual texture.
   - Then address shape: filter or stratify sources by mask ratio/contour quality before adjusting `strength`. Shape failures caused by tiny or irregular source breasts should not be treated as a pure diffusion parameter problem.

5. **Update the tuning/reporting loop.**
   - In `scripts/assistant/run_generate_eval_advise.py` and `run_full_report.py`, make reports clearly state whether evaluation used real-baseline calibration, strict defect gates, Qwen-VL, and fixed source seed.
   - Prevent advisor recommendations from being based on inconsistent metrics. If auto-calibrated pass is high but strict defects or visual review fail, report the run as not solved.
   - Keep `LATEST_NEXT_RUN.json` useful, but require it to include the evaluation mode and source-seed assumptions that produced the recommendation.

### Suggested validation commands

```bash
# Strict diagnostic evaluation on a fixed generated batch
python3 scripts/evaluation/review_generated_images.py \
  --images-dir outputs/generated/<batch_dir> \
  --output-dir outputs/eval/<eval_name> \
  --no-recursive --eval-profile full --enable-seam-check \
  --no-auto-calibrate

# Calibrated evaluation with a density-matched real pool (example: scattered jpeg pool)
python3 scripts/evaluation/review_generated_images.py \
  --images-dir outputs/generated/<batch_dir> \
  --output-dir outputs/eval/<eval_name> \
  --no-recursive --eval-profile full --enable-seam-check \
  --real-images-dir datasets/jpeg

# Full-image generation (default mode) with capped save resolution
python3 scripts/generation/run_mammo_sd15.py \
  --base-model-local hf_cache/sd15 \
  --lora-path outputs/lora/mammo_sd15_v6_allMLO/final_lora \
  --metadata-csv datasets/CBIS_CLEAN_V2/metadata_clean.csv \
  --filter-view MLO --filter-density scattered \
  --num-images 4 --seed 2026 --source-seed 20260513 \
  --mode full-image \
  --fullimage-min-short-side 384 \
  --fullimage-output-long-side 2048 \
  --scheduler dpm --num-steps 20 \
  --strength 0.5 --guidance-scale 8.5
```

### Done criteria

- A fixed-source A/B run can be reproduced with the same `source_map.json`.
- Reports show strict and calibrated evaluation results side by side.
- A run is only considered improved if strict defects decrease, visual review no longer rejects obvious plastic texture/skin-line failures, and BRISQUE or texture metrics do not regress materially.
- Documentation in `docs/developer/开发日志.md` and `CLAUDE.md` matches the implemented defaults and the current recommended tuning protocol.

## Code quality standards

- **Keep it simple** — single responsibility per module; no premature abstraction; no dead code; flat call chains (≤3 levels deep)
- **Type hints** — annotate function signatures; use `| None` not `Optional`; prefer dataclass over raw dict
- **Logging over print** — use `logging.getLogger(__name__)` for production paths; `print()` only in CLI entry points and debug scripts
- **Fail fast** — validate at system boundaries (user input, file I/O, API responses); no `except: pass`; error messages must be actionable
- **Resource cleanup** — use `with` / context managers for files, GPU memory, network connections
- **No secrets in code or logs** — keys and tokens live in `.env` only
- **Trust internal callers, validate external inputs** — don't re-validate your own output; do validate everything from outside
