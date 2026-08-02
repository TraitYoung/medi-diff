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

