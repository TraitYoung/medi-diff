# Evaluation (dual track)

**Audience:** ML / medical-imaging engineers reading MammoGen QC outputs.

Chinese detail: [`../评价体系说明.md`](../评价体系说明.md) · formulas: [`../评估标准说明.md`](../评估标准说明.md) · showcase gates: [`../乳腺钼靶生成图评价标准_v3.md`](../乳腺钼靶生成图评价标准_v3.md).

> Do **not** merge the two tracks into one “clinical quality” score. They answer different questions.

---

## 1. Why two tracks

| Concern | Question | MammoGen track |
|---------|----------|----------------|
| In-domain anatomy / imaging proxies | Does this look like a plausible FFDM MLO (shape, texture, artifacts)? | **Rule metrics** (16 dims, hard/soft tags, `ok`, pass rate) |
| Generative-model comparability | How do batches compare under FID-like distances? | **Generic deep features** (`academic_metrics`: FID/KID/PRC/sFID) |

Inception features are ImageNet-pretrained and **not** mammography-optimal. Use FID-family metrics for relative ablations, not as radiology acceptance.

---

## 2. Rule track (primary)

Implemented in `scripts/evaluation/review_generated_images.py`.

- Groups **A–F** (composition/anatomy, grayscale, texture/spectrum, artifacts, real-distribution deviation, no-reference proxies).
- Aggregate **0–100** `total_score` plus per-image `ok` / `hard_tags` / `soft_reasons`.
- With `--real-images-dir`, thresholds can auto-calibrate from a real pool — useful for soft scores, but **must not erase** hard defect tags such as `BANDING`, `SHAPE_ODD`, `CONTOUR_FRACTURED`.
- Under `eval_profile=full`, some seam/skin-line tags are often **soft** penalties (lower semantic score without hard veto).

Honest expectation: uncalibrated strict `pass_rate` around **~50%** is common.

### Suggested CLI

```bash
# Strict diagnostic
python3 scripts/evaluation/review_generated_images.py \
  --images-dir outputs/generated/samples/<batch> \
  --output-dir outputs/eval/<name>_strict \
  --no-recursive --eval-profile full --enable-seam-check \
  --no-auto-calibrate

# With density-matched real baseline
python3 scripts/evaluation/review_generated_images.py \
  --images-dir outputs/generated/samples/<batch> \
  --output-dir outputs/eval/<name>_cal \
  --no-recursive --eval-profile full --enable-seam-check \
  --real-images-dir datasets/jpeg
```

Outputs: `summary.json`, `review_report.csv`, `top_k_*.txt`.

---

## 3. Generic track (supplementary)

When `--real-images-dir` is set and deps are installed (`piq`, `torch-fidelity`, `pytorch-fid`), batch-level fields land in `summary.json → academic_metrics`:

| Metric | Notes |
|--------|--------|
| FID / KID | Distribution distance; not mammography-specific |
| Precision / Recall / F (PRC) | Manifold PR via torch-fidelity — not clinical PR curves |
| `sfid_spatial768` | Patch-space Fréchet; for **within-project** relative compares |

Tiny batches may skip or degrade some metrics — that is expected.

---

## 4. How to report results

1. Lead with rule-track pass rate, dominant failure tags, and calibration assumptions (`source-seed`, density filter, auto-calibrate on/off).
2. Optionally report `academic_metrics` for ablation tables; state Inception domain shift.
3. Avoid “FID improved ⇒ clinically usable” or “high total_score replaces radiologist review”.

Related scripts: `compare_runs.py` (active); ablation/top-5 helpers under `archive/evaluation/` (unsupported).
