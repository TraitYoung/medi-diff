# Roadmap

Public product roadmap for **MammoGen**. Author migration notes live under `docs/private/` and are not part of the user-facing docs index.

## Shipped (1.0)

- [x] MIT license, bilingual README, architecture overview
- [x] `pip install -e .` / `.[dev]` / `.[full]` via `pyproject.toml`
- [x] CI: `ruff` + `pytest -m "not heavy"`
- [x] Default output root `outputs/generated/samples/`
- [x] HF packaging under `hf/` + `publish_hf_assets.py`

## Next

- [ ] Publish HF remotes: LoRA weights, cleaned metadata, static gallery Space
- [ ] Standalone QC CLI (rule engine usable without the generation stack)
- [ ] Standalone label-guard CLI (DICOM burn-in cleanup)
- [ ] Golden source set (`datasets/golden_sources.json`) for reproducible A/B runs
- [ ] Blog series: burn-in labels, full-image vs patch, rule-based QC lessons
- [ ] Optional Zenodo DOI for stable citation

## Principles

- Prefer honest failure modes (banding, shape oddities, ~50% strict pass rate) over inflated metrics.
- Do not claim clinical diagnostic use.
- Stabilize evaluation before blind hyperparameter search.
