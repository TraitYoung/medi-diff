# Changelog

All notable changes to **MammoGen** (`medi-diff`) are documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [1.0.0] — 2026-08-01

### Added

- MIT `LICENSE` and open-source project branding (**MammoGen**).
- `pyproject.toml` editable install (`pip install -e .` / `.[dev]` / `.[full]`).
- GitHub Actions CI: `ruff` + CPU `pytest` (`-m "not heavy"`).
- Unit tests for `pipeline_config`, `image_utils`, `label_guard`, `tuning_state`.
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, issue / PR templates.
- Hugging Face packaging under `hf/` + `scripts/tools/publish_hf_assets.py`
  (model card, dataset card, static gallery Space).

### Changed

- Default generated-image root: `outputs/generated/samples/`
  (was `outputs/generated/毕业论文_生成图像/`).
- README rewritten for open-source onboarding (EN primary + Chinese positioning).
- Defense / thesis-only docs excluded from git tracking via `.gitignore`.

### Notes

- Not for clinical diagnosis.
- Hugging Face remote upload requires `HF_TOKEN` and local LoRA / metadata assets;
  see `hf/README.md`.
