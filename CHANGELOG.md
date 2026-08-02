# Changelog

All notable changes to **MammoGen** (`medi-diff`) are documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `SECURITY.md`, `CITATION.cff`, public `ROADMAP.md`, `docs/README.md` index.
- UI screenshots under `docs/assets/ui-*.png` (English filenames).

### Changed

- Gradio / `start.sh` branding unified to **MammoGen**.
- Removed unrelated SpecForge / job-description docs from the public tree.
- Moved author migration notes to `docs/private/`.
- Scrubbed thesis-oriented wording from public evaluation docs and FAQ.
- `apps/README.md` defaults aligned with `GenParams` (`strength=0.44`, CFG `7.5`, steps `40`, output long side `2048`).
- Dropped cloud-host hard-coded JPEG candidate paths.

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

- Default generated-image root: `outputs/generated/samples/`.
- README rewritten for open-source onboarding (EN primary + Chinese positioning).
- Internal-only drafts excluded from git tracking via `.gitignore`.

### Notes

- Not for clinical diagnosis.
- Hugging Face remote upload requires `HF_TOKEN` and local LoRA / metadata assets;
  see `hf/README.md`.
