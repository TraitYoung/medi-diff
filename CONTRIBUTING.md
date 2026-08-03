# Contributing to MammoGen

Thanks for interest in improving this project. Short rules keep reviews fast.

Project front doors: [README (English)](README.md) · [README (中文)](README.zh-CN.md) · [docs/en](docs/en/README.md).

## Before you start

1. Open an issue for non-trivial changes (API, eval thresholds, training defaults).
2. Keep PRs focused — one concern per PR.
3. Do **not** commit weights, datasets, `.env`, or generated batches (`outputs/`, `datasets/`, `hf_cache/`).
4. Security issues: follow [`SECURITY.md`](SECURITY.md) (do not file public issues for vulns).

## Dev setup

```bash
git clone https://github.com/TraitYoung/medi-diff.git
cd medi-diff
pip install -r requirements-dev.txt   # editable install + pytest/ruff/numpy/opencv
# Full GPU stack (optional on the machine that actually trains/infers):
pip install -r requirements.txt
```

## Checks (same as CI)

```bash
ruff check scripts/core scripts/assistant/tuning_state.py scripts/tests apps/__init__.py
pytest -m "not heavy"
```

Heavy tests (torch / fullimg path) on a GPU box:

```bash
pytest -m heavy
```

## Code style

- Prefer small, readable functions; match existing naming in `scripts/core/`.
- Generation defaults live in `scripts/core/pipeline_config.py` (`GenParams`) — do not fork silent defaults in the UI.
- Medical/ethics: never claim clinical diagnostic use; keep the README disclaimer intact.

## Pull requests

- Describe **why** and how you tested.
- Update docs when behavior or paths change.
- New Python modules under `scripts/` should include at least one light unit test when practical.
