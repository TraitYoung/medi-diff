# `apps/` — Gradio UI & FastAPI

| File | Role |
|------|------|
| `start.sh` / `stop.sh` | Launch **FastAPI (`api_server`)** + **Gradio (`app_gradio`)**, or either alone; port cleanup and Gradio listen self-check. |
| `app_gradio.py` | **Daily entry**: calls in-repo `scripts/` via `subprocess` — does **not** call the local HTTP API (parallel, independent). |
| `api_server.py` | REST v2: async jobs wrapping the same scripts for automation clients. |
| `port_util.sh` | Port helpers used by `start.sh`. |
| `listen_port_of_pid.py` | Resolve actual `LISTEN` port when Gradio drifts. |

**Output root:** Gradio「生成」writes under `outputs/generated/samples/`. One-click pipeline (`run_full_report.py`) passes the same `--output-base` so gallery / eval dropdowns stay aligned.

**Dataset attribution:** header links [`CBIS-DDSM-DATASET`](https://github.com/sposso/CBIS-DDSM-DATASET). Canonical cleaned metadata: `datasets/CBIS_CLEAN_V2/metadata_clean.csv`.

**「生成」tab (Gradio)**

- Streamlined panel: speed-oriented controls, view / **density (default `scattered`)**. Fixed **full-image** mode (no patch UI).
- Passes `--mode full-image` and **`--fullimage-output-long-side 2048`** (same as CLI defaults).
- Defaults aligned with `GenParams` / CLI: **`strength=0.44`**, **`guidance_scale=7.5`**, **`num_steps=40`**, scheduler DPM.
- LoRA fixed to `outputs/lora/mammo_sd15_v6_allMLO/final_lora`. Switch weights / negative prompts via CLI or API.
- Advisor JSON `outputs/reports/LATEST_NEXT_RUN.json`: re-run with CLI `run_full_report.py --from-latest-tuning`. `load_latest_next_run_into_tuning()` is **not** wired to a UI button (experimental). Use **「调参历史」** for rollback (`PARAM_HISTORY.json`).

Smoke check: `python3 scripts/tools/verify_ui_wiring.py` (see [用户操作手册](../docs/developer/用户操作手册.md)).
