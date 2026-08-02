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

**「生成」 / 「一键流水线」**

- Speed preset (default **本地流畅**):
  - **本地流畅** → `num_steps=28`, `--mem-profile local` (attention/VAE slicing; for ~12–16 GiB GPUs such as 5070 Ti)
  - **云端质量** → `num_steps=40`, `--mem-profile auto`
- Sampling defaults otherwise match `GenParams`: **`strength=0.44`**, **`guidance_scale=7.5`**, full-image, **`--fullimage-output-long-side 2048`**.
- LoRA fixed to `outputs/lora/mammo_sd15_v6_allMLO/final_lora`. Switch weights / negative prompts via CLI or API.
- Advisor JSON `outputs/reports/LATEST_NEXT_RUN.json`: re-run with CLI `run_full_report.py --from-latest-tuning`. `load_latest_next_run_into_tuning()` is **not** wired to a UI button (experimental). Use **「调参历史」** for rollback (`PARAM_HISTORY.json`).

Smoke checks:

```bash
python3 scripts/tools/verify_ui_wiring.py
python3 scripts/tools/check_local_gpu.py
```
