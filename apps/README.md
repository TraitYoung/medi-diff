# `apps/` 一览

| 文件 | 作用 |
|------|------|
| `start.sh` / `stop.sh` | 一键拉起 **FastAPI (`api_server`)** + **Gradio (`app_gradio`)**，或仅其一；端口清理与 Gradio 监听自检。 |
| `app_gradio.py` | **答辩/日常主入口**：通过 `subprocess` 调用仓库内 `scripts/`，**不经由** HTTP 调用本机 API（与 API 并行、逻辑独立）。 |
| `api_server.py` | REST v2：异步任务封装同一套脚本，供联调或自动化客户端使用。 |
| `port_util.sh` | 被 `start.sh` 引用的端口释放工具。 |
| `listen_port_of_pid.py` | 解析进程实际 `LISTEN` 端口（Gradio 漂移端口时打印正确 URL）。 |

**输出目录约定**：Gradio「生成」沿用 `run_mammo_sd15.py` 默认根目录 `outputs/generated/毕业论文_生成图像/`。「一键流水线」经 `run_full_report.py` **显式传入同一 `--output-base`**，与画廊/评估下拉一致。

**训练数据集来源**：界面顶栏展示 [`CBIS-DDSM-DATASET`](https://github.com/sposso/CBIS-DDSM-DATASET)。本项目使用其 CBIS-DDSM 数据基础，并在仓库内维护清洗后的元数据路径 `datasets/CBIS_CLEAN_V2/metadata_clean.csv`。

**「生成」标签（Gradio）**

- **单一流程页**：速度预设（快速 / 精细）、体位与 **密度（默认 `scattered`）**、频域后处理等。答辩版固定走 **全图模式**，不再展示 Patch 拼接、LoRA 路径、LoRA scale 或 Negative Prompt。
- **全图** 模式会传 `--mode full-image` 与 **`--fullimage-output-long-side 1024`**，便于 Gradio 预览与快速评估；论文留档需要大图时可在 CLI 中手动改回 2048。
- **质量调优默认**：快速预设 `strength=0.34, guidance=6.2`；精细预设 `strength=0.36, guidance=6.5`，优先保留 MLO 胸大肌、IMF 和乳腺条索结构。
- **LoRA** 由程序内部固定为 `outputs/lora/mammo_sd15_v6_allMLO/final_lora`；如需研究性切换权重或负向提示词，请使用 CLI/API，不放在答辩界面中。
- 顾问写入的 `outputs/reports/LATEST_NEXT_RUN.json`：推荐用 CLI **`run_full_report.py --from-latest-tuning`** 复跑；源码中 `load_latest_next_run_into_tuning()` 目前**未绑定**到前端按钮。历史与回滚见 **「调参历史」** Tab（`PARAM_HISTORY.json` / `--rollback` 逻辑）。

上线前可执行：`python3 scripts/tools/verify_ui_wiring.py`（参见 `docs/用户操作手册.md` 「上线前自检」）。
