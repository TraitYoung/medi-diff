# 基于扩散模型的乳腺钼靶图像生成系统

> 山东建筑大学计算机科学与技术学院 · 本科毕业设计  
> 课题：基于扩散模型的乳腺钼靶图像生成系统设计与实现

---

## 目录

| 章节 | 内容 |
|------|------|
| [系统简介](#系统简介) | 能力与模块总览 |
| [快速开始](#快速开始) | 环境、可选 NLP API、流水线、界面与 API |
| [目录结构](#目录结构) | 仓库文件树导读 |
| [主要参数说明](#主要参数说明) | 生成侧推荐区间 |
| [评审指标说明](#评审指标说明) | 违规类型与分组 |
| [归档路线说明](#归档路线说明) | 历史实验脚本 |
| [论文展示建议](#论文展示建议) | 定稿展示流程 |

### 配套文档（`docs/`）

| 文件 | 说明 |
|------|------|
| [用户操作手册](docs/用户操作手册.md) | 安装、界面、FAQ |
| [apps 入口说明](apps/README.md) | `start.sh`、Gradio vs FastAPI、`--output-base` 约定 |
| [评价体系说明](docs/评价体系说明.md) | **双轨指标**：域内规则体系与 FID/PRC/sFID 等通用指标的分工 |
| [API 接口文档](docs/API接口文档.md) | REST、`JobRecord`、`curl` 示例 |
| [项目结构与目录职责](docs/项目结构说明_交接版.md) | 路径职责、归档与迁移记录 |
| [开发日志](docs/开发日志.md) | 按日期的工程与实验纪要 |
| [项目交付检查清单](docs/项目交付检查清单.md) | 任务书与交付核对 |
| [评估标准说明（分项公式）](docs/评估标准说明_交接版.md) | 16 维分项与校准 |
| [模拟专家评分表](docs/模拟专家评分表.md) | 主观评分模板 |
| [开发者自用备忘](docs/developer/README.md) | **个人维护**：协作心法、指标体系怎么记、常见坑（非答辩口径） |

---

## 系统简介

本系统以 **Stable Diffusion 1.5 + 自训 LoRA（CBIS-DDSM）** 为核心：答辩版前端默认 **全图单次 img2img（`--mode full-image`）**，无 Patch 网格接缝；Patch-Overlap 仅保留为 CLI 研究回退路线。

### 核心能力

| 模块 | 说明 |
|------|------|
| **数据预处理** | CBIS-DDSM JPEG 筛选、乳腺区域 mask 提取、密度/体位/侧别标签结构化 |
| **条件生成** | SD1.5 + LoRA；**默认全图单次推理**（`fullimage_generate`）；CLI 保留 Patch overlap + 金字塔融合等研究参数 |
| **质量评审** | **域内**：16 维规则分项 + **`hard_tags`** / **`soft_reasons`**（如 **`SKIN_LINE_MISSING`**、**`GRID_SEAM`** 在 `eval_profile=full` 多为软惩罚，不参与硬否决）；**`--real-images-dir`** 驱动真实分布校准与学术指标。**通用**：FID/KID、P/R、sFID |
| **自动化报告（可选）** | `run_full_report.py` 等脚本可经由 `.env` 调用第三方 **文本 API**（默认 DeepSeek，可改用 Qwen）与 **DashScope Qwen‑VL** 看图，产出 Markdown 报告与下一轮参数的 JSON（实现见 `scripts/assistant/`） |
| **前端界面** | Gradio：**「生成」**为单面板（预设 + 全图；密度默认 **scattered**；全图默认保存长边 **1024** 便于预览）；另有 **画廊 / 评估 / 一键流水线 / 调参历史** 等 |
| **后端 API** | FastAPI RESTful API，支持异步任务队列 |

### 主线数据、权重与推理默认

- **训练数据集来源**：[`sposso/CBIS-DDSM-DATASET`](https://github.com/sposso/CBIS-DDSM-DATASET)；本项目在此基础上进行 JPEG 筛选、标签清洗、mask 提取与 `CBIS_CLEAN_V2/metadata_clean.csv` 结构化整理。
- **主线权重（Gradio 默认）**：`outputs/lora/mammo_sd15_v6_allMLO/final_lora`；答辩版前端已隐藏 LoRA 路径、LoRA scale 与 Negative Prompt，避免误操作。
- **实验线**：已从 **均衡密度 captions** 构建 **LoRA v6** 数据集与训练脚本（参见 [`docs/开发日志.md`](docs/开发日志.md) 同日条目）；如需切换权重，请通过 CLI 或 API 显式指定 **`--lora-path`**。
- **默认行为**：`run_mammo_sd15.py` 默认 **全图模式**、默认开启 `--source-quality-sort`、可选后处理与标签守护；关闭分别用 **`--mode patch`**、`--no-postprocess`、`--no-legacy-label-guard`
- **后处理（生成脚本 `--postprocess`）**：沿用频域链路，但其中 **`blend=0.4`**、**`fill_voids=False`**（避免评审 **ARTIFACT_BUBBLES** 假阳性）；与独立 `postprocess_freq.py` CLI 的默认 **`PostprocessParams`** 不尽相同
- **密度与 Prompt**：当 **`--filter-density`** 恰好为单个档位（fatty/scattered/heterogeneous/dense）时，正向 prompt 会与训练 caption **英文密度短语**及对 **`--filter-view`（MLO/CC）** 的体位措辞 **自动对齐**
- **源图**：**`filter_source_pool`** 会筛掉极高长宽比图源（默认 **`--max-source-aspect-ratio 2.2`**）；全图推理隐式 **`fullimage`** 缩放带 **较长边推理 ≤1024** 的安全封顶，并按需保证短边 **`--fullimage-min-short-side`（默认 384）**
- **金属标记**：检出生成图中的 **_BB 状高亮标记** → 写入前整条丢弃并重抽图源； **`--postprocess`** 末尾再次扫描并对残留做 **原位涂补中值**
- **标签守护**：开启 **`--legacy-label-guard`** 且有 **`QWEN_API_KEY`** 时，批次结束跑 **POST Qwen‑VL**，**每张图 HTTP 超时 60s**，**patch / 全图均执行**（已移除「仅因全图就跳过 VL」的旧逻辑）；角落 OCR 清除函数保留于仓库但 **主流程已不再调用**，改依赖 VL

---

## 快速开始

### 环境安装

```bash
pip install -r requirements.txt
```

### 配置可选 API（`.env`）

流水线脚本可通过第三方文本 API（DeepSeek > GLM > Qwen）和 DashScope Qwen-VL 生成分析报告。密钥放在 `.env`，勿提交仓库。

```bash
cp .env.example .env
# 编辑 .env，至少配置一个文本 API key
# 默认 --advisor-mode text_only（仅文本）；加 both 时调 Qwen-VL 看图
```

### 一键全自动流水线（推荐入口）

```bash
# 主入口：生成 6 张 MLO dense → 自动评审 → 文本/视觉分析报告 → FINAL_REPORT.md
python3 scripts/assistant/run_generate_eval_advise.py \
  --metadata-csv datasets/CBIS_CLEAN_V2/metadata_clean.csv \
  --filter-view MLO --filter-density dense \
  --num-images 6 --tag-prefix my_run

# 便捷包装器（通过子进程调用 run_generate_eval_advise；默认生成模式为 full-image）
python3 scripts/assistant/run_full_report.py \
  --metadata-csv datasets/CBIS_CLEAN_V2/metadata_clean.csv \
  --filter-view MLO --filter-density dense \
  --num-images 6 --tag-prefix my_run

# 查看调参历史
python3 scripts/assistant/run_full_report.py --show-history

# 基于最近一次 API 输出的参数 JSON 重跑下一轮
python3 scripts/assistant/run_full_report.py --from-latest-tuning \
  --metadata-csv datasets/CBIS_CLEAN_V2/metadata_clean.csv \
  --filter-view MLO --filter-density dense

# 回滚到历史第 2 条参数重跑
python3 scripts/assistant/run_full_report.py --rollback 2 \
  --metadata-csv datasets/CBIS_CLEAN_V2/metadata_clean.csv
```

- **成本控制**：流水线默认 **`--advisor-mode text_only`**（仅调用文本接口）；需在报告中加入视觉分析时使用 **`--advisor-mode both`**（将调用 DashScope **Qwen‑VL**，费用与配额以云控制台为准）。
- **`--compact-advisor`**：压缩报送 API 的历史与统计数据行数以降低单次调用上下文长度。
- **`--no-qwen-vl`**：显式跳过多模态阶段。

### 单独生成

```bash
# 主线：全图（默认）；保存长边 capped 便于 UI/评测内存（0=不缩放）
python3 scripts/generation/run_mammo_sd15.py \
  --base-model-local hf_cache/sd15 \
  --lora-path outputs/lora/mammo_sd15_v4_clean/final_lora \
  --metadata-csv datasets/CBIS_CLEAN_V2/metadata_clean.csv \
  --filter-view MLO --filter-density scattered \
  --num-images 6 --seed 2026 \
  --mode full-image \
  --fullimage-output-long-side 2048 \
  --fullimage-min-short-side 384 \
  --scheduler dpm --num-steps 50 \
  --strength 0.36 --guidance-scale 6.5 \
  --output-subdir-prefix demo_run

# 研究回退：Patch 拼接（仅 CLI，Gradio 答辩版已隐藏）
python3 scripts/generation/run_mammo_sd15.py \
  --mode patch \
  --base-model-local hf_cache/sd15 \
  --lora-path outputs/lora/mammo_sd15_v4_clean/final_lora \
  --metadata-csv datasets/CBIS_CLEAN_V2/metadata_clean.csv \
  --filter-view MLO --filter-density dense \
  --num-images 6 --seed 2026 \
  --strength 0.47 --overlap-ratio 0.90 \
  --scheduler dpm --num-steps 50 --patch-size 640 \
  --output-subdir-prefix demo_patch
```

### 单独评审

```bash
# 不传 real-images-dir 则不做该路径下的域内分项基线校准与 FID 等 academic_metrics（若依赖未齐也会置空）
python3 scripts/evaluation/review_generated_images.py \
  --images-dir outputs/generated/demo_run_20260503_000000_000 \
  --output-dir outputs/eval/demo_run \
  --no-recursive --eval-profile full --enable-seam-check \
  --real-images-dir datasets/jpeg
```

### 多轮对比表

```bash
python3 scripts/evaluation/compare_runs.py --runs-dir outputs/eval
```

### 一键启动 Gradio + FastAPI（推荐）

在项目根目录执行：

```bash
bash apps/start.sh
```

| 服务 | 地址 | 说明 |
|------|------|------|
| **Gradio** | `http://127.0.0.1:7860` | 「生成」单页 + 画廊 / 流水线 / **调参历史**（回滚）等；详见 [`apps/README.md`](apps/README.md) |
| **FastAPI** | `http://127.0.0.1:8000` | Swagger：`/docs`；健康检查：`GET /health` |

停止：`bash apps/stop.sh` 或 Ctrl+C。开发热重载：`bash apps/start.sh --api-reload`。完整参数见 `bash apps/start.sh --help`。

### 单独启动（调试）

```bash
# 仅 Gradio
python3 apps/app_gradio.py --host 0.0.0.0 --port 7860

# 仅 FastAPI（是否加 --reload 自行决定）
uvicorn apps.api_server:app --host 0.0.0.0 --port 8000
# 接口文档：http://127.0.0.1:8000/docs
```

---

## 目录结构

| 目录 | 说明 |
|------|------|
| `apps/` | Gradio 前端（生成单面板 + Tab 拆分）+ FastAPI REST v2 + 启停脚本 |
| `scripts/core/` | **共享核心库**：PipelineConfig、SDPipeline、后处理、标签守护、编排器 |
| `scripts/generation/` | 主线生成（`run_mammo_sd15.py`）+ 归档路线（SDXL/ControlNet 等） |
| `scripts/training/` | LoRA 训练（`train_lora_quick.py`）+ 数据集准备 |
| `scripts/evaluation/` | 评审（域内规则 + FID/PRC/sFID）、多轮对比、消融 |
| `scripts/preprocessing/` | CBIS-DDSM 清洗、mask 提取、burn-in 清除（CRAFT+LaMa） |
| `scripts/postprocess/` | 频域后处理 CLI（核心在 `scripts/core/postprocess_pipeline.py`） |
| `scripts/assistant/` | 全流程报告（可选 DeepSeek/Qwen 文本/VL 顾问） |
| `scripts/tests/` | 含 `test_fullimage_output_size.py`（全图输出长边封顶）与 `run_ablation.py` 等 |
| `docs/` | 任务书、开发日志、用户手册、API 文档、评价体系说明 |
| `datasets/` | CBIS-DDSM JPEG + `CBIS_CLEAN_V2/metadata_clean.csv`（主线；训练数据集来源见 [`sposso/CBIS-DDSM-DATASET`](https://github.com/sposso/CBIS-DDSM-DATASET)） |
| `outputs/` | 生成图、评审、LoRA 权重、报告、日志 |
| `hf_cache/` | SD1.5 本地离线快照 |
| `archive/` | 废案与旧备份（不参与主流程） |

详细路径说明见 [`docs/项目结构说明_交接版.md`](docs/项目结构说明_交接版.md)。

---

## 主要参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--mode` | `full-image`（默认）单次整图 diffusion；`patch` 仅用于 CLI 研究回退 | full-image |
| `--fullimage-long-side` | 推理前缩放目标的**长边**（8 对齐），且隐式不会超过 **1024** 推断安全顶 | 768 |
| `--fullimage-min-short-side` | 高长宽比片保证短边不低于该值（设为 **0** 关闭） | 384 |
| `--fullimage-output-long-side` | 保存 PNG **长边上界**（**0**=按原图源最大边）；Gradio 默认传 **1024**，论文留档可手动传 **2048** | CLI 默认 2048 |
| `--max-source-aspect-ratio` | `filter_source_pool` 丢弃更极端比例的源 JPEG | 2.2 |
| `--strength` | img2img；答辩版为保留胸大肌/IMF 轮廓下调 | 0.34–0.36 |
| `--overlap-ratio` | Patch 重叠率；仅 CLI patch 模式使用，Gradio 不暴露 | **0.88–0.90** |
| `--guidance-scale` | CFG 强度；过高易放大水墨纹理、白点和边缘伪影 | 6.2–6.5 |
| `--global-guide-blend` | 全局引导混合比；**超过 0.70 极易导致 TOO_UNIFORM/BANDING** | 0.30–0.44 |
| `--global-guide-strength` | 全局引导低分辨率 img2img 强度；控制布局约束程度 | 0.25–0.30 |
| `--blend-sigma-divisor` | 接缝高斯 σ = overlap_px/divisor；越小σ越大融合越柔 | 1.20–1.50 |
| `--gabor-alpha` | Gabor 方向纹理增强强度 | 0.50–0.55 |
| `--scheduler` | 扩散调度器：dpm=DPM-Solver++(推荐), ddim, pndm | dpm |
| `--num-steps` | DPM-Solver++ 步数；全图主线常见 **50**（省时可试 40） | 50 |
| `--patch-size` | Patch 边长；仅 CLI patch 模式使用，Gradio 不暴露 | 640 |
| `--blend-mode` | 融合模式：hann(默认), pyramid(多频带Laplacian), gaussian, linear | hann |

---

## 评审指标说明

系统评审使用 **两套互补指标**，详见 **[`docs/评价体系说明.md`](docs/评价体系说明.md)**：

1. **域内规则体系**：A～F 组 16 维分项（构图/灰度/纹理/伪影/分布/解剖）+ BRISQUE + 功率谱 β，产出 `ok`/`total_score`，用于筛图与调参。
2. **通用深度学习指标**：FID/KID、Precision–Recall、sfid_spatial768（写入 `summary.json → academic_metrics`），用于横向对比。

分项公式见 [`docs/评估标准说明_交接版.md`](docs/评估标准说明_交接版.md)。  
**注意**：`eval_profile=full` 下 **`GRID_SEAM`**、**`SKIN_LINE_MISSING`** 等多走 **`soft_reasons`**，只影响语义分与排序，**不作为 `hard_tags` 一票否决**；调参请同时看 **`strict_pass_rate`** 与肉眼。自动校准请让 **`--real-images-dir`**（或流水线/Gradio 自动解析的目录）与 **`--filter-density`** 所代表的 **真实钼靶子集**一致，避免 dense 生成却用 scattered 基线（或反之）。
---

## 归档路线说明

以下路线在早期实验中使用，结论已记录在 `docs/开发日志.md`，脚本保留用于复现：

| 路线 | 主要脚本 | 结论 |
|------|----------|------|
| SDXL Inpaint | `run_mammo_inpaint.py` | 生成较自然，但计算成本高，接缝更严重 |
| SDXL img2img | `run_mammo_img2img.py` | 结构更弱，不如主线 |
| T2I-Adapter | `run_mammo_adapter.py` | 条件控制精准，但训练成本高 |
| ControlNet | `run_mammo_controlnet.py` | 解剖约束强，但对 mask 质量敏感 |

---

## 论文展示建议

1. **固定参数生成**：使用 **`mammo_sd15_v4_clean`** + **`datasets/CBIS_CLEAN_V2/metadata_clean.csv`** 条件源图，`run_mammo_sd15.py` 固定 **`--seed`** / **`--source-seed`**，生成 ≥30 张候选。
2. **自动筛选**：`review_generated_images.py` 评审后，仅取 `rank_tier=1` 的图像进入展示。
3. **指标写法**：参见 `docs/评价体系说明.md`：规则分项为主，`academic_metrics` 中与 FID/PRC 等可比指标并列说明及局限。
4. **多轮对比**：`compare_runs.py` 汇总各轮 pass_rate / mean_score / BRISQUE 对比表。
5. **可视化**：`make_top5_compare.py` 生成真实图 vs 生成图对比条。
6. **消融**：`scripts/tests/run_ablation.py` 可复现单变量对照实验（固定 seed、同源图、逐变量改动），自动汇总对比表；历史消融记录见 `docs/开发日志.md` 2026-05-08 条目。
