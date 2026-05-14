# API 接口文档

> **版本**：REST API v2.0 · FastAPI  
> **本地默认**：Gradio **7860**、API **8000**（参见 `README.md`、`bash apps/start.sh --help`）  
> **交互式文档**：`http://127.0.0.1:8000/docs`  
> **Gradio（7860）**：**「生成」** 为**单面板**（速度预设 · 密度默认 scattered · 高级采样控件隐藏）；一键流水线 / 画廊 / **调参历史（回滚）** 等仍为独立 Tab。顾问 JSON 载入 `LATEST_NEXT_RUN.json` 以 CLI **`run_full_report.py --from-latest-tuning`** 为主。Gradio 与 API **并行入口**，不经 REST 调脚本——详见 **`apps/README.md`**。

---

## 目录（自上而下）

| 小节 | 说明 |
|------|------|
| [启动](#api-start) | `bash apps/start.sh` / `uvicorn` |
| [基础约定](#api-meta) | URL、异步任务 |
| [系统](#api-health) | `GET /health` |
| [生成](#api-generate) | `POST /generate/sd15` · `sdxl` |
| [评估](#api-review) | `POST /review` |
| [批次与评估](#api-batches) | `GET /batches`、`/evaluations`、`/results` |
| [报告](#api-reports) | `/reports/latest`、`/history` |
| [任务](#api-jobs) | `JobRecord`、`GET /jobs` |
| [静态](#api-static) | `/static/*` |
| [错误码](#api-errors) | HTTP |

> **提示**：请以 **`/docs`**（Swagger）为辅，对照本文字段表联调。

---

<a id="api-start"></a>

## 启动 API 服务

| 方式 | 命令 |
|------|------|
| 一键（API + Gradio） | `bash apps/start.sh` |
| 停止 / 释口 | `bash apps/stop.sh`（或结束 `start.sh` 前台进程） |
| 仅 API（开发，**默认不建议**对整仓 `--reload`） | `uvicorn apps.api_server:app --host 0.0.0.0 --port 8000` |
| 需要 API 热重载（与 Gradio 同机时优先） | `bash apps/start.sh --api-reload` |

**说明**：仓库内 **`apps/start.sh` 默认关闭** uvicorn **`--reload`**，以免与 Gradio 同改文件时反复抢占 **8000** 导致子进程异常退出；与界面联调仍用一键启动即可。

以下 `curl` 示例默认 **`http://localhost:8000`**；在云主机或映射端口时请替换 BASE URL。

---

<a id="api-meta"></a>

## 基础信息

| 项目 | 值 |
|------|----|
| 基础 URL | `http://localhost:8000` |
| 数据格式 | JSON |
| 认证 | 无（演示版本，内网部署建议） |
| 任务模型 | 异步：`POST` 返回 `job_id`，用 `GET /jobs/{id}` 轮询状态 |

---

<a id="api-health"></a>

## 系统接口（`GET /health`）

#### `GET /health` — 健康检查

检查系统各关键资源是否就绪。

**响应示例**

```json
{
  "ok": true,
  "root": "/root/autodl-tmp/medi-diff",
  "sd15_model_exists": true,
  "sd15_lora_exists": true,
  "sdxl_model_exists": false,
  "metadata_csv_exists": true,
  "real_images_dir_exists": true,
  "generated_dir_exists": true,
  "eval_dir_exists": true
}
```

---

<a id="api-generate"></a>

## 生成接口（`POST /generate/*`）

#### `POST /generate/sd15` — SD1.5 主线生成（推荐）

使用 SD1.5 + LoRA + Patch-Overlap img2img 生成乳腺钼靶图像。

**请求体（JSON）**

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `num_images` | int | 6 | 生成张数（1–50） |
| `seed` | int | 2026 | 随机种子 |
| `num_steps` | int | 52 | DDIM 推理步数（20–80） |
| `strength` | float | 0.42 | img2img 重绘强度（0.05–0.80） |
| `guidance_scale` | float | 7.9 | CFG 强度（1.0–12.0） |
| `overlap_ratio` | float | 0.86 | Patch 重叠率（0.40–0.95） |
| `global_guide_blend` | float | 0.58 | 全局引导混合比（0.0–0.70） |
| `blend_sigma_divisor` | float | 1.50 | 接缝高斯 σ 控制（1.05–3.0） |
| `lora_path` | string | 自动检测 | LoRA 权重目录；**省略时**服务端检测：若存在则优先 **`outputs/lora/mammo_sd15_v4_clean/final_lora`**，否则回退历史 v3 等路径（与 `apps/api_server.py` 一致） |
| `output_subdir_prefix` | string | `"api_sd15"` | 输出目录前缀 |
| `filter_view` | string | `"MLO"` | 体位筛选（MLO/CC/空=不限） |
| `filter_density` | string | `"dense"` | 密度筛选（dense/scattered/…/空=不限） |
| `eval_profile` | string | `"full"` | 评审档位（full/patch） |

**元数据 CSV（条件源图）**：本仓库 **FastAPI 实现**将条件源图 CSV 固定为模块级 **`METADATA_CSV`**：**若** `datasets/CBIS_CLEAN_V2/metadata_clean.csv` **存在则优先选用**，否则回退 `datasets/CBIS_CLEAN/metadata_clean.csv`（见 `apps/api_server.py`）。若需其它 CSV，应改后端常量或通过命令行 `run_mammo_sd15.py --metadata-csv` 独立跑批。

**请求示例**

```bash
curl -X POST http://localhost:8000/generate/sd15 \
  -H "Content-Type: application/json" \
  -d '{
    "num_images": 4,
    "seed": 42,
    "filter_view": "MLO",
    "filter_density": "dense",
    "output_subdir_prefix": "api_demo"
  }'
```

**响应**：`JobRecord`（见 [任务](#api-jobs)）。

---

#### `POST /generate/sdxl` — SDXL Inpaint 生成（归档）

历史路线，保留用于复现实验。主线请使用 `/generate/sd15`。

**请求体（JSON）**

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `num_images` | int | 3 | 生成张数 |
| `seed` | int | 2026 | 随机种子 |
| `steps` | int | 30 | 采样步数 |
| `strength` | float | 0.35 | 重绘强度 |
| `guidance_scale` | float | 2.0 | CFG 强度 |
| `output_subdir_prefix` | string | `"api_inpaint"` | 输出前缀 |
| `base_model` | string | 自动 | SDXL Inpaint 模型路径 |
| `lora_path` | string | `""` | LoRA 路径（留空不加载） |
| `lora_scale` | float | 0.70 | LoRA 权重缩放 |
| `local_files_only` | bool | true | 仅使用本地缓存 |
| `negative_prompt` | string | 默认 | 负向 prompt |

---

<a id="api-review"></a>

## 评估接口（`POST /review`）

评审后端调用 **`review_generated_images.py`**：`summary.json` 同时包含 **域内分项**（`pass_rate`、`group_*`…）与可选 **`academic_metrics`**（依赖 `requirements.txt` 中的 `piq` / `torch-fidelity` / `pytorch-fid`）。两套指标的解释见 **`docs/评价体系说明.md`**。

#### `POST /review` — 提交评估任务

对已生成批次运行质量评审。

**请求体（JSON）**

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `images_dir` | string | 必填 | 待评估图像目录 |
| `output_dir` | string | 自动 | 评审结果输出目录 |
| `recursive` | bool | false | 是否递归扫描子目录 |
| `real_images_dir` | string | 自动 | 真实图库路径；启用域内分项基线校准，并写入 `academic_metrics`（FID/KID/PRC/sFID\* 等在依赖与样本数满足时） |
| `real_baseline_json` | string | `""` | 直接使用已计算的基线 JSON；与手写 `real_images_dir` **二选一**，用于加速重复评审 |
| `top_k` | int | 30 | Top-K 推荐数 |
| `compute_fid` | bool | true | 为 `false` 时跳过需拉取真实库的批级 Inception 度量（可能影响 `academic_metrics` 整块） |

**请求示例**

```bash
curl -X POST http://localhost:8000/review \
  -H "Content-Type: application/json" \
  -d '{
    "images_dir": "outputs/generated/api_demo_20260503_000000_000",
    "output_dir": "outputs/eval/api_demo_review",
    "top_k": 20,
    "compute_fid": false
  }'
```

---

<a id="api-batches"></a>

## 批次与评估查询

#### `GET /batches` — 列出所有生成批次

```
GET /batches?limit=20
```

**响应示例**

```json
[
  {
    "name": "api_demo_20260503_120000_000",
    "path": "/root/autodl-tmp/medi-diff/outputs/generated/...",
    "image_count": 6,
    "images": ["0001.png", "0002.png", ...],
    "has_source_mapping": false,
    "mtime": 1746288000.0
  }
]
```

#### `GET /batches/{batch_name}` — 查询指定批次详情

#### `GET /batches/{batch_name}/images/{filename}` — 获取单张图像文件

#### `GET /evaluations` — 列出所有评估结果

#### `GET /evaluations/{eval_name}` — 查询指定评估详情

**响应字段**：`name`、`path`、`summary`（完整 summary.json 内容）、`has_advisor`、`has_report`。

#### `GET /results` — 批次 + 评估汇总（一次性）

---

<a id="api-reports"></a>

## 报告接口

#### `GET /reports/latest` — 获取最新 FINAL_REPORT 路径

```json
{
  "latest_report_path": "/root/autodl-tmp/medi-diff/outputs/reports/my_run_20260503/FINAL_REPORT.md",
  "exists": true
}
```

#### `GET /reports/history` — 获取调参历史（最近 5 轮）

返回数组，每项结构：

```json
{
  "index": 1,
  "source_tag": "round10_dense_mlo",
  "timestamp": "2026-05-03T10:00:00",
  "metrics": {
    "pass_rate": 0.62,
    "mean_brisque": 34.2,
    "mean_total_score": 6.8
  },
  "parameters": {
    "strength": 0.42,
    "overlap_ratio": 0.86,
    "global_guide_blend": 0.58,
    "blend_sigma_divisor": 1.50,
    "guidance_scale": 7.9,
    "num_steps": 52
  }
}
```

---

<a id="api-jobs"></a>

## 任务接口（`JobRecord`）

所有生成/评审操作异步执行，成功后返回 **`JobRecord`**（见下方字段表）。

#### `GET /jobs` — 列出所有任务

```
GET /jobs?limit=50
```

#### `GET /jobs/{job_id}` — 查询单个任务状态

**JobRecord 结构**

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 任务 ID（12位 hex） |
| `kind` | string | `"generate"` 或 `"review"` |
| `status` | string | `queued` / `running` / `succeeded` / `failed` |
| `command` | list | 实际执行的命令 |
| `created_at` | float | Unix 时间戳 |
| `started_at` | float? | 开始时间 |
| `finished_at` | float? | 完成时间 |
| `elapsed_seconds` | float? | 耗时（秒） |
| `return_code` | int? | 进程退出码 |
| `stdout` | string | 标准输出（最多 15000 字符） |
| `stderr` | string | 错误输出（最多 5000 字符） |

**轮询示例**

```bash
JOB_ID=$(curl -s -X POST http://localhost:8000/generate/sd15 \
  -H "Content-Type: application/json" \
  -d '{"num_images": 2}' | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")

# 轮询直到完成
while true; do
  STATUS=$(curl -s http://localhost:8000/jobs/$JOB_ID | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")
  echo "状态: $STATUS"
  [ "$STATUS" = "succeeded" ] || [ "$STATUS" = "failed" ] && break
  sleep 10
done
```

#### `DELETE /jobs/{job_id}` — 删除任务记录（仅内存，不终止进程）

---

<a id="api-static"></a>

## 静态文件

| 路径 | 映射目录 | 说明 |
|------|----------|------|
| `/static/generated/*` | `outputs/generated/` | 生成图像直接访问 |
| `/static/eval/*` | `outputs/eval/` | 评审结果文件 |
| `/static/reviews/*` | `outputs/reviews/` | 旧版评审结果 |

**示例**：`http://localhost:8000/static/generated/api_demo_20260503_000000_000/0001.png`

---

<a id="api-errors"></a>

## 错误码

| HTTP 状态码 | 说明 |
|-------------|------|
| 400 | 请求参数错误（路径越界、格式不合法） |
| 404 | 资源不存在（批次/评估/任务 ID 未找到） |
| 422 | Pydantic 参数校验失败（字段类型/范围错误） |
| 500 | 服务器内部错误 |
