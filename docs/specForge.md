# SpecForge - AI Software Engineering Spec Generator

SpecForge is a full-stack resume project for AI Coding workflows. It turns rough product ideas or code snippets into structured engineering artifacts: requirement discovery, sprint planning, implementation sketches, test plans, generated test drafts, and code review reports.

Highlights:

- FastAPI + Next.js end-to-end app with SSE streaming and session history.
- Pydantic v2 structured outputs to reduce drift in multi-step LLM workflows.
- SQLite FTS5 lightweight retrieval memory for prior specs and recurring review issues.
- Full trace output for pipeline step duration, summaries, and runtime metrics.
- Backend structure split into API routes, services, pipeline orchestration, and memory storage so the engineering boundaries are easy to explain in interviews.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the implementation map.

---
🔗 Live Demo: https://spec-forge-phi.vercel.app/

🔎 Backend Health: https://ishowrelx5-specforge-api.hf.space/api/v1/health

# SpecForge — AI 软件工程规格锻造

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-green.svg)](https://fastapi.tiangolo.com/)
[![Pydantic v2](https://img.shields.io/badge/Protocol-Pydantic_V2-red.svg)](https://docs.pydantic.dev/)
[![Next.js 16](https://img.shields.io/badge/Frontend-Next.js_16-black.svg)](https://nextjs.org/)
[![LangChain](https://img.shields.io/badge/LLM-LangChain-orange.svg)](https://www.langchain.com/)
[![DeepSeek](https://img.shields.io/badge/Model-DeepSeek_V4-purple.svg)](https://platform.deepseek.com/)
[![Redis](https://img.shields.io/badge/Cache-Redis-red.svg)](https://redis.io/)
[![TypeScript](https://img.shields.io/badge/Lang-TypeScript-3178c6.svg)](https://www.typescriptlang.org/)
[![Tailwind CSS](https://img.shields.io/badge/CSS-Tailwind_v4-06b6d4.svg)](https://tailwindcss.com/)
[![CI](https://github.com/TraitYoung/specForge/actions/workflows/ci.yml/badge.svg)](https://github.com/TraitYoung/specForge/actions/workflows/ci.yml)

**公开仓库：** [github.com/TraitYoung/specForge](https://github.com/TraitYoung/specForge) · **实现映射：** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

**Vibe coding 的 demo → 可维护的生产级项目。**

让 AI 写代码很快，但代码写完之后的麻烦事更多——没需求文档、没测试、架构说不清，中断几天回来就不记得做到哪了。SpecForge 在代码生成的同时补上工程规格：需求拆解、Sprint 规划、实现草案、测试方案，让项目能持续迭代而不是烂在初期。

## 项目概述

**技术栈：** Python、FastAPI、Next.js 16、LangChain、DeepSeek V4、Pydantic v2、Redis、SQLite FTS5、SSE、TypeScript、Tailwind CSS v4

**项目背景：** 针对 AI Coding 场景中"代码可生成但难以工程化"的问题，构建多 Agent 协同开发系统，实现从自然语言想法到结构化需求、开发计划、测试方案的闭环，同时支持逆向分析现有代码，发现架构风险与质量问题，提高工程可控性。

**主要工作：**

- 设计双模式 AI 流水线——正向工程：想法 → 需求发现 → Sprint 规划 → 实现草案 + 测试交付并行 → Merge 汇总；逆向审查：代码 → 需求推测 → 架构/质量分析 → 优先级改进计划
- 基于 ThreadPoolExecutor 并行阶段执行，降低端到端延迟
- 基于 Pydantic v2 构建 5 个结构化输出模型（需求规格、Sprint 规划、代码草案、测试与 DoD、逆向分析），字段级约束确保 LLM 输出可校验且步骤间只传 JSON 摘要控制 Token 开销
- 实现项目类型自动识别（Web / Mobile / API / 数据流水线 / 游戏工具），注入领域专属工程关注点与输出偏好
- 设计步骤级模型路由与上下文预算管理器，实现模型热切换、摘要截断，避免 Prompt 膨胀与成本失控
- **RAG 检索增强生成**——基于 SQLite FTS5 构建本地规格检索引擎，正向工程自动检索历史相似需求规格作为参考上下文提升生成一致性，逆向审查积累高频代码问题模式形成知识库，实现"越用越准"的持续学习机制
- FastAPI + Next.js 端到端落地，支持 SSE 流式输出、Redis 会话缓存、全链路 Trace（步骤耗时与摘要），生成 Prompt 可直接用于 Cursor

**项目成果：**

- 完成 AI 软件工程自动化闭环，覆盖需求生成、开发规划、并行实现交付及代码审查
- 输出成果可直接用于实际开发环境，实现从自然语言想法到可执行工程计划的端到端交付
- 通过结构化输出约束与上下文预算控制，显著提升多步骤 AI 工作流的稳定性与 Token 效率

## 工程亮点

- **结构化输出**：Pydantic v2 强类型契约 + field_validator 字段级约束，每步可验证，不会产生协议漂移
- **Token 成本内建**：步骤间只传 JSON 摘要，上下文预算统一管理，用户原文与摘要均有硬截断上限
- **全链路追踪**：每轮返回 trace_id + trace[]，记录每步骤耗时与关键输出，支持性能复盘
- **项目画像自动匹配**：关键词检测识别 Web/Mobile/API/Data/Game Tools 共 5 种项目类型，注入对应工程关注点与输出偏好
- **步骤级模型路由**：发现/设计/实现/交付/合并各步骤可独立配置不同 LLM 模型，按步骤粒度调配成本与能力
- **串行对齐的测试方案**：实现草案完成后再生成测试方案与测试代码草稿，确保用例与代码一致
- **RAG 检索增强**：基于 SQLite FTS5 全文检索引擎，正向工程自动检索历史相似需求规格作为 Few-shot 参考上下文，提升生成一致性与结构化输出质量；逆向审查积累高频代码问题模式形成知识库，实现"越用越准"的持续学习机制
- **SSE 流式输出**：聊天气泡仅展示短摘要；完整 **SPEC.md / REVIEW.md** 实现包落盘至 `output/chats/`，支持一键复制 Cursor Prompt 或下载

## 两种工作模式

### 正向工程 (Spec)：想法 → 工程规格

输入你的需求描述，系统输出一份结构化的软件工程交付包：

1. **需求发现** — 用户故事、验收标准、Sprint 目标、可度量结果
2. **Sprint 设计** — 模块拆分、数据流、有序待办、技术探针、停车场
3. **实现草案** — MVP 核心路径代码草稿（含语言标识与依赖说明）
4. **测试与交付** — 测试用例、完成定义 (DoD)、CI/CD 提示、CHANGELOG、Sprint 回顾（对照实现草案生成）
5. **测试代码草稿** — 2~3 个可粘贴的测试文件（vitest/pytest 等），写入 SPEC.md 的 `Generated Test Files`

聊天气泡为**摘要**（含测试用例与 DoD 预览）；完整 **SPEC.md** 含 Implementation / Test Prompt、代码草稿、Generated Test Files 与 Release Notes，保存为 `output/chats/*_SPEC.md`，可在界面复制或下载后拖入 Cursor/Copilot。

### 三种「测试」概念（避免混淆）

| 类型 | 说明 |
|------|------|
| **流水线测试方案** | Delivery 步产出的用例标题、DoD、CI 说明；不自动执行 |
| **测试代码草稿** | Test Code 步产出的可粘贴测试文件；需在用户项目中自行运行 |
| **本仓库 pytest** | [`tests/`](tests/) 验证 SpecForge 自身；与业务产出无关 |

### 逆向审查 (Review)：代码 → 审查报告

粘贴你通过 vibe coding 产出的代码，系统反向推导：

- 推测的业务目标与用户故事
- 缺失的测试用例
- 架构问题（耦合、职责不清、缺少抽象）
- 代码质量问题（命名、错误处理、硬编码）
- 按优先级排列的改进计划

附带**可直接粘贴到 Cursor 的重构 prompt**。

## 快速开始

### 一键启动（推荐）

双击项目根目录的 **`start_dev.bat`**，自动启动 Redis + Backend + Frontend，端口就绪后打开浏览器。

### 手动启动

**1. 环境配置**

```bash
pip install -r requirements.txt
cp .env.example .env
```

`.env` 中至少配置（默认 **DeepSeek V4 Flash**）：
```bash
LLM_API_KEY=你的_DeepSeek_API_Key
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-v4-pro
LLM_STRUCTURED_MODE=native
LLM_THINKING=enabled
LLM_REASONING_EFFORT=max
```
Key 在 [platform.deepseek.com](https://platform.deepseek.com/) 创建。轻量流式步骤（`test_code` / `merge`）建议 `deepseek-v4-flash`，structured 步骤用 `deepseek-v4-pro`（均为 1M 上下文）。旧版 `QWEN_*` / Moonshot `LLM_*` 仍兼容。

**2. 启动服务**

```powershell
# 终端 1 — Redis（可选，会话缓存）
redis-server --port 6379

# 终端 2 — 后端 FastAPI
python -m uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000 --reload

# 终端 3 — 前端 Next.js
cd frontend && npm install && npm run dev
```

浏览器打开 `http://localhost:3000`。

## 项目结构

```
├── backend/
│   ├── main.py                          # FastAPI 入口
│   ├── agents/dev_pipeline/
│   │   ├── orchestrator.py              # 核心编排：正向 5 步 + 逆向审查
│   │   └── step_agents.py               # 各步骤 Agent 配置与执行
│   ├── config/
│   │   ├── context_budget.py            # Token 预算控制
│   │   └── step_model_routing.py        # 步骤级模型路由
│   ├── schemas/
│   │   ├── workflows.py                 # 5 个流水线 Pydantic 模型 + Prompt 生成
│   │   ├── protocols.py                 # TaskIntent
│   │   └── trace.py                     # TraceStep
│   ├── memory/
│   │   ├── session_cache.py             # Redis 会话缓存
│   │   └── spec_store.py                # SQLite FTS5 规格检索
│   └── prompts/
│       └── dev_pipeline_profiles.py     # 5 种项目画像检测
├── frontend/                            # Next.js 16 + Tailwind v4
│   ├── app/
│   │   ├── page.tsx                     # 聊天主页面（SSE + Trace 面板）
│   │   └── api/                         # API 反向代理到 FastAPI
│   └── lib/backend.ts
├── tests/                               # pytest（32 用例）
├── scripts/
│   ├── dev_stack.ps1                    # 开发环境管理
│   └── locustfile.py                    # 压测入口
├── start_dev.bat                        # 一键启动
├── docker-compose.yml
└── requirements.txt
```

## API

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/health` | GET | 健康探活 |
| `/api/v1/chat` | POST | 非流式对话 |
| `/api/v1/chat/stream` | POST | SSE 流式输出（推荐） |
| `/api/v1/chat/history` | GET | 会话历史（需 `x-session-id`） |
| `/api/v1/chat/export` | POST | 导出会话为 JSONL |

```json
{
  "text": "做一个个人记账 App，支持多账户、分类预算、月度报表...",
  "mode": "spec"
}
```

`mode` 取值：`"spec"`（正向工程，默认）或 `"review"`（逆向审查）。

## 开发约定

- 遵循 [CONTRIBUTING.md](CONTRIBUTING.md) 中的工程规范
- 后端：`python -m compileall backend -q && pytest`
- 前端：`cd frontend && npx tsc --noEmit && npm run build`
- 架构细节见 [docs/项目结构与技术要点.md](docs/项目结构与技术要点.md)
## 部署指南

### 推荐拓扑

- Frontend: Vercel, root directory 选择 `frontend`, 构建命令 `npm run build`, 安装命令 `npm ci`.
- Backend: Hugging Face Spaces Docker, 当前后端地址为 `https://ishowrelx5-specforge-api.hf.space`.
- Redis: Upstash Redis 免费层, 将 TLS Redis 连接串填入 Hugging Face Space Secrets 的 `REDIS_URL`. Redis 不可用时系统会降级为无会话缓存, 核心生成能力仍可用.
- Database: SQLite FTS5, 首次启动由 `SpecStore._migrate()` 自动初始化 `data/spec_store.db`. Hugging Face 免费 Space 的磁盘不保证长期持久化, 作品集 demo 可接受; 生产环境应迁移到托管数据库.

### 环境变量

Backend(Hugging Face Space Secrets):

```bash
LLM_API_KEY=your_deepseek_api_key
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-v4-flash
LLM_STRUCTURED_MODE=native
LLM_THINKING=disabled
LLM_REQUEST_TIMEOUT=300
REDIS_URL=rediss://default:<password>@<host>:6379
```

DeepSeek V4 结构化步骤建议 `LLM_STRUCTURED_MODE=native`；关闭 thinking 可显著提速。

Frontend(Vercel):

```bash
BACKEND_URL=https://ishowrelx5-specforge-api.hf.space
```

### Docker Compose

本地或单机部署可以直接使用:

```bash
cp .env.example .env
docker compose up --build -d
curl http://127.0.0.1:8000/api/v1/health
```

日志查看:

```bash
docker compose logs -f api
docker compose logs -f frontend
```

### Hugging Face Spaces 后端

1. 后端使用独立的 Hugging Face Space 仓库 `iShowRelx5/specForge-api`, 由本地 `hf-space/` 目录推送。
2. Space SDK 选择 Docker, 监听端口 `7860`; `hf-space/README.md` 顶部保留 `sdk: docker` / `app_port: 7860` metadata.
3. 在 Space **Settings → Repository secrets** 中设置：
   | Secret | 值 |
   |--------|-----|
   | `LLM_API_KEY` | DeepSeek 控制台 API Key |
   | `LLM_BASE_URL` | `https://api.deepseek.com/v1` |
   | `LLM_MODEL` | `deepseek-v4-flash`（或 `deepseek-v4-pro`） |
   | `LLM_STRUCTURED_MODE` | `native` |
   | `REDIS_URL` | Upstash 等 Redis 连接串（可选） |
   旧 Secret 名 `QWEN_*` 仍可用，建议改名为 `LLM_*`。
4. 保存后 **Restart Space**。访问 health 应看到 `env.has_llm_key: true`、`env.llm_model: "deepseek-v4-flash"`。

常用推送命令:

```powershell
.\scripts\sync_hf_space.ps1
cd hf-space
git add -A
git commit -m "deploy backend update"
git push origin main
```

### Vercel 前端

1. Import Git Repository, root directory 填 `frontend`.
2. Environment Variables 填 `BACKEND_URL=https://ishowrelx5-specforge-api.hf.space`.
3. 部署后访问 `/api/health`, 确认能代理到后端.
4. 部署成功后, 将本 README 第一行 Live Demo 替换为真实 Vercel URL; 在此之前不要填写会 404 的占位域名.

前端推送到 GitHub `main` 后 Vercel 会自动部署; 如未触发, 在 Vercel -> Deployments 手动 Redeploy。

## 监控与日志

`GET /api/v1/health` 返回服务版本、启动时长、Redis/SQLite 状态、进程内存和关键环境变量是否存在。请求日志会记录 `trace_id`, path, status, duration_ms 和 client_ip; 流水线日志会记录每一步耗时、估算 token 和内存快照。

![SpecForge observability trace](docs/assets/observability-trace.png)

## 容错设计

完整说明见 [docs/RELIABILITY.md](docs/RELIABILITY.md). 当前版本覆盖 Pydantic 结构化校验、API 限流、Redis 降级、SQLite 写入失败保护、上下文预算裁剪、SSE 错误事件和生产健康检查。
