# AI 全栈工程师 · 面试准备（毕设项目 × JD 对齐版）

> 目标岗位：未名集团 AI 事业部 · AI 全栈工程师（120 AI 急救系统）  
> 核心项目：**medi-diff** — 独立完成的毕业设计（端到端全栈 + 多模态 + LLM 闭环）  
> 面试风格预期：节奏紧凑、重实操轻理论、含上机小题

---

## 零、JD → 你的项目：诚实对齐表

面试前先过一遍这张表：**强项主动讲，缺口提前准备话术，不夸大。**

| JD 要求 | 你的毕设对应 | 匹配度 | 怎么说 |
|---------|-------------|--------|--------|
| 端到端（前端+后端+架构） | Gradio UI + FastAPI + subprocess 脚本层 + 数据/训练/推理全链路 | ★★★★☆ | 「从数据到界面到 API 都是我独立搭的」 |
| 医疗垂直场景 | 乳腺钼靶（FFDM）生成与质控，CBIS-DDSM | ★★★★★ | 「和贵司医疗 AI 方向一致，我做过影像域的质量门禁」 |
| 多模态（图像+文本） | SD 图像生成 + Qwen-VL 看图顾问 + DeepSeek/Qwen 文本顾问 | ★★★★☆ | 「图像理解 + 文本推理 + 结构化指标输入 LLM」 |
| 多 Agent 协作 | `run_generate_eval_advise.py` 三阶段编排（生成→评估→顾问） | ★★★☆☆ | 「目前是 **Workflow 编排**，不是 CrewAI 多 Agent；但任务拆解、状态持久化、闭环迭代思路一致」 |
| RAG / 知识库 | 评估摘要 + 指标 JSON + 调参历史 → 喂给顾问 LLM | ★★★☆☆ | 「轻量 **Context 注入**，未上向量库；急救指南 RAG 是我下一步想做的」 |
| Prompt / Tool / Workflow | system prompt、多后端切换、JSON 参数建议输出 | ★★★★☆ | 「顾问输出结构化 `parameters`，写入 `LATEST_NEXT_RUN.json` 驱动下一轮」 |
| Python + FastAPI | 主线语言；`api_server.py` REST + 异步 Job 队列 | ★★★★★ | 直接展示 `/docs` 和 job 轮询 |
| React + TypeScript | **未使用**（前端是 Gradio） | ★★☆☆☆ | 见下文「缺口话术」 |
| 低延迟 / 高并发 / 弱网 | GPU 推理优化、Job 异步、健康检查与日志 | ★★★☆☆ | 「急救实时性更高；我做过 **重任务异步化 + 可观测性** 的基础」 |
| WebSocket / 流式 | **未实现** | ★☆☆☆☆ | 「API 是 submit + poll；流式 LLM 输出是我愿意在入职后快速补的」 |
| LangChain / CrewAI | **未使用**（自研 subprocess 编排） | ★★☆☆☆ | 「理解范式，项目里用更轻量的编排；框架上手快」 |
| Git / 测试 / 日志 | pytest、`logging`、请求中间件、`/health` GPU 信息 | ★★★★☆ | 举 `test_source_artifact_filter.py`、API 日志示例 |
| Cursor / AI 辅助开发 | 项目 `.cursor/rules`、顾问脚本、开发日志 | ★★★★★ | 「日常用 Cursor 做重构和测试生成」 |

---

## 一、自我介绍（90 秒 · 全栈 AI 版）

> 我是 XX 大学 XX 专业应届/XX 届，**Python 全栈 + AI 应用**方向。  
>  
> 我的代表项目是**独立完成的毕业设计**：一套**医疗影像 AI 应用系统**——基于 Stable Diffusion + LoRA 的乳腺钼靶可控生成，覆盖 **数据预处理 → 模型训练 → 推理服务 → 自动质控 → Web 界面与 API → LLM 顾问闭环**。  
>  
> **全栈部分**：后端用 **FastAPI** 封装生成/评估接口，异步 Job 队列 + 静态资源挂载；前端用 **Gradio** 做答辩级操作台（生成、评估、画廊、一键流水线、调参历史）。  
> **AI 部分**：图像侧是扩散模型 img2img + 域内 16 维规则评估；LLM 侧接 DeepSeek/Qwen 文本顾问和 Qwen-VL 视觉顾问，把评估指标转成**结构化调参建议**并持久化，形成「生成—评估—改进」闭环。  
>  
> 选这个项目是因为我想证明：**不只是调模型，而是能把 AI 做成可交付的产品**——参数可复现、质量可量化、接口可调用。这和贵司 120 AI 急救「高可靠、可解释、端到端落地」的方向是一致的。我前端目前是 Gradio，**React + FastAPI 的分工我很熟悉**，入职后可以快速切到贵司技术栈。

**背诵要点（3 个关键词）：** 医疗影像 · 端到端 · LLM 闭环

---

## 二、把毕设讲成「AI 全栈项目」而不是「纯算法实验」

### 2.1 一句话定位

> 一个 **SaaS 形态的 medical AI 工具**：用户通过 UI/API 提交条件 → 后台 GPU 推理 → 自动质控打分 → 可选 LLM 顾问给下一轮参数——**产品闭环完整，算法是核心引擎之一**。

### 2.2 全栈分层（面试官问架构时用）

```
┌─────────────────────────────────────────────────────────┐
│  表现层   Gradio（生成/评估/画廊/流水线/调参历史）          │
│           FastAPI（/generate/sd15, /jobs/{id}, /health） │
├─────────────────────────────────────────────────────────┤
│  编排层   run_generate_eval_advise.py                    │
│           run_full_report.py · tuning_state.py           │
├─────────────────────────────────────────────────────────┤
│  AI 引擎  run_mammo_sd15.py（SD1.5+LoRA img2img）        │
│           review_generated_images.py（16维+hard_tags）    │
│           ask_advisor.py（文本/VL 多模态顾问）            │
├─────────────────────────────────────────────────────────┤
│  数据层   CBIS_CLEAN_V2 · LoRA 权重 · outputs/ 批次产物   │
└─────────────────────────────────────────────────────────┘
```

**强调的工程决策：**

| 决策 | 理由 |
|------|------|
| UI/API **subprocess 调脚本** | GPU 进程隔离；CLI 可单独复现；Gradio 不拖垮模型内存 |
| `GenParams` dataclass 单一真相源 | 前后端/CLI 参数一致，避免「界面一套、命令行一套」 |
| 每批次 `source_map.json` + `run_params.json` | 可追溯、可 A/B、可审计——急救系统同样需要 |
| `.env` 管 API Key，代码零硬编码 | 生产安全基线 |

### 2.3 和「120 AI 急救」的迁移叙事（必背）

| 急救场景 | 你的毕设类比 |
|----------|-------------|
| 院前伤情图像/描述输入 | 钼靶源图 + 体位/密度条件筛选 |
| 实时感知与理解 | Qwen-VL 看图 + 规则 CV 指标双轨 |
| 调度/协同 | 生成→评估→顾问 三阶段 workflow |
| 知识库支撑决策 | 评估摘要 + 历史调参 JSON 注入 LLM |
| 高可靠 | hard_tags 一票否决 = 「不可用就不放行」 |
| 可解释 | A-F 分组扣分项、tags 可追溯，不是黑盒 pass/fail |

**标准话术：**

> 「急救是 **多模态输入 + 规则/模型混合决策 + 低延迟编排**；我的毕设是 **图像生成 + 规则质控 + LLM 建议** 的同构问题，区别主要是延迟要求和 Agent 复杂度。架构上我已经习惯 **把重推理和 Web 层解耦、用 Job 异步、用结构化 JSON 在模块间传递**。」

---

## 三、JD 逐条深挖 · 推荐回答

### 职责 1：端到端开发

**问：** 你具体负责什么？

**答：** 独立毕设，**全链路 owner**。数据清洗脚本、LoRA 训练配置、推理管线、16 维评估器、FastAPI 服务、Gradio 界面、一键流水线、LLM 顾问对接、单元测试和文档都是我做的。没有「只写模型」或「只画 UI」的分工。

**可展示：** `bash apps/start.sh` → Gradio 7860 + API 8000；OpenAPI `/docs`。

---

### 职责 2：多模态 AI

**问：** 多模态怎么做的？

**答：** 三条线：

1. **图像生成**：SD1.5 + LoRA，img2img 保留解剖结构  
2. **图像理解**：Qwen-VL 顾问直接看生成图 vs 真实钼靶，给视觉诊断  
3. **文本推理**：DeepSeek/Qwen 读 `summary.json` 里的通过率、A-F 均分、hard_tags，输出中文建议 + 下一轮 `parameters` JSON  

**加分：** negative_prompt 和评估规则里显式抑制「肿块/钙化簇」等医学语义——说明我考虑 **下游临床误导风险**，不是只追求「像照片」。

---

### 职责 3：多 Agent 协作

**问：** 用过 Agent 吗？

**答（诚实版）：**

> 毕设里是 **Workflow 编排**，不是 LangChain 多 Agent。`run_generate_eval_advise.py` 串行调用三个「专家模块」：生成器、评估器、顾问——类似 **单协调者 + 专用 Tool** 模式。  
> 状态通过 `LATEST_NEXT_RUN.json` / `PARAM_HISTORY.json` 持久化，支持 `--skip-generate` 等分段调试。  
> 我理解急救场景的 Agent 需要 **任务拆解、并行调度、失败重试**；我的编排层已经处理了 **步骤依赖、skip 开关、子进程 rc 检查**，差的是框架化和实时通信，这是我入职后优先补的。

**如果被追问 CrewAI/LangChain：** 说范式（Planner → Tools → Memory），承认项目为控制复杂度没用框架，**2 周内可基于 FastAPI 迁移**。

---

### 职责 4：RAG / 数据管道

**问：** RAG 做过吗？

**答（诚实 + 积极）：**

> 完整向量 RAG **还没上 Pinecone/Milvus**，但有 **结构化知识注入**：  
> - 管道：`summary.json` → 压缩指标 → `ask_advisor.py` system prompt  
> - 持久化：`PARAM_HISTORY.json` 最近 5 轮，相当于 **session memory**  
> - 数据管道：CBIS 清洗 → mask → caption JSONL → 训练 → 批次产物  
>  
> 急救指南 RAG 我会用 **分块 + metadata（场景/优先级）+ 引用溯源** 来做，毕设的「评估指标喂 LLM」本质上是同一种 **retrieve context → generate action** 模式。

---

### 职责 5：Prompt / Tool / Memory / Workflow

| 概念 | 毕设实例 |
|------|----------|
| **Prompt** | 生成 prompt 含 MLO 解剖约束；顾问 system prompt 要求输出 JSON |
| **Tool** | subprocess 调 `run_mammo_sd15.py`、`review_generated_images.py` |
| **Memory** | `LATEST_NEXT_RUN.json`、`PARAM_HISTORY.json` |
| **Workflow** | `run_generate_eval_advise.py` 三阶段 + `--from-latest-tuning` |

**难点故事（1 分钟）：** 顾问若只看 auto-calibrated pass_rate 会给出错误建议 → 我在报告里区分 **strict vs calibrated**，顾问输入优先 strict 缺陷标签 → 对应急救里「不能只看单一指标，要结合规则 veto」。

---

### 职责 6：性能 / 稳定性 / 体验

**已实现：**

- FastAPI **异步 Job**（提交即返 job_id，轮询状态）  
- `/health` 含 GPU、内存、uptime  
- HTTP 请求日志中间件  
- 生成脚本逐步耗时日志（load model / per-image / total）  
- 子进程失败不 silent pass（check rc）

**急救场景差距（主动说）：**

> 我的瓶颈在 **GPU 推理秒级**；急救要 **毫秒~秒级 API + 弱网降级**。我理解方向：**流式 LLM 首 token、边缘轻量模型、关键路径缓存、超时熔断**——毕设里 Job 异步和 health check 是同一套工程思维的起手式。

---

## 四、缺口话术（React / WebSocket / LangChain）

### React 没用过怎么办？

> 「毕设前端是 **Gradio**，为了快速交付医疗影像 demo 和答辩演示。后端 API 是标准 **REST + Pydantic 模型**，和 React 对接没有耦合。我有 JavaScript/TypeScript 基础（若属实补充：课程/小项目），**组件化、状态管理、调用 FastAPI** 的学习曲线对我可控。我可以入职第一周用 React 消费现有 `/jobs` API 做一个最小控制台。」

**不要说：** 「Gradio 就是 React」——会被识破。

### WebSocket / 流式

> 「当前是 submit + poll；LLM 顾问是一次性返回。我知道急救指导需要 **流式输出 + 断线重连**，实现上会用 FastAPI WebSocket 或 SSE + 前端 incremental render。」

### LangChain

> 「我读过 RAG/Agent 文档，项目里 deliberately 用 subprocess 保持简单可控。愿意在团队规范下用 LangChain/LlamaIndex 统一 Tool 和 Memory。」

---

## 五、项目深挖 · 三个必讲难点（STAR）

### 1. 架构收敛：Patch 接缝 → 全图 img2img

- **S/T：** Patch 拼接有 GRID_SEAM/BANDING，评估 strict pass 低  
- **A：** FFT 检测条带；对比全图单次推理；移除 Patch 主线  
- **R：** 接缝类 hard defect 消除；用 768 推理 + 2048 上采样平衡显存与分辨率  

### 2. 医疗伪影：DICOM 标签 + inpaint 几何块

- **S/T：** 生成图出现「R MLO」、菱形修补块  
- **A：** 训练数据 mask 外清零；`label_guard.py`；去掉组织内 TELEA；源图 artifact 预筛  
- **R：** 伪影类失败下降；**可复现排查路径**（对比 inpaint 前后 → 定位算法 → 改策略 + 诊断脚本）  

### 3. 评估驱动调参：避免「虚高通过率」

- **S/T：** auto-calibrate 后 pass_rate 高但目视仍差  
- **A：** strict/calibrated 双口径；固定 `--source-seed`；一次只改 strength/guidance 一个变量  
- **R：** 调参可追踪；**和急救「可解释 + 不误报」同构**  

---

## 六、上机测试准备（贴合 JD + 本仓库）

### 高概率题型

| 类型 | 考察点 | 你项目中的参照 |
|------|--------|----------------|
| FastAPI 小题 | POST 创建 Job、GET 查状态、Pydantic 校验 | `api_server.py` |
| 异步 / 线程 | 长任务不阻塞请求 | Job 队列 + threading |
| LLM 调用 | HTTP 调 OpenAI 兼容 API、解析 JSON | `ask_advisor.py` |
| 工作流 | 顺序调用、失败短路 | `run_generate_eval_advise.py` |
| OpenCV / 数据 | 掩膜、轮廓、阈值判定 | `image_utils.py`、评估脚本 |
| 单元测试 | pytest 边界 case | `scripts/tests/` |

### 模拟题（建议手写练一遍）

**题 1：** 实现 `POST /assist`：入参 `{symptom: str, vitals: dict}`，mock 调 LLM，返回 `{triage_level, advice, citations[]}`。

**题 2：** 内存 JobStore：`create_job()` → `running` → `done/failed`，带 `created_at` 和超时标记。

**题 3：** 简化 RAG：`search(query, docs[]) -> top_k` 用关键词或 TF-IDF，拼进 prompt 模板。

**题 4：** 纯函数：`should_veto(tags: set[str], hard_tags: frozenset) -> bool` + 2 个 pytest。

### 上机原则

1. 先跑通 happy path，再补校验和测试  
2. 类型注解 + 清晰函数名  
3. 边写边讲：**输入输出、失败策略、如何扩展**  
4. 主动提「生产还会加 logging、超时、idempotency」

---

## 七、反向提问（体现全栈 + 医疗 AI 认知）

1. 120 急救系统当前 **端侧形态** 是 Web、小程序还是车载屏？前端技术栈定了吗？  
2. 多 Agent 是 **统一 orchestrator** 还是业务侧各自集成？有没有标准 Tool 协议？  
3. 知识库 RAG 的 **更新频率与溯源** 要求？院前弱网时 RAG 如何降级？  
4. 图像/语音模块是 **自研还是第三方 API**？延迟 SLA 大概多少？  
5. 团队对 **可解释性** 的产品要求（引用指南段落、置信度展示）？

---

## 八、演示 Checklist（若带电脑）

- [ ] `bash apps/start.sh` 正常，7860 / 8000 可访问  
- [ ] 浏览器开好：`/docs`、`生成` Tab、已有画廊批次  
- [ ] 准备 1 份 `outputs/eval/*/summary.json`（通过率、hard_tags）  
- [ ] 准备 1 份 `LATEST_NEXT_RUN.json`（LLM 结构化建议）  
- [ ] 备用：GPU 不可用时用已有截图 + 评估报告  

---

## 九、30 秒 / 90 秒 / 3 分钟三个版本

| 时长 | 内容 |
|------|------|
| **30s** | 医疗影像 AI 全栈毕设 · FastAPI+Gradio · SD+LoRA 生成 · 16 维质控 · LLM 闭环 |
| **90s** | 见第一节完整自我介绍 |
| **3min** | 90s + 架构图 + 一个难点 STAR + 和 120 急救的迁移叙事 |

---

## 十、心态提醒

- **毕设 = 独立全栈项目**，不是「学生作业」——你有 API、有质控、有闭环，比多数「只训了个模型」的候选人完整。  
- **Gradio 不是减分项**，关键是后端架构和 AI 工程化；React 是 **可快速补的工具**，不是能力边界。  
- 未名集团看重 **医疗落地 + 产业化**——多提 **可靠性、可解释、质控门禁、与临床风险**，少提 FID 刷榜。  
- 面试节奏快：**结论先行，细节等追问**。

---

*文档版本：2026-05-24 · 基于 `docs/AI全栈工程师_职位描述.md` 与仓库主线实现整理*
