# PLAN：从毕业设计到个人品牌项目

> 目标：把 `medi-diff`（基于扩散模型的乳腺钼靶图像生成系统）从一个"毕设仓库"改造成
> **一个可以作为个人品牌起点、技术博客素材库、GitHub 开源名片的中型项目**。
> 原则：**不重写代码，重写叙事；不删功能，清理口径；不堆新特性，先立门面。**

---

## 0. 这份计划的目标

| 维度 | 现状 | 目标 |
|------|------|------|
| 叙事 | 毕业设计课题（学校、任务书、答辩） | 开源项目（问题、方案、结果、局限） |
| 门面 | 中文单语 README，无徽章无许可证 | 双语 README + License + 架构图 + 徽章 |
| 工程 | 无 CI、无打包、requirements 不锁版 | CI 绿灯、可 `pip install -e .`、版本化发布 |
| 资产 | 权重/数据躺在本地 gitignore 里 | Hugging Face 发布（权重 + 数据 + 画廊 Demo） |
| 内容 | 答辩文档（私密用途） | 公开博客系列（7 篇），知乎/掘金/自建站分发 |
| 记忆点 | "毕设做了一个乳腺图像生成" | "开源了乳腺钼靶生成 + 自动化质控评估工具链" |

**最终验收一句话**：一个陌生人点进 GitHub 仓库，3 分钟内能明白"这是什么、为什么值得看、怎么跑起来"，并且愿意 star / 提问 / 转载。

---

## 1. 现状盘点（我们有什么牌）

### 1.1 真正的资产（保留并放大）

| 资产 | 为什么值钱 | 对应的叙事点 |
|------|-----------|-------------|
| 全链路管线（清洗→LoRA 训练→生成→评估→LLM 报告） | 市面上多数开源项目只覆盖单环节 | "从原始 DICOM 到评估报告的完整闭环" |
| 16 维规则评估体系 + 双轨指标（域内规则 vs FID/sFID） | 罕见：多数人只报 FID，没有可解释的医学质控 | "不靠 GPT-4V 的医学图像自动质控" |
| 标签守护（DICOM 烧录文字擦除 + Qwen-VL 验证） | 独立可复用，是预处理里的硬骨头 | "被所有论文忽略的烧录文字问题" |
| `source-seed` / `source_map.json` 可复现源选择 | 展示工程意识：实验可复现 | "调参的前提是固定变量" |
| 全图单次 img2img（放弃 patch 拼图的决策记录） | 有"踩坑—决策—结果"的完整故事 | 博客第 4 篇的好素材 |
| 评估校准的坑（auto-calibrate 会放松 hard gates） | 诚实记录失败是技术博客最稀缺的内容 | 博客第 5 篇的好素材 |

### 1.2 学生气来源（要清理）

- 文档口径：`docs/developer/答辩必备.md`、`模拟专家评分表.md`、`毕业设计任务书_*.md`、`毕业设计说明书大纲.md`、`项目交付检查清单.md`
- README 章节：`论文展示建议`、开头"山东建筑大学本科毕业设计"署名
- 代码硬编码：`apps/app_gradio.py` 中 `毕业论文_生成图像` 目录名
- 无 LICENSE / 无 CI / requirements 不锁版本 / 无打包
- 中文单语文档，README 无架构图无徽章

---

## 2. 定位与品牌

### 2.1 一句话定位（用于 README 首屏、博客简介、社交简介）

> **MammoGen — 面向乳腺钼靶（FFDM）图像的扩散模型生成与自动化质控开源工具链。**
> 从 CBIS-DDSM 数据清洗、SD1.5+LoRA 训练，到 16 维规则评估与 LLM 报告，一条命令跑通。

备选定位句（择一使用）：
- "Synthetic mammography generation with automated quality control"
- "让医学图像生成的每一步都可解释、可复现、可评估"

### 2.2 命名决策（M0 已确认）

- **仓库 slug 保持 `medi-diff`**
- **对外品牌名：`MammoGen`**（检索友好；README / 博客 / HF 统一使用）
- **Hugging Face 命名空间：个人账号 `TraitYoung`**（不用组织）
  - 权重：`TraitYoung/mammo-sd15-lora-v6`
  - 数据：`TraitYoung/cbis-clean-v2`
  - Gallery：`TraitYoung/mammo-gallery`
- 落地方式：README 标题写 `MammoGen` + 副标题保留仓库描述。

### 2.3 叙事主线（三支柱，贯穿 README / 博客 / 演讲）

```
数据（Data）──> 生成（Generation）──> 评估（Evaluation）
清洗烧录文字     全图 img2img + LoRA    16 维规则 + 双轨指标
mask 提取        可复现源选择           hard/soft 分轨否决
```

对外故事一句话版：
> "乳腺钼靶数据稀缺、标注昂贵、生成结果没人敢信。我做了一条让每一步都**可解释、可复现、可自动质检**的生成管线——并全部开源。"

---

## 3. 第一阶段：仓库工程化（约 1 周，P0）

> 目标：仓库本身"像个正经开源项目"。这一阶段不做任何新功能。

### 3.1 仓库门面

- [x] **LICENSE**：MIT；README 已声明 CBIS-DDSM/TCIA 数据归属与代码/数据许可分离
- [x] **README 重写**（英主 + 中文定位句）：
  - 首屏：`MammoGen` + 一句话定位 + 徽章（License / Python / GitHub / HF）
  - 架构图（Mermaid：清洗 → 训练 → 生成 → 评估 → 报告）
  - Demo 区：本地 UI 截图 + HF Gallery 路线图（真实对比条待 M3 补真图）
  - 快速开始（3 步）；FAQ；bibtex
- [x] **答辩口径文档进 `.gitignore`（本地保留、不再公开跟踪）**：已 `git rm --cached`（待随 M1 一并 commit）
- [x] README / 公开文档中"答辩版 / 论文 / 课题"措辞清除（`开发日志` 等个人备忘中的历史纪要保留）

### 3.2 代码与工程

- [x] `pyproject.toml`：`mammogen` 元信息 + `pip install -e .` / `.[dev]` / `.[full]`；`scripts/` / `apps/` 包化（`__init__.py`）
- [x] requirements 下限针 + `requirements-lock.txt`（CI CPU 工具链）
- [x] **GitHub Actions CI**（CPU）：`ruff` + `pytest -m "not heavy"`；单测覆盖 `label_guard` / `image_utils` / `pipeline_config` / `tuning_state`
- [x] 硬编码清理：`毕业论文_生成图像` → `outputs/generated/samples`（Gradio / API / review / run_full_report / verify_ui_wiring）
- [x] 提交模板：`CONTRIBUTING.md`、`CODE_OF_CONDUCT.md`、issue/PR 模板
- [x] `CHANGELOG.md`（`[1.0.0]`）；**git tag `v1.0.0` 待 commit 后打**
- [x] CLAUDE.md 同步更新（MammoGen 定位；输出根 `samples/`；CI/HF 命令）

### 3.3 文档整理（docs/ 重新分组）

```
docs/
├── guides/        # 用户手册、API 文档、安装说明（对外）
├── design/        # 架构说明、评估体系、参数说明（对外，技术深挖）
└── private/       # 个人备忘、开发日志（可保留在仓库，但明确标注"作者个人笔记"）
```

> M2 未强制物理搬迁 `docs/`（避免大挪路径打断链接）；公开口径已清，重组可并入后续 PR。

---

## 4. 第二阶段：资产发布 Hugging Face（P1，2-3 天）

> 权重和数据从"躺在本地的 gitignore"变成"可引用的公开资产"，这是从毕设到项目的关键一跃。

- [x] **LoRA Model Card 包装**：`hf/mammo-sd15-lora-v6/` → 目标 `TraitYoung/mammo-sd15-lora-v6`
- [x] **数据集 Card 包装**：`hf/cbis-clean-v2/`（含 SCHEMA + TCIA 归属提醒）
- [x] **Gallery Space 包装**：`hf/mammo-gallery/`（CPU 静态截图；无 GPU 推理）
- [x] **发布脚本**：`scripts/tools/publish_hf_assets.py`（`HF_TOKEN` + 可选本地权重/CSV）
- [ ] **远端实际上传**：本机暂无 `outputs/lora`、`datasets/`、`HF_TOKEN` — 备齐后执行 `python3 scripts/tools/publish_hf_assets.py --all`
- [ ] （可选）Zenodo DOI：为论文/博客引用提供稳定标识
- [x] README 已写明 HF ID 与下载/发布命令（远端 live 后徽章即绿）

---

## 5. 第三阶段：内容输出——博客系列（P1，2-4 周）

> 核心策略：**不写"项目介绍"，写"踩坑与决策"**。技术博客的传播力来自失败案例和可复现细节。

### 5.1 文章规划（7 篇，与管线一一对应）

| # | 标题（工作名） | 核心素材 | 传播点 |
|---|---------------|---------|--------|
| 1 | 为什么我要做乳腺钼靶图像生成 | 数据稀缺、标注贵、隐私敏感、合成数据用途 | 行业痛点 |
| 2 | 被所有人忽略的 DICOM 烧录文字 | 标签守护、CRAFT+LaMa 实验、mask 内文字 | 冷门细节，强差异 |
| 3 | LoRA 训练配方：r=32 全 MLO 视图 | 训练超参、数据清洗决策、SDXL/ControlNet 路线的失败结论 | 可复现配方 |
| 4 | 全图 img2img 为何优于 Patch 拼图 | patch 接缝问题、金字塔融合、strength/CFG 调参表、source-seed 可复现 | 工程决策复盘 |
| 5 | 不靠 GPT-4V 的医学图像自动质控 | 16 维规则体系、hard/soft 分轨、auto-calibrate 放松硬门禁的教训 | **最差异化的一篇** |
| 6 | Gradio + FastAPI 双入口的工程化 | 异步任务、容错降级、健康检查、Pydantic 校验 | 全栈工程感 |
| 7 | 医学生成模型的边界与未来 | 不做诊断用途、伦理、良恶性条件生成的路线图 | 价值观收尾 |

### 5.2 渠道与分发

- **主阵地**：自建技术博客（VitePress/Hugo，部署 GitHub Pages/Cloudflare Pages），域名 + 统一头像/简介/标语
- **国内分发**：知乎专栏、掘金（每篇同步）；公众号（可选）
- **国际分发**：英文精简版 → dev.to / Medium（可选，先看国内反馈）
- 每篇结尾统一 CTA：GitHub star、HF 权重下载、评论区提问

### 5.3 内容铁律

- 每篇必须含：**真实图像对比（含失败案例）**、可复现命令、指标数字
- 不吹"生成效果完美"，诚实呈现 BANDING/SHAPE_ODD 等失败与当前 pass_rate ~50% 的现实
- 明确声明"不用于临床诊断"（医疗内容的基本伦理底线，也是品牌可信度的来源）

---

## 6. 差异化强化（把亮点从"功能"变成"产品"）

按性价比排序，**不全部做**，先做前两项：

- [ ] **P1 评估体系独立化**：把 `review_generated_images.py` 的规则引擎拆成独立模块（输入：图像目录；输出：`summary.json`），让"不懂生成、只想给医学图像做质控"的人也能用。这是对外最有辨识度的资产
- [ ] **P1 标签守护独立化**：`label_guard.py` 已经是纯函数库，补一个独立 CLI + 文档即可对外
- [ ] **P2 数据子集白名单**：CLAUDE.md 里提到的"golden source set（4-8 张代表性 MLO dense 源图）"，做成 `datasets/golden_sources.json`，既服务调参复现，也是博客的可复现素材
- [ ] **P3 良恶性条件生成**：作为路线图写进 README/博客，不实现（避免项目范围失控）

---

## 7. 长期运营（P2，持续）

- [ ] GitHub 仓库每周 1 次 issue 巡检；star 增长记录在开发日志
- [ ] 每发布一篇博客，同步更新 README 的 "Blog/Publications" 区
- [ ] 收到外部 PR/issue 后，按开源协作流程响应（这是"个人品牌"最直接的证据）
- [ ] 复盘节点：发布 1 个月后，根据 star/阅读量决定英文版节奏与下一迭代方向

---

## 8. 里程碑与时间线

| 里程碑 | 内容 | 预计耗时 | 验收 |
|--------|------|---------|------|
| M0 审计 | 现状盘点、命名决策、文档 ignore 清单 | 1 天 | **已确认**：品牌 `MammoGen`；HF=`TraitYoung`；答辩文档 gitignore |
| M1 门面 | LICENSE、README 重写（MammoGen）、答辩文档 untrack、硬编码清理 | 2-3 天 | **已完成（待 commit）** |
| M2 工程 | pyproject、CI、锁版、单测、模板、CHANGELOG | 3-5 天 | **已完成（待 commit / tag `v1.0.0`）**；本地 `36 passed` |
| M3 资产 | HF cards + Space + publish 脚本 | 2-3 天 | **包装完成**；远端 upload 阻塞于本地权重/CSV + `HF_TOKEN` |
| M4 内容 | 博客 7 篇（建议 2 篇/周节奏） | 2-4 周 | 主阵地 4+ 篇发布，含失败案例 |
| M5 运营 | 渠道分发、issue 巡检、复盘 | 持续 | 发布 1 个月复盘报告 |

**建议顺序**：M1 → M2 与 M3 可并行 → M4 依赖 M1-M3（博客里要贴最终版 README/HF 链接）→ M5。

---

## 9. 行动清单（按优先级）

| 优先级 | 任务 | 产出 | 备注 |
|--------|------|------|------|
| P0 | LICENSE（MIT）+ 数据许可声明 | `LICENSE` | **M1 已完成** |
| P0 | README 重写（EN 主 + 架构图 + 徽章） | `README.md` | **M1 已完成**；真对比图待 M3 |
| P0 | 答辩文档进 `.gitignore` + `git rm --cached` | 本地保留、公开树不再跟踪 | **M1 已完成（待 commit）** |
| P0 | `毕业论文_生成图像` → `samples` | 代码 + 文档 | **M1 已完成** |
| P0 | pyproject + CI（ruff + pytest） | 配置 + Actions | **M2 已完成** |
| P1 | HF：LoRA 权重 + Model Card | `TraitYoung/mammo-sd15-lora-v6` | Card 就绪；upload 待资产 |
| P1 | HF：清洗数据 + 许可说明 | `TraitYoung/cbis-clean-v2` | Card 就绪；upload 待 CSV |
| P1 | HF：Gallery Space | `TraitYoung/mammo-gallery` | 本地包就绪；upload 待 token |
| P1 | 博客系列 7 篇 | 主阵地 + 知乎/掘金 | 每篇含失败案例与命令 |
| P1 | 评估体系/标签守护独立化 | 模块 + CLI + 文档 | 博客 2/5 篇的配套 |
| P2 | golden source set、CHANGELOG、CONTRIBUTING、模板 | 仓库文件 | 随 M2 一并做 |
| P2 | 英文版内容、Zenodo DOI、路线图更新 | 持续 | 看 M4 反馈再定 |

---

## 10. 完成标准（Done criteria）

- [x] GitHub 公开仓库中**无任何"毕业设计/答辩/论文"口径**的措辞与文档（个人备忘 / 开发日志历史纪要除外）— M1，待 commit 后生效于远程
- [x] 英文 README 完整（定位、架构图、快速开始、引用、许可）；陌生人 3 分钟可理解 — M1
- [x] CI 配置 + 本地 `pytest -m "not heavy"` / `pip install -e .[dev]` 可跑 — M2（远程 CI 待 push）
- [ ] LoRA 权重、清洗数据集、Gallery Space 三个 HF 资产**远端在线**（包装/脚本已就绪，待 `HF_TOKEN` + 本地文件）
- [ ] 博客主阵地已发布 ≥4 篇（含评估体系那篇差异化文章），篇篇有失败案例与可复现命令
- [ ] 发布 1 个月后完成复盘：star/阅读数据、issue 处理情况、下一步迭代方向

---

*M0–M2 代码侧已落地；M3 包装已就绪。下一步：commit → tag `v1.0.0` → push（触发 CI）→ 备齐 LoRA/CSV 后 `publish_hf_assets.py --all`。*
