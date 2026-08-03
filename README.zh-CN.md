# MammoGen

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](requirements.txt)
[![GitHub](https://img.shields.io/badge/GitHub-TraitYoung%2Fmedi--diff-181717?logo=github)](https://github.com/TraitYoung/medi-diff)
[![CI](https://github.com/TraitYoung/medi-diff/actions/workflows/ci.yml/badge.svg)](https://github.com/TraitYoung/medi-diff/actions/workflows/ci.yml)
[![HF](https://img.shields.io/badge/Hugging%20Face-TraitYoung-yellow?logo=huggingface)](https://huggingface.co/TraitYoung)

[English](README.md) · **简体中文**

**Synthetic mammography generation with automated quality control.**

**MammoGen** — 面向乳腺钼靶（FFDM）图像的扩散模型生成与自动化质控开源工具链。  
从 CBIS-DDSM 数据清洗、SD1.5+LoRA 训练，到 16 维规则评估与可选 LLM 报告，一条命令跑通。

> **不用于临床诊断。** 仅供研究、教学与合成数据实验。  
> **Not for clinical diagnosis.** Research / education / synthetic-data tooling only.

仓库 slug：[`medi-diff`](https://github.com/TraitYoung/medi-diff) · 品牌名：**MammoGen**

完整英文门面与英文文档索引见根目录 [`README.md`](README.md) 与 [`docs/en/`](docs/en/README.md)。下文为中文速览；细节以中文文档树为准。

---

## 为什么做这个项目

多数开源钼靶 / 医学生成仓库停在「训个 LoRA 再倒出图」。MammoGen 提供**闭环**：

| 支柱 | 内容 |
|------|------|
| **数据** | CBIS-DDSM 清洗、乳腺掩膜、烧录标签清除 |
| **生成** | SD1.5 + LoRA，默认**全图** img2img（无拼缝），可复现 `--source-seed` |
| **评估** | 16 维规则分 + 硬/软门禁，可选 FID/sFID——质控不依赖 GPT-4V |

诚实基线：严格 `pass_rate` 常在 **~50%**。失败模式（banding、形状异常等）会写进文档，不粉饰。

---

## 快速开始

```bash
git clone https://github.com/TraitYoung/medi-diff.git
cd medi-diff
pip install -r requirements-dev.txt   # CI / 轻量
pip install -r requirements.txt       # 完整 GPU 运行时
cp .env.example .env                  # 可选：顾问 API

# RTX 50 系（Blackwell）建议：
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python3 scripts/tools/check_local_gpu.py

bash apps/start.sh
```

权重与数据放置、HF 发布、默认超参见英文 [`README.md`](README.md)。  
中文操作手册：[`docs/developer/用户操作手册.md`](docs/developer/用户操作手册.md)。

生成默认：`strength=0.44` / CFG `7.5` / steps `40` / `--mem-profile auto`；Gradio 默认 **本地流畅**（28 steps + `local`）。

---

## 文档

| 文档 | 说明 |
|------|------|
| [中文文档索引](docs/README.md) | 手册、API、评价体系、结构说明 |
| [英文文档索引](docs/en/README.md) | User guide / API / evaluation |
| [apps README](apps/README.md) | Gradio 与 API、预设与输出目录 |
| [SECURITY](SECURITY.md) | 安全漏洞报告 |

---

## 许可与引用

- **代码**：[MIT](LICENSE) © 2026 TraitYoung  
- **训练数据**：源自 [CBIS-DDSM](https://www.cancerimagingarchive.net/collection/cbis-ddsm/)（TCIA）；再分发须遵守原始数据条款。本仓库**不**重新授权影像。  
- **基座模型**：Stable Diffusion 1.5，再分发权重时遵循其原许可。

BibTeX / [`CITATION.cff`](CITATION.cff) 见英文 README。

路线图：[`ROADMAP.md`](ROADMAP.md) · 变更：[`CHANGELOG.md`](CHANGELOG.md) · 贡献：[`CONTRIBUTING.md`](CONTRIBUTING.md)。
