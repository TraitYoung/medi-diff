# Maintainer notes

These files help maintain **MammoGen**. They are **not** the public product docs — start from the root [README (EN)](../../README.md) / [README.zh-CN](../../README.zh-CN.md) and [docs/en](../en/README.md) / [docs/README.md](../README.md).

**Mainline defaults:** `CBIS_CLEAN_V2` + LoRA `mammo_sd15_v6_allMLO` (r=32, all MLO); label guard on by default; generated batches under `outputs/generated/samples/`.

**Pre-release smoke check:** `python3 scripts/tools/verify_ui_wiring.py` · [用户操作手册](用户操作手册.md).

| File | Role |
|------|------|
| [用户操作手册.md](用户操作手册.md) | Operator guide (also linked publicly) |
| [API接口文档.md](API接口文档.md) | REST API |
| [项目结构说明.md](项目结构说明.md) | Layout |
| [容错设计.md](容错设计.md) | Failure / degrade paths |
| [开发日志.md](开发日志.md) | Experiment journal (historical) |
| [craft-lama-设计规格.md](craft-lama-设计规格.md) | Archived CRAFT+LaMa design |
| [craft-lama-实施计划.md](craft-lama-实施计划.md) | Archived CRAFT+LaMa plan |

Personal migration notes and study scratchpads live under [`docs/private/`](../private/).
