# 开发者自用文档（个人备忘）

这套内容**只对维护本仓库的自己**有用：**不保证**文风适合答辩、合作方或审稿人。对外请以根目录 **`README.md`** 与 **`docs/评价体系说明.md`** 等为口径。

**主线约定速查**（2026-05）：**CBIS_CLEAN_V2** + **LoRA `mammo_sd15_v6_allMLO`**（r=32，全 MLO 训练）；**标签守护**默认开启（`--no-legacy-label-guard` 可关）；**`run_full_report` / `run_generate_eval_advise` 默认开启后处理**（`--no-postprocess` 可选）。

**答辩前自检**：`python3 scripts/tools/verify_ui_wiring.py`；手册 `docs/用户操作手册.md` §7。

| 文件 | 干嘛用 |
|------|--------|
| [`工作台备忘.md`](工作台备忘.md) | 怎么快速理解系统在干什么、和大模型会话时怎么省事、评审/流水线常见坑 |
| [`答辩必备.md`](答辩必备.md) | **答辩最少要知道什么**——架构、选型论证、评估体系、演示流程、问答记录 |
| [`craft-lama-设计规格.md`](craft-lama-设计规格.md) | CRAFT+LaMa 文字擦除方案设计（2026-05-10） |
| [`craft-lama-实施计划.md`](craft-lama-实施计划.md) | CRAFT+LaMa 实施步骤记录（2026-05-10） |

之后如果笔记变长，可以再拆：`命令速查.md`、`评审调试.md` 等，索引仍放在本 README。
