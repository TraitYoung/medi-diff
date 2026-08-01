# 开发者自用文档（个人备忘）

这套内容**只对维护本仓库的自己**有用：**不保证**文风适合对外宣传或合作方。对外请以根目录 **`README.md`**（品牌 **MammoGen**）与 **`docs/评价体系说明.md`** 等为口径。

**主线约定速查**：**CBIS_CLEAN_V2** + **LoRA `mammo_sd15_v6_allMLO`**（r=32，全 MLO 训练）；**标签守护**默认开启（`--no-legacy-label-guard` 可关）；默认生成输出根 **`outputs/generated/samples/`**。

**上线前自检**：`python3 scripts/tools/verify_ui_wiring.py`；手册 [`用户操作手册.md`](用户操作手册.md)。

| 文件 | 干嘛用 |
|------|--------|
| [`工作台备忘.md`](工作台备忘.md) | 怎么快速理解系统在干什么、和大模型会话时怎么省事、评审/流水线常见坑 |
| [`craft-lama-设计规格.md`](craft-lama-设计规格.md) | CRAFT+LaMa 文字擦除方案设计（2026-05-10） |
| [`craft-lama-实施计划.md`](craft-lama-实施计划.md) | CRAFT+LaMa 实施步骤记录（2026-05-10） |

> 若干毕设/答辩口径文档已加入根目录 `.gitignore`（本地可保留，不进入公开跟踪）。

之后如果笔记变长，可以再拆：`命令速查.md`、`评审调试.md` 等，索引仍放在本 README。
