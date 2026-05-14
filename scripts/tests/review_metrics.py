#!/usr/bin/env python3
"""
评审管线验证指标：从 review 输出（CSV 行或 summary per_image）计算 Ground Truth 对齐的混淆与验收指标。

命名约定（与测试方案一致）：
- `good_*.png`：肉眼正常钼靶
- `bad_*.png`：肉眼鬼图（可带类别前缀如 `bad_nonmammo_095.png`）

指标定义：
- **Veto 召回（对 bad）**：命中 = ok=False 或 tier≥2（rank_tier）
- **正常误杀（对 good）**：误杀 = ok=False 或 tier≥2
- **Top5 纯度**：全集中按 final_rank_score 降序取 Top5，须全部为 good 且 tier==1 且 ok
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def label_from_filename(path_str: str) -> str | None:
    name = Path(path_str).name.lower()
    if name.startswith("good_"):
        return "good"
    if name.startswith("bad_"):
        return "bad"
    return None


def bad_category(path_str: str) -> str:
    """bad_nonmammo_xxx → nonmammo；否则为 other。"""
    stem = Path(path_str).stem.lower()
    if not stem.startswith("bad_"):
        return ""
    rest = stem[4:]  # after "bad_"
    if "_" in rest:
        return rest.split("_", 1)[0]
    return "generic"


def _tier(row: dict[str, Any]) -> int:
    for k in ("rank_tier", "tier"):
        if k in row and row[k] is not None and str(row[k]).strip() != "":
            try:
                return int(float(row[k]))
            except (TypeError, ValueError):
                pass
    return 3


def _ok(row: dict[str, Any]) -> bool:
    v = row.get("ok")
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ("1", "true", "yes")
    return bool(v)


def _score(row: dict[str, Any]) -> float:
    for k in ("final_rank_score", "final_rank"):
        if k in row and row[k] is not None and str(row[k]).strip() != "":
            try:
                return float(row[k])
            except (TypeError, ValueError):
                pass
    return 0.0


@dataclass
class ValidationResult:
    n_good: int = 0
    n_bad: int = 0
    n_unlabeled: int = 0
    bad_hits: int = 0
    bad_misses: int = 0
    good_false_kill: int = 0
    good_ok: int = 0
    top5_purity: bool = False
    top5_basenames: list[str] = field(default_factory=list)
    bad_miss_files: list[str] = field(default_factory=list)
    good_false_kill_files: list[str] = field(default_factory=list)
    per_bad_category: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def veto_recall(self) -> float:
        t = self.bad_hits + self.bad_misses
        return self.bad_hits / t if t else 0.0

    @property
    def good_false_kill_rate(self) -> float:
        return self.good_false_kill / self.n_good if self.n_good else 0.0


def rows_from_csv(csv_path: Path) -> list[dict[str, Any]]:
    with csv_path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def rows_from_summary(summary_path: Path) -> list[dict[str, Any]]:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    return list(data.get("per_image") or [])


def compute_validation_metrics(rows: list[dict[str, Any]]) -> ValidationResult:
    """rows 须含 image, ok, rank_tier/tier, final_rank_score。"""
    res = ValidationResult()
    keyed: list[tuple[str, dict[str, Any]]] = []
    for row in rows:
        img = row.get("image") or row.get("path") or ""
        lab = label_from_filename(str(img))
        keyed.append((str(img), row))
        if lab == "good":
            res.n_good += 1
        elif lab == "bad":
            res.n_bad += 1
        else:
            res.n_unlabeled += 1

    for img, row in keyed:
        lab = label_from_filename(img)
        if lab is None:
            continue
        tier = _tier(row)
        ok = _ok(row)
        if lab == "bad":
            # 命中鬼图：未以「好图」蒙混过关（tier1 且 ok 视为漏网）
            hit = (not ok) or (tier >= 2)
            if hit:
                res.bad_hits += 1
            else:
                res.bad_misses += 1
                res.bad_miss_files.append(Path(img).name)
            cat = bad_category(img)
            if cat:
                bucket = res.per_bad_category.setdefault(cat, {"hit": 0, "miss": 0})
                if hit:
                    bucket["hit"] += 1
                else:
                    bucket["miss"] += 1
        else:
            # good：误杀 = 被判差或挡在 tier2+
            fk = (not ok) or (tier >= 2)
            if fk:
                res.good_false_kill += 1
                res.good_false_kill_files.append(Path(img).name)
            else:
                res.good_ok += 1

    # Top5 纯度：仅对带标签的子集排序；若混有无标签，仅对有 final_rank 的排序
    labeled = [(img, row) for img, row in keyed if label_from_filename(img) in ("good", "bad")]
    # 与 review_generated_images.py 保持一致：先 tier 升序，再 final_rank_score 降序。
    labeled.sort(key=lambda x: (_tier(x[1]), -_score(x[1])))
    top5 = labeled[:5]
    res.top5_basenames = [Path(img).name for img, _ in top5]
    if len(top5) < 5:
        res.top5_purity = False
    else:
        res.top5_purity = all(
            label_from_filename(img) == "good" and _ok(row) and _tier(row) == 1 for img, row in top5
        )

    return res


def format_report_md(
    res: ValidationResult,
    title: str = "评审管线验证报告",
) -> str:
    lines = [
        f"# {title}",
        "",
        "## 样本统计",
        "",
        f"| 类型 | 数量 |",
        f"|------|------|",
        f"| good | {res.n_good} |",
        f"| bad | {res.n_bad} |",
        f"| 未按规则命名（忽略 GT） | {res.n_unlabeled} |",
        "",
        "## 验收指标",
        "",
        f"| 指标 | 数值 | 目标 |",
        f"|------|------|------|",
        f"| Veto 召回率（bad） | {res.veto_recall:.2%} | ≥90% |",
        f"| 正常误杀率（good） | {res.good_false_kill_rate:.2%} | ≤10% |",
        f"| Top5 纯度 | {'通过' if res.top5_purity else '未通过'} | 100% |",
        "",
        "## Top5（按 final_rank_score）",
        "",
        "\n".join(f"- {n}" for n in res.top5_basenames) if res.top5_basenames else "- （不足 5 张）",
        "",
    ]
    if res.bad_miss_files:
        lines.extend(
            [
                "## 漏网 bad（应优先调阈值/检查器）",
                "",
                "\n".join(f"- {x}" for x in res.bad_miss_files),
                "",
            ]
        )
    if res.good_false_kill_files:
        lines.extend(
            [
                "## 误杀 good（阈值过严）",
                "",
                "\n".join(f"- {x}" for x in res.good_false_kill_files),
                "",
            ]
        )
    if res.per_bad_category:
        lines.append("## bad 子类命中情况")
        lines.append("")
        lines.append("| 子类前缀 | 命中 | 漏网 |")
        lines.append("|----------|------|------|")
        for k, v in sorted(res.per_bad_category.items()):
            lines.append(f"| {k} | {v.get('hit', 0)} | {v.get('miss', 0)} |")
        lines.append("")
    return "\n".join(lines)
