#!/usr/bin/env python3
"""Shared read/write helpers for the advisor tuning loop state.

The latest tuning state intentionally keeps the historical
``LATEST_NEXT_RUN.json`` top-level contract so older readers can keep using
``parameters`` while newer UI/reporting code can also inspect evaluation mode
and source sampling metadata.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

LATEST_NEXT_RUN_FILE = "LATEST_NEXT_RUN.json"


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return data


def _parameters_from_next_run(path: Path) -> dict[str, Any]:
    data = _read_json(path)
    params = data.get("parameters")
    if isinstance(params, dict):
        return dict(params)
    return {str(k): v for k, v in data.items() if not str(k).startswith("_")}


def _read_optional_dict(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return _read_json(path)
    except (json.JSONDecodeError, OSError, ValueError):
        return {}


def _infer_run_params(eval_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    """Read run_params.json from eval dir first, then generated images dir."""
    paths = [eval_dir / "run_params.json"]
    images_dir = str(summary.get("images_dir") or "").strip()
    if images_dir:
        paths.append(Path(images_dir) / "run_params.json")
    for path in paths:
        if path.is_file():
            data = _read_optional_dict(path)
            if data:
                return data
    return {}


def _status_note_zh(summary: dict[str, Any]) -> str:
    strict = summary.get("strict_pass_rate")
    technical = summary.get("technical_pass_rate")
    passed = summary.get("pass_rate")
    if summary.get("auto_calibrated") and strict is not None and strict != passed:
        return "本轮含真实基线校准；调参请同时查看 strict_pass_rate，避免只按校准 pass_rate 判断。"
    if strict is not None and strict == 0:
        return "严格复盘仍未通过，下一轮应优先处理严格缺陷标签。"
    if technical is not None and technical != passed:
        return "pass_rate 与 technical_pass_rate 不一致，建议以 technical/strict 口径辅助调参。"
    return "顾问建议已同步，可用于 Gradio 调参页或 --from-latest-tuning。"


def load_latest_tuning_state(report_base: Path) -> dict[str, Any] | None:
    """Load ``outputs/reports/LATEST_NEXT_RUN.json`` if it exists."""
    path = report_base / LATEST_NEXT_RUN_FILE
    if not path.is_file():
        return None
    return _read_json(path)


def write_latest_tuning_state(report_base: Path, state: dict[str, Any]) -> Path:
    """Write the latest tuning state and return its path."""
    report_base.mkdir(parents=True, exist_ok=True)
    path = report_base / LATEST_NEXT_RUN_FILE
    path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def build_latest_tuning_state(
    eval_dir: Path,
    report_base: Path,
    *,
    source_tag: str | None = None,
    source_seed: int | None = None,
    source_quality_sort: bool | None = None,
    eval_mode: str | None = None,
) -> dict[str, Any] | None:
    """Build a compatible latest tuning state from an evaluation directory.

    Missing ``next_run_parameters.json`` means there is no advisor JSON to
    publish, so this returns ``None`` instead of failing the caller's workflow.
    Invalid JSON still raises, because a corrupt file should be fixed.
    """
    eval_dir = eval_dir.resolve()
    report_base = report_base.resolve()
    next_run_path = eval_dir / "next_run_parameters.json"
    if not next_run_path.is_file():
        return None

    params = _parameters_from_next_run(next_run_path)
    summary = _read_optional_dict(eval_dir / "summary.json")
    run_params = _infer_run_params(eval_dir, summary)

    inferred_source_seed = source_seed
    if inferred_source_seed is None and run_params.get("source_seed") is not None:
        try:
            inferred_source_seed = int(run_params["source_seed"])
        except (TypeError, ValueError):
            inferred_source_seed = None

    inferred_quality_sort = source_quality_sort
    if inferred_quality_sort is None and run_params.get("source_quality_sort") is not None:
        inferred_quality_sort = bool(run_params["source_quality_sort"])

    inferred_eval_mode = eval_mode
    if inferred_eval_mode is None:
        inferred_eval_mode = "calibrated" if bool(summary.get("auto_calibrated")) else "strict"

    state: dict[str, Any] = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_tag": source_tag or eval_dir.name,
        "source_eval_dir": str(eval_dir),
        "next_run_file": str(next_run_path.resolve()),
        "eval_mode": inferred_eval_mode,
        "source_seed": inferred_source_seed,
        "source_quality_sort": inferred_quality_sort,
        "parameters": params,
    }

    for key in (
        "auto_calibrated",
        "technical_pass_rate",
        "strict_pass_rate",
        "pass_rate",
        "strict_violation_rates",
        "technical_violation_rates",
    ):
        if key in summary:
            state[key] = summary.get(key)

    if "technical_violation_rates" not in state and isinstance(summary.get("violation_rates"), dict):
        state["technical_violation_rates"] = summary.get("violation_rates")

    if summary:
        state["status_note_zh"] = _status_note_zh(summary)

    # Keep report_base in the signature intentionally: callers pass the same
    # pair to build/write helpers, and tests can assert the resolved latest path.
    _ = report_base
    return state


def sync_latest_tuning_state(
    eval_dir: Path,
    report_base: Path,
    *,
    source_tag: str | None = None,
    source_seed: int | None = None,
    source_quality_sort: bool | None = None,
    eval_mode: str | None = None,
) -> Path | None:
    """Build and write latest tuning state; return ``None`` when no next-run JSON exists."""
    state = build_latest_tuning_state(
        eval_dir,
        report_base,
        source_tag=source_tag,
        source_seed=source_seed,
        source_quality_sort=source_quality_sort,
        eval_mode=eval_mode,
    )
    if state is None:
        return None
    return write_latest_tuning_state(report_base, state)
