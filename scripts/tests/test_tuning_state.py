"""Unit tests for advisor tuning_state helpers."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.assistant.tuning_state import (
    append_param_history,
    build_latest_tuning_state,
    load_latest_tuning_state,
    sync_latest_tuning_state,
    write_latest_tuning_state,
)


def test_write_and_load_latest_tuning_state(tmp_path: Path):
    state = {"parameters": {"strength": 0.44}, "source_tag": "demo"}
    path = write_latest_tuning_state(tmp_path, state)
    assert path.name == "LATEST_NEXT_RUN.json"
    loaded = load_latest_tuning_state(tmp_path)
    assert loaded is not None
    assert loaded["parameters"]["strength"] == 0.44


def test_build_latest_tuning_state_from_next_run(tmp_path: Path):
    eval_dir = tmp_path / "eval_run"
    report_base = tmp_path / "reports"
    eval_dir.mkdir()
    (eval_dir / "next_run_parameters.json").write_text(
        json.dumps({"parameters": {"strength": 0.4, "guidance_scale": 7.0}}),
        encoding="utf-8",
    )
    (eval_dir / "summary.json").write_text(
        json.dumps(
            {
                "pass_rate": 0.5,
                "strict_pass_rate": 0.5,
                "auto_calibrated": False,
            }
        ),
        encoding="utf-8",
    )
    state = build_latest_tuning_state(eval_dir, report_base, source_tag="t1")
    assert state is not None
    assert state["parameters"]["strength"] == 0.4
    assert state["eval_mode"] == "strict"
    assert "status_note_zh" in state


def test_build_latest_tuning_state_returns_none_without_params(tmp_path: Path):
    eval_dir = tmp_path / "eval_empty"
    eval_dir.mkdir()
    assert build_latest_tuning_state(eval_dir, tmp_path / "reports") is None


def test_append_param_history_keeps_last_five(tmp_path: Path):
    for i in range(7):
        append_param_history(
            tmp_path,
            source_tag=f"run_{i}",
            parameters={"strength": 0.3 + i * 0.01},
            metrics={"pass_rate": 0.1 * i},
        )
    hist = json.loads((tmp_path / "PARAM_HISTORY.json").read_text(encoding="utf-8"))
    assert len(hist) == 5
    assert hist[0]["index"] == 1
    assert hist[-1]["source_tag"] == "run_6"


def test_sync_latest_tuning_state_writes_files(tmp_path: Path):
    eval_dir = tmp_path / "eval_sync"
    report_base = tmp_path / "reports"
    eval_dir.mkdir()
    (eval_dir / "next_run_parameters.json").write_text(
        json.dumps({"parameters": {"num_steps": 40}}),
        encoding="utf-8",
    )
    (eval_dir / "summary.json").write_text(
        json.dumps({"pass_rate": 0.6, "mean_total_score": 0.7}),
        encoding="utf-8",
    )
    out = sync_latest_tuning_state(eval_dir, report_base, source_tag="sync")
    assert out is not None
    assert (report_base / "LATEST_NEXT_RUN.json").is_file()
    assert (report_base / "PARAM_HISTORY.json").is_file()
