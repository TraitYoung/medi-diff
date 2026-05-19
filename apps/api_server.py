#!/usr/bin/env python3
"""FastAPI 后端：封装钼靶图生成、评估和结果查询 API。"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
_START_TIME = time.time()

ROOT = Path(__file__).resolve().parents[1]
GENERATED_DIR   = ROOT / "outputs/generated"
EVAL_DIR        = ROOT / "outputs/eval"
REVIEWS_DIR     = ROOT / "outputs/reviews"
REPORT_DIR      = ROOT / "outputs/reports"
# 主线：SD1.5 + LoRA（优先 v4 标签清洗后权重，否则退回 v3 retrain）
_v4_final = ROOT / "outputs/lora/mammo_sd15_v4_clean/final_lora"
_v3_ck500 = ROOT / "outputs/lora/mammo_sd15_v3_retrain_20260502_1723/checkpoint-500"
DEFAULT_LORA = _v4_final if (_v4_final / "adapter_model.safetensors").is_file() else _v3_ck500
_mc_v2 = ROOT / "datasets/CBIS_CLEAN_V2/metadata_clean.csv"
_mc_v1 = ROOT / "datasets/CBIS_CLEAN/metadata_clean.csv"
METADATA_CSV = _mc_v2 if _mc_v2.is_file() else _mc_v1
DEFAULT_MODEL   = ROOT / "hf_cache/sd15"
# 归档路线：SDXL Inpaint（仍可用，见 /generate/sdxl）
SDXL_MODEL      = ROOT / "hf_cache/sdxl_inpaint_01"
SDXL_LORA       = ROOT / "outputs/lora/sdxl_lora_mammo_formal_v2/final"
REAL_IMAGES_DIR = ROOT / "datasets/jpeg"

app = FastAPI(
    title="Mammography Diffusion Generation API",
    description=(
        "乳腺钼靶扩散生成系统 REST API（毕业设计）。\n\n"
        "**主线**：`POST /generate/sd15` — SD1.5 + LoRA + Patch-Overlap img2img。\n"
        "**归档**：`POST /generate/sdxl` — SDXL Inpaint（历史路线，仍可复现）。\n"
        "异步任务：提交后用 `GET /jobs/{id}` 轮询状态；完成后用 `GET /batches` 查看结果。"
    ),
    version="2.0.0",
)

# 挂载静态目录（供前端直接访问图片）
if GENERATED_DIR.exists():
    app.mount("/static/generated", StaticFiles(directory=str(GENERATED_DIR)), name="generated")
if EVAL_DIR.exists():
    app.mount("/static/eval", StaticFiles(directory=str(EVAL_DIR)), name="eval")
if REVIEWS_DIR.exists():
    app.mount("/static/reviews", StaticFiles(directory=str(REVIEWS_DIR)), name="reviews")


# ─── 数据模型 ─────────────────────────────────────────────────────────────────

class GenerateSD15Request(BaseModel):
    """SD1.5 + LoRA + Patch-Overlap 主线生成请求（推荐）。"""
    num_images: int = Field(default=6, ge=1, le=50, description="生成张数")
    seed: int = Field(default=2026, description="随机种子")
    num_steps: int = Field(default=50, ge=10, le=80, description="扩散推理步数")
    strength: float = Field(default=0.42, ge=0.05, le=0.80, description="img2img 重绘强度")
    guidance_scale: float = Field(default=8.5, ge=1.0, le=12.0, description="CFG 强度")
    overlap_ratio: float = Field(default=0.90, ge=0.40, le=0.95, description="Patch 重叠率")
    global_guide_blend: float = Field(default=0.35, ge=0.0, le=0.70, description="全局引导混合比（≤0.70）")
    global_guide_strength: float = Field(default=0.20, ge=0.0, le=0.60, description="全局引导 img2img 强度")
    blend_sigma_divisor: float = Field(default=1.55, ge=1.05, le=3.0, description="接缝高斯 σ 控制")
    lora_path: str = Field(default=str(DEFAULT_LORA), description="LoRA 权重目录")
    output_subdir_prefix: str = Field(default="api_sd15", min_length=1, max_length=80)
    filter_view: str = Field(default="MLO", description="体位筛选：MLO/CC/空=不限")
    filter_density: str = Field(default="dense", description="密度筛选：dense/scattered/…/空=不限")
    eval_profile: str = Field(default="full", description="评审档位：full/patch")


class GenerateRequest(BaseModel):
    """SDXL Inpaint 归档路线（兼容旧接口，请优先使用 /generate/sd15）。"""
    num_images: int = Field(default=3, ge=1, le=50)
    seed: int = 2026
    steps: int = Field(default=30, ge=1, le=100)
    strength: float = Field(default=0.35, ge=0.0, le=1.0)
    guidance_scale: float = Field(default=2.0, ge=0.0, le=20.0)
    output_subdir_prefix: str = Field(default="api_inpaint", min_length=1, max_length=80)
    base_model: str = str(SDXL_MODEL)
    lora_path: str = Field(default="", description="LoRA 目录；留空则不加载")
    lora_scale: float = Field(default=0.70, ge=0.0, le=2.0)
    local_files_only: bool = True
    negative_prompt: str = "blurry, artifacts, text, watermark, oversaturated"


class ReviewRequest(BaseModel):
    images_dir: str
    output_dir: str = "outputs/reviews/review_output_api"
    recursive: bool = False
    real_images_dir: str = str(REAL_IMAGES_DIR)
    real_baseline_json: str = ""
    top_k: int = Field(default=30, ge=1, le=200)
    compute_fid: bool = True


class JobRecord(BaseModel):
    id: str
    kind: Literal["generate", "review"]
    status: Literal["queued", "running", "succeeded", "failed"]
    command: list[str]
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    elapsed_seconds: float | None = None
    return_code: int | None = None
    stdout: str = ""
    stderr: str = ""


jobs: dict[str, JobRecord] = {}
jobs_lock = threading.Lock()


# ─── 内部工具 ─────────────────────────────────────────────────────────────────

def _resolve_under_root(value: str, *, must_exist: bool = False) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"路径必须位于项目目录内: {value}") from exc
    if must_exist and not path.exists():
        raise HTTPException(status_code=404, detail=f"路径不存在: {path}")
    return path


def _start_job(kind: Literal["generate", "review"], command: list[str]) -> JobRecord:
    job = JobRecord(
        id=uuid.uuid4().hex[:12],
        kind=kind,
        status="queued",
        command=command,
        created_at=time.time(),
    )
    with jobs_lock:
        jobs[job.id] = job
    threading.Thread(target=_run_job, args=(job.id,), daemon=True).start()
    return job


def _run_job(job_id: str) -> None:
    with jobs_lock:
        job = jobs[job_id]
        job.status = "running"
        job.started_at = time.time()
        command = list(job.command)

    logger.info("Job %s started: %s", job_id, " ".join(command[:6]))
    proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)

    with jobs_lock:
        job = jobs[job_id]
        job.return_code = proc.returncode
        job.stdout = proc.stdout[-15000:]
        job.stderr = proc.stderr[-5000:]
        job.finished_at = time.time()
        if job.started_at is not None:
            job.elapsed_seconds = round(job.finished_at - job.started_at, 1)
        job.status = "succeeded" if proc.returncode == 0 else "failed"
        logger.info("Job %s %s in %.1fs (rc=%d)",
                    job_id, job.status, job.elapsed_seconds or 0,
                    proc.returncode)


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _batch_info(path: Path) -> dict:
    images = sorted([*path.glob("*.png"), *path.glob("*.jpg"), *path.glob("*.jpeg")])
    return {
        "name": path.name,
        "path": str(path),
        "image_count": len(images),
        "images": [img.name for img in images],
        "has_source_mapping": (path / "source_mapping.json").exists(),
        "mtime": path.stat().st_mtime,
    }


def _latest_batches(limit: int = 20) -> list[dict]:
    if not GENERATED_DIR.exists():
        return []
    rows: list[dict] = []
    for path in sorted(GENERATED_DIR.rglob("*_000"), key=lambda p: p.stat().st_mtime, reverse=True):
        if not path.is_dir():
            continue
        rows.append(_batch_info(path))
        if len(rows) >= limit:
            break
    return rows


def _eval_summaries(limit: int = 30) -> list[dict]:
    if not EVAL_DIR.exists():
        return []
    rows: list[dict] = []
    for path in sorted(EVAL_DIR.rglob("summary.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        if not (path.parent / "review_report.csv").exists():
            continue
        summary = _load_json(path)
        rows.append({
            "name": path.parent.name,
            "path": str(path.parent),
            "total_images": summary.get("total_images", 0),
            "pass_rate": summary.get("pass_rate", 0.0),
            "mean_total_score": summary.get("mean_total_score", 0.0),
            "mean_brisque": (summary.get("academic_metrics") or {}).get("mean_brisque"),
            "group_mean_scores": summary.get("group_mean_scores", {}),
            "violation_rates_top": dict(
                sorted((summary.get("violation_rates") or {}).items(), key=lambda x: -x[1])[:6]
            ),
        })
        if len(rows) >= limit:
            break
    return rows


def _review_summaries(limit: int = 30) -> list[dict]:
    rows = []
    if not REVIEWS_DIR.exists():
        return []
    for path in sorted(REVIEWS_DIR.rglob("summary.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        if not (path.parent / "review_report.csv").exists():
            continue
        summary = _load_json(path)
        rows.append({
            "name": path.parent.name,
            "path": str(path.parent),
            "total_images": summary.get("total_images", 0),
            "pass_rate": summary.get("pass_rate", 0.0),
            "mean_total_score": summary.get("mean_total_score", 0.0),
            "group_mean_scores": summary.get("group_mean_scores", {}),
            "academic_metrics": summary.get("academic_metrics", {}),
            "top_violations": summary.get("top_violations", []),
        })
        if len(rows) >= limit:
            break
    return rows


# ─── API 路由 ─────────────────────────────────────────────────────────────────

@app.get("/health", summary="健康检查")
def health() -> dict:
    gpu_info = {}
    try:
        import torch
        gpu_info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            gpu_info["cuda_device_count"] = torch.cuda.device_count()
            gpu_info["cuda_device_name"] = torch.cuda.get_device_name(0)
            gpu_info["cuda_memory_allocated_gb"] = round(
                torch.cuda.memory_allocated(0) / 1024**3, 2)
            gpu_info["cuda_memory_reserved_gb"] = round(
                torch.cuda.memory_reserved(0) / 1024**3, 2)
    except Exception:
        gpu_info["cuda_available"] = False
        gpu_info["error"] = "torch not available"

    mem_info = {}
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        mem_info["rss_mb"] = round(proc.memory_info().rss / 1024**2, 1)
        mem_info["vms_mb"] = round(proc.memory_info().vms / 1024**2, 1)
        mem_info["cpu_percent"] = proc.cpu_percent(interval=0.1)
    except Exception:
        mem_info["error"] = "psutil not available"
        try:
            with open(f"/proc/{os.getpid()}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        mem_info["rss_mb"] = round(
                            int(line.split()[1]) / 1024, 1)
                        break
        except Exception:
            pass

    return {
        "ok": True,
        "uptime_seconds": round(time.time() - _START_TIME, 1),
        "root": str(ROOT),
        "gpu": gpu_info,
        "memory": mem_info,
        "sd15_model_exists": DEFAULT_MODEL.exists(),
        "sd15_lora_exists": DEFAULT_LORA.exists(),
        "sdxl_model_exists": SDXL_MODEL.exists(),
        "metadata_csv_exists": METADATA_CSV.is_file(),
        "real_images_dir_exists": REAL_IMAGES_DIR.exists(),
        "generated_dir_exists": GENERATED_DIR.exists(),
        "eval_dir_exists": EVAL_DIR.exists(),
    }


@app.middleware("http")
async def _log_requests(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    dt = time.time() - t0
    logger.info("%s %s → %s %.3fs", request.method,
                request.url.path, response.status_code, dt)
    return response


@app.get("/results", summary="查询所有生成批次与评估汇总")
def results() -> dict:
    return {
        "batches": _latest_batches(),
        "evaluations": _eval_summaries(),
    }


@app.get("/batches", summary="列出所有生成批次")
def list_batches(limit: int = 20) -> list[dict]:
    return _latest_batches(limit=limit)


@app.get("/batches/{batch_name}", summary="查询指定批次详情")
def get_batch(batch_name: str) -> dict:
    path = _resolve_under_root(f"outputs/generated/毕业论文_生成图像/{batch_name}", must_exist=True)
    if not path.is_dir():
        raise HTTPException(status_code=404, detail="不是一个有效的批次目录")
    return _batch_info(path)


@app.get("/batches/{batch_name}/images/{filename}", summary="获取批次中的单张图像文件")
def get_image(batch_name: str, filename: str) -> FileResponse:
    img_path = _resolve_under_root(
        f"outputs/generated/毕业论文_生成图像/{batch_name}/{filename}", must_exist=True
    )
    if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise HTTPException(status_code=400, detail="不支持的文件类型")
    return FileResponse(str(img_path))


@app.get("/reviews", summary="列出旧版评估报告（兼容）")
def list_reviews(limit: int = 30) -> list[dict]:
    return _review_summaries(limit=limit)


@app.get("/evaluations", summary="列出所有评估结果（主线 eval 目录）")
def list_evaluations(limit: int = 30) -> list[dict]:
    return _eval_summaries(limit=limit)


@app.get("/evaluations/{eval_name}", summary="查询指定评估详情")
def get_evaluation(eval_name: str) -> dict:
    p = _resolve_under_root(f"outputs/eval/{eval_name}", must_exist=True)
    summary = _load_json(p / "summary.json")
    return {
        "name": eval_name,
        "path": str(p),
        "summary": summary,
        "has_advisor": (p / "advisor_suggestions.md").exists(),
        "has_report": (p / "review_report.csv").exists(),
    }


@app.get("/reports/latest", summary="获取最新 FINAL_REPORT 路径")
def get_latest_report() -> dict:
    p = REPORT_DIR / "LATEST_REPORT.txt"
    if not p.is_file():
        raise HTTPException(status_code=404, detail="暂无报告，请先运行全自动流水线")
    path = p.read_text(encoding="utf-8").strip()
    return {"latest_report_path": path, "exists": Path(path).is_file()}


@app.get("/reports/history", summary="获取参数调优历史（最近 5 轮）")
def get_param_history() -> list[dict]:
    p = REPORT_DIR / "PARAM_HISTORY.json"
    if not p.is_file():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []


@app.post("/generate/sd15", summary="提交 SD1.5 主线生成任务（推荐，异步）")
def generate_sd15(req: GenerateSD15Request) -> JobRecord:
    """使用 SD1.5 + LoRA + Patch-Overlap 生成乳腺钼靶图像（主线方案）。"""
    lora_abs = _resolve_under_root(req.lora_path or str(DEFAULT_LORA), must_exist=False)
    command = [
        sys.executable, "scripts/generation/run_mammo_sd15.py",
        "--base-model-local", str(DEFAULT_MODEL),
        "--lora-path", str(lora_abs),
        "--num-images", str(req.num_images),
        "--seed", str(req.seed),
        "--num-steps", str(req.num_steps),
        "--strength", str(req.strength),
        "--guidance-scale", str(req.guidance_scale),
        "--overlap-ratio", str(req.overlap_ratio),
        "--global-guide-blend", str(req.global_guide_blend),
        "--global-guide-strength", str(req.global_guide_strength),
        "--blend-sigma-divisor", str(req.blend_sigma_divisor),
        "--output-subdir-prefix", req.output_subdir_prefix,
    ]
    if METADATA_CSV.is_file():
        command.extend(["--metadata-csv", str(METADATA_CSV)])
        if req.filter_view.strip():
            command.extend(["--filter-view", req.filter_view.strip()])
        if req.filter_density.strip():
            command.extend(["--filter-density", req.filter_density.strip()])
    return _start_job("generate", command)


@app.post("/generate", summary="提交 SDXL Inpaint 生成任务（归档路线，异步）",
          deprecated=True)
@app.post("/generate/sdxl", summary="提交 SDXL Inpaint 生成任务（归档路线，异步）")
def generate_sdxl(req: GenerateRequest) -> JobRecord:
    """SDXL Inpaint 路线已归档。主线请使用 /generate/sd15。"""
    job_id = f"archived-{uuid.uuid4().hex[:12]}"
    return JobRecord(
        job_id=job_id,
        status="completed",
        message="SDXL Inpaint 路线已归档，脚本已移除。请使用 POST /generate/sd15。",
        created_at=datetime.now().isoformat(),
    )


@app.post("/review", summary="提交评估任务（异步）")
def review(req: ReviewRequest) -> JobRecord:
    images_dir = _resolve_under_root(req.images_dir, must_exist=True)
    output_dir = _resolve_under_root(req.output_dir, must_exist=False)
    command = [
        sys.executable, "scripts/evaluation/review_generated_images.py",
        "--images-dir", str(images_dir),
        "--output-dir", str(output_dir),
        "--top-k", str(req.top_k),
    ]
    command.append("--recursive" if req.recursive else "--no-recursive")
    if req.compute_fid and req.real_images_dir.strip():
        real_dir = _resolve_under_root(req.real_images_dir, must_exist=True)
        command.extend(["--real-images-dir", str(real_dir)])
    elif req.real_baseline_json.strip():
        command.extend(["--real-baseline-json", str(_resolve_under_root(req.real_baseline_json, must_exist=True))])
    return _start_job("review", command)


@app.get("/jobs", summary="列出所有任务")
def list_jobs(limit: int = 50) -> list[JobRecord]:
    with jobs_lock:
        return sorted(jobs.values(), key=lambda j: j.created_at, reverse=True)[:limit]


@app.get("/jobs/{job_id}", summary="查询单个任务状态")
def get_job(job_id: str) -> JobRecord:
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    return job


@app.delete("/jobs/{job_id}", summary="删除任务记录（仅内存，不终止进程）")
def delete_job(job_id: str) -> dict:
    with jobs_lock:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="任务不存在")
        del jobs[job_id]
    return {"deleted": job_id}
