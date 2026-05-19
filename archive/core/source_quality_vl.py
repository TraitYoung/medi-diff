"""VL-based source image quality screening for mammography generation.

Filters source images using Qwen-VL to assess contrast, exposure, artifacts,
and overall suitability as SD img2img input. Designed to be cost-efficient:
384px thumbnails, JPEG Q=70, YES/NO output (max 8 tokens), with file-based cache.

Usage:
    from scripts.core.source_quality_vl import filter_pool_by_vl
    good_pool = filter_pool_by_vl(pool, target_count=6)
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

# Reuse JPEG encoding from label heuristic (avoid duplication)
import sys as _sys
_path_preproc = str(Path(__file__).resolve().parents[2] / "scripts" / "preprocessing")
if _path_preproc not in _sys.path:
    _sys.path.insert(0, _path_preproc)
from mammo_label_heuristic import _gray_to_base64_jpeg  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "outputs" / ".vl_source_cache"

_VL_SOURCE_MODEL_DEFAULT = "qwen3-vl-plus"


def _resolve_vl_model() -> str:
    return (
        os.environ.get("QWEN_VL_MODEL", "").strip()
        or _VL_SOURCE_MODEL_DEFAULT
    )


# ── VL prompt ──────────────────────────────────────────────────────────────────

_VL_SOURCE_SYSTEM = (
    "You are a mammography quality control specialist. "
    "Answer ONLY with the single word YES or NO. No punctuation, no explanation."
)

_VL_SOURCE_USER = (
    "Evaluate this mammogram as a SOURCE IMAGE for AI image generation (Stable Diffusion img2img). "
    "A good source image should have: "
    "adequate tissue contrast (not severely underexposed or overexposed), "
    "visible breast parenchyma and structural details, "
    "no severe compression artifacts, grid patterns, or scanner noise that obscures anatomy. "
    "Minor imperfections are acceptable. "
    "Answer NO only if the image is severely degraded: nearly all black (underexposed), "
    "nearly all white (overexposed/saturated), dominated by non-anatomical artifacts, "
    "or the breast tissue is barely visible (covers <5% of the frame). "
    "Answer YES if the image is usable as a generation source, even if not perfect."
)

_VL_MAX_SIDE = 384
_VL_JPEG_QUAL = 70
_VL_MAX_TOKENS = 8
_VL_TIMEOUT = 25


# ── Cache ──────────────────────────────────────────────────────────────────────

def _cache_key(file_path: Path) -> str:
    try:
        st = file_path.stat()
        raw = f"{file_path.resolve()}:{st.st_mtime_ns}:{st.st_size}"
    except OSError:
        raw = str(file_path.resolve())
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def _cache_path() -> Path:
    return CACHE_DIR / "cache.json"


def _load_cache() -> dict:
    cp = _cache_path()
    if cp.is_file():
        try:
            return json.loads(cp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path().write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def _cache_get(file_path: Path) -> bool | None:
    key = _cache_key(file_path)
    entry = _load_cache().get(key)
    if entry and isinstance(entry, dict) and "pass" in entry:
        return bool(entry["pass"])
    return None


def _cache_set(file_path: Path, passed: bool) -> None:
    key = _cache_key(file_path)
    cache = _load_cache()
    cache[key] = {"pass": passed, "ts": time.time()}
    # Prune entries older than 7 days
    cutoff = time.time() - 7 * 86400
    cache = {k: v for k, v in cache.items() if v.get("ts", 0) > cutoff}
    _save_cache(cache)


# ── VL call ────────────────────────────────────────────────────────────────────

def ask_vl_source_quality(
    gray: np.ndarray,
    *,
    max_side: int = _VL_MAX_SIDE,
    jpeg_quality: int = _VL_JPEG_QUAL,
    timeout: int = _VL_TIMEOUT,
) -> tuple[bool, str]:
    """Returns (pass: bool, reason: str). Pass=True means the image is suitable as SD source."""
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    b64 = _gray_to_base64_jpeg(gray, max_side, jpeg_quality)
    data_url = f"data:image/jpeg;base64,{b64}"

    # Load .env
    dotenv = ROOT / ".env"
    if dotenv.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(dotenv, override=False)
        except ImportError:
            pass

    api_key = os.environ.get("QWEN_API_KEY", "").strip()
    base_url = os.environ.get(
        "QWEN_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ).strip()
    model = _resolve_vl_model()

    if not api_key:
        return False, "no QWEN_API_KEY"

    try:
        from openai import OpenAI
    except ImportError:
        return False, "openai not installed"

    client = OpenAI(api_key=api_key, base_url=base_url)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _VL_SOURCE_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": _VL_SOURCE_USER},
                    ],
                },
            ],
            max_tokens=_VL_MAX_TOKENS,
            timeout=timeout,
        )
    except Exception as e:
        return False, f"VL call failed: {e}"

    raw = (resp.choices[0].message.content or "").strip().upper()
    has_yes = "YES" in raw
    has_no = "NO" in raw

    if has_yes and not has_no:
        return True, f"VL=YES"
    if has_no and not has_yes:
        return False, f"VL=NO"
    return False, f"VL_ambiguous={raw!r}"


# ── Pool filter ────────────────────────────────────────────────────────────────

def filter_pool_by_vl(
    pool: list[Path],
    target_count: int,
    *,
    batch_size: int = 12,
    max_vl_calls: int = 60,
    verbose: bool = True,
) -> list[Path]:
    """Iteratively screen source images with VL until target_count approved.

    Args:
        pool: shuffled list of source image paths.
        target_count: how many approved images to collect.
        batch_size: images to screen per iteration.
        max_vl_calls: hard cap on total VL calls (cost guard).
        verbose: print progress.

    Returns:
        Approved source paths (length ≤ target_count).
    """
    approved: list[Path] = []
    remaining = list(pool)
    vl_calls = 0
    cached_hits = 0

    while len(approved) < target_count and remaining and vl_calls < max_vl_calls:
        batch = remaining[:batch_size]
        remaining = remaining[batch_size:]

        for p in batch:
            if len(approved) >= target_count:
                break

            # Check cache first
            cached = _cache_get(p)
            if cached is not None:
                cached_hits += 1
                if cached:
                    approved.append(p)
                continue

            # VL call
            vl_calls += 1
            gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                _cache_set(p, False)
                continue

            passed, reason = ask_vl_source_quality(gray)
            _cache_set(p, passed)

            if passed:
                approved.append(p)
                if verbose:
                    print(f"  [VL source] ✓ {p.name} ({reason}) — approved ({len(approved)}/{target_count})")
            else:
                if verbose:
                    print(f"  [VL source] ✗ {p.name} ({reason}) — rejected")

            # Small delay between calls to avoid rate limiting
            if vl_calls > 0 and vl_calls % 8 == 0:
                time.sleep(0.3)

    if verbose:
        print(
            f"  [VL source] screening done: {vl_calls} VL calls, "
            f"{cached_hits} cached, {len(approved)} approved (needed {target_count})"
        )

    if len(approved) < target_count:
        # Top up from remaining pool (non-VL-checked) if we ran out
        shortage = target_count - len(approved)
        extra = remaining[:shortage]
        if extra:
            approved.extend(extra)
            if verbose:
                print(f"  [VL source] topped up {len(extra)} unchecked images (VL calls exhausted)")

    return approved
