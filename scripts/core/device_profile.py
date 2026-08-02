"""GPU memory profiles for MammoGen generation (cloud / local / tight).

Resolves an auto profile from CUDA VRAM and applies Diffusers memory opts so
consumer GPUs (e.g. RTX 5070 Ti, 12–16 GiB) can run the SD1.5 + LoRA mainline
without rewriting quality defaults.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

logger = logging.getLogger(__name__)

MemProfile = Literal["cloud", "local", "tight"]
MemProfileChoice = Literal["auto", "cloud", "local", "tight"]

VALID_PROFILES: frozenset[str] = frozenset({"cloud", "local", "tight"})
VALID_CHOICES: frozenset[str] = frozenset({"auto", "cloud", "local", "tight"})

_GIB = 1024**3
_CLOUD_MIN_BYTES = 20 * _GIB
_LOCAL_MIN_BYTES = 10 * _GIB

CU128_INSTALL_HINT = (
    "pip install torch torchvision --index-url "
    "https://download.pytorch.org/whl/cu128"
)


@dataclass(frozen=True)
class CudaEnvInfo:
    available: bool
    device_name: str
    total_memory_bytes: int
    total_memory_gib: float
    torch_version: str
    torch_cuda_version: str | None
    capability: tuple[int, int] | None


def resolve_mem_profile(
    requested: str = "auto",
    *,
    total_memory_bytes: int | None = None,
) -> MemProfile:
    """Map ``auto|cloud|local|tight`` to a concrete profile.

    Auto thresholds (total VRAM):
      - ``>= 20 GiB`` → cloud
      - ``10–20 GiB`` → local
      - ``< 10 GiB`` → tight
      - no CUDA / unknown → local (safer consumer default)
    """
    choice = (requested or "auto").strip().lower()
    if choice not in VALID_CHOICES:
        raise ValueError(
            f"Unknown mem profile {requested!r}; expected one of {sorted(VALID_CHOICES)}"
        )
    if choice != "auto":
        return choice  # type: ignore[return-value]

    if total_memory_bytes is None:
        total_memory_bytes = _probe_total_memory_bytes()
    if total_memory_bytes is None or total_memory_bytes <= 0:
        logger.info("mem-profile auto: no CUDA VRAM info → local")
        return "local"
    if total_memory_bytes >= _CLOUD_MIN_BYTES:
        return "cloud"
    if total_memory_bytes >= _LOCAL_MIN_BYTES:
        return "local"
    return "tight"


def _probe_total_memory_bytes() -> int | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    try:
        return int(torch.cuda.get_device_properties(0).total_memory)
    except Exception as exc:  # noqa: BLE001 — probe only
        logger.warning("Failed to read CUDA device properties: %s", exc)
        return None


def describe_cuda_environment() -> CudaEnvInfo:
    try:
        import torch
    except ImportError:
        return CudaEnvInfo(
            available=False,
            device_name="",
            total_memory_bytes=0,
            total_memory_gib=0.0,
            torch_version="(not installed)",
            torch_cuda_version=None,
            capability=None,
        )

    torch_version = str(torch.__version__)
    torch_cuda_version = getattr(torch.version, "cuda", None)
    if not torch.cuda.is_available():
        return CudaEnvInfo(
            available=False,
            device_name="",
            total_memory_bytes=0,
            total_memory_gib=0.0,
            torch_version=torch_version,
            torch_cuda_version=torch_cuda_version,
            capability=None,
        )

    props = torch.cuda.get_device_properties(0)
    total = int(props.total_memory)
    try:
        capability = tuple(int(x) for x in torch.cuda.get_device_capability(0))  # type: ignore[misc]
    except Exception:  # noqa: BLE001
        capability = None
    return CudaEnvInfo(
        available=True,
        device_name=str(props.name),
        total_memory_bytes=total,
        total_memory_gib=round(total / _GIB, 2),
        torch_version=torch_version,
        torch_cuda_version=torch_cuda_version,
        capability=capability,  # type: ignore[arg-type]
    )


def _parse_cuda_major_minor(cuda_version: str | None) -> tuple[int, int] | None:
    if not cuda_version:
        return None
    parts = str(cuda_version).split(".")
    try:
        return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
    except ValueError:
        return None


def needs_cu128_for_device(env: CudaEnvInfo | None = None) -> bool:
    """True when GPU looks like RTX 50-series / Blackwell but torch CUDA < 12.8."""
    info = env or describe_cuda_environment()
    if not info.available:
        return False
    name = info.device_name or ""
    looks_50 = (
        "50" in name
        or "Blackwell" in name
        or "blackwell" in name.lower()
        or (info.capability is not None and info.capability[0] >= 12)
    )
    if not looks_50:
        return False
    parsed = _parse_cuda_major_minor(info.torch_cuda_version)
    if parsed is None:
        return False
    return parsed < (12, 8)


def warn_if_blackwell_needs_cu128(env: CudaEnvInfo | None = None) -> None:
    """Warn when a 50-series / Blackwell GPU is paired with torch CUDA < 12.8."""
    info = env or describe_cuda_environment()
    if not needs_cu128_for_device(info):
        return
    logger.warning(
        "GPU %s looks like RTX 50-series / Blackwell (sm_120) but "
        "torch.version.cuda=%s (< 12.8). Expect 'no kernel image' errors. "
        "Reinstall with: %s",
        info.device_name,
        info.torch_cuda_version,
        CU128_INSTALL_HINT,
    )


def _try_enable_xformers(pipe: Any) -> bool:
    enable = getattr(pipe, "enable_xformers_memory_efficient_attention", None)
    if not callable(enable):
        return False
    try:
        enable()
        logger.info("Enabled xformers memory-efficient attention")
        return True
    except Exception as exc:  # noqa: BLE001 — optional accel
        logger.info("xformers unavailable (%s); using default attention / SDPA", exc)
        return False


def place_and_optimize_pipe(pipe: Any, profile: MemProfile, device: Any) -> MemProfile:
    """Move ``pipe`` onto device (or CPU-offload) and apply profile memory opts.

    - cloud: full ``pipe.to(device)`` + optional xformers
    - local: ``pipe.to(device)`` + attention/VAE slicing & tiling + optional xformers
    - tight: local opts + ``enable_model_cpu_offload()`` (do not keep full model on GPU)
    """
    if profile not in VALID_PROFILES:
        raise ValueError(f"Invalid resolved profile: {profile!r}")

    device_type = getattr(device, "type", str(device))
    use_cuda = device_type == "cuda"

    if profile == "tight" and use_cuda:
        _apply_local_slicing(pipe)
        _try_enable_xformers(pipe)
        offload = getattr(pipe, "enable_model_cpu_offload", None)
        if callable(offload):
            offload()
            logger.info("mem-profile=tight: model CPU offload enabled")
        else:
            logger.warning(
                "enable_model_cpu_offload unavailable; falling back to pipe.to(%s)",
                device,
            )
            pipe.to(device)
        return profile

    pipe.to(device)
    if profile == "local":
        _apply_local_slicing(pipe)
        _try_enable_xformers(pipe)
        logger.info("mem-profile=local: attention/VAE slicing+tiling enabled")
    else:
        _try_enable_xformers(pipe)
        logger.info("mem-profile=cloud: full GPU residency")
    return profile


def _apply_local_slicing(pipe: Any) -> None:
    for name in ("enable_attention_slicing", "enable_vae_slicing", "enable_vae_tiling"):
        fn = getattr(pipe, name, None)
        if callable(fn):
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                logger.debug("%s failed: %s", name, exc)


def should_empty_cache_between_images(profile: MemProfile) -> bool:
    return profile in ("local", "tight")


def log_profile_banner(
    requested: str,
    resolved: MemProfile,
    env: CudaEnvInfo | None = None,
) -> None:
    info = env or describe_cuda_environment()
    logger.info(
        "mem-profile requested=%s resolved=%s | GPU=%s VRAM=%.2f GiB | "
        "torch=%s cuda=%s",
        requested,
        resolved,
        info.device_name or "(none)",
        info.total_memory_gib,
        info.torch_version,
        info.torch_cuda_version or "(n/a)",
    )
