#!/usr/bin/env python3
"""Smoke-check local CUDA / Blackwell readiness for MammoGen.

Exit codes:
  0 — OK (CPU-only host also exits 0 with a note; no device to validate)
  1 — CUDA device present but matmul / kernel execution failed
  2 — unexpected import / runtime error outside the matmul probe
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.core.device_profile import (  # noqa: E402
    CU128_INSTALL_HINT,
    describe_cuda_environment,
    needs_cu128_for_device,
    resolve_mem_profile,
    warn_if_blackwell_needs_cu128,
)


def main() -> int:
    env = describe_cuda_environment()
    print(f"torch={env.torch_version}")
    print(f"torch.version.cuda={env.torch_cuda_version or '(n/a)'}")
    if not env.available:
        print("CUDA: not available (CPU-only). Generation will be extremely slow.")
        print("mem-profile(auto) →", resolve_mem_profile("auto", total_memory_bytes=None))
        return 0

    print(f"GPU: {env.device_name}")
    print(f"VRAM: {env.total_memory_gib} GiB ({env.total_memory_bytes} bytes)")
    if env.capability:
        print(f"capability: sm_{env.capability[0]}{env.capability[1]}")
    profile = resolve_mem_profile("auto", total_memory_bytes=env.total_memory_bytes)
    print(f"mem-profile(auto) → {profile}")
    warn_if_blackwell_needs_cu128(env)
    if needs_cu128_for_device(env):
        print(f"WARNING: Blackwell/50-series likely needs CUDA 12.8+: {CU128_INSTALL_HINT}")

    try:
        import torch

        a = torch.randn(64, 64, device="cuda", dtype=torch.float16)
        b = torch.randn(64, 64, device="cuda", dtype=torch.float16)
        _ = (a @ b).sum().item()
        torch.cuda.synchronize()
        print("matmul probe: OK")
        return 0
    except Exception as exc:  # noqa: BLE001 — CLI probe
        print(f"matmul probe: FAILED — {exc}", file=sys.stderr)
        print(
            "If this is an RTX 50-series / Blackwell GPU, reinstall PyTorch with CUDA 12.8+:\n"
            "  pip install torch torchvision --index-url "
            "https://download.pytorch.org/whl/cu128",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"check_local_gpu error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
