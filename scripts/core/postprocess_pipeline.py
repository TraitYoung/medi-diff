"""Postprocessing pipeline: importable functions (no subprocess needed).

These pure numpy/CV2 functions are extracted from scripts/postprocess/postprocess_freq.py.
The CLI script becomes a thin wrapper that calls run_postprocess_on_dir().
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage


# ═══════════════════════════════════════════════════════════════════════════════
# Individual processing functions
# ═══════════════════════════════════════════════════════════════════════════════

def winsorize(gray: np.ndarray, low_pct: float, high_pct: float) -> np.ndarray:
    """Percentile clipping then linear remap to [0,255]."""
    lo = np.percentile(gray, low_pct)
    hi = np.percentile(gray, high_pct)
    if hi <= lo:
        return gray
    clipped = np.clip(gray.astype(np.float32), lo, hi)
    return ((clipped - lo) / (hi - lo) * 255).astype(np.uint8)


def fill_voids(
    gray: np.ndarray,
    min_area: int,
    max_area: int,
    min_circ: float,
) -> np.ndarray:
    """Detect isolated dark round holes in breast foreground, fill with local mean.

    Strategy:
      1. Otsu threshold on foreground interior to find dark regions
      2. Connected-component analysis: filter by area [min_area, max_area] & circularity >= min_circ
      3. Fill each void with dilated neighbourhood mean (gradient blend towards centre)
    """
    result = gray.copy().astype(np.float32)

    _, fg_mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_ERODE, k, iterations=3)

    fg_pixels = gray[fg_mask > 0]
    if len(fg_pixels) < 100:
        return gray
    dark_thresh = int(np.percentile(fg_pixels, 15))
    dark_mask = ((gray < dark_thresh) & (fg_mask > 0)).astype(np.uint8) * 255

    num, labels, stats, _ = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)
    filled = 0
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue
        component = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        perimeter = cv2.arcLength(contours[0], True)
        if perimeter < 1:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < min_circ:
            continue

        dilated = cv2.dilate(component, k, iterations=4)
        ring_mask = (dilated > 0) & (component == 0) & (fg_mask > 0)
        ring_vals = gray[ring_mask]
        if len(ring_vals) < 5:
            continue
        fill_val = float(np.mean(ring_vals))

        dist = ndimage.distance_transform_edt(component)
        max_dist = dist.max()
        if max_dist > 0:
            alpha = np.clip(dist / max_dist, 0, 1)
            void_region = (component > 0)
            result[void_region] = (
                fill_val * alpha[void_region]
                + result[void_region] * (1 - alpha[void_region])
            )
        filled += 1

    if filled:
        print(f"    填充空洞: {filled} 个")
    return np.clip(result, 0, 255).astype(np.uint8)


def radial_power_spectrum(gray_f32: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute 1D radial power spectrum from a 2D FFT."""
    h, w = gray_f32.shape
    fft_shift = np.fft.fftshift(np.fft.fft2(gray_f32))
    power = np.abs(fft_shift) ** 2
    cy, cx = h // 2, w // 2
    ys, xs = np.mgrid[0:h, 0:w]
    r = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2).astype(np.float32)
    max_r = int(min(cx, cy))
    radii = np.arange(1, max_r)
    ps = np.array([power[(r >= rv - 0.5) & (r < rv + 0.5)].mean() for rv in radii])
    return radii.astype(np.float32), ps


def estimate_beta(radii: np.ndarray, ps: np.ndarray) -> float:
    """Estimate power-law slope from log-log radial power spectrum."""
    valid = (radii > 0) & (ps > 0)
    coeffs = np.polyfit(np.log(radii[valid]), np.log(ps[valid]), 1)
    return float(-coeffs[0])


def correct_beta(gray: np.ndarray, beta_tgt: float, blend: float) -> np.ndarray:
    """Frequency-domain beta-slope correction."""
    f64 = gray.astype(np.float64)
    radii, ps = radial_power_spectrum(f64.astype(np.float32))
    beta_src = estimate_beta(radii, ps)
    print(f"    β: {beta_src:.3f} → {beta_tgt:.3f}  (Δ={beta_src - beta_tgt:+.3f})")

    h, w = gray.shape
    cy, cx = h // 2, w // 2
    ys, xs = np.mgrid[0:h, 0:w]
    r = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2).astype(np.float64)
    r[cy, cx] = 1.0
    exponent = (beta_src - beta_tgt) / 2.0
    H = r ** exponent
    H = np.clip(H, None, 3.0)
    H[cy, cx] = 1.0

    fft_shift = np.fft.fftshift(np.fft.fft2(f64))
    corrected = np.real(np.fft.ifft2(np.fft.ifftshift(fft_shift * H)))

    c_min, c_max = corrected.min(), corrected.max()
    if c_max > c_min:
        corrected = (corrected - c_min) / (c_max - c_min) * 255.0

    if blend < 1.0:
        corrected = corrected * blend + f64 * (1.0 - blend)

    return np.clip(corrected, 0, 255).astype(np.uint8)


def apply_clahe(gray: np.ndarray, clip: float, grid: int) -> np.ndarray:
    """CLAHE local contrast enhancement."""
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(gray)


def edge_feather(gray: np.ndarray, ksize: int) -> np.ndarray:
    """Gaussian-smooth the breast contour edge to reduce saw-tooth effect."""
    k = ksize if ksize % 2 == 1 else ksize + 1
    _, fg = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
    ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k * 3, k * 3))
    edge_zone = cv2.dilate(fg, ke) - cv2.erode(fg, ke)
    smoothed = cv2.GaussianBlur(gray, (k, k), 0)
    alpha = (edge_zone.astype(np.float32) / 255.0)
    result = gray.astype(np.float32) * (1 - alpha) + smoothed.astype(np.float32) * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def unsharp_mask(gray: np.ndarray, radius: int, strength: float) -> np.ndarray:
    """Mild unsharp-mask sharpening for tissue texture."""
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=radius)
    sharpened = cv2.addWeighted(gray, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
# Orchestration (importable, no argparse / subprocess)
# ═══════════════════════════════════════════════════════════════════════════════

def run_postprocess_on_image(gray: np.ndarray, config) -> np.ndarray:
    """Apply the full postprocessing pipeline to a single image.

    Args:
        gray: (H, W) uint8 grayscale image
        config: PostprocessParams dataclass (or any object with matching attributes)
    """
    if config.winsorize:
        gray = winsorize(gray, config.winsorize_low, config.winsorize_high)
    if config.fill_voids:
        gray = fill_voids(gray, config.void_min_area, config.void_max_area,
                          config.void_circularity)
    if not config.no_freq:
        gray = correct_beta(gray, config.target_beta, config.blend)
    if config.clahe:
        gray = apply_clahe(gray, config.clahe_clip, config.clahe_grid)
    if getattr(config, "sharpen", True):
        gray = unsharp_mask(gray, getattr(config, "sharpen_radius", 3),
                            getattr(config, "sharpen_strength", 0.25))
    if getattr(config, "bilateral", False):
        gray = cv2.bilateralFilter(gray, getattr(config, "bilateral_d", 5),
                                   getattr(config, "bilateral_sigma_color", 30),
                                   getattr(config, "bilateral_sigma_space", 30))
    if getattr(config, "edge_feather", False):
        gray = edge_feather(gray, getattr(config, "feather_ksize", 5))
    return gray


def run_postprocess_on_dir(
    input_dir: Path,
    output_dir: Path,
    config,
    *,
    ext: str | None = None,
) -> None:
    """Apply the full postprocessing pipeline to every image in a directory.

    Args:
        input_dir: directory with PNG/JPG images
        output_dir: destination directory (created if missing)
        config: PostprocessParams dataclass
        ext: output filename suffix override (default: config.ext or '.png')
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_ext = ext or getattr(config, "ext", ".png")

    exts = {".png", ".jpg", ".jpeg"}
    images = [p for p in sorted(input_dir.iterdir()) if p.suffix.lower() in exts]
    if not images:
        print(f"[postprocess] 未找到图像：{input_dir}")
        return

    steps = []
    if config.winsorize:
        steps.append(f"Winsorize({config.winsorize_low:.1f}%~{config.winsorize_high:.1f}%)")
    if config.fill_voids:
        steps.append("空洞填充")
    if not config.no_freq:
        steps.append(f"β校正→{config.target_beta} blend={config.blend}")
    if config.clahe:
        steps.append(f"CLAHE(clip={config.clahe_clip})")
    if getattr(config, "sharpen", True):
        steps.append(f"Unsharp(strength={getattr(config, 'sharpen_strength', 0.25)})")
    if getattr(config, "bilateral", False):
        steps.append("Bilateral")
    if getattr(config, "edge_feather", False):
        steps.append(f"边缘羽化(k={getattr(config, 'feather_ksize', 5)})")
    print(f"[postprocess] {len(images)} 张  流水线: {' → '.join(steps)}")

    for i, path in enumerate(images):
        print(f"[{i + 1}/{len(images)}] {path.name}")
        try:
            gray = np.asarray(Image.open(path).convert("L"))
            gray = run_postprocess_on_image(gray, config)
            Image.fromarray(gray, mode="L").save(output_dir / (path.stem + out_ext))
        except Exception as e:
            print(f"  [错误] {e}")

    print(f"\n完成 → {output_dir}")
