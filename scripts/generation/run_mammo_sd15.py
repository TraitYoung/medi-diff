#!/usr/bin/env python3
"""Mammography image generation via Stable Diffusion 1.5 + LoRA img2img.

Default mode: full-image single-pass img2img (no patch grid, fast, no seams).
Legacy fallback: patch-overlap img2img with global guide + latent smooth field.

Usage:
    python3 scripts/generation/run_mammo_sd15.py \
        --base-model-local hf_cache/sd15 \
        --lora-path outputs/lora/mammo_sd15_v4_clean/final_lora \
        --metadata-csv datasets/CBIS_CLEAN_V2/metadata_clean.csv \
        --filter-view MLO --filter-density scattered \
        --num-images 6 --seed 2026 \
        --mode full-image \
        --fullimage-long-side 768 \
        --fullimage-output-long-side 2048 \
        --scheduler dpm --num-steps 50
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

# ── Utility functions (imported by scripts/core/ modules) ───────────────────

def resize_long_side_gray(gray: np.ndarray, long_side: int) -> np.ndarray:
    """Resize so long side == long_side, keeping aspect ratio. long_side ≤ 0 means no-op."""
    if long_side <= 0:
        return gray
    h, w = gray.shape
    cur_long = max(h, w)
    if cur_long == long_side:
        return gray
    scale = long_side / cur_long
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    interp = cv2.INTER_LANCZOS4 if scale > 1 else cv2.INTER_AREA
    return cv2.resize(gray, (new_w, new_h), interpolation=interp)


def enhance_input_contrast(gray: np.ndarray, clahe_clip: float = 0.8,
                           clahe_grid: int = 8) -> np.ndarray:
    """Apply CLAHE if median < 40 (dark images)."""
    if np.median(gray) >= 40:
        return gray
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
    return clahe.apply(gray)


def _detect_metal_markers(gray: np.ndarray, min_radius: int = 4,
                          max_radius: int = 30, param1: int = 40,
                          param2: int = 18) -> list[tuple[int, int, int]]:
    """Detect BB marker circles via HoughCircles. Returns [(x, y, r), ...]."""
    blur = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
        param1=param1, param2=param2,
        minRadius=min_radius, maxRadius=max_radius,
    )
    if circles is None:
        return []
    return [(int(c[0]), int(c[1]), int(c[2])) for c in circles[0]]


def _inpaint_circles(gray: np.ndarray,
                     circles: list[tuple[int, int, int]]) -> np.ndarray:
    """Softly suppress detected circle regions without TELEA geometry traces."""
    result = gray.copy()
    for x, y, r in circles:
        pad = max(10, int(r * 4))
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(gray.shape[1], x + pad), min(gray.shape[0], y + pad)
        roi = result[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        yy, xx = np.ogrid[y1:y2, x1:x2]
        core = ((xx - x) ** 2 + (yy - y) ** 2) <= (r + 2) ** 2
        ring = (((xx - x) ** 2 + (yy - y) ** 2) <= (r + pad // 2) ** 2) & ~core
        fill = float(np.median(result[y1:y2, x1:x2][ring])) if np.any(ring) else float(np.median(roi))
        alpha = cv2.GaussianBlur(core.astype(np.float32), (0, 0), sigmaX=max(1.5, r / 2))
        alpha = np.clip(alpha, 0, 1)
        result[y1:y2, x1:x2] = (
            roi.astype(np.float32) * (1 - alpha)
            + fill * alpha
        ).astype(np.uint8)
    return result


def _apply_upscale(gray: np.ndarray, mode: str = "none",
                   factor: float = 2.0) -> np.ndarray:
    """Upscale result image."""
    if mode == "none" or factor <= 1.0:
        return gray
    h, w = gray.shape
    new_h, new_w = int(h * factor), int(w * factor)
    interp = cv2.INTER_LINEAR if mode == "bilinear" else (
        cv2.INTER_LANCZOS4 if mode == "lanczos" else cv2.INTER_CUBIC)
    return cv2.resize(gray, (new_w, new_h), interpolation=interp)


# ── Source pool ─────────────────────────────────────────────────────────────

def _has_source_artifact_burden(
    marker_score: float = 0.0,
    bg_marker_count: int = 0,
    bg_marker_frac: float = 0.0,
    fg_dot_count: int = 0,
    texture_lap_p75: float | None = None,
    texture_grad_p75: float | None = None,
    circumscribed_mass_count: int = 0,
    calc_cluster_count: int = 0,
    calc_dot_count: int = 0,
) -> bool:
    """Return True for source images likely to imprint visible dot artifacts."""
    low_texture = (
        texture_lap_p75 is not None
        and texture_grad_p75 is not None
        and (texture_lap_p75 < 14 or texture_grad_p75 < 14)
    )
    return (
        marker_score >= 8
        or bg_marker_count >= 20
        or bg_marker_frac >= 0.018
        or fg_dot_count >= 10
        or low_texture
        or circumscribed_mass_count > 0
        or calc_cluster_count > 0
        or calc_dot_count >= 6
    )


def _main_tissue_mask(gray: np.ndarray) -> np.ndarray:
    """Build a coarse mask for the main breast body."""
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    otsu_v, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, bin_img = cv2.threshold(gray, max(int(otsu_v * 0.5), 8), 255, cv2.THRESH_BINARY)
    bin_img = cv2.morphologyEx(
        bin_img,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)),
        iterations=2,
    )
    cnts, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros_like(gray, dtype=np.uint8)
    if cnts:
        cv2.drawContours(mask, [max(cnts, key=cv2.contourArea)], -1, 255, -1)
    return mask


def _suspicious_lesion_stats(gray: np.ndarray, tissue_mask: np.ndarray) -> dict:
    """Detect mass-like blobs and clustered calcification-like bright dots."""
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    tissue = tissue_mask > 0
    tissue_vals = gray[tissue]
    if len(tissue_vals) < 500:
        return {
            "circumscribed_mass_count": 0,
            "calc_cluster_count": 0,
            "calc_dot_count": 0,
        }

    h, w = gray.shape[:2]
    tissue_area = max(int(np.sum(tissue)), 1)
    otsu_thr, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    mass_thr = max(float(np.percentile(tissue_vals, 88.0)), float(otsu_thr) * 1.08)
    mass_bin = ((gray >= mass_thr) & tissue).astype(np.uint8) * 255
    mass_bin = cv2.morphologyEx(
        mass_bin,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    n_mass, labels_mass, stats_mass, _ = cv2.connectedComponentsWithStats(mass_bin, connectivity=8)
    circumscribed_mass_count = 0
    min_mass_area = max(120, int(tissue_area * 0.0015))
    max_mass_area = max(min_mass_area + 1, int(tissue_area * 0.06))
    for j in range(1, n_mass):
        area = int(stats_mass[j, cv2.CC_STAT_AREA])
        if area < min_mass_area or area > max_mass_area:
            continue
        bw = int(stats_mass[j, cv2.CC_STAT_WIDTH])
        bh = int(stats_mass[j, cv2.CC_STAT_HEIGHT])
        if bw < 10 or bh < 10:
            continue
        aspect = bw / max(float(bh), 1.0)
        if aspect < 0.45 or aspect > 2.2:
            continue
        comp = (labels_mass == j).astype(np.uint8) * 255
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        contour_area = float(cv2.contourArea(contour))
        peri = float(cv2.arcLength(contour, True))
        if contour_area <= 0 or peri <= 0:
            continue
        circularity = 4 * np.pi * contour_area / max(peri ** 2, 1.0)
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        solidity = contour_area / max(hull_area, 1.0)
        if circularity < 0.42 or solidity < 0.82:
            continue
        x0 = int(stats_mass[j, cv2.CC_STAT_LEFT])
        y0 = int(stats_mass[j, cv2.CC_STAT_TOP])
        pad = max(8, min(32, max(bw, bh) // 2))
        x1, y1 = max(0, x0 - pad), max(0, y0 - pad)
        x2, y2 = min(w, x0 + bw + pad), min(h, y0 + bh + pad)
        roi = gray[y1:y2, x1:x2]
        comp_roi = labels_mass[y1:y2, x1:x2] == j
        tissue_roi = tissue[y1:y2, x1:x2]
        ring = tissue_roi & ~comp_roi
        if np.sum(ring) < 20:
            continue
        contrast = float(np.median(gray[labels_mass == j])) - float(np.median(roi[ring]))
        if contrast < 10:
            continue
        circumscribed_mass_count += 1

    calc_thr = max(float(np.percentile(tissue_vals, 99.75)), float(otsu_thr) * 1.25)
    calc_bin = ((gray >= calc_thr) & tissue).astype(np.uint8) * 255
    n_calc, labels_calc, stats_calc, cent = cv2.connectedComponentsWithStats(calc_bin, connectivity=8)
    centers = []
    for j in range(1, n_calc):
        area = int(stats_calc[j, cv2.CC_STAT_AREA])
        bw = int(stats_calc[j, cv2.CC_STAT_WIDTH])
        bh = int(stats_calc[j, cv2.CC_STAT_HEIGHT])
        if 2 <= area <= 80 and max(bw, bh) <= 14:
            centers.append((float(cent[j][0]), float(cent[j][1])))

    calc_cluster_count = 0
    cluster_radius = max(28.0, min(h, w) * 0.055)
    for cx, cy in centers:
        nearby = 0
        for ox, oy in centers:
            if (cx - ox) ** 2 + (cy - oy) ** 2 <= cluster_radius ** 2:
                nearby += 1
        if nearby >= 4:
            calc_cluster_count += 1
            break

    return {
        "circumscribed_mass_count": circumscribed_mass_count,
        "calc_cluster_count": calc_cluster_count,
        "calc_dot_count": len(centers),
    }


def filter_source_pool(metadata_csv: Path, filter_view: str = "",
                       filter_density: str = "", num_images: int = 8,
                       source_seed: int | None = None,
                       source_quality_sort: bool = False,
                       max_aspect_ratio: float = 2.2) -> list[dict]:
    """Filter metadata CSV, return list of source entries with image_path etc."""
    import pandas as pd

    df = pd.read_csv(metadata_csv)
    # Support both 'image_path' (legacy) and 'src' (CBIS_CLEAN_V2) column names
    if "image_path" not in df.columns and "src" in df.columns:
        df = df.rename(columns={"src": "image_path"})
    needed = ["image_path", "view", "density"]
    for col in needed:
        if col not in df.columns:
            raise KeyError(f"Metadata CSV missing column: {col}")

    if filter_view:
        df = df[df["view"].str.upper() == filter_view.upper()]
    if filter_density:
        density_map = {"fatty": 1, "scattered": 2,
                       "heterogeneous": 3, "dense": 4}
        target = density_map.get(filter_density.lower())
        if target is not None:
            df = df[df["density"].apply(
                lambda x: density_map.get(str(x).lower(), -1)) == target]

    # Filter extreme aspect ratios
    if "width" in df.columns and "height" in df.columns:
        df = df[df.apply(
            lambda r: (max(r["width"], r["height"]) /
                       max(min(r["width"], r["height"]), 1)) <= max_aspect_ratio,
            axis=1)]

    entries = list(df.to_dict("records"))
    rng = np.random.RandomState(
        source_seed if source_seed is not None
        else int(time.time() * 1000) % 2**31)
    rng.shuffle(entries)

    # Do not rank by file size. Large JPEGs often include wide black canvas,
    # burned-in labels, or marker-heavy views; rank by image-derived geometry
    # after the actual file is loaded instead.

    # Shape quality pre-filter: reject sources with badly fractured or non-oval contours
    good_entries = []
    artifact_entries = []
    rejected = 0
    artifact_rejected = 0
    for e in entries:
        path = e.get("image_path", "")
        if not path or not os.path.exists(path):
            good_entries.append(e)
            if not source_quality_sort and len(good_entries) >= num_images:
                break
            continue
        try:
            src = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if src is None:
                good_entries.append(e)
                continue
            src = resize_long_side_gray(src, 1024)
            h, w = src.shape[:2]
            actual_aspect = max(h, w) / max(min(h, w), 1)
            if actual_aspect > max_aspect_ratio:
                rejected += 1
                continue
            otsu_v, _ = cv2.threshold(src, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            _, bin_img = cv2.threshold(src, max(int(otsu_v * 0.5), 8),
                                       255, cv2.THRESH_BINARY)
            k_cl = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
            bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, k_cl, iterations=2)
            cnts, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
            if not cnts:
                good_entries.append(e)
                continue
            c = max(cnts, key=cv2.contourArea)
            area = float(cv2.contourArea(c))
            if area < 100:
                good_entries.append(e)
                continue
            area_frac = area / max(float(h * w), 1.0)
            if area_frac < 0.08 or area_frac > 0.85:
                rejected += 1
                continue
            peri = float(cv2.arcLength(c, True))
            circ = 4 * np.pi * area / max(peri ** 2, 1.0)
            hull = cv2.convexHull(c)
            hull_area = float(cv2.contourArea(hull))
            convex_defect = (hull_area - area) / max(hull_area, 1.0)
            # Reject: too non-oval (circ<0.30) or too concave (>45% convex hull area missing)
            if circ < 0.30 or convex_defect > 0.45:
                rejected += 1
                continue
            tissue_mask = np.zeros_like(src, dtype=np.uint8)
            cv2.drawContours(tissue_mask, [c], -1, 255, -1)
            tissue_protect = cv2.dilate(
                tissue_mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
                iterations=1,
            )
            bg_vals = src[tissue_protect == 0]
            bright_thr = (
                max(float(np.percentile(bg_vals, 96.0)), 45.0)
                if len(bg_vals) > 500 else max(float(np.percentile(src, 99.2)), 95.0)
            )
            bg_bright = ((src >= bright_thr) & (tissue_protect == 0)).astype(np.uint8) * 255
            n_bg, _, stats_bg, _ = cv2.connectedComponentsWithStats(bg_bright, connectivity=8)
            bg_marker_area = 0
            bg_marker_count = 0
            bg_max_area = int(max(8000, h * w * 0.06))
            for j in range(1, n_bg):
                a = int(stats_bg[j, cv2.CC_STAT_AREA])
                bw = int(stats_bg[j, cv2.CC_STAT_WIDTH])
                bh = int(stats_bg[j, cv2.CC_STAT_HEIGHT])
                if 8 <= a <= bg_max_area and bw >= 3 and bh >= 3:
                    bg_marker_area += a
                    bg_marker_count += 1
            bg_marker_frac = bg_marker_area / max(float(h * w), 1.0)
            bg_marker_penalty = bg_marker_count * 0.18 + bg_marker_frac * 60.0

            tissue_vals = src[tissue_mask > 0]
            fg_dot_count = 0
            texture_lap_p75 = None
            texture_grad_p75 = None
            if len(tissue_vals) > 500:
                lap = cv2.Laplacian(src, cv2.CV_32F, ksize=3)
                texture_lap_p75 = float(np.percentile(np.abs(lap)[tissue_mask > 0], 75))
                sobx = cv2.Sobel(src, cv2.CV_32F, 1, 0, ksize=3)
                soby = cv2.Sobel(src, cv2.CV_32F, 0, 1, ksize=3)
                grad = np.sqrt(sobx * sobx + soby * soby)
                texture_grad_p75 = float(np.percentile(grad[tissue_mask > 0], 75))
                fg_thr = max(float(np.percentile(tissue_vals, 99.85)), bright_thr)
                fg_bright = ((src >= fg_thr) & (tissue_mask > 0)).astype(np.uint8) * 255
                n_fg, _, stats_fg, _ = cv2.connectedComponentsWithStats(fg_bright, connectivity=8)
                for j in range(1, n_fg):
                    a = int(stats_fg[j, cv2.CC_STAT_AREA])
                    bw = int(stats_fg[j, cv2.CC_STAT_WIDTH])
                    bh = int(stats_fg[j, cv2.CC_STAT_HEIGHT])
                    if 3 <= a <= 90 and max(bw, bh) <= 14:
                        fg_dot_count += 1
            lesion_stats = _suspicious_lesion_stats(src, tissue_mask)
            fg_dot_penalty = max(0, fg_dot_count - 3) * 0.06
            e = dict(e)
            try:
                marker_score = float(e.get("marker_score", 0) or 0)
            except Exception:
                marker_score = 0.0
            e["_source_quality_score"] = (
                area_frac * 1.2
                + circ * 0.35
                - convex_defect * 0.45
                - abs(actual_aspect - 1.55) * 0.08
                - bg_marker_penalty
                - fg_dot_penalty
            )
            e["_source_artifact_stats"] = {
                "marker_score": marker_score,
                "bg_marker_count": bg_marker_count,
                "bg_marker_frac": bg_marker_frac,
                "fg_dot_count": fg_dot_count,
                "texture_lap_p75": texture_lap_p75,
                "texture_grad_p75": texture_grad_p75,
                **lesion_stats,
            }
            if _has_source_artifact_burden(
                marker_score=marker_score,
                bg_marker_count=bg_marker_count,
                bg_marker_frac=bg_marker_frac,
                fg_dot_count=fg_dot_count,
                texture_lap_p75=texture_lap_p75,
                texture_grad_p75=texture_grad_p75,
                circumscribed_mass_count=int(lesion_stats["circumscribed_mass_count"]),
                calc_cluster_count=int(lesion_stats["calc_cluster_count"]),
                calc_dot_count=int(lesion_stats["calc_dot_count"]),
            ):
                artifact_rejected += 1
                artifact_entries.append(e)
                continue
            good_entries.append(e)
            if not source_quality_sort and len(good_entries) >= num_images:
                break
        except Exception:
            good_entries.append(e)
            if not source_quality_sort and len(good_entries) >= num_images:
                break

    if rejected > 0:
        logger.debug("Shape filter: rejected %d/%d sources (bad aspect/area/shape)",
                     rejected, len(entries))
    if artifact_rejected > 0:
        logger.info("Artifact filter: skipped %d/%d source candidates",
                    artifact_rejected, len(entries))

    if source_quality_sort:
        good_entries.sort(
            key=lambda e: float(e.get("_source_quality_score", 0.0)),
            reverse=True,
        )

    if len(good_entries) < num_images and artifact_entries:
        logger.warning(
            "Artifact filter left only %d clean sources; using %d fallback sources",
            len(good_entries), min(len(artifact_entries), num_images - len(good_entries)),
        )
        good_entries.extend(artifact_entries[: num_images - len(good_entries)])

    return good_entries[:num_images]


def load_source_gray(image_path: str | Path) -> np.ndarray:
    """Load source mammogram as uint8 grayscale."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read source image: {image_path}")
    return img


# ── Prompt alignment ────────────────────────────────────────────────────────

def _build_prompt(filter_view: str, filter_density: str,
                  base_prompt: str) -> str:
    """Rewrite prompt to match LoRA caption phrasing for a single density tier."""
    density_tiers = {"fatty", "scattered", "heterogeneous", "dense"}
    if filter_density.lower() not in density_tiers:
        return base_prompt
    view_map = {"MLO": "MLO view", "CC": "CC view"}
    view_phrase = view_map.get(filter_view.upper(), "")
    density_phrase = f"{filter_density} density"
    anatomy_phrase = ""
    if filter_view.upper() == "MLO":
        anatomy_phrase = (
            "triangular pectoralis muscle, clear inframammary fold, "
            "directional fibroglandular strands toward nipple"
        )
    parts = [p for p in [view_phrase, density_phrase, anatomy_phrase] if p]
    if parts:
        return (
            f"medical grayscale mammogram, {' '.join(parts)}, "
            "fine ligament texture, FFDM, no text, no labels"
        )
    return base_prompt


# ── Core generation functions (imported by scripts/core/ modules) ───────────

def fullimage_generate(
    src_gray: np.ndarray,
    pipe,
    prompt: str,
    negative_prompt: str,
    strength: float = 0.42,
    guidance_scale: float = 7.9,
    num_inference_steps: int = 50,
    generator: torch.Generator | None = None,
    fullimage_long_side: int = 768,
    fullimage_min_short_side: int = 384,
    fullimage_output_long_side: int = 2048,
) -> np.ndarray:
    """Single-pass full-image img2img.

    Scales source so long side ≤ fullimage_long_side, short side ≥
    fullimage_min_short_side when feasible, hard cap 1024 px.  Output
    resized so saved long side ≤ fullimage_output_long_side (0 = native).
    """
    h, w = src_gray.shape
    long_edge = max(h, w)
    short_edge = min(h, w)
    target_long = min(fullimage_long_side, 1024)

    scale = target_long / long_edge if long_edge > target_long else 1.0
    new_long = int(round(long_edge * scale))
    new_short = int(round(short_edge * scale))

    if new_short < fullimage_min_short_side:
        alt_scale = fullimage_min_short_side / short_edge
        alt_long = int(round(long_edge * alt_scale))
        if alt_long <= target_long:
            new_long, new_short = alt_long, fullimage_min_short_side

    new_long, new_short = (new_long // 8) * 8, (new_short // 8) * 8
    target_h, target_w = (new_long, new_short) if h >= w else (new_short, new_long)

    src_rgb = cv2.cvtColor(src_gray, cv2.COLOR_GRAY2RGB)
    src_pil = Image.fromarray(src_rgb).resize((target_w, target_h), Image.LANCZOS)

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=src_pil,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
    ).images[0]

    result_gray = np.array(result.convert("L"))

    if fullimage_output_long_side > 0:
        result_gray = resize_long_side_gray(result_gray, fullimage_output_long_side)

    return result_gray


# ── Post-generation helpers ─────────────────────────────────────────────────

def _final_metal_sweep(gray: np.ndarray, max_retries: int = 3) -> np.ndarray:
    """Detect and inpaint metal BB markers on final image."""
    for _ in range(max_retries):
        circles = _detect_metal_markers(gray)
        if not circles:
            break
        gray = _inpaint_circles(gray, circles)
    return gray


def _inpaint_background_bright_markers(gray: np.ndarray) -> np.ndarray:
    """Darken bright letters/markers in background without inpainting artefacts."""
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    if h * w < 1000:
        return gray

    otsu_v, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, fg = cv2.threshold(gray, max(int(otsu_v * 0.45), 8), 255, cv2.THRESH_BINARY)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k_close, iterations=2)
    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return gray

    tissue = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(tissue, [max(cnts, key=cv2.contourArea)], -1, 255, -1)
    protect_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    tissue = cv2.dilate(tissue, protect_k, iterations=1)

    bg_vals = gray[tissue == 0]
    bright_thr = (
        max(float(np.percentile(bg_vals, 94.0)), 32.0)
        if len(bg_vals) > 500 else max(float(np.percentile(gray, 97.0)), 65.0)
    )
    candidates = ((gray >= bright_thr) & (tissue == 0)).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(candidates, connectivity=8)
    if n <= 1:
        return gray

    result = gray.copy()
    max_area = int(max(1200, h * w * 0.06))
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 6 or area > max_area:
            continue
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        if bw < 2 or bh < 2:
            continue
        if max(bw, bh) > max(h, w) * 0.20:
            continue

        x0 = int(stats[i, cv2.CC_STAT_LEFT])
        y0 = int(stats[i, cv2.CC_STAT_TOP])
        pad = max(10, min(30, max(bw, bh)))
        x1, y1 = max(0, x0 - pad), max(0, y0 - pad)
        x2, y2 = min(w, x0 + bw + pad), min(h, y0 + bh + pad)
        roi = result[y1:y2, x1:x2]
        component = labels[y1:y2, x1:x2] == i
        component = cv2.dilate(
            component.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        ).astype(bool)
        context = roi[(~component) & (tissue[y1:y2, x1:x2] == 0)]
        fill = float(np.percentile(context, 10)) if len(context) else float(np.percentile(roi, 10))
        alpha = cv2.GaussianBlur(component.astype(np.float32), (0, 0), sigmaX=2.5)
        alpha = np.clip(alpha * 1.15, 0, 1.0)
        result[y1:y2, x1:x2] = (
            roi.astype(np.float32) * (1 - alpha)
            + fill * alpha
        ).astype(np.uint8)

    return result


def _soften_isolated_bright_spots(gray: np.ndarray) -> np.ndarray:
    """Reduce tiny isolated bright dots while avoiding polygonal inpaint masks."""
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    result = gray.copy()
    h, w = gray.shape[:2]
    otsu_v, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, fg = cv2.threshold(gray, max(int(otsu_v * 0.55), 12), 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=1)
    fg = cv2.erode(fg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1)
    vals = gray[fg > 0]
    if len(vals) < 500:
        return result

    hi_thr = max(float(np.percentile(vals, 99.75)), float(otsu_v) * 1.15)
    bright = ((gray >= hi_thr) & (fg > 0)).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 3 or area > 120:
            continue
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        if max(bw, bh) > 18:
            continue
        x0 = int(stats[i, cv2.CC_STAT_LEFT])
        y0 = int(stats[i, cv2.CC_STAT_TOP])
        pad = 12
        x1, y1 = max(0, x0 - pad), max(0, y0 - pad)
        x2, y2 = min(w, x0 + bw + pad), min(h, y0 + bh + pad)
        roi = result[y1:y2, x1:x2]
        component = labels[y1:y2, x1:x2] == i
        context = roi[(~component) & (fg[y1:y2, x1:x2] > 0)]
        if len(context) < 12:
            continue
        fill = float(np.percentile(context, 65))
        alpha = cv2.GaussianBlur(component.astype(np.float32), (0, 0), sigmaX=1.6)
        alpha = np.clip(alpha * 0.93, 0, 0.93)
        result[y1:y2, x1:x2] = (
            roi.astype(np.float32) * (1 - alpha)
            + fill * alpha
        ).astype(np.uint8)
    return result


# ── Test pattern ────────────────────────────────────────────────────────────

def _make_test_pattern(h: int = 768, w: int = 1024) -> np.ndarray:
    """Synthetic breast-shaped test pattern for debugging."""
    gray = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2 - 30)
    axes = (w // 3, h * 2 // 5)
    cv2.ellipse(gray, center, axes, 0, 0, 360, 200, -1)
    noise = np.random.RandomState(42).randint(0, 30, (h, w), dtype=np.uint8)
    mask = gray > 0
    gray[mask] = np.clip(gray[mask].astype(int) + noise[mask].astype(int) - 15,
                         0, 255).astype(np.uint8)
    return gray


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    from scripts.core.pipeline_config import GenParams
    from scripts.core.label_guard import (
        erase_background_labels, erase_bright_border_labels,
        feather_canvas_edge)
    p = argparse.ArgumentParser(
        description="SD1.5+LoRA mammography image generation",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Model
    g = p.add_argument_group("Model")
    g.add_argument("--base-model-local", type=str, default=None)
    g.add_argument("--base-model", type=str, default="runwayml/stable-diffusion-v1-5")
    g.add_argument("--lora-path", type=str,
                   default="outputs/lora/mammo_sd15_v6_allMLO/final_lora")
    g.add_argument("--scheduler", type=str, default="dpm",
                   choices=["dpm", "pndm", "ddim"])

    # Mode
    g = p.add_argument_group("Generation Mode")
    g.add_argument("--mode", type=str, default="full-image",
                   choices=["full-image"])
    g.add_argument("--fullimage-long-side", type=int, default=768)
    g.add_argument("--fullimage-min-short-side", type=int, default=384)
    g.add_argument("--fullimage-output-long-side", type=int, default=2048)

    # Sampling
    g = p.add_argument_group("Sampling")
    g.add_argument("--num-steps", type=int, default=50)
    g.add_argument("--strength", type=float, default=GenParams.strength)
    g.add_argument("--guidance-scale", type=float, default=GenParams.guidance_scale)
    g.add_argument("--negative-prompt", type=str, default=None,
                   help="覆盖 GenParams.negative_prompt")

    # Source
    g = p.add_argument_group("Source Selection")
    g.add_argument("--metadata-csv", type=str,
                   default="datasets/CBIS_CLEAN_V2/metadata_clean.csv")
    g.add_argument("--filter-view", type=str, default="")
    g.add_argument("--filter-density", type=str, default="")
    g.add_argument("--num-images", type=int, default=6)
    g.add_argument("--seed", type=int, default=2026)
    g.add_argument("--source-seed", type=int, default=None)
    g.add_argument("--source-quality-sort", action="store_true", default=False)
    g.add_argument("--no-source-quality-sort", action="store_false",
                   dest="source_quality_sort")
    g.add_argument("--max-source-aspect-ratio", type=float, default=2.2)

    # Output
    g = p.add_argument_group("Output")
    g.add_argument("--output-base", type=str, default="outputs/generated")
    g.add_argument("--output-subdir-prefix", type=str, default="sd15")

    # Postprocess archived — see archive/postprocess/postprocess_freq.py

    # Label guard
    g = p.add_argument_group("Label Guard")
    g.add_argument("--legacy-label-guard", action="store_true", default=True)
    g.add_argument("--no-legacy-label-guard", action="store_false",
                   dest="legacy_label_guard")
    g.add_argument("--preclean-border-labels", action="store_true", default=False)

    # VL
    g.add_argument("--no-qwen-vl", action="store_true", default=False)

    args = p.parse_args()

    # Resolve paths
    def _resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else ROOT / path

    base_model_local = _resolve(args.base_model_local) if args.base_model_local else None
    lora_path = _resolve(args.lora_path)
    metadata_csv = _resolve(args.metadata_csv)
    output_base = _resolve(args.output_base)

    t_pipeline_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s  Mode: %s", device, args.mode)

    # Load model
    from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler, DDIMScheduler

    t0 = time.time()
    logger.info("Loading SD1.5 img2img pipeline...")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        str(base_model_local) if base_model_local else args.base_model,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        safety_checker=None,
    )
    pipe.to(device)

    if args.scheduler == "dpm":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    elif args.scheduler == "ddim":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    if lora_path.exists():
        logger.info("Loading LoRA: %s", lora_path)
        if (lora_path / "adapter_model.safetensors").is_file():
            from peft import PeftModel

            pipe.unet = PeftModel.from_pretrained(pipe.unet, str(lora_path))
            logger.info("Loaded PEFT LoRA adapter into UNet")
        else:
            pipe.load_lora_weights(str(lora_path))
    else:
        logger.warning("LoRA path not found: %s", lora_path)

    logger.info("Model+LoRA loaded in %.1fs", time.time() - t0)

    # Source pool
    t0 = time.time()
    if metadata_csv.is_file():
        source_pool_size = max(args.num_images, args.num_images * 4)
        sources = filter_source_pool(
            metadata_csv=metadata_csv, filter_view=args.filter_view,
            filter_density=args.filter_density, num_images=source_pool_size,
            source_seed=args.source_seed,
            source_quality_sort=args.source_quality_sort,
            max_aspect_ratio=args.max_source_aspect_ratio)
        logger.info("Selected %d candidate sources for %d requested images (%.1fs)",
                    len(sources), args.num_images, time.time() - t0)
    else:
        logger.warning("Metadata not found: %s. Using test patterns.", metadata_csv)
        sources = [{"image_path": ""} for _ in range(args.num_images)]

    # Output dir
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = output_base / f"{args.output_subdir_prefix}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prompts
    prompt = _build_prompt(args.filter_view, args.filter_density, GenParams.prompt)
    neg_prompt = args.negative_prompt if args.negative_prompt else GenParams.negative_prompt

    # Save params
    run_params = {
        "base_model_local": str(base_model_local) if base_model_local else None,
        "lora_path": str(lora_path),
        "seed": args.seed,
        "source_seed": args.source_seed,
        "source_quality_sort": args.source_quality_sort,
        "num_images": args.num_images,
        "mode": args.mode,
        "strength": args.strength,
        "guidance_scale": args.guidance_scale,
        "num_steps": args.num_steps,
        "scheduler": args.scheduler,
        "legacy_label_guard": args.legacy_label_guard,
        "canvas_edge_feather": 3,
        "metadata_csv": str(metadata_csv),
        "filter_view": args.filter_view,
        "filter_density": args.filter_density,
        "fullimage_long_side": args.fullimage_long_side,
        "fullimage_output_long_side": args.fullimage_output_long_side,
        "fullimage_min_short_side": args.fullimage_min_short_side,
        "output_dir": str(out_dir),
    }
    with open(out_dir / "run_params.json", "w", encoding="utf-8") as f:
        json.dump(run_params, f, indent=2, ensure_ascii=False)

    source_map: dict[str, str] = {}

    logger.info("Generating %d images from %d candidates...", args.num_images, len(sources))
    saved_count = 0
    lesion_skip_count = 0
    for idx, src_entry in enumerate(tqdm(sources)):
        if saved_count >= args.num_images:
            break
        src_path = src_entry.get("image_path", "")
        if src_path and os.path.exists(src_path):
            src_gray = load_source_gray(src_path)
        else:
            src_gray = _make_test_pattern()

        # Pre-filter
        if args.legacy_label_guard and args.preclean_border_labels:
            src_gray = erase_bright_border_labels(src_gray)
        if args.legacy_label_guard:
            src_gray = _inpaint_background_bright_markers(src_gray)

        src_gray = enhance_input_contrast(src_gray)

        base_seed = args.seed + idx * 17
        t0 = time.time()

        gen = torch.Generator(device=device).manual_seed(base_seed)
        result = fullimage_generate(
            src_gray=src_gray, pipe=pipe, prompt=prompt,
            negative_prompt=neg_prompt, strength=args.strength,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_steps, generator=gen,
            fullimage_long_side=args.fullimage_long_side,
            fullimage_min_short_side=args.fullimage_min_short_side,
            fullimage_output_long_side=0)  # upscale deferred to after post-processing

        logger.info("  [%d/%d] %.1fs", idx + 1, len(sources),
                     time.time() - t0)

        # Post-filter
        if args.legacy_label_guard:
            result = erase_background_labels(result)
            result = erase_bright_border_labels(result)
            result = feather_canvas_edge(result, feather_px=3)

        result = _soften_isolated_bright_spots(result)

        result = _final_metal_sweep(result)
        result = _inpaint_background_bright_markers(result)

        # Postprocess hook removed (2026-05-18) — see archive/postprocess/
        # Recover by importing PostprocessParams + run_postprocess_on_image

        # Lesion check runs at native SD resolution (calibrated thresholds)
        lesion_stats = _suspicious_lesion_stats(result, _main_tissue_mask(result))
        if (
            int(lesion_stats["circumscribed_mass_count"]) > 0
            or int(lesion_stats["calc_cluster_count"]) > 0
            or int(lesion_stats["calc_dot_count"]) >= 4
        ):
            lesion_skip_count += 1
            logger.warning(
                "Skip candidate %d due to suspicious lesion-like pattern: %s",
                idx, lesion_stats,
            )
            continue

        # Upscale after cleanup and lesion check so all processing runs at native SD resolution
        if args.fullimage_output_long_side > 0:
            result = resize_long_side_gray(result, args.fullimage_output_long_side)

        out_name = f"sd15_{saved_count:04d}.png"
        Image.fromarray(result).save(out_dir / out_name)
        source_map[out_name] = src_path if src_path else f"pattern_{idx}"
        saved_count += 1

    if saved_count < args.num_images:
        logger.warning(
            "Generated only %d/%d images after lesion filtering; skipped %d candidates",
            saved_count, args.num_images, lesion_skip_count,
        )

    with open(out_dir / "source_map.json", "w", encoding="utf-8") as f:
        json.dump(source_map, f, indent=2, ensure_ascii=False)

    elapsed_total = time.time() - t_pipeline_start
    logger.info("Done. %d images → %s | total %.1fs (%.1fs/img)",
                len(source_map), out_dir, elapsed_total,
                elapsed_total / max(len(source_map), 1))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    main()
