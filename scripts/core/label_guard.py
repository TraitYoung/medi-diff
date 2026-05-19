"""Pluggable label/artifact guard for mammography generation.

Stateless post-generation filtering: erase DICOM burn-in text from background,
fade bright border labels, optionally clean entire background, feather canvas edges.
Each function takes and returns a (H,W) uint8 grayscale numpy array.

Removed from mainline (2026-05-19): strip_crop, inpaint_interior_bright_spots,
clean_background. These were never called by the active generation path.
"""

from __future__ import annotations

import cv2
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# Individual filter functions
# ═══════════════════════════════════════════════════════════════════════════════

def erase_background_labels(gray: np.ndarray) -> np.ndarray:
    """Erase DICOM burn-in labels (L/MLO/RMLO/device text) from background regions.

    Core heuristic: each bright small CC whose local neighbourhood is >60% dark
    background is classified as a label and removed. Tissue regions are protected
    by Otsu*0.95 threshold + 31px elliptical dilation.
    """
    h, w = gray.shape[:2]
    total_px = h * w
    result = gray.copy()

    # Bright layer CCs (label candidate pool)
    bright_thr = float(np.percentile(gray, 93))
    if bright_thr < 80:
        return result

    _, bin_bright = cv2.threshold(gray, int(bright_thr), 255, cv2.THRESH_BINARY)
    small_k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    bin_filled = cv2.dilate(bin_bright, small_k, iterations=1)

    n, lbl, stats, _ = cv2.connectedComponentsWithStats(bin_filled, connectivity=8)
    if n <= 1:
        return result

    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
    if len(areas) == 0:
        return result

    order = np.argsort(-areas)
    max_area = float(areas[order[0]])
    label_max_area = min(max_area * 0.05, total_px * 0.010)

    # Otsu boundary for dark background / tissue protection
    otsu_thr, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark_bg_thr = max(float(otsu_thr) * 0.7, 30.0)

    tissue_thr_high = int(float(otsu_thr) * 0.95)
    _, tissue_high = cv2.threshold(gray, tissue_thr_high, 255, cv2.THRESH_BINARY)
    n_tp, lbl_tp, stats_tp, _ = cv2.connectedComponentsWithStats(tissue_high, connectivity=8)
    if n_tp > 1:
        areas_tp = stats_tp[1:, cv2.CC_STAT_AREA]
        order_tp = np.argsort(-areas_tp)
        max_tp = float(areas_tp[order_tp[0]])
        tissue_mask_high = np.zeros((h, w), dtype=np.uint8)
        for rank in range(min(10, len(order_tp))):
            if float(areas_tp[order_tp[rank]]) < max_tp * 0.04:
                break
            tissue_mask_high[lbl_tp == int(order_tp[rank]) + 1] = 255
        protect_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        tissue_protected = cv2.dilate(tissue_mask_high, protect_k, iterations=2)
    else:
        tissue_protected = np.zeros((h, w), dtype=np.uint8)

    # Candidate CC filtering: neighbourhood dark-pixel ratio
    band = 30
    pad = 8
    label_regions = np.zeros((h, w), dtype=np.uint8)

    for i in range(len(areas)):
        area = float(areas[i])
        if area <= 0 or area > label_max_area:
            continue
        cc_label = i + 1
        cc_h = int(stats[cc_label, cv2.CC_STAT_HEIGHT])
        cc_w = int(stats[cc_label, cv2.CC_STAT_WIDTH])
        if cc_h < 10 or cc_w < 8:
            continue
        if max(cc_h, cc_w) / max(1, min(cc_h, cc_w)) > 7:
            continue

        x0 = int(stats[cc_label, cv2.CC_STAT_LEFT])
        y0 = int(stats[cc_label, cv2.CC_STAT_TOP])
        bw2 = int(stats[cc_label, cv2.CC_STAT_WIDTH])
        bh2 = int(stats[cc_label, cv2.CC_STAT_HEIGHT])

        nx1, ny1 = max(0, x0 - band), max(0, y0 - band)
        nx2, ny2 = min(w, x0 + bw2 + band), min(h, y0 + bh2 + band)
        nb_gray = gray[ny1:ny2, nx1:nx2]
        nb_lbl = lbl[ny1:ny2, nx1:nx2]
        nb_outside = nb_gray[nb_lbl != cc_label].flatten()
        if len(nb_outside) == 0:
            continue
        dark_frac = float(np.sum(nb_outside < dark_bg_thr)) / len(nb_outside)
        if dark_frac < 0.60:
            continue

        x1, y1p = max(0, x0 - pad), max(0, y0 - pad)
        x2, y2p = min(w, x0 + bw2 + pad), min(h, y0 + bh2 + pad)
        label_regions[y1p:y2p, x1:x2] = 255

    if np.any(label_regions):
        label_regions[tissue_protected > 0] = 0
        result[label_regions > 0] = (result[label_regions > 0].astype(np.float32) * 0.35).astype(np.uint8)

    return result


def erase_bright_border_labels(
    gray: np.ndarray,
    border_frac: float = 0.028,
    bright_pct: float = 99.0,
    *,
    mode: str = "cc",
) -> np.ndarray:
    """Remove bright border artefacts (device text / bright speckles) from the four edges.

    Args:
        border_frac: fraction of H/W for the border strip width
        bright_pct: percentile threshold for "bright" pixels
        mode: "cc" (small connected-component only, safest),
              "threshold" (old: zero everything above threshold in strip),
              "off" (passthrough)
    """
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    result = gray.copy()
    if mode == "off":
        return result

    h, w = gray.shape[:2]
    bh = max(4, int(h * border_frac))
    bw = max(4, int(w * border_frac))

    thr = float(np.percentile(gray, bright_pct))
    if thr < 30:
        return result

    if mode == "threshold":
        for region in [
            result[:bh, :],
            result[h - bh :, :],
            result[:, :bw],
            result[:, w - bw :],
        ]:
            region[region > thr] = 0
        return result

    # CC mode: only remove small bright CCs within the border strips
    dark_bg_thr = float(max(np.percentile(gray, 32), 22.0))
    min_cc_area = max(6, int(bh * bw * 0.00006))
    strip_px_max = max(bh * w, h * bw)
    max_cc_area = int(np.clip(strip_px_max * 0.022, 200, 4800))

    rois = [
        (0, bh, 0, w),
        (h - bh, h, 0, w),
        (0, h, 0, bw),
        (0, h, w - bw, w),
    ]

    bnd = max(8, min(bh, bw, 24) // 2 + 4)

    for y0, y1, x0, x1 in rois:
        if y1 <= y0 or x1 <= x0:
            continue
        patch = gray[y0:y1, x0:x1]
        ph, pw = patch.shape[:2]
        bright = (patch > thr).astype(np.uint8) * 255
        n, labels, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < min_cc_area or area > max_cc_area:
                continue
            cx = int(stats[i, cv2.CC_STAT_LEFT])
            cy = int(stats[i, cv2.CC_STAT_TOP])
            cw_ = int(stats[i, cv2.CC_STAT_WIDTH])
            ch_ = int(stats[i, cv2.CC_STAT_HEIGHT])
            nx1 = max(0, cx - bnd)
            ny1 = max(0, cy - bnd)
            nx2 = min(pw, cx + cw_ + bnd)
            ny2 = min(ph, cy + ch_ + bnd)
            nb = patch[ny1:ny2, nx1:nx2]
            lm = labels[ny1:ny2, nx1:nx2]
            outside = nb[lm != i]
            if len(outside) < 8:
                continue
            dark_frac = float(np.sum(outside < dark_bg_thr)) / len(outside)
            if dark_frac < 0.55:
                continue
            m = labels == i
            ys, xs = np.where(m)
            result[y0 + ys, x0 + xs] = (result[y0 + ys, x0 + xs].astype(np.float32) * 0.35).astype(np.uint8)

    return result


def feather_canvas_edge(
    gray: np.ndarray,
    feather_px: int = 3,
) -> np.ndarray:
    """Apply a narrow linear alpha ramp to the four canvas edges.

    Fixes the bright 1-pixel edge-line caused by outermost patch rows/cols having
    slightly higher weights (no symmetric neighbour). Does not affect central tissue.
    """
    if feather_px <= 0:
        return gray
    result = gray.astype(np.float32)
    h, w = result.shape[:2]
    fp = min(feather_px, h // 4, w // 4)
    ramp = np.linspace(0.0, 1.0, fp, dtype=np.float32)
    result[:fp, :] *= ramp[:, np.newaxis]
    result[h - fp :, :] *= ramp[::-1, np.newaxis]
    result[:, :fp] *= ramp[np.newaxis, :]
    result[:, w - fp :] *= ramp[np.newaxis, ::-1]
    return np.clip(result, 0, 255).astype(np.uint8)


