"""Pluggable label/artifact guard for mammography generation.

Provides stateless image filtering functions that can be applied as pre-filter
(before SD generation) or post-filter (after SD generation). Each function takes
and returns a (H,W) uint8 grayscale numpy array.

The L1 heuristic and Qwen-VL bbox integration remains in
scripts/preprocessing/mammo_label_heuristic.py; this module wraps the fixed-order
erasure and canvas-edge operations that were previously inline in run_mammo_sd15.py.
"""

from __future__ import annotations

import cv2
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# Individual filter functions
# ═══════════════════════════════════════════════════════════════════════════════

def strip_crop(
    gray: np.ndarray,
    top: float = 0.04,
    bottom: float = 0.04,
    left: float = 0.03,
    right: float = 0.03,
) -> np.ndarray:
    """Conservative border strip crop to remove DICOM annotation bands."""
    h, w = gray.shape[:2]
    y1 = int(round(h * top))
    y2 = h - int(round(h * bottom))
    x1 = int(round(w * left))
    x2 = w - int(round(w * right))
    if y2 > y1 + 16 and x2 > x1 + 16:
        return gray[y1:y2, x1:x2]
    return gray


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


def inpaint_interior_bright_spots(gray: np.ndarray) -> np.ndarray:
    """Inpaint isolated bright-spot artifacts inside the breast region.

    Targets small high-intensity CCs (5-300px) that are surrounded by
    relatively dark tissue — the hallmark of SD-generated phantom blobs.
    True calcifications (surrounded by dense tissue > Otsu threshold) are
    protected and left intact.
    """
    result = gray.copy()
    h, w = gray.shape[:2]

    # Build breast foreground mask (coarse)
    otsu_thr, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg_thr = max(int(otsu_thr * 0.5), 10)
    _, fg_mask = cv2.threshold(gray, fg_thr, 255, cv2.THRESH_BINARY)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, k_close, iterations=2)
    k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    fg_interior = cv2.erode(fg_mask, k_erode, iterations=3)

    breast_pixels = gray[fg_interior > 0]
    if len(breast_pixels) < 500:
        return result

    # Detect very bright spots: top 0.5% within breast
    hi_thr = float(np.percentile(breast_pixels, 99.5))
    global_median = float(np.median(breast_pixels))
    dense_thr = float(otsu_thr)  # threshold for "dense tissue"

    bright_bin = np.zeros((h, w), dtype=np.uint8)
    bright_bin[(gray >= hi_thr) & (fg_interior > 0)] = 255

    n, labels, stats, _ = cv2.connectedComponentsWithStats(bright_bin, connectivity=8)
    if n <= 1:
        return result

    inpaint_mask = np.zeros((h, w), dtype=np.uint8)
    band = 20  # neighbourhood radius for context check

    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 5 or area > 300:
            continue

        x0 = int(stats[i, cv2.CC_STAT_LEFT])
        y0 = int(stats[i, cv2.CC_STAT_TOP])
        bw_ = int(stats[i, cv2.CC_STAT_WIDTH])
        bh_ = int(stats[i, cv2.CC_STAT_HEIGHT])

        nx1, ny1 = max(0, x0 - band), max(0, y0 - band)
        nx2, ny2 = min(w, x0 + bw_ + band), min(h, y0 + bh_ + band)
        nb_gray = gray[ny1:ny2, nx1:nx2]
        nb_lbl = labels[ny1:ny2, nx1:nx2]
        outside = nb_gray[nb_lbl != i].flatten()

        if len(outside) < 10:
            continue

        # Skip if neighbourhood is predominantly dense tissue (likely real calcification)
        dense_frac = float(np.sum(outside > dense_thr)) / len(outside)
        if dense_frac > 0.40:
            continue

        # Check if CC is in an isolated/dark context relative to global median
        nb_median = float(np.median(outside))
        if nb_median > global_median * 1.4:
            continue  # surrounded by bright tissue — not isolated artifact

        inpaint_mask[labels == i] = 255

    if not np.any(inpaint_mask):
        return result

    # Dilate mask slightly for cleaner inpaint boundary
    k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    inpaint_mask = cv2.dilate(inpaint_mask, k_dil, iterations=1)

    result = cv2.inpaint(result, inpaint_mask, inpaintRadius=5,
                         flags=cv2.INPAINT_TELEA)
    return result


def clean_background(gray: np.ndarray) -> np.ndarray:
    """Zero out pure-background regions outside the breast main body.

    Protects edge-adjacent thin tissue via distance-transform gradient fade.
    """
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    lo = float(np.percentile(gray, 2))
    hi = float(np.percentile(gray, 98))
    if hi - lo < 5:
        return gray

    thresh_val = max(int(lo + (hi - lo) * 0.025), 3)
    _, bin_img = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    closed = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, close_k, iterations=2)

    n_cc, labels_map, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if n_cc <= 1:
        return gray

    areas = stats[1:, cv2.CC_STAT_AREA]
    order = np.argsort(-areas)
    li0 = int(order[0])
    breast_label = li0 + 1
    breast_mask = np.where(labels_map == breast_label, np.uint8(255), np.uint8(0))
    max_area = float(areas[li0])

    for rank in range(1, min(3, len(order))):
        li_r = int(order[rank])
        cc_label = li_r + 1
        ar = float(areas[li_r])
        if ar < max_area * 0.08:
            break
        cx1 = stats[breast_label, cv2.CC_STAT_LEFT] + stats[breast_label, cv2.CC_STAT_WIDTH] // 2
        cx2 = stats[cc_label, cv2.CC_STAT_LEFT] + stats[cc_label, cv2.CC_STAT_WIDTH] // 2
        if abs(int(cx2) - int(cx1)) <= int(w * 0.60):
            breast_mask = np.where(
                (breast_mask == 255) | (labels_map == cc_label),
                np.uint8(255),
                np.uint8(0),
            )

    protect_r = max(35, int(min(h, w) * 0.06))
    dil_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (protect_r * 2 + 1, protect_r * 2 + 1))
    protected = cv2.dilate(breast_mask, dil_k)

    hx = max(35, (w // 9) | 1)
    horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (hx, 5))
    protected = cv2.dilate(protected, horiz, iterations=2)

    dist = cv2.distanceTransform((protected > 0).astype(np.uint8), cv2.DIST_L2, 5)
    feather_px = float(max(18, int(min(h, w) * 0.022)))
    alpha = np.clip(dist / feather_px, 0.0, 1.0).astype(np.float32)
    result = (gray.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)
    return result
