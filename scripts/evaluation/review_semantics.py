"""
语义漏斗专用检查器：模态 CNN、解剖 CV、patch 接缝频谱、简化影像组学。

设计约束：
- 尽量不 import review_generated_images（由主脚本注入 build_mask，避免循环依赖）。
- 无权重 / 无基线时打印 warning 并降级，不抛致命错误。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# ModalityChecker（ResNet-18，P(mammography)）
# ---------------------------------------------------------------------------
# 训练数据说明（用户后续自训）：正样本 CBIS-DDSM 真实钼靶；负样本为
# ChestX-ray14（胸片）、BUSI（超声）、ImageNet 灰度化自然图各约 500 张。


@dataclass
class ModalityResult:
    """单张图的模态分类结果。"""

    p_mammography: float | None  # None 表示未启用（无有效权重）
    veto: bool
    veto_reason: str
    source: str  # "checkpoint" | "disabled" | "error"


class ModalityChecker:
    """轻量 ResNet-18，输出 P(钼靶)。无 checkpoint 时不做 hard veto（避免随机头误杀）。"""

    def __init__(
        self,
        weights_path: Path | str | None,
        device: str = "cpu",
        img_size: int = 256,
        veto_threshold: float = 0.85,
    ) -> None:
        self.weights_path = Path(str(weights_path)) if weights_path else None
        self.device = device
        self.img_size = img_size
        self.veto_threshold = veto_threshold
        self._model = None
        self._ok = False
        self._load_error: str | None = None

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from torchvision import models
        except Exception as e:
            self._load_error = str(e)
            return

        if self.weights_path is None or not self.weights_path.is_file():
            warnings.warn(
                "[ModalityChecker] 未提供有效权重路径，跳过模态 hard veto 与 P 估计。",
                stacklevel=2,
            )
            return

        m = models.resnet18(weights=None)
        m.fc = torch.nn.Linear(m.fc.in_features, 1)
        try:
            state = torch.load(self.weights_path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(self.weights_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict) and any(k.startswith("module.") for k in state):
            state = {k.replace("module.", ""): v for k, v in state.items()}
        try:
            m.load_state_dict(state, strict=False)
        except Exception as e:
            self._load_error = str(e)
            warnings.warn(f"[ModalityChecker] 权重加载失败，跳过模态检查: {e}", stacklevel=2)
            return

        m.eval()
        m.to(self.device)
        self._model = m
        self._ok = True

    def predict(self, gray_u8: np.ndarray) -> ModalityResult:
        """灰度 uint8 → P(mammography)。无模型时返回 p=None、不 veto。"""
        self._lazy_load()
        if not self._ok or self._model is None:
            return ModalityResult(
                p_mammography=None,
                veto=False,
                veto_reason="",
                source="disabled",
            )

        import torch

        g = gray_u8.astype(np.float32)
        if g.ndim != 2:
            g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY) if g.shape[-1] == 3 else g[..., 0]
        g = cv2.resize(g, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(g).unsqueeze(0).unsqueeze(0) / 255.0
        x = x.repeat(1, 3, 1, 1).to(self.device)
        try:
            with torch.no_grad():
                logit = self._model(x).squeeze(1)
                p = float(torch.sigmoid(logit).detach().cpu().numpy().reshape(-1)[0])
        except Exception as e:
            warnings.warn(f"[ModalityChecker] 推理失败: {e}", stacklevel=2)
            return ModalityResult(None, False, "", "error")

        veto = p < self.veto_threshold
        reason = "MODALITY_LOW" if veto else ""
        return ModalityResult(p, veto, reason, "checkpoint")


# ---------------------------------------------------------------------------
# AnatomyStructureChecker（传统 CV）
# ---------------------------------------------------------------------------


@dataclass
class AnatomyFlags:
    skin_line_present: bool
    closed_ring_detected: bool
    center_overexposed: bool
    mask_area_ratio: float


@dataclass
class AnatomyCheckResult:
    veto: bool
    veto_reasons: list[str]
    flags: AnatomyFlags


def _edge_skin_line_present(gray: np.ndarray, band_px: int = 20) -> bool:
    """皮肤线：Canny 边缘检测约束在 Otsu 乳腺边界带内，验证薄亮弧存在。

    旧版直接用全图 Canny 易被 letterbox 边框伪影欺骗；
    纯 Otsu 边界均值比较（mean(boundary) > 1.12 * mean(interior)）会因乳腺内部
    密度天然高于边界带而系统性漏检——皮肤线是局部边缘特征，不是区域亮度特征。
    """
    h, w = gray.shape[:2]

    # 1. 乳腺掩膜（Otsu）
    thresh_val = max(5.0, float(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]))
    _, breast = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
    if int(np.count_nonzero(breast)) < 500:
        return False

    # 2. 边界带：乳腺 - 形态学腐蚀
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    eroded = cv2.erode(breast, k, iterations=2)
    boundary_mask = (breast.astype(np.int16) - eroded.astype(np.int16)) > 0
    if np.count_nonzero(boundary_mask) < 40:
        return False

    # 3. Canny 边缘检测，约束在边界带内（排除 letterbox 伪影和内部组织边缘）
    edges = cv2.Canny(gray, 40, 120)
    boundary_edges = edges & boundary_mask.astype(np.uint8)
    boundary_edge_count = int(np.count_nonzero(boundary_edges))
    if boundary_edge_count < 50:
        return False

    # 4. 边界边缘点必须整体亮于紧邻内侧组织（皮肤线是亮线，不是暗线）
    interior_band_mask = (eroded.astype(np.int16) - cv2.erode(eroded, k, iterations=2).astype(np.int16)) > 0
    interior_near_px = gray[interior_band_mask]
    boundary_edge_px = gray[boundary_edges > 0]
    if len(interior_near_px) < 50:
        return False
    # 皮肤线是亮边缘，但生成图对比度天然偏低 → 放宽亮度比
    if float(np.median(boundary_edge_px)) < float(np.median(interior_near_px)) * 0.85:
        return False

    # 5. 沿边界边缘的连续弧长需足够（排除散落噪点）
    cnts, _ = cv2.findContours(boundary_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_len = max((float(cv2.arcLength(c, False)) for c in cnts), default=0.0)
    thr_len = float(max(100.0, 0.07 * max(h, w)))
    return max_len >= thr_len


def _closed_ring_veto(gray: np.ndarray, circ_min: float = 0.8, area_min: int = 1000) -> bool:
    """闭合圆环伪影：高圆形度 + 足够大面积（拦截「眼球/圆饼」）。"""
    # 偏亮区域上做轮廓分析
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.count_nonzero(th) > th.size * 0.85:
        th = cv2.bitwise_not(th)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        area = float(cv2.contourArea(c))
        if area < area_min:
            continue
        peri = float(cv2.arcLength(c, True))
        if peri < 1e-6:
            continue
        circ = float(4.0 * np.pi * area / (peri * peri))
        if circ >= circ_min:
            return True
    # 补充：Hough 圆（强响应时否决）
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min(gray.shape) // 4,
        param1=100,
        param2=28,
        minRadius=12,
        maxRadius=min(gray.shape) // 3,
    )
    if circles is not None and len(circles[0]) >= 1:
        for x, y, rad in circles[0]:
            a = np.pi * (rad ** 2)
            if a >= area_min:
                return True
    return False


def _center_overexposed(gray: np.ndarray, bright_thr: int = 240, frac_thr: float = 0.30) -> bool:
    """中心 50% 区域过曝占比（拦截拼贴式高光核）。"""
    h, w = gray.shape[:2]
    y0, y1 = h // 4, 3 * h // 4
    x0, x1 = w // 4, 3 * w // 4
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return False
    frac = float(np.count_nonzero(roi >= bright_thr)) / float(roi.size)
    return frac > frac_thr


def anatomy_structure_check(
    gray: np.ndarray,
    breast_mask: np.ndarray,
    mask_ratio_p5_p95: tuple[float, float] | None,
) -> AnatomyCheckResult:
    """解剖结构一票否决集合。mask_ratio 需在真实分布 [P5,P95] 内（由主脚本传入）。"""
    reasons: list[str] = []
    total = float(breast_mask.size)
    mask_area_ratio = float(np.count_nonzero(breast_mask)) / max(1.0, total)

    if mask_ratio_p5_p95 is not None:
        lo, hi = mask_ratio_p5_p95
        if mask_area_ratio < lo or mask_area_ratio > hi:
            reasons.append("MASK_AREA_OUT_OF_DIST")

    skin_ok = _edge_skin_line_present(gray)
    if not skin_ok:
        reasons.append("SKIN_LINE_MISSING")

    ring = _closed_ring_veto(gray)
    if ring:
        reasons.append("CLOSED_RING_ARTIFACT")

    center_bad = _center_overexposed(gray)
    if center_bad:
        reasons.append("CENTER_OVEREXPOSED")

    flags = AnatomyFlags(
        skin_line_present=skin_ok,
        closed_ring_detected=ring,
        center_overexposed=center_bad,
        mask_area_ratio=mask_area_ratio,
    )
    return AnatomyCheckResult(veto=len(reasons) > 0, veto_reasons=reasons, flags=flags)


# ---------------------------------------------------------------------------
# PatchSeamChecker（径向频谱尖峰）
# ---------------------------------------------------------------------------


@dataclass
class SeamFlags:
    grid_peak_detected: bool
    peak_frequencies: list[float]


def radial_spectrum_1d(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """二维 DFT 后按半径分桶得到 S(r)，归一化。返回 (r_centers, S)。物理意义：各空间频率能量占比。"""
    g = gray.astype(np.float32)
    g = (g - float(g.mean())) / (float(g.std()) + 1e-6)
    f = np.fft.fftshift(np.fft.fft2(g))
    p = (np.abs(f) ** 2).astype(np.float64)
    h, w = p.shape
    yy, xx = np.indices((h, w))
    cy, cx = h / 2.0, w / 2.0
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_max = float(rr.max()) + 1e-6
    rr_n = (rr / r_max).flatten()
    p_flat = p.flatten()
    n_bins = min(256, max(32, min(h, w) // 2))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    prof = np.zeros(n_bins, dtype=np.float64)
    for i in range(n_bins):
        m = (rr_n >= edges[i]) & (rr_n < edges[i + 1])
        if np.any(m):
            prof[i] = float(np.mean(p_flat[m]))
    centers = 0.5 * (edges[:-1] + edges[1:])
    prof = np.log1p(prof)
    return centers.astype(np.float64), prof.astype(np.float64)


def patch_seam_check(
    gray: np.ndarray,
    patch_size: int | None,
    *,
    enabled: bool = True,
    z_sigma: float = 3.0,
) -> tuple[bool, SeamFlags]:
    """检测 patch 网格基频/谐波尖峰；显著则 veto。"""
    if not enabled:
        return False, SeamFlags(False, [])

    r, s = radial_spectrum_1d(gray)
    if s.size < 8:
        return False, SeamFlags(False, [])

    # 局部平滑包络：移动均值近似背景趋势
    k = max(5, s.size // 16 | 1)
    if k % 2 == 0:
        k += 1
    kernel = np.ones(k, dtype=np.float64) / float(k)
    # 反射填充避免 mode="same" 的零填充边界假峰
    pad = k // 2
    s_padded = np.pad(s, pad, mode="reflect")
    smooth_full = np.convolve(s_padded, kernel, mode="valid")
    smooth = smooth_full[: len(s)]
    resid = s - smooth
    mad = float(np.median(np.abs(resid - np.median(resid))) + 1e-9)
    sigma_est = 1.4826 * mad
    thr = z_sigma * max(sigma_est, 1e-6)

    # 排除边界受填充影响的前 k//2 个 bin
    edge_margin = max(2, k // 2)
    peaks_idx: list[int] = []
    for i in range(edge_margin, len(resid) - edge_margin):
        if resid[i] == max(resid[i - 2 : i + 3]) and resid[i] > thr:
            peaks_idx.append(i)

    peak_freqs: list[float] = []
    H, W = gray.shape[:2]

    for i in peaks_idx:
        peak_freqs.append(float(r[i]))

    veto = False
    # 若 patch_size 已知且目标频率落在可分辨区域，检查网格谐波
    if patch_size and patch_size > 8 and len(s) > 16:
        n = float(max(H, W))
        for k_h in (1, 2, 3, 4):
            r_norm = (k_h * n / float(patch_size)) / n
            if r_norm > 0.92:
                break
            j = int(np.argmin(np.abs(r - r_norm)))
            # 低频 bins 受 1/f 频谱陡降主导，残差天然大，不适用于格线检测
            if j < edge_margin:
                continue
            j = int(np.clip(j, edge_margin, len(resid) - 1 - edge_margin))
            local = resid[j - 2 : j + 3]
            if float(np.max(local)) > thr:
                veto = True

    strong = sum(1 for i in peaks_idx if resid[i] > thr)
    if strong >= 3:
        veto = True

    return veto, SeamFlags(veto, peak_freqs[:12])


# ---------------------------------------------------------------------------
# Radiomics（20 维 → PCA-10，Mahalanobis）
# ---------------------------------------------------------------------------


def extract_radiomics_vector(gray: np.ndarray, breast_mask: np.ndarray) -> np.ndarray | None:
    """提取约 20 维简化特征；失败返回 None。"""
    try:
        from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
        from skimage.measure import label, regionprops
    except Exception:
        return None

    if int(np.count_nonzero(breast_mask)) < 400:
        return None

    g = gray.astype(np.float64)
    roi = (breast_mask > 0).astype(np.uint8)
    ys, xs = np.where(roi > 0)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    patch = g[y0:y1, x0:x1]
    m = roi[y0:y1, x0:x1]
    if patch.size < 64:
        return None

    gi = np.clip(patch, 0, 255).astype(np.uint8)
    mi = (m > 0).astype(np.uint8)
    # GLCM：4 维
    levels = 64
    gi_q = (gi.astype(np.float64) / 255.0 * (levels - 1)).astype(np.uint8)
    mi_bool = mi.astype(bool)
    # 旧版 skimage 无 mask 参数：背景填 0（对 GLCM 略有偏置，仅作兼容）
    try:
        glcm = graycomatrix(
            gi_q,
            distances=[1],
            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            levels=levels,
            symmetric=True,
            normed=True,
            mask=mi_bool,
        )
    except TypeError:
        gi_for_glcm = np.where(mi_bool, gi_q, np.uint8(0))
        glcm = graycomatrix(
            gi_for_glcm,
            distances=[1],
            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            levels=levels,
            symmetric=True,
            normed=True,
        )
    contrast = float(graycoprops(glcm, "contrast").mean())
    homogeneity = float(graycoprops(glcm, "homogeneity").mean())
    correlation = float(graycoprops(glcm, "correlation").mean())
    energy = float(graycoprops(glcm, "energy").mean())

    # LBP：8 统计量（非 59 维直方图）
    lbp = local_binary_pattern(gi, P=8, R=1, method="uniform")
    vals = lbp[mi > 0].astype(np.float64)
    if vals.size < 32:
        return None
    hist, _ = np.histogram(vals, bins=10, range=(0, 10), density=True)
    chi2_like = float(np.sum((hist - hist.mean()) ** 2 / (hist.mean() + 1e-6)))
    lbp_mean = float(vals.mean())
    lbp_var = float(vals.var())
    lbp_skew = float(np.mean(((vals - lbp_mean) / (np.sqrt(lbp_var) + 1e-6)) ** 3))
    lbp_kurt = float(np.mean(((vals - lbp_mean) / (np.sqrt(lbp_var) + 1e-6)) ** 4))
    lbp_ent = float(-np.sum(hist * np.log(hist + 1e-9)))

    # 形状：掩膜面积比、偏心率、周长面积比
    lbl = label(mi.astype(bool))
    props = regionprops(lbl)
    if not props:
        return None
    pr = max(props, key=lambda r: int(r.area))
    ecc = float(pr.eccentricity)
    perim = float(pr.perimeter)
    area_m = float(pr.area)
    par = float(perim / (2.0 * np.sqrt(np.pi * area_m + 1e-6)))

    # 灰度：均值、标准差、偏度、峰度、熵
    pv = gi[mi > 0].astype(np.float64) / 255.0
    mu = float(pv.mean())
    sd = float(pv.std())
    skew = float(np.mean(((pv - mu) / (sd + 1e-6)) ** 3))
    kurt = float(np.mean(((pv - mu) / (sd + 1e-6)) ** 4))
    hist_g, _ = np.histogram(pv, bins=32, range=(0, 1), density=True)
    hg = hist_g / (hist_g.sum() + 1e-12)
    ent_g = float(-np.sum(hg * np.log(hg + 1e-12)))

    # 频谱：高频能量比、主导方向性（简化：高频/全频）
    f = np.fft.rfft2(gi.astype(np.float32))
    ps = (np.abs(f) ** 2).astype(np.float64)
    h2, w2 = ps.shape
    cy = h2 // 2
    hf = float(ps[cy // 2 :].sum() / (ps.sum() + 1e-12))
    row_energy = float(ps[:, : w2 // 4].sum() / (ps.sum() + 1e-12))
    col_energy = float(ps[: h2 // 4, :].sum() / (ps.sum() + 1e-12))
    directionality = float(max(row_energy, col_energy))

    return np.array(
        [
            contrast,
            homogeneity,
            correlation,
            energy,
            lbp_mean,
            lbp_var,
            lbp_skew,
            lbp_kurt,
            lbp_ent,
            chi2_like,
            float(area_m / (mi.size + 1e-6)),
            ecc,
            par,
            mu,
            sd,
            skew,
            kurt,
            ent_g,
            hf,
            directionality,
        ],
        dtype=np.float64,
    )


@dataclass
class RadiomicsBaseline:
    """PCA 潜空间（10 维）上的 Mahalanobis：与 build_radiomics_baseline.py 写入字段一致。"""

    pca_mean: np.ndarray  # (20,) sklearn PCA.mean_
    pca_components: np.ndarray  # (10, 20) sklearn PCA.components_
    y_mean: np.ndarray  # (10,) 训练集潜向量均值（通常接近 0）
    inv_cov10: np.ndarray  # (10, 10)


def load_radiomics_baseline(path: Path | None) -> RadiomicsBaseline | None:
    if path is None or not path.is_file():
        return None
    try:
        z = np.load(path, allow_pickle=False)
        W = np.asarray(z["pca_components"], dtype=np.float64)
        mu = np.asarray(z["pca_mean"], dtype=np.float64).reshape(-1)
        y_mean = np.asarray(z["y_mean"], dtype=np.float64).reshape(-1)
        invc = np.asarray(z["inv_cov10"], dtype=np.float64)
        if W.shape != (10, 20) or mu.size != 20 or y_mean.size != 10 or invc.shape != (10, 10):
            warnings.warn(f"[Radiomics] 基线维度异常: {path}", stacklevel=2)
            return None
        return RadiomicsBaseline(pca_mean=mu, pca_components=W, y_mean=y_mean, inv_cov10=invc)
    except Exception as e:
        warnings.warn(f"[Radiomics] 读取基线失败 {path}: {e}", stacklevel=2)
        return None


def radiomics_mahalanobis_score(x20: np.ndarray, base: RadiomicsBaseline | None) -> tuple[float, float]:
    """返回 (d_squared, score)，score=exp(-d²/2)。d² 为潜空间 Mahalanobis。"""
    if base is None or x20 is None or x20.size != 20:
        return float("nan"), 0.5
    z = (x20.astype(np.float64) - base.pca_mean).reshape(-1)
    y = (base.pca_components @ z).reshape(-1)
    dy = y - base.y_mean.reshape(-1)
    d2 = float(dy @ base.inv_cov10 @ dy)
    score = float(np.exp(-0.5 * d2))
    return d2, score


def breast_mask_iou(mask_gen: np.ndarray, mask_real: np.ndarray) -> float:
    a = (mask_gen > 0).astype(np.uint8)
    b = (mask_real > 0).astype(np.uint8)
    inter = float(np.count_nonzero(a & b))
    union = float(np.count_nonzero(a | b)) + 1e-6
    return inter / union
