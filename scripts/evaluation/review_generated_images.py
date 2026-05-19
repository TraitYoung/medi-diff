"""
批量审查生成图（量化评估版）

在原有规则审查基础上，补齐乳腺钼靶生成图的 14 项量化指标，覆盖：
- A 构图与解剖   : A1 面积比 / A2 圆形度 / A3 胸肌线
- B 灰阶分布     : B1 均值方差 / B2 动态范围 / B3 高光-暗区占比
- C 纹理与频率   : C1 径向功率谱偏差 / C2 局部对比度熵
- D 伪影与幻觉   : D1 孤立圆形区 / D2 边缘黑洞 / D3 条带伪影
- E 真实分布偏离 : E1 直方图 Wasserstein / E2 密度分型匹配

每张图输出：
- 14 项原始指标
- 每项 0-1 小分 + 失败标签
- 5 个维度分组均分
- 加权总分 (0-100)

批次汇总：
- summary.json：平均总分、各维度平均、各违规标签出现率
    => 直接用于参数对比表（strength / CFG / cnet_scale 等）

CLI 保持向后兼容；想复用旧基线可用 --real-baseline-json 避免重新计算。
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import random
import tempfile
from collections import Counter
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from tqdm import tqdm

from review_semantics import (
    ModalityChecker,
    anatomy_structure_check,
    breast_mask_iou,
    extract_radiomics_vector,
    load_radiomics_baseline,
    patch_seam_check,
    radiomics_mahalanobis_score,
)

# ---------------------------------------------------------------------------
# Optional dependency status flags
# ---------------------------------------------------------------------------
try:
    from scipy.stats import wasserstein_distance as _scipy_wasserstein
except Exception:
    _scipy_wasserstein = None

try:
    import torch as _torch
    import piq as _piq
    _PIQ_OK = True
except Exception:
    _PIQ_OK = False
    _torch = None

try:
    from pytorch_fid.fid_score import calculate_fid_given_paths as _calc_fid
    _FID_OK = True
except Exception:
    _FID_OK = False

try:
    from torch_fidelity import calculate_metrics as _torch_fidelity_calculate_metrics
    _TORCH_FIDELITY_OK = True
except Exception:
    _TORCH_FIDELITY_OK = False

try:
    from pytorch_fid.fid_score import calculate_frechet_distance as _pt_frechet
    from pytorch_fid.inception import InceptionV3 as _PTInceptionV3
    from pytorch_fid.fid_score import ImagePathDataset as _PTImagePathDataset
    _PT_FID_UTILS_OK = True
except Exception:
    _PT_FID_UTILS_OK = False

try:
    import torch.nn.functional as _F_pt
except Exception:
    pass

try:
    import torch as _torch_clip
    from transformers import CLIPModel as _HFCLIPModel
    from transformers import CLIPProcessor as _HFCLIPProcessor
    _CLIP_TR_OK = True
    _CLIP_MODEL = None
    _CLIP_PROC = None
    _CLIP_MODEL_ID_LOADED: str | None = None
    _CLIP_LABELS_LAST: tuple[str, ...] = ()
    _CLIP_DISABLED_REASON: str | None = None
    _REVIEW_CLIP_MODEL_ID: str | None = None
except Exception:
    _CLIP_TR_OK = False
    _CLIP_MODEL = None
    _CLIP_PROC = None
    _CLIP_MODEL_ID_LOADED = None
    _CLIP_LABELS_LAST = ()
    _CLIP_DISABLED_REASON = None
    _REVIEW_CLIP_MODEL_ID = None
    _torch_clip = None
    _HFCLIPModel = None
    _HFCLIPProcessor = None

# ---------------------------------------------------------------------------
# Module-level globals
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGES_DIR = ROOT / 'outputs/generated/毕业论文_生成图像'
DEFAULT_GROUP_WEIGHTS = {
    'A': 0.2,
    'B': 0.1,
    'C': 0.2,
    'D': 0.25,
    'E': 0.05,
    'F': 0.2,
}
HIST_BINS = 100


# ---------------------------------------------------------------------------
# Academic metric helpers
# ---------------------------------------------------------------------------
def _gather_image_paths_for_academic_metrics(root, *, n_max, rng):
    """与批级 Inception 学术指标抽样一致的非递归文件名列表。"""
    exts = frozenset({'.jpg', '.png', '.jpeg'})
    cand = sorted(
        p for p in root.rglob('*')
        if p.suffix.lower() in exts and '.ipynb_checkpoints' not in p.parts
    )
    if n_max <= 0 or len(cand) <= n_max:
        return cand
    return rng.sample(cand, n_max)


@contextlib.contextmanager
def _temporary_symlink_dirs(gen_paths, real_paths):
    """为 torch-fidelity 提供仅含 symlink 的平面目录。"""
    with tempfile.TemporaryDirectory(prefix='review_gen_') as tg, \
         tempfile.TemporaryDirectory(prefix='review_real_') as tr:
        for i, p in enumerate(gen_paths):
            ext = p.suffix.lower() if p.suffix else '.png'
            link = Path(tg) / f'g_{i:05d}{ext}'
            os.symlink(p.resolve(), link)
        for i, p in enumerate(real_paths):
            ext = p.suffix.lower() if p.suffix else '.png'
            link = Path(tr) / f'r_{i:05d}{ext}'
            os.symlink(p.resolve(), link)
        yield Path(tg), Path(tr)


def _spatial_sfid_patch768(gen_paths_str, real_paths_str, *, device, batch_size):
    """空间块级 Frechet 距离（Inception Mixed_6e，768 维、17x17 网格）。
    与 StyleGAN TF 定义的 pool 前 2048-d sFID 非逐比特一致；用于「空间感受野」层面的分布距离。
    """
    import torchvision.transforms as T

    if not gen_paths_str or not real_paths_str or not _PT_FID_UTILS_OK:
        return None

    block_idx = 2
    model = _PTInceptionV3([block_idx], resize_input=True, normalize_input=True).to(device)
    model.eval()
    to_tensor = T.Compose([
        T.Resize((299, 299), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
    ])

    def _stack_feats(paths):
        ds = _PTImagePathDataset(paths, transforms=to_tensor)
        if len(ds) == 0:
            return None
        loader = _torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0,
        )
        feats = []
        for batch in loader:
            batch = batch.to(device)
            with _torch.no_grad():
                outp = model(batch)[0]
            bsz, _, h, w = outp.shape
            fl = outp.permute(0, 2, 3, 1).reshape(bsz, h * w, -1).cpu().numpy().astype(np.float64)
            feats.append(fl)
        if not feats:
            return None
        return np.vstack(feats)

    g = _stack_feats(gen_paths_str)
    r = _stack_feats(real_paths_str)
    if g is None or r is None or g.shape[0] < 2 or r.shape[0] < 2:
        return None
    mu_g = np.mean(g, axis=0)
    sg = np.cov(g, rowvar=False)
    mu_r = np.mean(r, axis=0)
    sr = np.cov(r, rowvar=False)
    return float(_pt_frechet(mu_g, sg, mu_r, sr))


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description='量化审查生成钼铛图')
    p.add_argument('--images-dir', type=Path, default=DEFAULT_IMAGES_DIR,
                   help=f'待审查图像目录（默认：{DEFAULT_IMAGES_DIR.name}）')
    p.add_argument('--recursive', action=argparse.BooleanOptionalAction, default=True,
                   help='是否递归扫描子目录（默认开）。仅顶层用 --no-recursive。')
    p.add_argument('--include-compare', action='store_true',
                   help='将 *_ffdm_compare.png 一并纳入审查（默认排除）。')
    p.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/reviews/review_output',
                   help='输出目录')
    p.add_argument('--name-prefix', type=str, default='',
                   help='仅审查文件名以该前缀开头的图片（如 img2img_）。为空则不过滤。')
    p.add_argument('--real-images-dir', type=Path, default=None,
                   help='真实钼铛图目录。提供后会计算 RAPS 基线、直方图基线、密度先验。')
    p.add_argument('--real-baseline-json', type=Path, default=None,
                   help='已有真实基线 JSON，提供后跳过 real-images-dir 的重新计算。')
    p.add_argument('--no-prc-sweep', action='store_true',
                   help='仅计算 KNN k=3 的一点 PRC；不写多 k 的 pr_curve（省时间）。')
    p.add_argument('--prc-k-list', type=str, default='1,3,5,11,21',
                   help='PRC 扫描：逗号分隔的 prc_neighborhood（需安装 torch-fidelity）；与 --no-prc-sweep 互斥时以 --no-prc-sweep 为准。')
    p.add_argument('--resize-long-side', type=int, default=1024,
                   help='评测/建基线前把图缩到长边=该值（0 禁用）。默认 1024，显著加速且不影响统计稳定性。')
    p.add_argument('--max-baseline-samples', type=int, default=300,
                   help='构建真实基线时最多抽样多少张（均匀随机），默认 300。<=0 表示全量。')
    p.add_argument('--baseline-workers', type=int, default=0,
                   help='真实基线并行进程数；0=自动（min(8, CPU 数)），1=单进程。')
    p.add_argument('--review-workers', type=int, default=0,
                   help='生成图 review 的并行进程数；0=自动。')
    p.add_argument('--sample-seed', type=int, default=2026,
                   help='基线随机抽样的 seed。')
    p.add_argument('--top-k', type=int, default=30,
                   help='导出推荐前K张')
    p.add_argument('--auto-calibrate', action=argparse.BooleanOptionalAction, default=True,
                   help='从真实基线 percentiles 自动推导阈值和中心值（默认开）。关掉用 --no-auto-calibrate，以 CLI 指定的硬阈值为准。')
    p.add_argument('--min_mask_ratio', type=float, default=0.001)
    p.add_argument('--max_mask_ratio', type=float, default=0.75)
    p.add_argument('--min_circularity', type=float, default=0.32)
    p.add_argument('--max_circularity', type=float, default=0.9)
    p.add_argument('--require_pectoral', action='store_true',
                   help='若开启，MLO-like 图无胸肌线会判为失败')
    p.add_argument('--max_mean_intensity', type=float, default=0.7)
    p.add_argument('--min_std_intensity', type=float, default=0.05)
    p.add_argument('--min_dynamic_range', type=float, default=0.3)
    p.add_argument('--max_dynamic_range', type=float, default=0.95)
    p.add_argument('--max_bright_ratio', type=float, default=0.99,
                   help='过曝像素比例上限')
    p.add_argument('--max_dark_ratio', type=float, default=0.25,
                   help='死黑像素比例上限（仅乳腺区内）')
    p.add_argument('--min_local_entropy', type=float, default=0)
    p.add_argument('--max_local_entropy', type=float, default=4.5)
    p.add_argument('--max_ps_emd', type=float, default=0.22)
    p.add_argument('--max_grad_wass', type=float, default=0.2,
                   help='C3 梯度直方图 Wasserstein 上限')
    p.add_argument('--max_isolated_round', type=int, default=6,
                   help='孤立圆形亮区数量阈值')
    p.add_argument('--max_edge_voids', type=int, default=200)
    p.add_argument('--edge-density-floor', type=float, default=0.008,
                   help='边缘密度下限；低于此值 Group C（纹理）分数被硬封顶 0.5')
    p.add_argument('--min_banding_score', type=float, default=0.62)
    p.add_argument('--max_round_density', type=float, default=0.04,
                   help='圆形亮连通域总面积占乳腺区比例上限；超过判 ARTIFACT_TRYPOPHOBIA（密集伪影）。')
    p.add_argument('--max_contour_concavity', type=float, default=0.45,
                   help='乳腺最大凸性缺陷深度 / sqrt(area) 上限；超过判 CONTOUR_FRACTURED。')
    p.add_argument('--max_contour_perim_ratio', type=float, default=1.35,
                   help='轮廓周长 / 凸包周长上限；超过判 CONTOUR_FRACTURED（锯齿震荡）。')
    p.add_argument('--veto-group-min', type=float, default=0.3,
                   help='一票否决：任一维度（A/B/C/D/E）得分低于此值直接判 Fail。设为 0 关闭。')
    p.add_argument('--max_hist_wass', type=float, default=0.15)
    p.add_argument('--max_ref_z', type=float, default=2.6)
    p.add_argument('--max_cavity_ratio', type=float, default=0.28)
    p.add_argument('--max_bright_spots', type=int, default=20)
    p.add_argument('--max_mirror_sim', type=float, default=0.84)
    p.add_argument('--max_hu_dist', type=float, default=1.8)
    p.add_argument('--anatomy-chaos-hard', type=float, default=0.4,
                   help='解剖混沌度上限；≥此值打硬伤 ANATOMY_NON_MAMMO 并一票否决（默认对齐 CBIS×鬼图校准）。')
    p.add_argument('--clip-anatomy', action='store_true',
                   help='启用 CLIP 文本对比补强（区分钼铛 vs 胸片/手足）；需 transformers + 权重，离线请配 --clip-model。')
    p.add_argument('--clip-model', type=str, default='openai/clip-vit-base-patch32',
                   help='HuggingFace CLIP id 或本地目录（配合 --clip-anatomy）。')
    p.add_argument('--clip-anatomy-fail-margin', type=float, default=0.04,
                   help='CLIP：钼铛类 prompt  softmax 概率需高于次优类至少该阈值，否则打 CLIP_CROSS_ANATOMY。')
    p.add_argument('--weights', type=str, default='',
                   help='JSON 覆盖分组权重，如 \'{"A":0.3,"B":0.2,"C":0.15,"D":0.25,"E":0.1}\'')
    p.add_argument('--modality-classifier-path', type=Path, default=None,
                   help='模态分类器权重 .pth；不存在则跳过模态 hard veto（warning）。')
    p.add_argument('--radiomics-baseline', type=Path, default=None,
                   help='影像组学基线 radiomics_baseline.npz；不存在则 semantic 中 radiomics 用中性分。')
    p.add_argument('--enable-seam-check', action=argparse.BooleanOptionalAction, default=True,
                   help='patch 网格频域接缝检测（默认开）。')
    p.add_argument('--patch-size-eval', type=int, default=512,
                   help='接缝检测假定 patch 边长（与生成脚本 patch-size 对齐）。')
    p.add_argument('--tier1-semantic-threshold', type=float, default=0.8)
    p.add_argument('--tier2-semantic-threshold', type=float, default=0.5)
    p.add_argument('--top-k-tier-priority', action=argparse.BooleanOptionalAction, default=True,
                   help='Top-K 先填满 Tier1；不足时用 Tier2 递补（默认真）。')
    p.add_argument('--paired-source-dir', type=Path, default=None,
                   help='若提供且存在同名源图，则计算乳腺掩膜 IoU 并入 semantic_score。')
    p.add_argument('--modality-device', type=str, default='cpu',
                   help='模态分类器设备：cpu 或 cuda（默认 cpu，单张 batch=1）。')
    p.add_argument('--eval-profile', choices=('full', 'patch'), default='full',
                   help='评估画像：full=全幅 FFDM，默认把皮肤线/圆环/接缝作为语义降级而非一票否决；patch=patch 生成图，保留更严格 hard veto。')
    p.add_argument('--strict-stat-veto', action=argparse.BooleanOptionalAction, default=False,
                   help='是否让旧统计 ok=False 直接参与最终 ok。默认关，避免全幅真实图被 BRISQUE/BANDING 批量误杀。')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
def is_image(path):
    return path.suffix.lower() in frozenset({'.bmp', '.webp', '.jpg', '.png', '.jpeg'})


def resize_long_side(gray, long_side):
    """把灰度图按长边等比缩放到 long_side。<=0 或比原图大则原样返回。
    FFT/卷积是平方级开销，缩一下对真实基线是 6-30x 提速。
    """
    if long_side is None or long_side <= 0:
        return gray
    h, w = gray.shape[:2]
    m = max(h, w)
    if m <= long_side:
        return gray
    scale = long_side / float(m)
    nh = max(16, int(round(h * scale)))
    nw = max(16, int(round(w * scale)))
    return cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)


def largest_component(binary):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return np.zeros_like(binary, dtype=np.uint8)
    idx = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    return (labels == idx).astype(np.uint8) * 255


def build_mask(gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.count_nonzero(th) > th.size * 0.8:
        th = cv2.bitwise_not(th)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)
    return largest_component(th)


# ---------------------------------------------------------------------------
# Distance / shape helpers
# ---------------------------------------------------------------------------
def wasserstein_1d(u, v):
    """对齐 bin 的两直方图的一维 Wasserstein。"""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if u.sum() <= 0 or v.sum() <= 0:
        return 1
    u = u / u.sum()
    v = v / v.sum()
    if _scipy_wasserstein:
        support = np.linspace(0, 1, len(u))
        return float(_scipy_wasserstein(support, support, u_weights=u, v_weights=v))
    return float(np.mean(np.abs(np.cumsum(u) - np.cumsum(v))))


def emd_1d_profile(p, q):
    """留给 RAPS 的归一化 EMD。"""
    p = p.astype(np.float64)
    q = q.astype(np.float64)
    p /= np.sum(p) + 1e-12
    q /= np.sum(q) + 1e-12
    return float(np.mean(np.abs(np.cumsum(p) - np.cumsum(q))))


def contour_hu_distance(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 99
    c = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    peri = float(cv2.arcLength(c, True))
    if area < 10 or peri < 5:
        return 99
    x, y, w, h = cv2.boundingRect(c)
    tmpl = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(tmpl, (w // 2, h // 2), (max(2, w // 2 - 1), max(2, h // 2 - 1)), 0, 0, 360, 255, -1)
    tmpl_cnts, _ = cv2.findContours(tmpl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not tmpl_cnts:
        return 99
    return float(cv2.matchShapes(c, tmpl_cnts[0], cv2.CONTOURS_MATCH_I1, 0))


def compute_circularity(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0
    c = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    peri = float(cv2.arcLength(c, True))
    if peri < 1e-06:
        return 0
    return float(4 * np.pi * area / (peri * peri))


# ---------------------------------------------------------------------------
# Pectoral line detection
# ---------------------------------------------------------------------------
def detect_pectoral_line(gray, mask):
    """粗略检测 MLO 位胸肌线。
    返回 (found, line_count)。若检测不到，不一定说明崩坏（CC 位本就无），只是 A3 不加分。
    """
    if int(np.count_nonzero(mask)) < 500:
        return (False, 0)
    h, w = gray.shape
    left_density = float(np.count_nonzero(mask[:, :w // 3]))
    right_density = float(np.count_nonzero(mask[:, -w // 3:]))
    if left_density >= right_density:
        roi = gray[:, :w // 3]
    else:
        roi = gray[:, -w // 3:]
    _, high = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(high, 50, 150)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=60,
                            minLineLength=max(80, h // 6), maxLineGap=12)
    if lines is None:
        return (False, 0)
    ok = 0
    for ln in lines.reshape(-1, 4):
        x1, y1, x2, y2 = ln
        dx = x2 - x1
        dy = y2 - y1
        ang = np.degrees(np.arctan2(dy, dx + 1e-06))
        if 15 < abs(ang) < 75:
            ok += 1
    return (ok >= 2, int(ok))


# ---------------------------------------------------------------------------
# Entropy / spectral profile / round detection
# ---------------------------------------------------------------------------
def compute_local_entropy(gray, mask):
    """用 5x5 局部方差分布的熵近似「局部对比度混乱度」。"""
    if int(np.count_nonzero(mask)) < 500:
        return 0
    g = gray.astype(np.float32) / 255
    k = (5, 5)
    mean = cv2.blur(g, k)
    mean_sq = cv2.blur(g * g, k)
    local_var = np.clip(mean_sq - mean * mean, 0, None)
    vals = local_var[mask > 0]
    if vals.size < 64:
        return 0
    hist, _ = np.histogram(vals, bins=50, range=(0, float(vals.max() + 1e-06)))
    prob = hist.astype(np.float64)
    s = prob.sum()
    if s <= 0:
        return 0
    prob /= s
    prob = prob[prob > 0]
    return float(-np.sum(prob * np.log(prob)))


def radial_power_profile(gray, mask, bins=64):
    if int(np.count_nonzero(mask)) < 100:
        return np.zeros(bins, dtype=np.float32)
    x = gray.astype(np.float32)
    x = (x - float(x.mean())) / (float(x.std()) + 1e-06)
    x = x * (mask > 0).astype(np.float32)
    f = np.fft.fftshift(np.fft.fft2(x))
    p = (np.abs(f) ** 2).astype(np.float64)
    h, w = p.shape
    yy, xx = np.indices((h, w))
    cx = w / 2
    cy = h / 2
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rr = rr / (np.max(rr) + 1e-06)
    edges = np.linspace(0, 1, bins + 1)
    prof = np.zeros(bins, dtype=np.float64)
    for i in range(bins):
        m = (rr >= edges[i]) & (rr < edges[i + 1])
        if not np.any(m):
            continue
        prof[i] = float(np.mean(p[m]))
    prof = np.log1p(prof)
    s = float(np.sum(prof))
    if s <= 1e-12:
        return np.zeros(bins, dtype=np.float32)
    return (prof / s).astype(np.float32)


def detect_isolated_round(gray, mask):
    """在乳腺区内找近圆形、20~800px 的亮连通域。

    返回 (suspicious_round, isolated_round, round_total_area, round_density):
      - suspicious_round: 圆形亮连通域数量
      - isolated_round:   其中孤立（60px 内无同类）的数量（D1 用）
      - round_total_area: 所有圆形亮连通域总面积（像素）
      - round_density:    round_total_area / breast_mask_area  （D4 用，密集伪影/气泡阵列指标）
    """
    mask_area = int(np.count_nonzero(mask))
    if mask_area < 500:
        return (0, 0, 0, 0)
    vals = gray[mask > 0]
    if vals.size == 0:
        return (0, 0, 0, 0)
    thr = float(np.percentile(vals, 85))
    bright = ((gray >= thr) & (mask > 0)).astype(np.uint8) * 255
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(bright, connectivity=8)
    round_idx = []
    centers = []
    total_area = 0
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 20 or area > 800:
            continue
        x, y, w, h, _ = stats[i]
        if w < 3 or h < 3:
            continue
        aspect = float(min(w, h)) / float(max(w, h))
        if aspect < 0.6:
            continue
        round_idx.append(i)
        centers.append((float(cents[i, 0]), float(cents[i, 1])))
        total_area += area
    density = float(total_area) / float(max(1, mask_area))
    if not round_idx:
        return (0, 0, 0, density)
    isolated = 0
    if len(centers) == 1:
        isolated = 1
    else:
        pts = np.asarray(centers, dtype=np.float32)
        for i, c in enumerate(pts):
            d = np.linalg.norm(pts - c, axis=1)
            d[i] = 1e+09
            if float(d.min()) > 60:
                isolated += 1
    return (len(round_idx), isolated, int(total_area), density)


# ---------------------------------------------------------------------------
# Contour / edge / banding / vertical structure / grid detection
# ---------------------------------------------------------------------------
def compute_contour_irregularity(mask):
    """乳腺外轮廓的「断裂/锯齿」量化。

    - concavity_depth_norm: 最大凸性缺陷深度 / sqrt(mask_area)。
      真实下垂乳房外缘基本是凸的，MLO 位略有胸肌方向的小凹陷；一旦生成图里出现
      「被咬一口」式的深凹陷，这个值会显著变大。归一化到 sqrt(面积) 消除分辨率影响。
    - perimeter_ratio: 实际轮廓周长 / 凸包周长。
      凸曲线约 1.0；锯齿/震荡越剧烈，比值越大。

    两者都用「真实图 p99」作硬阈值，超过任一即 CONTOUR_FRACTURED。
    """
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return (0, 1)
    c = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    if area < 100:
        return (0, 1)
    peri = float(cv2.arcLength(c, True))
    try:
        hull_pts = cv2.convexHull(c, returnPoints=True)
        hull_peri = float(cv2.arcLength(hull_pts, True))
    except Exception:
        hull_peri = peri
    perimeter_ratio = float(peri / max(1e-06, hull_peri))
    concavity = 0
    try:
        hull_idx = cv2.convexHull(c, returnPoints=False)
        if hull_idx is not None and len(hull_idx) >= 3 and len(c) >= 4:
            defects = cv2.convexityDefects(c, hull_idx)
            if defects is not None and len(defects) > 0:
                depths = defects[:, 0, 3].astype(np.float64) / 256
                concavity = float(np.max(depths))
    except Exception:
        pass
    concavity_norm = float(concavity / (np.sqrt(area) + 1e-06))
    return (concavity_norm, perimeter_ratio)


def detect_edge_voids(gray, mask):
    if int(np.count_nonzero(mask)) < 500:
        return 0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dil = cv2.dilate(mask, k)
    band = cv2.bitwise_xor(dil, mask)
    low = (gray < 13) & (band > 0)
    return int(np.count_nonzero(low))


def compute_banding_score(gray, mask):
    """检查乳腺区内部的条带/扫描线。越接近 1 越健康。

    只看掩膜内的行列切片，避免纯黑背景把阈值带偏。
    异常行定义：方差 < 0.15 x 掩膜内中位数方差（真·死带），
    或 > 6 x 中位数方差（高亮扫描线）。
    """
    if int(np.count_nonzero(mask)) < 500:
        return 1
    g = gray.astype(np.float32)
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return 1
    y1 = int(ys.max())
    y0 = int(ys.min())
    x1 = int(xs.max())
    x0 = int(xs.min())
    if y1 - y0 < 20 or x1 - x0 < 20:
        return 1
    sub = g[y0:y1 + 1, x0:x1 + 1]
    sub_mask = mask[y0:y1 + 1, x0:x1 + 1] > 0

    def _axis_var(m, mm, axis):
        mm_f = mm.astype(np.float32)
        cnt = mm_f.sum(axis=axis)
        s = (m * mm_f).sum(axis=axis)
        sq = ((m * mm_f) ** 2).sum(axis=axis)
        mean = np.where(cnt > 0, s / np.maximum(cnt, 1e-06), 0)
        var = np.where(cnt > 0, sq / np.maximum(cnt, 1e-06) - mean * mean, 0)
        var = np.clip(var, 0, None)
        var[cnt < 5] = np.nan
        return var

    row_var = _axis_var(sub, sub_mask, axis=1)
    col_var = _axis_var(sub, sub_mask, axis=0)
    row_valid = row_var[~np.isnan(row_var)]
    col_valid = col_var[~np.isnan(col_var)]
    if row_valid.size < 10 or col_valid.size < 10:
        return 1
    r_med = float(np.median(row_valid))
    c_med = float(np.median(col_valid))
    r_dead = int(np.sum(row_valid < 0.15 * r_med))
    c_dead = int(np.sum(col_valid < 0.15 * c_med))
    r_spike = int(np.sum(row_valid > 6 * r_med))
    c_spike = int(np.sum(col_valid > 6 * c_med))
    total = r_dead + c_dead + r_spike + c_spike
    denom = max(20, 0.05 * (row_valid.size + col_valid.size))
    return float(max(0, 1 - total / denom))


def compute_vertical_structure_score(gray, mask):
    """
    检测乳腺内部是否存在异常的竖向高亮结构（脊柱幻觉）。

    方法：
      1. 提取前景内的高亮区域（top 15% 像素）
      2. 用 PCA 分析高亮区的主轴方向
      3. 若主轴为竖向（角度接近 90 度）且细长比 > 3，判为竖向结构
      4. 同时检查中心列占比：真实乳腺高密度区多偏向一侧，居中竖条高度可疑

    返回 0~1，越低越可疑（1=健康，0=明显竖向高亮结构）。
    """
    if int(np.count_nonzero(mask)) < 500:
        return 1
    g = gray.astype(np.float32)
    fg = g[mask > 0]
    thresh = float(np.percentile(fg, 85))
    bright_mask = ((g >= thresh) & (mask > 0)).astype(np.uint8)
    n_bright = int(np.count_nonzero(bright_mask))
    if n_bright < 50:
        return 1
    coords = np.column_stack(np.where(bright_mask > 0)).astype(np.float32)
    mean = coords.mean(axis=0)
    centered = coords - mean
    cov = np.cov(centered.T)
    if cov.ndim < 2:
        return 1
    eigvals, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, -1]
    angle_deg = float(abs(np.degrees(np.arctan2(abs(major[1]), abs(major[0] + 1e-09)))))
    eigvals = np.maximum(eigvals, 1e-06)
    elongation = float(np.sqrt(eigvals[-1] / eigvals[-2])) if eigvals[-2] > 0 else 1
    H, W = gray.shape
    cx = W // 2
    center_strip_w = W // 6
    center_bright = int(np.sum(bright_mask[:, max(0, cx - center_strip_w):cx + center_strip_w]))
    center_ratio = center_bright / max(1, n_bright)
    vert_score = max(0, (angle_deg - 60) / 30)
    elong_score = max(0, (elongation - 2) / 4)
    center_score = max(0, (center_ratio - 0.3) / 0.4)
    danger = vert_score * 0.4 + elong_score * 0.4 + center_score * 0.2
    return float(max(0, 1 - min(1, danger)))


def compute_grid_pattern_score(gray, mask):
    """
    检测乳腺内部是否存在规则格子/蜂窝状纹理（latent grid lock 幻觉）。

    方法：列采样归一化自相关（NACF），只有极强周期信号（NACF > 0.60）才触发。
    - 真实乳腺纹理为随机分形，NACF 快速衰减，不存在强二次峰
    - 蜂窝/格子纹理在固定间距出现多个 NACF > 0.60 的强峰
    - 需要 >= 3/6 采样行同时满足条件，避免偶发伪峰
    - 纯软指标（记录分值）；GRID_ARTIFACT tag 已移出 Hard Tag 列表

    返回 0~1，越低越有周期性（1=健康，0=极强格子）。
    """
    if int(np.count_nonzero(mask)) < 500:
        return 1
    g = gray.astype(np.float32)
    g_masked = g * (mask > 0).astype(np.float32)
    H2, W2 = gray.shape
    nacf_periodic_rows = 0
    total_rows_checked = 0
    for row_frac in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        r = int(row_frac * H2)
        row_fg = g_masked[r].astype(np.float64)
        nz = np.count_nonzero(row_fg)
        if nz < 48:
            continue
        total_rows_checked += 1
        row_fg = row_fg - row_fg[row_fg != 0].mean()
        var = float(np.sum(row_fg ** 2))
        if var < 1:
            continue
        acorr = np.correlate(row_fg, row_fg, mode='full')
        mid = len(acorr) // 2
        nacf = acorr[mid:] / var
        nacf = nacf[16:]
        if len(nacf) < 32:
            continue
        strong_peaks = np.where(
            (nacf[1:-1] > 0.6) & (nacf[1:-1] > nacf[:-2]) & (nacf[1:-1] > nacf[2:])
        )[0]
        if len(strong_peaks) >= 2:
            nacf_periodic_rows += 1
    if total_rows_checked < 2:
        return 1
    periodic_ratio = nacf_periodic_rows / total_rows_checked
    danger = max(0, (periodic_ratio - 0.5) / 0.5)
    return float(max(0, 1 - min(1, danger)))


# ---------------------------------------------------------------------------
# CLIP anatomy / semantic chaos
# ---------------------------------------------------------------------------
def clip_breast_margin_and_risk(model_id, gray_u8):
    """CLIP 钼靶语义 margin（越大越确信是乳腺钼靶）；risk=1-margin；失败返回 (1.0, 0.0, '')。"""
    if not _CLIP_TR_OK:
        return (100, 0, '')
    if _CLIP_DISABLED_REASON:
        return (100, 0, 'clip_disabled')
    from PIL import Image as _PILImage

    texts = (
        'a mammogram x-ray of breast tissue only, medical grayscale',
        'a chest x-ray with ribs spine and lungs',
        'an x-ray showing fingers hands or wrists',
        'a dental or skull bone x-ray with teeth',
        'multiple overlapping xrays collage abstract medical scan',
        'fine abstract texture noise without anatomy',
    )
    device = _torch_clip.device('cuda' if _torch_clip.cuda.is_available() else 'cpu')
    try:
        global _CLIP_MODEL, _CLIP_PROC, _CLIP_MODEL_ID_LOADED, _CLIP_LABELS_LAST
        if _CLIP_MODEL is None or _CLIP_PROC is None or str(model_id) != _CLIP_MODEL_ID_LOADED:
            print(f'[CLIP anatomy] loading {model_id} ...')
            _CLIP_PROC = _HFCLIPProcessor.from_pretrained(model_id, local_files_only=True)
            _CLIP_MODEL = _HFCLIPModel.from_pretrained(model_id, local_files_only=True).to(device)
            _CLIP_MODEL_ID_LOADED = str(model_id)
            _CLIP_LABELS_LAST = texts
            _CLIP_MODEL.eval()

        rgb = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2RGB)
        inp = _CLIP_PROC(
            images=[_PILImage.fromarray(rgb)],
            text=list(texts),
            return_tensors='pt',
            padding=True,
            truncation=True,
        )
        inp = {k: v.to(device) for k, v in inp.items()}
        with _torch_clip.no_grad():
            out = _CLIP_MODEL(**inp)
        logits = out.logits_per_image.squeeze(0)
        probs = logits.softmax(dim=-1)
        probs_cpu = probs.detach().float().cpu().numpy()
        p0 = float(probs_cpu[0])
        rest_max = float(np.max(probs_cpu[1:])) if probs_cpu.size > 1 else 0
        margin = p0 - rest_max
        risk = float(np.clip(1 - max(0, margin) / 0.18, 0, 1))
        return (margin, risk, '')
    except Exception as e:
        _CLIP_DISABLED_REASON = str(e)
        print(f'[CLIP anatomy] 跳过并禁用本批 CLIP: {e}')
        return (100, 0, 'clip_disabled')


def compute_anatomy_semantic_chaos(gray, mask, mirror_sim, hu_dist, edge_density, clip_risk=0.0):
    """检测「非钼靶」式内容与全局拼图感（胸片/手足幻觉、左右镜像复制等）。

    组合：(1) 碎片化 (2) 镜像/Hu/Canny、(3) 可选 CLIP 跨器官风险 clip_risk。
    anatomy_chaos 属于 [0,1]，越大越可疑。
    """
    dbg = {}

    def _count_elong(binary, amin, amax, ar_thresh):
        n_c, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        cnt = 0
        for i in range(1, n_c):
            area_i = int(stats[i, cv2.CC_STAT_AREA])
            if area_i < amin or area_i > amax:
                continue
            w = max(1, int(stats[i, cv2.CC_STAT_WIDTH]))
            h = max(1, int(stats[i, cv2.CC_STAT_HEIGHT]))
            asp = float(min(w, h)) / float(max(w, h))
            if asp < ar_thresh:
                cnt += 1
        return cnt

    mask_area = int(np.count_nonzero(mask))
    if mask_area < 500:
        dbg = {'elong_bright': 0, 'elong_dark': 0, 'edge_mid_cc': 0, 'tiny_bright_cc': 0}
        return (0, dbg)

    sqrt_ma = float(np.sqrt(mask_area))
    vals = gray[mask > 0]
    # Bright elongated segments (bone-like, "rib" or "vertebra" hallucination)
    thr_hi = max(1, int(np.percentile(vals, 87)))
    bright_bin = ((gray >= thr_hi) & (mask > 0)).astype(np.uint8) * 255
    bright_bin = cv2.morphologyEx(bright_bin, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    # area range: [13, 255] pixels, aspect ratio < 0.26
    elong_bright = _count_elong(bright_bin, 13, 255, 0.26)

    # Dark elongated segments
    thr_lo = max(1, int(np.percentile(vals, 18)))
    dark_bin = ((gray <= thr_lo) & (mask > 0)).astype(np.uint8) * 255
    dark_bin = cv2.morphologyEx(dark_bin, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    elong_dark = _count_elong(dark_bin, 18, 380, 0.26)

    # Tiny bright spots (mosaic or sand-like)
    # count components with area <=140 within the bright_bin
    _, _, stats_b, _ = cv2.connectedComponentsWithStats(bright_bin, connectivity=8)
    tiny_bright = sum(1 for i in range(1, len(stats_b)) if int(stats_b[i, cv2.CC_STAT_AREA]) <= 140)

    # Mid-frequency edge count in Canny
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 45, 110)
    masked = edges & (mask > 0).astype(np.uint8)
    edge_mid_n, _, stats_e, _ = cv2.connectedComponentsWithStats(masked, connectivity=8)
    edge_mid_cc = sum(
        1 for i in range(1, edge_mid_n)
        if 45 <= int(stats_e[i, cv2.CC_STAT_AREA]) <= 5200
    )
    # The edge density fraction among those mid-frequency edges
    bridge_px = sum(
        int(stats_e[i, cv2.CC_STAT_AREA])
        for i in range(1, edge_mid_n)
        if 45 <= int(stats_e[i, cv2.CC_STAT_AREA]) <= 5200
    )
    n_e = bridge_px / max(1, mask_area)
    # ratio of "mid-frequency edge components" occupying >0.025 of mask
    edge_mid_dense = sum(
        1 for i in range(1, edge_mid_n)
        if 45 <= int(stats_e[i, cv2.CC_STAT_AREA]) <= 5200
        and int(stats_e[i, cv2.CC_STAT_AREA]) / max(1, mask_area) > 0.025
    )

    # Composite scores
    t_frag = min(1, (elong_bright * 2.2 + elong_dark * 1.5 + tiny_bright * 0.3) / max(1, sqrt_ma))
    t_mirror_raw = max(0, 0.5 - mirror_sim) * 2
    t_hu = min(1, hu_dist / 8)
    t_edge_d = min(1, n_e * 3)
    fuse = t_frag * 0.35 + t_mirror_raw * 0.20 + t_hu * 0.15 + t_edge_d * 0.10 + clip_risk * 0.20
    # Mirror boost
    t_mirror = max(0, 0.55 - mirror_sim) * 3
    mir_boost = max(0, t_mirror - 0.4) * 0.15
    chaos_geo = min(1, fuse + mir_boost)
    anatomy_chaos = float(np.clip(chaos_geo, 0, 1))
    dbg = {
        'elong_bright': elong_bright,
        'elong_dark': elong_dark,
        'edge_mid_cc': edge_mid_cc,
        'tiny_bright_cc': tiny_bright,
    }
    return (anatomy_chaos, dbg)


# ---------------------------------------------------------------------------
# Power spectrum slope / BRISQUE / histogram / gradient helpers
# ---------------------------------------------------------------------------
def compute_ps_slope(gray, mask):
    """计算功率谱斜率 beta（1/f^beta 定律）。

    真实钼靶图功率谱服从幂律分布，对数-对数空间下斜率 beta 约 2.0~3.5。
    - beta 过高（>4）：纹理过于平滑（对应 TOO_UNIFORM）
    - beta 过低（<1.5）：高频噪声过多（对应 TOO_NOISY）
    参考：Burgess et al. (1999) Medical Physics 26(4)，确认乳腺钼靶图功率谱 beta 约 3。
    """
    if int(np.count_nonzero(mask)) < 500:
        return 0
    x = gray.astype(np.float32)
    x = (x - float(x.mean())) / (float(x.std()) + 1e-06)
    x = x * (mask > 0).astype(np.float32)
    f = np.fft.fftshift(np.fft.fft2(x))
    p = (np.abs(f) ** 2).astype(np.float64)
    h, w = p.shape
    yy, xx = np.indices((h, w))
    cx = w / 2
    cy = h / 2
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).flatten()
    p_flat = p.flatten()
    min_r = max(2, 0.02 * min(h, w))
    max_r = 0.45 * min(h, w)
    sel = (rr >= min_r) & (rr <= max_r)
    if np.sum(sel) < 50:
        return 0
    try:
        slope, _ = np.polyfit(np.log(rr[sel] + 1e-06), np.log(p_flat[sel] + 1e-06), 1)
        return float(-slope)
    except Exception:
        return 0


def compute_brisque(gray):
    """计算 BRISQUE 无参考图像质量分（0=最好，100=最差）。

    依赖 piq 库；不可用时返回 -1.0（表示跳过）。
    改为在乳腺 mask 内计算，排除 letterbox 纯黑背景的虚假边缘对 MSCN 系数的干扰。
    """
    if not _PIQ_OK:
        return -1
    try:
        mask = build_mask(gray)
        if np.count_nonzero(mask) < 500:
            return -1
        ys, xs = np.where(mask > 0)
        y1 = int(ys.max()) + 1
        y0 = int(ys.min())
        x1 = int(xs.max()) + 1
        x0 = int(xs.min())
        y0 = max(0, y0 - 8)
        y1 = min(gray.shape[0], y1 + 8)
        x0 = max(0, x0 - 8)
        x1 = min(gray.shape[1], x1 + 8)
        cropped = gray[y0:y1, x0:x1].copy()
        crop_mask = mask[y0:y1, x0:x1]
        cropped[crop_mask == 0] = 0
        t = _torch.tensor(cropped, dtype=_torch.float32).unsqueeze(0).unsqueeze(0) / 255
        score = float(_piq.brisque(t, data_range=1, reduction='none')[0])
        return min(100, max(0, score))
    except Exception:
        return -1


def normalized_histogram(breast_vals_u8, bins=HIST_BINS):
    if breast_vals_u8.size == 0:
        return np.zeros(bins, dtype=np.float32)
    v = breast_vals_u8.astype(np.float32) / 255
    hist, _ = np.histogram(v, bins=bins, range=(0, 1), density=True)
    s = float(hist.sum())
    if s <= 1e-12:
        return np.zeros(bins, dtype=np.float32)
    return (hist / s).astype(np.float32)


def normalized_gradient_hist(gray, mask, bins=64):
    """乳腺区梯度幅值分布直方图。补抓「过平滑/假塑料纹理」."""
    if int(np.count_nonzero(mask)) < 500:
        return np.zeros(bins, dtype=np.float32)
    g = gray.astype(np.float32) / 255
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    vals = mag[mask > 0]
    if vals.size < 64:
        return np.zeros(bins, dtype=np.float32)
    vmax = max(float(np.percentile(vals, 99.5)), 1e-06)
    vals = np.clip(vals / vmax, 0, 1)
    hist, _ = np.histogram(vals, bins=bins, range=(0, 1), density=False)
    hist = hist.astype(np.float32)
    s = float(hist.sum())
    if s <= 1e-12:
        return np.zeros(bins, dtype=np.float32)
    return hist / s


def density_type(mean_norm, std_norm):
    """非常粗的 BI-RADS 风格四分型，仅基于乳腺区均值/方差。用于 E2 匹配度的分桶。"""
    if mean_norm < 0.33 and std_norm < 0.1:
        return 'a_fatty'
    if mean_norm < 0.45 and std_norm < 0.12:
        return 'b_scattered'
    if std_norm >= 0.14 or mean_norm >= 0.55:
        if std_norm >= 0.16 and mean_norm >= 0.55:
            return 'd_extreme'
        return 'c_hetero'
    return None


# ---------------------------------------------------------------------------
# extract_metrics: 全量指标提取，返回 dict
# ---------------------------------------------------------------------------
def extract_metrics(gray):
    mask = build_mask(gray)
    total_px = mask.size
    mask_ratio = float(np.count_nonzero(mask) / max(1, total_px))
    vals = gray[mask > 0]
    empty_mask = vals.size == 0
    mean_norm = float(vals.mean() / 255) if not empty_mask else 0
    std_norm = float(vals.std() / 255) if not empty_mask else 0

    if empty_mask:
        dr = 0
        bright_ratio = 1
        dark_ratio = 1
    else:
        v_norm = vals.astype(np.float32) / 255
        p1, p99 = np.percentile(v_norm, [1, 99])
        dr = float(p99 - p1)
        bright_ratio = float((v_norm > 0.9).sum() / v_norm.size)
        dark_ratio = float((v_norm < 0.1).sum() / v_norm.size)

    if empty_mask:
        cavity_ratio = 1
        bright_spots = 999
    else:
        low_thr = np.percentile(vals, 8)
        low = ((gray <= low_thr) & (mask > 0)).astype(np.uint8) * 255
        n, _, stats, _ = cv2.connectedComponentsWithStats(low, connectivity=8)
        largest_low = int(np.max(stats[1:, cv2.CC_STAT_AREA])) if n > 1 else 0
        cavity_ratio = float(largest_low / max(1, np.count_nonzero(mask)))
        hi_thr = np.percentile(vals, 99.5)
        hi = ((gray >= hi_thr) & (mask > 0)).astype(np.uint8) * 255
        n2, _, stats2, _ = cv2.connectedComponentsWithStats(hi, connectivity=8)
        bright_spots = 0
        if n2 > 1:
            for i in range(1, n2):
                a = int(stats2[i, cv2.CC_STAT_AREA])
                if 5 <= a <= 220:
                    bright_spots += 1

    h, w = gray.shape
    left = gray[:, :w // 2].astype(np.float32)
    right = cv2.flip(gray[:, w - w // 2:], 1).astype(np.float32)
    if left.shape == right.shape:
        l = (left - left.mean()) / (left.std() + 1e-06)
        r = (right - right.mean()) / (right.std() + 1e-06)
        mirror_sim = float((l * r).mean())
    else:
        mirror_sim = 0

    edges = cv2.Canny(gray, 80, 160)
    edge_density = float(np.count_nonzero((edges > 0) & (mask > 0)) / max(1, np.count_nonzero(mask)))
    hu_dist = contour_hu_distance(mask)
    circ = compute_circularity(mask)
    ps_profile = radial_power_profile(gray, mask, bins=64)
    local_ent = compute_local_entropy(gray, mask)
    round_total, round_isolated, round_total_area, round_density = detect_isolated_round(gray, mask)
    edge_voids = detect_edge_voids(gray, mask)
    banding = compute_banding_score(gray, mask)
    vert_struct = compute_vertical_structure_score(gray, mask)
    grid_pattern = compute_grid_pattern_score(gray, mask)
    clip_margin = 100
    clip_risk = 0
    if _REVIEW_CLIP_MODEL_ID:
        clip_margin, clip_risk, _ = clip_breast_margin_and_risk(_REVIEW_CLIP_MODEL_ID, gray)
    anatomy_chaos, anatomy_dbg = compute_anatomy_semantic_chaos(
        gray, mask, mirror_sim, hu_dist, edge_density, clip_risk=clip_risk,
    )
    pect_found, pect_lines = detect_pectoral_line(gray, mask)
    contour_concavity, contour_perim_ratio = compute_contour_irregularity(mask)
    hist = normalized_histogram(vals, bins=HIST_BINS)
    grad_hist = normalized_gradient_hist(gray, mask, bins=64)
    d_type = density_type(mean_norm, std_norm)
    ps_slope = compute_ps_slope(gray, mask)
    brisque_score = compute_brisque(gray)

    return {
        'mask_ratio': mask_ratio,
        'circularity': circ,
        'contour_concavity': float(contour_concavity),
        'contour_perim_ratio': float(contour_perim_ratio),
        'pectoral_found': bool(pect_found),
        'pectoral_lines': int(pect_lines),
        'mean_intensity': mean_norm,
        'std_intensity': std_norm,
        'dynamic_range': dr,
        'bright_ratio': bright_ratio,
        'dark_ratio': dark_ratio,
        'local_entropy': local_ent,
        'ps_profile': ps_profile,
        'round_total': int(round_total),
        'round_isolated': int(round_isolated),
        'round_total_area': int(round_total_area),
        'round_density': float(round_density),
        'edge_voids_px': int(edge_voids),
        'banding_score': float(banding),
        'vert_struct_score': float(vert_struct),
        'grid_pattern_score': float(grid_pattern),
        'anatomy_chaos': float(anatomy_chaos),
        'elong_bright_segments': int(anatomy_dbg.get('elong_bright', 0)),
        'elong_dark_segments': int(anatomy_dbg.get('elong_dark', 0)),
        'edge_mid_segments': int(anatomy_dbg.get('edge_mid_cc', 0)),
        'tiny_bright_cc': int(anatomy_dbg.get('tiny_bright_cc', 0)),
        'clip_margin': float(clip_margin),
        'clip_risk': float(clip_risk),
        'hist': hist,
        'grad_hist': grad_hist,
        'density_type': d_type,
        'ps_slope': ps_slope,
        'brisque_score': brisque_score,
        'cavity_ratio': cavity_ratio,
        'bright_spots': int(bright_spots),
        'mirror_sim': mirror_sim,
        'edge_density': edge_density,
        'hu_dist': hu_dist,
    }


# ---------------------------------------------------------------------------
# Worker helpers
# ---------------------------------------------------------------------------
def _auto_workers(n_items, user_val):
    if user_val and user_val > 0:
        return min(user_val, max(1, n_items))
    cpu = cpu_count() or 1
    return max(1, min(8, cpu, max(1, n_items)))


def _baseline_worker(args_tuple):
    """并行进程用：读取+缩放+抽取基线所需字段。失败返回 None。"""
    path_str, long_side = args_tuple
    try:
        gray = cv2.imread(path_str, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return None
        if long_side and long_side > 0:
            gray = resize_long_side(gray, long_side)
        m = extract_metrics(gray)
        if m.get('bright_spots', 999) >= 999:
            return None
        return {
            'scalars': [
                m['mask_ratio'], m['mean_intensity'], m['std_intensity'],
                m['edge_density'], m['cavity_ratio'],
                float(m['bright_spots']), m['hu_dist'],
                m['dynamic_range'], m['local_entropy'], m['circularity'],
            ],
            'extras': {
                'bright_ratio': float(m['bright_ratio']),
                'dark_ratio': float(m['dark_ratio']),
                'banding_score': float(m['banding_score']),
                'edge_voids_px': float(m['edge_voids_px']),
                'round_isolated': float(m['round_isolated']),
                'round_density': float(m['round_density']),
                'contour_concavity': float(m['contour_concavity']),
                'contour_perim_ratio': float(m['contour_perim_ratio']),
            },
            'ps': np.asarray(m['ps_profile'], dtype=np.float32),
            'hist': np.asarray(m['hist'], dtype=np.float32),
            'grad_hist': np.asarray(m['grad_hist'], dtype=np.float32),
            'density': m['density_type'],
        }
    except Exception:
        return None


def _review_worker(args_tuple):
    path_str, long_side = args_tuple
    try:
        gray = cv2.imread(path_str, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return None
        if long_side and long_side > 0:
            gray = resize_long_side(gray, long_side)
        m = extract_metrics(gray)
        return m
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Reference statistics / baseline
# ---------------------------------------------------------------------------
def build_reference_stats(real_dir, *, long_side, max_samples, workers, sample_seed):
    if real_dir is None:
        return None
    all_paths = sorted(
        p for p in real_dir.rglob('*')
        if p.is_file() and is_image(p)
    )
    if not all_paths:
        return None

    if max_samples and max_samples > 0 and len(all_paths) > max_samples:
        rng = random.Random(sample_seed)
        paths = rng.sample(all_paths, max_samples)
    else:
        paths = all_paths

    print(f'[baseline] 候选 {len(all_paths)} 张，实际抽样 {len(paths)} 张；长边={long_side}')
    feats = []
    extras_all = []
    ps_all = []
    hist_all = []
    grad_hist_all = []
    density_cnt = {}

    n_workers = _auto_workers(len(paths), workers)
    tasks = [(str(p), long_side) for p in paths]

    if n_workers > 1:
        print(f'[baseline] 并行进程数 = {n_workers}')
        with Pool(processes=n_workers) as pool:
            it = pool.imap_unordered(_baseline_worker, tasks, chunksize=4)
            for res in tqdm(it, total=len(tasks), desc='Build real baseline'):
                if res is None:
                    continue
                feats.append(res['scalars'])
                extras_all.append(res['extras'])
                ps_all.append(res['ps'])
                hist_all.append(res['hist'])
                grad_hist_all.append(res['grad_hist'])
                density_cnt[res['density']] = density_cnt.get(res['density'], 0) + 1
    else:
        for t in tqdm(tasks, desc='Build real baseline'):
            res = _baseline_worker(t)
            if res is None:
                continue
            feats.append(res['scalars'])
            extras_all.append(res['extras'])
            ps_all.append(res['ps'])
            hist_all.append(res['hist'])
            grad_hist_all.append(res['grad_hist'])
            density_cnt[res['density']] = density_cnt.get(res['density'], 0) + 1

    if not feats:
        return None

    arr = np.asarray(feats, dtype=np.float32)
    ps_arr = np.asarray(ps_all, dtype=np.float32) if ps_all else np.zeros((1, 64), dtype=np.float32)
    hist_arr = np.asarray(hist_all, dtype=np.float32) if hist_all else np.zeros((1, HIST_BINS), dtype=np.float32)
    grad_hist_arr = np.asarray(grad_hist_all, dtype=np.float32) if grad_hist_all else np.zeros((1, 64), dtype=np.float32)

    ps_mean = ps_arr.mean(axis=0)
    ps_mean = ps_mean / (ps_mean.sum() + 1e-12)
    hist_mean = hist_arr.mean(axis=0)
    hist_mean = hist_mean / (hist_mean.sum() + 1e-12)
    grad_mean = grad_hist_arr.mean(axis=0)
    grad_mean = grad_mean / (grad_mean.sum() + 1e-12)

    total = sum(density_cnt.values()) or 1
    density_prior = {k: v / total for k, v in density_cnt.items()}

    hist_dists = np.asarray([wasserstein_1d(h, hist_mean) for h in hist_arr], dtype=np.float32)
    ps_dists = np.asarray([emd_1d_profile(p, ps_mean) for p in ps_arr], dtype=np.float32)
    grad_dists = np.asarray([wasserstein_1d(g, grad_mean) for g in grad_hist_arr], dtype=np.float32)

    def _pct(vals):
        vals = np.asarray(vals, dtype=np.float64)
        if vals.size == 0:
            return {'p1': 0, 'p5': 0, 'p50': 0, 'p95': 0, 'p99': 0, 'mean': 0, 'std': 0}
        return {
            'p1': float(np.percentile(vals, 1)),
            'p5': float(np.percentile(vals, 5)),
            'p50': float(np.percentile(vals, 50)),
            'p95': float(np.percentile(vals, 95)),
            'p99': float(np.percentile(vals, 99)),
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals)),
        }

    percentiles = {
        'mask_ratio': _pct(arr[:, 0]),
        'mean_intensity': _pct(arr[:, 1]),
        'std_intensity': _pct(arr[:, 2]),
        'edge_density': _pct(arr[:, 3]),
        'cavity_ratio': _pct(arr[:, 4]),
        'bright_spots': _pct(arr[:, 5]),
        'hu_dist': _pct(arr[:, 6]),
        'dynamic_range': _pct(arr[:, 7]),
        'local_entropy': _pct(arr[:, 8]),
        'circularity': _pct(arr[:, 9]),
        'bright_ratio': _pct(np.asarray([e['bright_ratio'] for e in extras_all])),
        'dark_ratio': _pct(np.asarray([e['dark_ratio'] for e in extras_all])),
        'banding_score': _pct(np.asarray([e['banding_score'] for e in extras_all])),
        'edge_voids_px': _pct(np.asarray([e['edge_voids_px'] for e in extras_all])),
        'round_isolated': _pct(np.asarray([e['round_isolated'] for e in extras_all])),
        'round_density': _pct(np.asarray([e['round_density'] for e in extras_all])),
        'contour_concavity': _pct(np.asarray([e['contour_concavity'] for e in extras_all])),
        'contour_perim_ratio': _pct(np.asarray([e['contour_perim_ratio'] for e in extras_all])),
        'hist_wass': _pct(hist_dists),
        'ps_emd': _pct(ps_dists),
        'grad_wass': _pct(grad_dists),
    }

    return {
        'count': int(arr.shape[0]),
        'feature_names': [
            'mask_ratio', 'mean_intensity', 'std_intensity', 'edge_density',
            'cavity_ratio', 'bright_spots', 'hu_dist', 'dynamic_range',
            'local_entropy', 'circularity',
        ],
        'mean': arr.mean(axis=0),
        'std': arr.std(axis=0) + 1e-06,
        'ps_mean': ps_mean,
        'hist_mean': hist_mean,
        'grad_mean': grad_mean,
        'density_prior': density_prior,
        'percentiles': percentiles,
    }


# ---------------------------------------------------------------------------
# Baseline JSON I/O
# ---------------------------------------------------------------------------
def load_baseline_json(path):
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding='utf-8'))
    try:
        return {
            'count': int(data['count']),
            'feature_names': list(data['feature_names']),
            'mean': np.asarray(data['mean'], dtype=np.float32),
            'std': np.asarray(data['std'], dtype=np.float32) + 1e-06,
            'ps_mean': np.asarray(data.get('ps_mean', []), dtype=np.float32),
            'hist_mean': np.asarray(data.get('hist_mean', []), dtype=np.float32),
            'grad_mean': np.asarray(data.get('grad_mean', []), dtype=np.float32),
            'density_prior': dict(data.get('density_prior', {})),
            'percentiles': dict(data.get('percentiles', {})),
        }
    except KeyError:
        return None


def save_baseline_json(ref, path, args):
    """Save reference stats as JSON for reuse."""
    if ref is None:
        return
    out = {
        'count': ref['count'],
        'feature_names': ref['feature_names'],
        'mean': ref['mean'].tolist(),
        'std': ref['std'].tolist(),
        'ps_mean': ref.get('ps_mean', np.zeros(64, dtype=np.float32)).tolist(),
        'hist_mean': ref.get('hist_mean', np.zeros(HIST_BINS, dtype=np.float32)).tolist(),
        'grad_mean': ref.get('grad_mean', np.zeros(64, dtype=np.float32)).tolist(),
        'density_prior': ref.get('density_prior', {}),
        'percentiles': ref.get('percentiles', {}),
        'args': {
            'resize_long_side': args.resize_long_side,
            'max_baseline_samples': args.max_baseline_samples,
            'sample_seed': args.sample_seed,
        },
    }
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def apply_calibration(args, ref):
    """用真实分布的 percentiles 自动覆写阈值/中心值。

    设计：硬阈值（触发 tag）用 p1/p99，只把真实分布里最离群的 ~2% 判违规；
    软评分斜率（0-1 分）用 p5/p95 的宽度，生成图有坏值会被连续扣分。
    这样真实图自检通过率 >= 95%，而生成图里的结构性偏差仍能被稳稳抓到。
    """
    if ref is None:
        return args, {}
    pct = ref.get('percentiles', {})
    if not pct:
        return args, {}
    changes = {}
    pct_mask = pct.get('mask_ratio', {})
    if 'p1' in pct_mask and 'p99' in pct_mask and 'p5' in pct_mask and 'p95' in pct_mask:
        args.min_mask_ratio = float(pct_mask['p1'])
        args.max_mask_ratio = float(pct_mask['p99'])
        setattr(args, '_cal_mask_soft_lo', float(pct_mask['p5']))
        setattr(args, '_cal_mask_soft_hi', float(pct_mask['p95']))
        changes['mask_ratio'] = (f"p1={pct_mask['p1']:.4g}", f"p99={pct_mask['p99']:.4g}")
    pct_circ = pct.get('circularity', {})
    if 'p1' in pct_circ and 'p99' in pct_circ and 'p5' in pct_circ and 'p95' in pct_circ:
        args.min_circularity = max(0.1, float(pct_circ['p1']))
        args.max_circularity = float(pct_circ['p99'])
        setattr(args, '_cal_circ_soft_lo', float(pct_circ['p5']))
        setattr(args, '_cal_circ_soft_hi', float(pct_circ['p95']))
        changes['circularity'] = (f"p1={pct_circ['p1']:.4g}", f"p99={pct_circ['p99']:.4g}")
    pct_mean = pct.get('mean_intensity', {})
    if 'p1' in pct_mean and 'p99' in pct_mean and 'p5' in pct_mean and 'p95' in pct_mean:
        setattr(args, '_cal_mean_p1', float(pct_mean['p1']))
        setattr(args, '_cal_mean_p5', float(pct_mean['p5']))
        setattr(args, '_cal_mean_p95', float(pct_mean['p95']))
        setattr(args, '_cal_mean_p99', float(pct_mean['p99']))
        changes['mean_intensity'] = (f"p1={pct_mean['p1']:.4g}", f"p99={pct_mean['p99']:.4g}")
        args.max_mean_intensity = min(0.99, float(pct_mean['p99']))
    pct_std = pct.get('std_intensity', {})
    if 'p1' in pct_std and 'p99' in pct_std and 'p5' in pct_std and 'p95' in pct_std:
        setattr(args, '_cal_std_p1', float(pct_std['p1']))
        setattr(args, '_cal_std_p5', float(pct_std['p5']))
        setattr(args, '_cal_std_p95', float(pct_std['p95']))
        setattr(args, '_cal_std_p99', float(pct_std['p99']))
        changes['std_intensity'] = (f"p1={pct_std['p1']:.4g}", f"p99={pct_std['p99']:.4g}")
    pct_dr = pct.get('dynamic_range', {})
    if 'p1' in pct_dr and 'p99' in pct_dr and 'p5' in pct_dr and 'p95' in pct_dr:
        setattr(args, '_cal_dr_p1', float(pct_dr['p1']))
        setattr(args, '_cal_dr_p5', float(pct_dr['p5']))
        setattr(args, '_cal_dr_p95', float(pct_dr['p95']))
        setattr(args, '_cal_dr_p99', float(pct_dr['p99']))
        changes['dynamic_range'] = (f"p1={pct_dr['p1']:.4g}", f"p99={pct_dr['p99']:.4g}")
    pct_ent = pct.get('local_entropy', {})
    if 'p1' in pct_ent and 'p99' in pct_ent and 'p5' in pct_ent and 'p95' in pct_ent:
        setattr(args, '_cal_ent_p1', float(pct_ent['p1']))
        setattr(args, '_cal_ent_p5', float(pct_ent['p5']))
        setattr(args, '_cal_ent_p95', float(pct_ent['p95']))
        setattr(args, '_cal_ent_p99', float(pct_ent['p99']))
        changes['local_entropy'] = (f"p1={pct_ent['p1']:.4g}", f"p99={pct_ent['p99']:.4g}")
    pct_a4_c = pct.get('contour_concavity', {})
    if 'p99' in pct_a4_c:
        scale = float(pct_a4_c['p99'])
        if scale > 0:
            setattr(args, '_cal_a4_concavity_scale', scale)
            args.max_contour_concavity = scale
            changes['contour_concavity'] = (f"p99={scale:.4g}",)
    pct_a4_p = pct.get('contour_perim_ratio', {})
    if 'p99' in pct_a4_p:
        p_scale = float(pct_a4_p['p99'])
        if p_scale > 1:
            setattr(args, '_cal_a4_perim_scale', p_scale)
            args.max_contour_perim_ratio = p_scale
            changes['contour_perim_ratio'] = (f"p99={p_scale:.4g}",)
    pct_d4 = pct.get('round_density', {})
    if 'p99' in pct_d4:
        d4 = float(pct_d4['p99'])
        if d4 > 0:
            setattr(args, '_cal_d4_density_scale', d4)
            args.max_round_density = d4
            changes['round_density'] = (f"p99={d4:.4g}",)
    return args, changes


# ---------------------------------------------------------------------------
# Grade functions A1-F2
# ---------------------------------------------------------------------------
def _clip01(x):
    return float(max(0, min(1, x)))


def grade_a1(mask_ratio, args):
    hard_hi = args.max_mask_ratio
    hard_lo = args.min_mask_ratio
    soft_lo = getattr(args, '_cal_mask_soft_lo', hard_lo)
    soft_hi = getattr(args, '_cal_mask_soft_hi', hard_hi)
    if soft_lo <= mask_ratio <= soft_hi:
        return (1, '')
    if hard_lo <= mask_ratio <= hard_hi:
        return (0.6, 'AREA_BORDER')
    return (0, 'AREA_BAD')


def grade_a2(circ, args):
    hard_hi = args.max_circularity
    hard_lo = args.min_circularity
    soft_lo = getattr(args, '_cal_circ_soft_lo', hard_lo)
    soft_hi = getattr(args, '_cal_circ_soft_hi', hard_hi)
    center = (soft_lo + soft_hi) / 2
    if soft_lo <= circ <= soft_hi:
        return (1, '')
    if hard_lo <= circ <= hard_hi:
        if circ < soft_lo:
            return (_clip01(1 - abs(circ - center) * 3), 'SHAPE_ODD')
        return (_clip01(1 - abs(circ - center) * 3), 'SHAPE_ODD')
    return (_clip01(1 - abs(circ - center) * 3), 'SHAPE_ODD')


def grade_a3(found, args):
    if found:
        return (1, '')
    if args.require_pectoral:
        return (0, 'NO_PECTORAL')
    return (0.7, '')


def grade_a4(concavity, perim_ratio, args):
    """轮廓断裂/锯齿检测。两个子指标任一超限即 CONTOUR_FRACTURED（hard tag）。

    - concavity: 最大凸性缺陷深度 / sqrt(area)。真实约 0.05~0.20，生成图被「咬一口」时 >> 0.30。
    - perim_ratio: 周长 / 凸包周长。凸曲线约 1；锯齿震荡时显著 > 1.2。
    """
    c_scale = max(0.001, getattr(args, '_cal_a4_concavity_scale', args.max_contour_concavity))
    p_scale_hard = max(1.01, getattr(args, '_cal_a4_perim_scale', args.max_contour_perim_ratio))
    c_score = _clip01(1 - concavity / c_scale)
    p_score = _clip01(1 - max(0, perim_ratio - 1) / max(0.001, p_scale_hard - 1))
    score = (c_score + p_score) / 2
    if concavity > args.max_contour_concavity or perim_ratio > args.max_contour_perim_ratio:
        return (score, 'CONTOUR_FRACTURED')
    return (score, '')


def _double_band_score(x, p1, p5, p95, p99):
    """双档区间评分：[p5,p95] 为健康带给满分，[p1,p5] 与 [p95,p99] 线性衰减到 0.3，
    超出 [p1,p99] 给 <=0 分。替代原来"偏离 p50 就线性扣分"的陡峭函数，避免误杀真实图。
    """
    if p5 <= x <= p95:
        return 1
    if x < p5:
        if x >= p1:
            return 0.3 + 0.7 * (x - p1) / max(1e-06, p5 - p1)
        return max(0, 0.3 - 0.3 * (p1 - x) / max(1e-06, p5 - p1))
    if x <= p99:
        return 0.3 + 0.7 * (p99 - x) / max(1e-06, p99 - p95)
    return max(0, 0.3 - 0.3 * (x - p99) / max(1e-06, p99 - p95))


def grade_b1(mean, std, args):
    mean_score = _double_band_score(
        mean,
        getattr(args, '_cal_mean_p1', 0.1),
        getattr(args, '_cal_mean_p5', 0.2),
        getattr(args, '_cal_mean_p95', 0.7),
        getattr(args, '_cal_mean_p99', 0.85),
    )
    std_score = _double_band_score(
        std,
        getattr(args, '_cal_std_p1', 0.03),
        getattr(args, '_cal_std_p5', 0.05),
        getattr(args, '_cal_std_p95', 0.21),
        getattr(args, '_cal_std_p99', 0.28),
    )
    score = (mean_score + std_score) / 2
    if std < args.min_std_intensity:
        return (score, 'LOW_CONTRAST')
    if mean > args.max_mean_intensity:
        return (score, 'OVEREXPOSED')
    return (score, '')


def grade_b2(dr, args):
    score = _double_band_score(
        dr,
        getattr(args, '_cal_dr_p1', 0.25),
        getattr(args, '_cal_dr_p5', 0.35),
        getattr(args, '_cal_dr_p95', 0.95),
        getattr(args, '_cal_dr_p99', 1),
    )
    if dr < args.min_dynamic_range:
        return (score, 'LOW_DR')
    if dr > args.max_dynamic_range:
        return (score, 'OVER_STRETCHED')
    return (score, '')


def grade_b3(bright_ratio, dark_ratio, args):
    b_budget = max(0.0001, args.max_bright_ratio)
    d_budget = max(0.0001, args.max_dark_ratio)
    bs = _clip01(1 - max(0, bright_ratio - 0.5 * b_budget) / b_budget)
    ds = _clip01(1 - max(0, dark_ratio - 0.5 * d_budget) / d_budget)
    score = bs * ds
    tag = ''
    if bright_ratio > args.max_bright_ratio:
        tag = 'BLOWOUT'
        return (score, tag)
    if dark_ratio > args.max_dark_ratio:
        tag = 'DEAD_DARK'
    return (score, tag)


def grade_c1(ps_profile, ref_ps, args):
    """功率谱 EMD (RAPS)：径向功率谱分布与基线均值之间的 EMD。"""
    if ref_ps is None or len(ref_ps) == 0:
        return (1, '', 0)
    ps_profile = np.asarray(ps_profile, dtype=np.float64)
    ref_ps = np.asarray(ref_ps, dtype=np.float64)
    if ps_profile.sum() <= 0 or ref_ps.sum() <= 0:
        return (1, '', 0)
    emd = emd_1d_profile(ps_profile, ref_ps)
    thr = float(args.max_ps_emd)
    if emd <= thr:
        return (_clip01(1 - emd / (thr + 1e-06)), '', emd)
    return (_clip01(1 - emd / (thr + 1e-06)), 'PSD_OFF', emd)


def grade_c2(entropy, args):
    score = _double_band_score(
        entropy,
        getattr(args, '_cal_ent_p1', 0.5),
        getattr(args, '_cal_ent_p5', 1),
        getattr(args, '_cal_ent_p95', 4.5),
        getattr(args, '_cal_ent_p99', 5.5),
    )
    if entropy < args.min_local_entropy:
        return (score, 'TOO_UNIFORM')
    if entropy > args.max_local_entropy:
        return (score, 'TOO_NOISY')
    return (score, '')


def grade_c3(grad_hist, ref_grad, args):
    """梯度直方图 Wasserstein 距离。"""
    if ref_grad is None or len(ref_grad) == 0:
        return (1, '', 0)
    grad_hist = np.asarray(grad_hist, dtype=np.float64)
    ref_grad = np.asarray(ref_grad, dtype=np.float64)
    if grad_hist.sum() <= 0 or ref_grad.sum() <= 0:
        return (1, '', 0)
    wass = wasserstein_1d(grad_hist, ref_grad)
    thr = float(args.max_grad_wass)
    if wass <= thr:
        return (_clip01(1 - wass / (thr + 1e-06)), '', wass)
    return (_clip01(1 - wass / (thr + 1e-06)), 'GRAD_OFF', wass)


def grade_d1(isolated_round, args):
    if isolated_round <= args.max_isolated_round:
        return (1, '')
    extra = isolated_round - args.max_isolated_round
    return (_clip01(1 - extra * 0.2), 'ARTIFACT_BUBBLES')


def grade_d4(round_density, args):
    """密集圆形伪影检测。不论孤立与否：当圆形亮连通域总面积占乳腺比例超过真实 p99，
    判定为「气泡阵列 / 马赛克网格」，触发 ARTIFACT_TRYPOPHOBIA（hard tag）。
    """
    scale = max(0.0001, getattr(args, '_cal_d4_density_scale', args.max_round_density))
    score = _clip01(1 - round_density / scale)
    if round_density > args.max_round_density:
        return (score, 'ARTIFACT_TRYPOPHOBIA')
    return (score, '')


def grade_d2(edge_voids, args):
    scale = max(50, 2 * max(1, args.max_edge_voids))
    score = _clip01(1 - edge_voids / scale)
    tag = 'EDGE_VOIDS' if edge_voids > args.max_edge_voids else ''
    return (score, tag)


def grade_d3(banding_score, args):
    tag = 'BANDING' if banding_score < args.min_banding_score else ''
    return (banding_score, tag)


def grade_a5_vert(vert_struct_score):
    """竖向高亮结构（脊柱幻觉）检测。
    score<0.25 触发软 tag（仅记录，不一票否决；待校准后升为 Hard Tag）。
    """
    tag = 'VERTICAL_SPINE' if vert_struct_score < 0.25 else ''
    return (vert_struct_score, tag)


def grade_d5_grid(grid_pattern_score):
    """周期性格子纹理检测（蜂窝/阶梯幻觉）。
    score<0.30 触发软 tag（仅记录，不一票否决；待校准后升为 Hard Tag）。
    """
    tag = 'GRID_ARTIFACT' if grid_pattern_score < 0.3 else ''
    return (grid_pattern_score, tag)


def grade_a6_anatomy(anatomy_chaos, args):
    """解剖碎片化 / 狭长高对比斑块过多 → 「胸片拼图/手足类」幻觉检测（非 CLIP）。
    anatomy_chaos 越高越离谱；>= anatomy_chaos_hard 打硬伤一票否决。"""
    hard = float(getattr(args, 'anatomy_chaos_hard', 0.58))
    if anatomy_chaos >= hard:
        return (0, 'ANATOMY_NON_MAMMO')
    return (_clip01(1 - anatomy_chaos / max(hard, 1e-06)), '')


def grade_clip_cross(margin, args):
    """CLIP：钼靶 prompt 未明显胜出 → 可能为胸片/手足/拼贴。"""
    if margin > 50:
        return (1, '')
    thr = float(getattr(args, 'clip_anatomy_fail_margin', 0.04))
    if margin < thr:
        return (0, 'CLIP_CROSS_ANATOMY')
    return (_clip01((margin - thr) / max(1e-06, 0.14 - thr)), '')


def grade_e1(hist, ref_hist, args):
    """直方图 Wasserstein 距离。"""
    if ref_hist is None or len(ref_hist) == 0:
        return (1, '', 0)
    hist = np.asarray(hist, dtype=np.float64)
    ref_hist = np.asarray(ref_hist, dtype=np.float64)
    if hist.sum() <= 0 or ref_hist.sum() <= 0:
        return (1, '', 0)
    wass = wasserstein_1d(hist, ref_hist)
    thr = float(args.max_hist_wass)
    if wass <= thr:
        return (_clip01(1 - wass / (thr + 1e-06)), '', wass)
    return (_clip01(1 - wass / (thr + 1e-06)), 'HIST_OFF', wass)


def grade_e2(density_type_gen, density_prior):
    if not density_prior:
        return (1, '')
    prior = float(density_prior.get(density_type_gen, 0))
    if prior >= 0.2:
        return (1, '')
    if prior >= 0.05:
        return (0.6, '')
    return (0.3, 'DENSITY_MISMATCH')


def grade_f1(ps_slope):
    """功率谱斜率 beta 评分。
    真实钼靶 beta 约 2.0~3.5（参考 Burgess 1999, Medical Physics）。
    beta > 4.2：过平滑（纹理缺失）；beta < 1.5：高频噪声主导。
    """
    lo_hard, hi_hard = 1.2, 5
    lo_soft, hi_soft = 2, 3.8
    if ps_slope <= 0:
        return (0.5, '')
    if lo_soft <= ps_slope <= hi_soft:
        return (1, '')
    if lo_hard <= ps_slope <= hi_hard:
        if ps_slope < lo_soft:
            score = 0.4 + 0.6 * (ps_slope - lo_hard) / max(1e-06, lo_soft - lo_hard)
            tag = 'PS_SLOPE_LOW'
        else:
            score = 0.4 + 0.6 * (hi_hard - ps_slope) / max(1e-06, hi_hard - hi_soft)
            tag = 'PS_SLOPE_HIGH'
        return (_clip01(score), tag)
    if ps_slope < lo_hard:
        return (0, 'PS_SLOPE_LOW')
    return (0, 'PS_SLOPE_HIGH')


def grade_f2(brisque):
    """BRISQUE 无参考质量评分（0=最好，100=最差）。

    未安装 piq 时 brisque=-1，返回中性分 0.5。
    SD1.5+LANCZOS 放大校准：<=20 优秀，>70 不可接受，>60 触发 HIGH_BRISQUE。
    """
    if brisque < 0:
        return (0.5, '')
    if brisque <= 20:
        return (1, '')
    if brisque <= 60:
        return (_clip01(1 - ((brisque - 20) / 40) * 0.5), '')
    if brisque <= 70:
        return (_clip01(0.5 - ((brisque - 60) / 10) * 0.5), 'HIGH_BRISQUE')
    return (0, 'HIGH_BRISQUE')


# ---------------------------------------------------------------------------
# score_image: 核心评分函数
# ---------------------------------------------------------------------------
def score_image(m, ref, args):
    """m: extract_metrics() 返回的 dict
    ref: load_baseline_json() / build_reference_stats() 返回的参考统计
    args: argparse.Namespace
    """
    a1, t_a1 = grade_a1(m['mask_ratio'], args)
    a2, t_a2 = grade_a2(m['circularity'], args)
    a3, t_a3 = grade_a3(m['pectoral_found'], args)
    a4, t_a4 = grade_a4(m['contour_concavity'], m['contour_perim_ratio'], args)
    b1, t_b1 = grade_b1(m['mean_intensity'], m['std_intensity'], args)
    b2, t_b2 = grade_b2(m['dynamic_range'], args)
    b3, t_b3 = grade_b3(m['bright_ratio'], m['dark_ratio'], args)

    ref_ps = ref['ps_mean'] if ref else None
    ref_grad = ref['grad_mean'] if ref else None
    ref_hist = ref['hist_mean'] if ref else None
    density_prior = ref.get('density_prior', {}) if ref else {}

    c1, t_c1, ps_emd = grade_c1(m['ps_profile'], ref_ps, args)
    c2, t_c2 = grade_c2(m['local_entropy'], args)
    c3, t_c3, grad_wass = grade_c3(m['grad_hist'], ref_grad, args)
    d1, t_d1 = grade_d1(m['round_isolated'], args)
    d2, t_d2 = grade_d2(m['edge_voids_px'], args)
    d3, t_d3 = grade_d3(m['banding_score'], args)
    d4, t_d4 = grade_d4(m['round_density'], args)
    a5, t_a5 = grade_a5_vert(m.get('vert_struct_score', 1))
    d5, t_d5 = grade_d5_grid(m.get('grid_pattern_score', 1))
    a6, t_a6 = grade_a6_anatomy(float(m.get('anatomy_chaos', 0)), args)
    a7, t_a7 = grade_clip_cross(float(m.get('clip_margin', 100)), args)
    e1, t_e1, hist_wass = grade_e1(m['hist'], ref_hist, args)
    e2, t_e2 = grade_e2(m['density_type'], density_prior)
    f1, t_f1 = grade_f1(m.get('ps_slope', 0))
    f2, t_f2 = grade_f2(m.get('brisque_score', -1))

    group_mean = {
        'A': float(np.mean([a1, a2, a3, a4, a5, a6, a7])),
        'B': float(np.mean([b1, b2, b3])),
        'C': float(np.mean([c1, c2, c3])),
        'D': float(np.mean([d1, d2, d3, d4, d5])),
        'E': float(np.mean([e1, e2])),
        'F': float(np.mean([f1, f2])),
    }

    # Edge density floor cap for Group C
    _ed = float(m.get('edge_density', 1))
    _ed_floor = float(getattr(args, 'edge_density_floor', 0.008))
    if _ed < _ed_floor:
        group_mean['C'] = min(group_mean['C'], 0.5)

    # Weight override
    weights = dict(DEFAULT_GROUP_WEIGHTS)
    if args.weights:
        try:
            weights.update(json.loads(args.weights))
        except Exception:
            pass

    total = 100 * sum(group_mean[k] * weights.get(k, 0) for k in 'ABCDEF')

    # Collect non-empty tags
    tags = [t for t in (
        t_a1, t_a2, t_a3, t_a4, t_a5, t_a6, t_a7,
        t_b1, t_b2, t_b3,
        t_c1, t_c2, t_c3,
        t_d1, t_d2, t_d3, t_d4, t_d5,
        t_e1, t_e2,
        t_f1, t_f2,
    ) if t]

    # Legacy reasons
    legacy = []
    if m['mask_ratio'] < args.min_mask_ratio or m['mask_ratio'] > args.max_mask_ratio:
        legacy.append('mask_ratio')
    if m['cavity_ratio'] > args.max_cavity_ratio:
        legacy.append('large_cavity')
    if m['bright_spots'] > args.max_bright_spots:
        legacy.append('bright_spots')
    if m['mirror_sim'] > args.max_mirror_sim:
        legacy.append('bilateral_symmetry')
    if m['hu_dist'] > args.max_hu_dist:
        legacy.append('shape_hu')

    # Off-real-distribution check
    ref_z_mean_abs = 0
    if ref is not None:
        vec = np.asarray([
            m['mask_ratio'], m['mean_intensity'], m['std_intensity'],
            m['edge_density'], m['cavity_ratio'], float(m['bright_spots']),
            m['hu_dist'], m['dynamic_range'], m['local_entropy'], m['circularity'],
        ], dtype=np.float32)
        z = np.abs((vec - ref['mean']) / ref['std'])
        ref_z_mean_abs = float(z.mean())
        if ref_z_mean_abs > args.max_ref_z:
            legacy.append('off_real_distribution')
        if ps_emd > args.max_ps_emd:
            legacy.append('psd_off_real')

    # Hard tag set and final ok
    hard_tags = frozenset({
        'BANDING', 'BLOWOUT', 'AREA_BAD', 'DEAD_DARK', 'GRID_SEAM',
        'SHAPE_ODD', 'EDGE_VOIDS', 'NO_PECTORAL', 'MODALITY_LOW',
        'ARTIFACT_BUBBLES', 'ANATOMY_NON_MAMMO', 'CONTOUR_FRACTURED',
        'SKIN_LINE_MISSING', 'CENTER_OVEREXPOSED', 'CLIP_CROSS_ANATOMY',
        'ARTIFACT_TRYPOPHOBIA', 'CLOSED_RING_ARTIFACT', 'MASK_AREA_OUT_OF_DIST',
    })
    veto_threshold = float(getattr(args, 'veto_group_min', 0.3))
    failed_groups = [g for g in 'ABCDE' if group_mean[g] < veto_threshold] if veto_threshold > 0 else []

    ok = bool(total >= 50 and not any(t in hard_tags for t in tags) and not failed_groups)
    veto_reason = ','.join(f'{g}<{veto_threshold:.2f}' for g in failed_groups) if failed_groups else ''

    row = {
        'ok': ok,
        'total_score': float(total),
        'group_A': group_mean['A'],
        'group_B': group_mean['B'],
        'group_C': group_mean['C'],
        'group_D': group_mean['D'],
        'group_E': group_mean['E'],
        'group_F': group_mean['F'],
        'A1_area': a1, 'A2_circ': a2, 'A3_pect': a3, 'A4_contour': a4,
        'A5_vert': a5, 'A6_anatomy': a6, 'A7_clip': a7,
        'B1_stats': b1, 'B2_dr': b2, 'B3_extreme': b3,
        'C1_raps': c1, 'C2_entropy': c2, 'C3_grad': c3,
        'D1_round': d1, 'D2_void': d2, 'D3_band': d3, 'D4_round_density': d4, 'D5_grid': d5,
        'vert_struct_score': float(m.get('vert_struct_score', 1)),
        'grid_pattern_score': float(m.get('grid_pattern_score', 1)),
        'anatomy_chaos': float(m.get('anatomy_chaos', 0)),
        'elong_bright_segments': int(m.get('elong_bright_segments', 0)),
        'elong_dark_segments': int(m.get('elong_dark_segments', 0)),
        'edge_mid_segments': int(m.get('edge_mid_segments', 0)),
        'clip_margin': float(m.get('clip_margin', 100)),
        'clip_risk': float(m.get('clip_risk', 0)),
        'E1_hist': e1, 'E2_density': e2,
        'F1_ps_slope': f1, 'F2_brisque': f2,
        'ps_emd': float(ps_emd),
        'grad_wass': float(grad_wass),
        'hist_wass': float(hist_wass),
        'ref_z_mean_abs': ref_z_mean_abs,
        'contour_concavity': float(m['contour_concavity']),
        'contour_perim_ratio': float(m['contour_perim_ratio']),
        'round_density': float(m['round_density']),
        'tags': '|'.join(tags) if tags else 'pass',
        'veto': veto_reason,
        'legacy_reasons': '|'.join(legacy) if legacy else 'pass',
    }
    return row


# ---------------------------------------------------------------------------
# Semantic tier / funnel helpers
# ---------------------------------------------------------------------------
def _mask_ratio_p5_p95_from_ref(ref):
    if not ref:
        return None
    pct = ref.get('percentiles') or {}
    mr = pct.get('mask_ratio') or {}
    if 'p5' in mr and 'p95' in mr:
        return (float(mr['p5']), float(mr['p95']))
    return None


def run_stage1_hard_funnel(gray, args, ref=None, modality_checker=None, radiomics_base=None):
    """Stage1：模态 + 解剖 + 接缝；返回 veto 列表与诊断字典。"""
    veto_reasons = []
    soft_reasons = []
    mask = build_mask(gray)
    eval_profile = str(getattr(args, 'eval_profile', 'full'))
    p_mammo = None

    # Modality check
    if modality_checker is not None:
        mr = modality_checker.predict(gray)
        p_mammo = mr.p_mammography
        if mr.veto and mr.veto_reason:
            veto_reasons.append(mr.veto_reason)

    # Anatomy structure
    band = _mask_ratio_p5_p95_from_ref(ref)
    an = anatomy_structure_check(gray, mask, band)
    if an.veto:
        if eval_profile == 'full':
            for r in an.veto_reasons:
                if r == 'CENTER_OVEREXPOSED':
                    veto_reasons.append(r)
                else:
                    soft_reasons.append(r)
        else:
            veto_reasons.extend(an.veto_reasons)

    # Seam check
    patch_size = int(getattr(args, 'patch_size_eval', 640)) or 640
    seam_veto, seam_flags = patch_seam_check(
        gray, patch_size,
        enabled=bool(getattr(args, 'enable_seam_check', True)),
    )
    if seam_veto:
        if eval_profile == 'full':
            soft_reasons.append('GRID_SEAM')
        else:
            veto_reasons.append('GRID_SEAM')

    # Radiomics
    vec = extract_radiomics_vector(gray, mask)
    if vec is not None and radiomics_base is not None:
        d2, rad_score = radiomics_mahalanobis_score(vec, radiomics_base)
    elif vec is not None:
        d2 = float('nan')
        rad_score = 0.5
    else:
        d2 = float('nan')
        rad_score = 0.5

    return {
        'veto_reasons': veto_reasons,
        'soft_reasons': soft_reasons,
        'all_stage1_reasons': veto_reasons + soft_reasons,
        'stage1_veto': len(veto_reasons) > 0,
        'modality_confidence': p_mammo,
        'radiomics_distance': float(d2) if (vec is not None and not np.isnan(d2)) else None,
        'radiomics_score': float(rad_score) if (radiomics_base is not None and vec is not None) else 0.5,
        'anatomy_flags': {
            'skin_line_present': an.flags.skin_line_present,
            'closed_ring_detected': an.flags.closed_ring_detected,
            'center_overexposed': an.flags.center_overexposed,
            'mask_area_ratio': round(an.flags.mask_area_ratio, 6),
        },
        'seam_flags': {
            'grid_peak_detected': seam_flags.grid_peak_detected,
            'peak_frequencies': [round(float(x), 6) for x in seam_flags.peak_frequencies],
        },
        'breast_mask_for_iou': mask,
    }


def _find_paired_source(gen_path, src_dir):
    """在 src_dir 中查找与 gen_path 同名的源图（匹配 `*_<patient_id>_*` 模式）。"""
    if src_dir is None or not src_dir.is_dir():
        return None
    stem = gen_path.name
    candidates = sorted(src_dir.rglob(stem))
    for c in candidates:
        if c.is_file():
            return c
    # Try matching by patient ID pattern: find everything with same name stem
    for c in sorted(src_dir.rglob('*')):
        if c.is_file() and c.name == gen_path.name:
            return c
    return None


def compute_semantic_tier_and_final(funnel, quality_score, iou, args):
    """semantic_score [0,1], tier 1/2/3, final_rank_score 按规格。"""
    p = funnel.get('modality_confidence')
    rad_s_val = funnel.get('radiomics_score', 0.5)
    rad_s = float(rad_s_val) if rad_s_val else 0.5
    eval_profile = str(getattr(args, 'eval_profile', 'full'))
    norm_q = float(np.clip(quality_score / 100, 0, 1))

    if p is None and funnel.get('radiomics_distance') is None:
        # No modality model, no radiomics: use profile fallback
        if eval_profile == 'full':
            semantic = 0.55 + 0.45 * norm_q
        else:
            semantic = 0.5 + 0.3 * norm_q
    else:
        p_use = 0.72 if p is None else float(p)
        if iou is None:
            semantic = 0.58 * p_use + 0.42 * rad_s
        else:
            semantic = 0.5 * p_use + 0.35 * rad_s + 0.15 * float(np.clip(iou, 0, 1))

    # Soft penalties
    penalty_map = {
        'MASK_AREA_OUT_OF_DIST': 0.08,
        'SKIN_LINE_MISSING': 0.07,
        'CLOSED_RING_ARTIFACT': 0.12,
        'GRID_SEAM': 0.05 if eval_profile == 'full' else 0.12,
    }
    for r in set(funnel.get('soft_reasons') or []):
        semantic -= penalty_map.get(str(r), 0)
    semantic = float(np.clip(semantic, 0, 1))

    t1 = float(getattr(args, 'tier1_semantic_threshold', 0.75))
    t2 = float(getattr(args, 'tier2_semantic_threshold', 0.5))
    if semantic > t1:
        tier = 1
    elif semantic > t2:
        tier = 2
    else:
        tier = 3

    if tier == 1:
        final_rank = 0.6 * semantic + 0.4 * norm_q
    else:
        final_rank = semantic * 0.3

    return (semantic, tier, float(final_rank))


# ---------------------------------------------------------------------------
# Image collection
# ---------------------------------------------------------------------------
def collect_images(img_dir, args):
    candidates = [
        p for p in (
            img_dir.rglob('*') if args.recursive else img_dir.iterdir()
        )
        if p.is_file() and is_image(p)
    ]
    paths = sorted(p for p in candidates if '.ipynb_checkpoints' not in p.parts)
    if not args.include_compare:
        paths = [p for p in paths if '_ffdm_compare' not in p.name.lower()]
    if args.name_prefix:
        paths = [p for p in paths if p.name.startswith(args.name_prefix)]
    return paths


def fail_row(path):
    """返回一条全零占位行（读取失败时用）。"""
    seam_flags = {'grid_peak_detected': False, 'peak_frequencies': []}
    return {
        'image': str(path),
        'ok': False,
        'total_score': 0,
        'tier': 4,
        'rank_tier': 4,
        'semantic_score': 0,
        'quality_score': 0,
        'final_rank_score': 0,
        'modality_confidence': None,
        'radiomics_distance': None,
        'anatomy_flags': {},
        'seam_flags': seam_flags,
        'veto_reasons': ['read_failed'],
        'recommend_notes': 'read_failed',
        'legacy_reasons': 'read_failed',
        'tags': 'read_failed',
        'veto': '',
        'group_A': 0, 'group_B': 0, 'group_C': 0, 'group_D': 0, 'group_E': 0, 'group_F': 0,
        'A1_area': 0, 'A2_circ': 0, 'A3_pect': 0, 'A4_contour': 0, 'A5_vert': 0,
        'A6_anatomy': 0, 'A7_clip': 1,
        'B1_stats': 0, 'B2_dr': 0, 'B3_extreme': 0,
        'C1_raps': 0, 'C2_entropy': 0, 'C3_grad': 0,
        'D1_round': 0, 'D2_void': 0, 'D3_band': 0, 'D4_round_density': 0, 'D5_grid': 0,
        'E1_hist': 0, 'E2_density': 0,
        'F1_ps_slope': 0, 'F2_brisque': 0,
        'mask_ratio': 0, 'circularity': 0, 'pectoral_found': False,
        'contour_concavity': 0, 'contour_perim_ratio': 0,
        'mean_intensity': 0, 'std_intensity': 0, 'dynamic_range': 0,
        'bright_ratio': 0, 'dark_ratio': 0, 'local_entropy': 0,
        'ps_emd': 0, 'grad_wass': 0, 'hist_wass': 0,
        'round_isolated': 0, 'round_density': 0, 'edge_voids_px': 0,
        'banding_score': 0, 'vert_struct_score': 1, 'grid_pattern_score': 1,
        'anatomy_chaos': 0,
        'elong_bright_segments': 0, 'elong_dark_segments': 0,
        'edge_mid_segments': 0, 'tiny_bright_cc': 0,
        'clip_margin': 100, 'clip_risk': 0,
        'density_type': '', 'ps_slope': 0, 'brisque_score': -1,
        'cavity_ratio': 0, 'bright_spots': 999, 'mirror_sim': 1,
        'edge_density': 0, 'hu_dist': 99, 'ref_z_mean_abs': 0,
    }


# ---------------------------------------------------------------------------
# CSV_FIELDS
# ---------------------------------------------------------------------------
CSV_FIELDS = [
    'image', 'ok', 'total_score', 'tier', 'rank_tier',
    'semantic_score', 'quality_score', 'final_rank_score',
    'modality_confidence', 'radiomics_distance',
    'anatomy_flags_json', 'seam_flags_json', 'veto_reasons_json',
    'recommend_notes', 'tags', 'veto', 'legacy_reasons',
    'group_A', 'group_B', 'group_C', 'group_D', 'group_E', 'group_F',
    'A1_area', 'A2_circ', 'A3_pect', 'A4_contour', 'A5_vert', 'A6_anatomy', 'A7_clip',
    'B1_stats', 'B2_dr', 'B3_extreme',
    'C1_raps', 'C2_entropy', 'C3_grad',
    'D1_round', 'D2_void', 'D3_band', 'D4_round_density', 'D5_grid',
    'E1_hist', 'E2_density',
    'F1_ps_slope', 'F2_brisque',
    'mask_ratio', 'circularity', 'pectoral_found',
    'contour_concavity', 'contour_perim_ratio',
    'mean_intensity', 'std_intensity', 'dynamic_range',
    'bright_ratio', 'dark_ratio', 'local_entropy',
    'ps_emd', 'grad_wass', 'hist_wass',
    'round_isolated', 'round_density', 'edge_voids_px',
    'banding_score', 'vert_struct_score', 'grid_pattern_score',
    'anatomy_chaos', 'elong_bright_segments', 'elong_dark_segments',
    'edge_mid_segments', 'clip_margin', 'clip_risk',
    'density_type', 'ps_slope', 'brisque_score',
    'cavity_ratio', 'bright_spots', 'mirror_sim', 'edge_density', 'hu_dist',
    'ref_z_mean_abs',
]


def _row_for_csv(row):
    """CSV 需纯文本：`*_json` 列从对应 dict/list 字段序列化。"""
    alias = {
        'anatomy_flags_json': 'anatomy_flags',
        'seam_flags_json': 'seam_flags',
        'veto_reasons_json': 'veto_reasons',
    }
    out = {}
    for k in CSV_FIELDS:
        sk = alias.get(k, k)
        v = row.get(sk)
        if v is None:
            out[k] = ''
        elif isinstance(v, (dict, list, tuple)):
            out[k] = json.dumps(v, ensure_ascii=False)
        else:
            out[k] = v
    return out


def round_row(row):
    r = {}
    for k, v in row.items():
        if isinstance(v, float):
            r[k] = round(v, 6)
        elif isinstance(v, np.floating):
            r[k] = round(float(v), 6)
        elif isinstance(v, (dict, list, tuple)):
            r[k] = v
        else:
            r[k] = v
    return r


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    img_dir = args.images_dir.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not img_dir.is_dir():
        print(f'图像目录不存在: {img_dir}')
        return

    paths = collect_images(img_dir, args)
    hint = f'前缀 {args.name_prefix} ' if args.name_prefix else ''
    print(f'{hint}在 {img_dir}（recursive={args.recursive}）找到 {len(paths)} 图片')
    if not paths:
        return

    # Reference stats
    ref_stats = None
    changes = {}
    if args.real_baseline_json:
        ref_stats = load_baseline_json(args.real_baseline_json)
        print(f'加载已有真实基线: {args.real_baseline_json}')
    elif args.real_images_dir:
        ref_stats = build_reference_stats(
            args.real_images_dir,
            long_side=args.resize_long_side,
            max_samples=args.max_baseline_samples,
            workers=args.baseline_workers,
            sample_seed=args.sample_seed,
        )
        if ref_stats:
            print(f'真实基线样本数: {ref_stats["count"]}')
            # Save baseline JSON alongside output
            save_baseline_json(ref_stats, out_dir / 'real_baseline.json', args)
    else:
        print('未提供 --real-images-dir 或 --real-baseline-json，不使用基线校准')

    # Auto-calibrate
    if args.auto_calibrate and ref_stats and ref_stats.get('percentiles'):
        args, changes = apply_calibration(args, ref_stats)
        if changes:
            print(f'[auto-calibrate] 用真实分布 percentiles 覆写阈值：')
            for k, v in changes.items():
                print(f'  {k}: {v}')
    elif args.auto_calibrate and (ref_stats is not None and not ref_stats.get('percentiles')):
        print('[auto-calibrate] 提示：加载的基线 JSON 里没有 percentiles 字段，请重建基线或改用 --no-auto-calibrate。')

    # CLIP setup
    if args.clip_anatomy:
        if _CLIP_TR_OK:
            os.environ.setdefault('TRANSFORMERS_OFFLINE', '0')
            _REVIEW_CLIP_MODEL_ID = str(args.clip_model).strip() or 'openai/clip-vit-base-patch32'
            print(f'[clip-anatomy] 启用 CLIP：{_REVIEW_CLIP_MODEL_ID}')
        else:
            print('[clip-anatomy] transformers/torch 不可用，已忽略 --clip-anatomy')

    # Modality / radiomics
    modality_checker = None
    if args.modality_classifier_path and args.modality_classifier_path.is_file():
        try:
            modality_checker = ModalityChecker(
                weights_path=args.modality_classifier_path,
                device=str(args.modality_device),
            )
        except Exception:
            pass
    elif args.modality_classifier_path:
        print(f'[semantic funnel] 模态权重不存在，跳过模态否决: {args.modality_classifier_path}')

    radiomics_base = None
    if args.radiomics_baseline and args.radiomics_baseline.is_file():
        radiomics_base = load_radiomics_baseline(args.radiomics_baseline)
        if radiomics_base is None:
            print(f'[semantic funnel] 影像组学基线加载失败: {args.radiomics_baseline}')
    elif args.radiomics_baseline:
        print(f'[semantic funnel] 影像组学基线不存在: {args.radiomics_baseline}')

    # Review uses stage-1 funnel and scorer objects that are only wired in the
    # sequential branch. Keep generated-image review single-process so rows are
    # always scored and written; baseline building still has its own workers.
    n_review_workers = 1
    rows = []
    metrics_map = {}

    if n_review_workers > 1:
        print(f'[review] 并行进程数 = {n_review_workers}（review 强制单进程）')
        # NOTE: review_worker is single-process due to GPU/CLIP constraints
        tasks = [(str(p), args.resize_long_side) for p in paths]
        with Pool(processes=n_review_workers) as pool:
            it = pool.imap_unordered(_review_worker, tasks, chunksize=8)
            for path_str, m in tqdm(
                zip([str(p) for p in paths], it),
                total=len(paths), desc='Review images',
            ):
                if m is None:
                    rows.append(fail_row(path_str))
                    continue
                metrics_map[path_str] = m
    else:
        for p in tqdm(paths, desc='Review images'):
            path_str = str(p)
            try:
                gray = cv2.imread(path_str, cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    rows.append(fail_row(path_str))
                    continue
                if args.resize_long_side and args.resize_long_side > 0:
                    gray = resize_long_side(gray, args.resize_long_side)
                m = extract_metrics(gray)
                metrics_map[path_str] = m
            except Exception:
                rows.append(fail_row(path_str))
                continue

            # Stage 1 funnel
            funnel = run_stage1_hard_funnel(
                gray, args, ref=ref_stats,
                modality_checker=modality_checker,
                radiomics_base=radiomics_base,
            )
            # Score
            s = score_image(m, ref_stats, args)
            tags_merged = s.get('tags', 'pass')
            veto_reasons = funnel.get('veto_reasons', [])
            vr = '|'.join(veto_reasons) if veto_reasons else ''
            leg = s.get('legacy_reasons', 'pass')

            # Pair IoU
            iou = None
            if args.paired_source_dir:
                srcp = _find_paired_source(p, args.paired_source_dir)
                if srcp and srcp.is_file():
                    src_gray = cv2.imread(str(srcp), cv2.IMREAD_GRAYSCALE)
                    if src_gray is not None:
                        iou = breast_mask_iou(build_mask(gray), build_mask(src_gray))

            # Semantic tier
            semantic, tier, final_rank = compute_semantic_tier_and_final(
                funnel, s['total_score'], iou, args,
            )
            ok_final = bool(s['ok'] and not funnel['stage1_veto'])

            # Recommend notes
            notes_parts = []
            if funnel['stage1_veto']:
                notes_parts.append('funnel_veto')
            if iou is not None:
                notes_parts.append(f'mask_iou={iou:.3f}')

            row = {
                'image': str(p.name),
                'ok': bool(ok_final),
                'total_score': float(s['total_score']),
                'tier': int(tier),
                'rank_tier': int(tier),
                'semantic_score': float(semantic),
                'quality_score': float(s['total_score']),
                'final_rank_score': float(final_rank),
                'modality_confidence': funnel.get('modality_confidence'),
                'radiomics_distance': funnel.get('radiomics_distance'),
                'anatomy_flags': funnel.get('anatomy_flags', {}),
                'seam_flags': funnel.get('seam_flags', {}),
                'veto_reasons': veto_reasons,
                'soft_reasons': funnel.get('soft_reasons', []),
                'recommend_notes': '; '.join(notes_parts) if notes_parts else '',
                'tags': tags_merged,
                'veto': vr,
                'legacy_reasons': leg,
                'group_A': s.get('group_A', 0),
                'group_B': s.get('group_B', 0),
                'group_C': s.get('group_C', 0),
                'group_D': s.get('group_D', 0),
                'group_E': s.get('group_E', 0),
                'group_F': s.get('group_F', 0),
                'A1_area': s.get('A1_area', 0),
                'A2_circ': s.get('A2_circ', 0),
                'A3_pect': s.get('A3_pect', 0),
                'A4_contour': s.get('A4_contour', 0),
                'A5_vert': s.get('A5_vert', 0),
                'A6_anatomy': s.get('A6_anatomy', 0),
                'A7_clip': s.get('A7_clip', 1),
                'B1_stats': s.get('B1_stats', 0),
                'B2_dr': s.get('B2_dr', 0),
                'B3_extreme': s.get('B3_extreme', 0),
                'C1_raps': s.get('C1_raps', 0),
                'C2_entropy': s.get('C2_entropy', 0),
                'C3_grad': s.get('C3_grad', 0),
                'D1_round': s.get('D1_round', 0),
                'D2_void': s.get('D2_void', 0),
                'D3_band': s.get('D3_band', 0),
                'D4_round_density': s.get('D4_round_density', 0),
                'D5_grid': s.get('D5_grid', 0),
                'E1_hist': s.get('E1_hist', 0),
                'E2_density': s.get('E2_density', 0),
                'F1_ps_slope': s.get('F1_ps_slope', 0),
                'F2_brisque': s.get('F2_brisque', 0),
                'mask_ratio': m.get('mask_ratio', 0),
                'circularity': m.get('circularity', 0),
                'pectoral_found': m.get('pectoral_found', False),
                'contour_concavity': m.get('contour_concavity', 0),
                'contour_perim_ratio': m.get('contour_perim_ratio', 0),
                'mean_intensity': m.get('mean_intensity', 0),
                'std_intensity': m.get('std_intensity', 0),
                'dynamic_range': m.get('dynamic_range', 0),
                'bright_ratio': m.get('bright_ratio', 0),
                'dark_ratio': m.get('dark_ratio', 0),
                'local_entropy': m.get('local_entropy', 0),
                'ps_emd': float(s.get('ps_emd', 0)),
                'grad_wass': float(s.get('grad_wass', 0)),
                'hist_wass': float(s.get('hist_wass', 0)),
                'round_isolated': m.get('round_isolated', 0),
                'round_density': m.get('round_density', 0),
                'edge_voids_px': m.get('edge_voids_px', 0),
                'banding_score': m.get('banding_score', 0),
                'vert_struct_score': m.get('vert_struct_score', 1),
                'grid_pattern_score': m.get('grid_pattern_score', 1),
                'anatomy_chaos': m.get('anatomy_chaos', 0),
                'elong_bright_segments': m.get('elong_bright_segments', 0),
                'elong_dark_segments': m.get('elong_dark_segments', 0),
                'edge_mid_segments': m.get('edge_mid_segments', 0),
                'clip_margin': float(m.get('clip_margin', 100)),
                'clip_risk': float(m.get('clip_risk', 0)),
                'density_type': m.get('density_type', '') or '',
                'ps_slope': m.get('ps_slope', 0),
                'brisque_score': m.get('brisque_score', -1),
                'cavity_ratio': m.get('cavity_ratio', 0),
                'bright_spots': m.get('bright_spots', 999),
                'mirror_sim': m.get('mirror_sim', 1),
                'edge_density': m.get('edge_density', 0),
                'hu_dist': m.get('hu_dist', 99),
                'ref_z_mean_abs': float(s.get('ref_z_mean_abs', 0)),
            }
            rows.append(row)

    # CSV output
    csv_path = out_dir / 'review_report.csv'
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(_row_for_csv(r))

    # Summary
    ok_count = sum(1 for r in rows if r.get('ok'))
    pass_rate = ok_count / max(1, len(rows))
    mean_total = float(np.mean([r.get('total_score', 0) for r in rows]))
    tag_counts = Counter()
    veto_count = 0
    veto_by_groups = Counter()
    for r in rows:
        tags_str = r.get('tags', '')
        if tags_str and tags_str != 'pass':
            for t in tags_str.split('|'):
                tag_counts[t.strip()] += 1
        if r.get('veto'):
            veto_count += 1
            for seg in r['veto'].split(','):
                veto_by_groups[seg.strip()] += 1
    violation_rates = {k: v / max(1, len(rows)) for k, v in tag_counts.items()}
    mean_ps_slope = float(np.mean([r.get('ps_slope', 0) for r in rows]))
    valid_brisque = [r.get('brisque_score', -1) for r in rows if r.get('brisque_score', -1) >= 0]
    mean_brisque = float(np.mean(valid_brisque)) if valid_brisque else -1
    tier_hist = Counter(r.get('tier', 4) for r in rows)

    per_image_summary = []
    for r in rows:
        per_image_summary.append({
            'image': r.get('image', ''),
            'ok': bool(r.get('ok', False)),
            'tier': int(r.get('tier', 4)),
            'total_score': float(r.get('total_score', 0)),
            'semantic_score': float(r.get('semantic_score', 0)),
            'tags': r.get('tags', ''),
            'veto': r.get('veto', ''),
            'legacy_reasons': r.get('legacy_reasons', ''),
        })

    summary = {
        'count': len(rows),
        'ok_count': ok_count,
        'pass_rate': pass_rate,
        'mean_total_score': round(mean_total, 4),
        'mean_ps_slope': round(mean_ps_slope, 4),
        'mean_brisque': round(mean_brisque, 4),
        'group_means': {
            g: float(np.mean([r.get(f'group_{g}', 0) for r in rows]))
            for g in 'ABCDEF'
        },
        'violation_rates': {k: round(v, 4) for k, v in sorted(violation_rates.items())},
        'veto_count': veto_count,
        'veto_by_group': dict(veto_by_groups),
        'tier_hist': dict(sorted(tier_hist.items())),
        'thresholds': {
            'min_mask_ratio': args.min_mask_ratio,
            'max_mask_ratio': args.max_mask_ratio,
            'min_circularity': args.min_circularity,
            'max_circularity': args.max_circularity,
            'max_contour_concavity': args.max_contour_concavity,
            'max_contour_perim_ratio': args.max_contour_perim_ratio,
            'max_mean_intensity': args.max_mean_intensity,
            'min_std_intensity': args.min_std_intensity,
            'min_dynamic_range': args.min_dynamic_range,
            'max_dynamic_range': args.max_dynamic_range,
            'max_bright_ratio': args.max_bright_ratio,
            'max_dark_ratio': args.max_dark_ratio,
            'max_isolated_round': args.max_isolated_round,
            'max_round_density': args.max_round_density,
            'max_edge_voids': args.max_edge_voids,
            'min_banding_score': args.min_banding_score,
            'max_hist_wass': args.max_hist_wass,
            'max_ps_emd': args.max_ps_emd,
            'max_grad_wass': args.max_grad_wass,
            'max_ref_z': args.max_ref_z,
            'veto_group_min': getattr(args, 'veto_group_min', 0.3),
        },
        'auto_calibrated': bool(changes),
        'calibration_changes': changes,
        'per_image': per_image_summary,
    }

    summary_path = out_dir / 'summary.json'
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding='utf-8',
    )
    print(f'摘要写入 {summary_path}')
    print(f'通过率: {ok_count}/{len(rows)} = {pass_rate:.2%}')

    # Top-K ranked
    ranked = sorted(
        [r for r in rows if r.get('ok')],
        key=lambda x: (int(x.get('rank_tier', 4)), -float(x.get('final_rank_score', 0)), -float(x.get('total_score', 0))),
    )
    top_k = min(args.top_k, len(ranked))
    top_txt = '\n'.join(
        f'{i + 1}. {r["image"]} tier={r["rank_tier"]} final={r["final_rank_score"]:.4f} score={r["total_score"]:.1f}'
        for i, r in enumerate(ranked[:top_k])
    ) if top_k > 0 else '(无合格图)'
    candidates_txt = f'共 {len(ranked)} 张合格，推荐 Top-{top_k}:\n{top_txt}'

    # Legacy ranked
    legacy_ranked = sorted(
        rows,
        key=lambda x: (
            int(x.get('tier', 4)),
            -float(x.get('final_rank_score', 0)),
            -float(x.get('total_score', 0)),
        ),
    )
    legacy_top_txt = '\n'.join(
        f'{i + 1}. {r["image"]} tier={r["tier"]} final={r["final_rank_score"]:.4f} score={r["total_score"]:.1f}'
        for i, r in enumerate(legacy_ranked[:top_k])
    )

    ranked_path = out_dir / 'top_k.txt'
    ranked_path.write_text(candidates_txt, encoding='utf-8')
    print(ranked_path)
    print(candidates_txt)


if __name__ == '__main__':
    main()
