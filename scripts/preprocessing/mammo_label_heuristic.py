"""钼靶边角标签检测：两级流水线。

一级（轻量拓扑，纯 NumPy/OpenCV，零 API 开销）
  用 H0 Betti 数思路：
    - 全图阈值 → 所有前景连通域（cc）
    - 最大的一块 = 乳腺主体（大、连续）
    - 除此以外、出现在"边角带"里的小亮块 = 候选标签（离散）
    - 统计边角带内可疑小 cc 的数量与分布 → label_score
  三态输出：CERTAIN_CLEAN / CERTAIN_LABELED / UNCERTAIN

二级（Qwen-VL，仅当一级返回 UNCERTAIN / CERTAIN_LABELED 时调用）
  - ask_vl_label_check : 快速 YES/NO 判断（保留向后兼容）
  - ask_vl_label_bbox  : 返回标签区域坐标框列表，可精确涂黑
    * 图缩到 512px、JPEG Q=80
    * 要求 VL 返回 JSON 坐标框（百分比 0-100），再换算为绝对像素
  - blackout_label_regions : 将坐标框区域归零，带可选 padding
  环境变量：QWEN_API_KEY；
    QWEN_VL_LABEL_MODEL（可选，仅标签守护优先）→ QWEN_VL_MODEL → 默认 qwen3.6-plus

  涂黑参数（仅当函数对应形参为 None 时读环境，显式传参优先）：
    LABEL_BLACKOUT_PAD、LABEL_BLACKOUT_MARGIN_FRAC、LABEL_MAX_BOX_AREA_FRAC、
    LABEL_MAX_BOX_SIDE_FRAC

  其它：LABEL_VL_BBOX_TIMEOUT_SEC 覆盖 ask_vl_label_bbox 的 HTTP 超时（秒）；
    LABEL_GUARD_DEBUG=1 时对 blackout_label_regions 输出 INFO 日志。
"""

from __future__ import annotations

import base64
import logging
import os
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.core.image_utils import resize_long_side

_LOG = logging.getLogger(__name__)

# 标签守护专用 VL：优先 QWEN_VL_LABEL_MODEL，否则与顾问等共用 QWEN_VL_MODEL
_VL_LABEL_MODEL_DEFAULT = "qwen3-vl-plus"


def _truthy_env(key: str) -> bool:
    return os.environ.get(key, "").strip().lower() in ("1", "true", "yes", "on")


def _optional_env_float(key: str, fallback: float) -> float:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return fallback
    try:
        return float(raw)
    except ValueError:
        return fallback


def _optional_env_int(key: str, fallback: int) -> int:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return fallback
    try:
        return int(raw, 10)
    except ValueError:
        return fallback


def _resolve_vl_label_model() -> str:
    return (
        os.environ.get("QWEN_VL_LABEL_MODEL", "").strip()
        or os.environ.get("QWEN_VL_MODEL", "").strip()
        or _VL_LABEL_MODEL_DEFAULT
    )


# ── 枚举与数据结构 ────────────────────────────────────────────────────────────

class LabelVerdict(Enum):
    CERTAIN_CLEAN   = "certain_clean"
    CERTAIN_LABELED = "certain_labeled"
    UNCERTAIN       = "uncertain"


@dataclass
class LabelHeuristicResult:
    verdict:     LabelVerdict
    label_score: float   # 0~1，越大越像有标签
    confidence:  float   # 对当前 verdict 的把握
    detail:      str = ""


# ── 一级：H0 拓扑分析 ─────────────────────────────────────────────────────────

def topological_label_score(
    gray_u8: np.ndarray,
    *,
    topo_max_side: int = 512,
    corner_frac: float = 0.28,    # 边角带宽度（占图像长/宽的比例）
    min_cc_area: int   = 6,       # 忽略低于此面积的噪点 cc
    max_cc_frac: float = 0.25,    # 候选标签 cc 面积上限（相对于最大 cc 即乳腺）
                                   # 由 0.10 → 0.25：防止较大箭头/字符被排除在外
    suspicious_for_certain: int = 3,   # ≥ N 个可疑 cc → CERTAIN_LABELED
    suspicious_for_uncertain: int = 1, # ≥ N → UNCERTAIN
) -> tuple[float, int, str]:
    """
    返回 (label_score ∈ [0,1], 可疑 cc 数量, 说明文字)。

    拓扑思路：
      乳腺主体 H0 连通分量数 = 1（大块连续亮域）
      标签/字符  H0 连通分量散布在黑色背景角落（多个小亮块）

    两轮扫描：
      Round-A（边角带）：原有逻辑，检测角落的文字/ID 标记
      Round-B（全图高亮 CC）：检测乳腺主体外任何较亮的独立块，
        覆盖箭头等位于图像中央但不属于腺体的标注
    """
    g = gray_u8
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    small = resize_long_side(g, topo_max_side, only_downscale=True, min_side=4)
    h, w  = small.shape

    # 去噪后 Otsu 分割
    blurred = cv2.GaussianBlur(small, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 所有前景连通域
    n_cc, labels_map, stats, centroids = cv2.connectedComponentsWithStats(
        thresh, connectivity=8
    )
    if n_cc <= 1:
        return 0.0, 0, "no foreground cc"

    # 找乳腺（最大前景 cc，label=0 是背景）
    areas = stats[1:, cv2.CC_STAT_AREA]
    breast_idx = int(areas.argmax()) + 1   # +1 offset（跳过背景）
    breast_area = float(stats[breast_idx, cv2.CC_STAT_AREA])

    # 边角带掩膜（上/下/左/右四条带的并集）
    ch = max(4, int(round(h * corner_frac)))
    cw = max(4, int(round(w * corner_frac)))
    corner_mask = np.zeros((h, w), dtype=bool)
    corner_mask[:ch, :]    = True
    corner_mask[h - ch:, :] = True
    corner_mask[:, :cw]    = True
    corner_mask[:, w - cw:] = True

    suspicious = 0
    details: list[str] = []

    # Round-A：边角带内可疑小 cc（原有逻辑）
    for i in range(1, n_cc):
        if i == breast_idx:
            continue
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_cc_area:
            continue
        if area > breast_area * max_cc_frac:
            continue
        cx, cy = centroids[i]
        if corner_mask[min(int(cy), h - 1), min(int(cx), w - 1)]:
            suspicious += 1
            details.append(f"cc#{i}(area={area},cx={cx:.0f},cy={cy:.0f},zone=corner)")

    # Round-B：全图高亮 cc（覆盖非边角的箭头/标注）
    # 策略：用高百分位阈值（而非 Otsu）找到极亮的孤立块，
    #   排除乳腺主体本身后，任何面积合理的高亮 cc 均为可疑
    bright_thr = float(np.percentile(small, 96))
    if bright_thr > 30:  # 图像本身有足够对比度才做高亮扫描
        _, bright_mask = cv2.threshold(blurred, int(bright_thr), 255, cv2.THRESH_BINARY)
        n_bcc, b_labels, b_stats, b_centroids = cv2.connectedComponentsWithStats(
            bright_mask, connectivity=8
        )
        if n_bcc > 1:
            b_areas = b_stats[1:, cv2.CC_STAT_AREA]
            b_breast_idx = int(b_areas.argmax()) + 1
            for i in range(1, n_bcc):
                if i == b_breast_idx:
                    continue
                area = int(b_stats[i, cv2.CC_STAT_AREA])
                # 面积范围：大于噪点，小于乳腺的 30%
                if area < min_cc_area * 3:
                    continue
                if area > breast_area * 0.30:
                    continue
                cx, cy = b_centroids[i]
                # 只报告不在边角带的高亮 cc（边角带已被 Round-A 处理）
                iy = min(int(cy), h - 1)
                ix = min(int(cx), w - 1)
                if not corner_mask[iy, ix]:
                    suspicious += 1
                    details.append(
                        f"cc#{i}(area={area},cx={cx:.0f},cy={cy:.0f},zone=bright_noncorner)"
                    )

    # 归一化为 0~1
    score = float(np.clip(suspicious / max(float(suspicious_for_certain), 1.0), 0.0, 1.0))
    detail_str = f"suspicious_cc={suspicious}; " + (", ".join(details[:5]) if details else "none")
    return score, suspicious, detail_str


def compute_label_heuristic(
    gray_u8: np.ndarray,
    *,
    topo_max_side: int  = 512,
    corner_frac: float  = 0.28,
    clean_below:  float = 0.20,   # label_score ≤ 此值 → CERTAIN_CLEAN
    labeled_above: float = 0.65,  # label_score ≥ 此值 → CERTAIN_LABELED
    analysis_max_side: int = 512, # 兼容旧调用签名（alias）
) -> LabelHeuristicResult:
    """一级轻量拓扑判别（不调用任何 API）。"""
    side = min(topo_max_side, analysis_max_side)
    score, n_susp, detail = topological_label_score(
        gray_u8,
        topo_max_side=side,
        corner_frac=corner_frac,
    )

    if score <= clean_below:
        conf = float(np.clip(1.0 - score / max(clean_below, 1e-6), 0.0, 1.0))
        return LabelHeuristicResult(LabelVerdict.CERTAIN_CLEAN, score, conf, detail)

    if score >= labeled_above:
        conf = float(np.clip((score - labeled_above) / max(1.0 - labeled_above, 1e-6), 0.0, 1.0))
        return LabelHeuristicResult(LabelVerdict.CERTAIN_LABELED, score, conf, detail)

    # 中段：UNCERTAIN
    half_band = (labeled_above - clean_below) / 2.0
    dist_to_edge = min(score - clean_below, labeled_above - score)
    conf = float(np.clip(1.0 - dist_to_edge / max(half_band, 1e-6), 0.0, 1.0))
    return LabelHeuristicResult(LabelVerdict.UNCERTAIN, score, conf, detail)


# ── 二级：Qwen-VL（仅 UNCERTAIN 时调用） ──────────────────────────────────────

_VL_SYSTEM = (
    "You are a specialist for mammography display metadata. "
    "Answer ONLY with the single word YES or NO. No punctuation, no explanation."
)
_VL_USER = (
    "Scan the ENTIRE mammogram systematically from corner to corner, including peripheral black bands "
    "and all margins. "
    "Answer YES only if you see definitive medical-imaging workstation / DICOM display labels: "
    "alphanumeric patient, study or accession IDs; dates/times; view or laterality codes printed as overlays "
    "(e.g. L, R, CC, MLO, RMLO, LMLO); facility or modality burn‑in text; small orientation markers explicitly "
    "printed on the acquisition (not anatomy). "
    "Answer NO if you only see breast anatomy: parenchyma, Cooper ligaments, skin line, nipple shadow, vascular "
    "structures, benign calcifications, noise, benign masses, benign asymmetry or any grayscale tissue-like texture - "
    "even where they are bright, linear or focal. When unsure, answer NO."
)

# 全图判别：略高于旧版分辨率，便于小字 ID（仍压缩 token）
_VL_MAX_SIDE    = 384
_VL_JPEG_QUAL   = 70
_VL_MAX_TOKENS  = 8     # 只需要 YES 或 NO


def _gray_to_base64_jpeg(gray: np.ndarray, max_side: int, quality: int) -> str:
    h, w = gray.shape[:2]
    m = max(h, w)
    if m > max_side:
        scale = max_side / float(m)
        nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
        gray = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", gray, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    return base64.standard_b64encode(buf.tobytes()).decode("ascii")


def ask_vl_label_check(
    gray_u8: np.ndarray,
    *,
    max_side: int   = _VL_MAX_SIDE,
    jpeg_quality: int = _VL_JPEG_QUAL,
    timeout: int    = 30,
) -> LabelHeuristicResult:
    """
    调用 Qwen-VL 对图像做二值判断（是否含标签）。
    仅在一级返回 UNCERTAIN 后作为兜底。
    环境变量：QWEN_API_KEY；模型见 QWEN_VL_LABEL_MODEL 或 QWEN_VL_MODEL（默认 qwen3.6-plus）。
    """
    g = gray_u8
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)

    b64 = _gray_to_base64_jpeg(g, max_side, jpeg_quality)
    data_url = f"data:image/jpeg;base64,{b64}"

    # 动态加载 dotenv + openai（避免在非 API 路径引入依赖）
    _root = Path(__file__).resolve().parents[2]
    _dotenv = _root / ".env"
    if _dotenv.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(_dotenv, override=False)
        except ImportError:
            pass

    api_key  = os.environ.get("QWEN_API_KEY", "").strip()
    base_url = os.environ.get(
        "QWEN_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ).strip()
    model = _resolve_vl_label_model()

    if not api_key:
        raise RuntimeError("缺少 QWEN_API_KEY，无法调用 Qwen-VL")

    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("未安装 openai 包：pip install openai")

    client = OpenAI(api_key=api_key, base_url=base_url)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _VL_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": _VL_USER},
                ],
            },
        ],
        max_tokens=_VL_MAX_TOKENS,
        timeout=timeout,
    )

    raw = (resp.choices[0].message.content or "").strip().upper()
    # 容忍带标点的回复：YES. / NO! 等
    has_yes = "YES" in raw
    has_no  = "NO"  in raw

    if has_yes and not has_no:
        return LabelHeuristicResult(
            LabelVerdict.CERTAIN_LABELED, 1.0, 1.0, f"VL={raw!r}"
        )
    if has_no and not has_yes:
        return LabelHeuristicResult(
            LabelVerdict.CERTAIN_CLEAN, 0.0, 1.0, f"VL={raw!r}"
        )
    # 回复含义不明（极少见）→ 保守裁条带
    return LabelHeuristicResult(
        LabelVerdict.CERTAIN_LABELED, 0.8, 0.5, f"VL_ambiguous={raw!r}"
    )


# ── 三级：VL 返回坐标框，精确涂黑 ─────────────────────────────────────────────

_VL_BBOX_SYSTEM = (
    "You localize only true medical-imaging display metadata on mammograms. "
    "Respond ONLY with a valid JSON array, no markdown, no explanation."
)

_VL_BBOX_USER = (
    "Systematically scan the FULL mammogram field-of-view: breast, pectoral region if present, "
    "and all black letterbox margins and corners. "
    "Return one tight bounding box per item, in percent of image width/height (0-100), ONLY for: "
    "workstation or DICOM burned-in text/symbols (patient ID, accession, dates/times); laterality or view codes "
    "(L, R, CC, MLO, RMLO, LMLO, RCC, LCC, etc.); facility or equipment markers (e.g. bead/orientation markers) "
    "that are clearly not breast tissue; numeric ruler scales in margins. "
    "Do NOT box breast parenchyma, skin, nipple, Cooper ligaments, vessels, clip markers inside tissue, masses, "
    "asymmetry, benign calcifications, or any region that could be normal anatomy or texture. "
    "Do NOT box large areas: each box should tightly wrap only the textual/symbolic overlay. "
    "If there is no such medical-imaging label overlay, output exactly: []\n"
    "Format: [{\"x1\":5,\"y1\":2,\"x2\":20,\"y2\":8}, ...]\n"
    "Output ONLY the JSON array, nothing else."
)

_VL_BBOX_MAX_SIDE  = 512   # 比 YES/NO 模式更大，确保看清文字位置
_VL_BBOX_JPEG_QUAL = 80
_VL_BBOX_MAX_TOK   = 256   # 坐标 JSON 通常不超过 150 token


def ask_vl_label_bbox(
    gray_u8: np.ndarray,
    *,
    max_side: int = _VL_BBOX_MAX_SIDE,
    jpeg_quality: int = _VL_BBOX_JPEG_QUAL,
    timeout: int = 40,
) -> list[tuple[int, int, int, int]]:
    """
    调用 Qwen-VL 定位图像中的标签区域，返回绝对像素坐标框列表。

    返回值: [(x1,y1,x2,y2), ...]，已换算为原始 gray_u8 的像素坐标。
    若无标签或调用失败，返回空列表。

    环境变量：
      QWEN_API_KEY   - DashScope API 密钥
      QWEN_VL_LABEL_MODEL / QWEN_VL_MODEL - 视觉模型（标签守护默认 qwen3.6-plus，见 _resolve_vl_label_model）
      QWEN_BASE_URL  - API 基础 URL（默认 DashScope 兼容端点）
      LABEL_VL_BBOX_TIMEOUT_SEC - 若设置则覆盖本函数 timeout 参数（秒）
    """
    import json

    g = gray_u8
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    orig_h, orig_w = g.shape[:2]

    env_timeout = os.environ.get("LABEL_VL_BBOX_TIMEOUT_SEC", "").strip()
    if env_timeout:
        try:
            timeout = int(env_timeout, 10)
        except ValueError:
            pass

    # 缩图发送（记录缩放比以还原坐标）
    m = max(orig_h, orig_w)
    if m > max_side:
        scale = max_side / float(m)
        nh = max(1, int(round(orig_h * scale)))
        nw = max(1, int(round(orig_w * scale)))
        g_small = cv2.resize(g, (nw, nh), interpolation=cv2.INTER_AREA)
    else:
        g_small = g
        scale = 1.0

    ok, buf = cv2.imencode(".jpg", g_small, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    if not ok:
        return []
    b64 = base64.standard_b64encode(buf.tobytes()).decode("ascii")
    data_url = f"data:image/jpeg;base64,{b64}"

    # 加载 .env
    _root = Path(__file__).resolve().parents[2]
    _dotenv = _root / ".env"
    if _dotenv.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(_dotenv, override=False)
        except ImportError:
            pass

    api_key  = os.environ.get("QWEN_API_KEY", "").strip()
    base_url = os.environ.get(
        "QWEN_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ).strip()
    model = _resolve_vl_label_model()

    if not api_key:
        return []

    try:
        from openai import OpenAI
    except ImportError:
        return []

    client = OpenAI(api_key=api_key, base_url=base_url)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _VL_BBOX_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": _VL_BBOX_USER},
                    ],
                },
            ],
            max_tokens=_VL_BBOX_MAX_TOK,
            timeout=timeout,
        )
    except Exception:
        return []

    raw = (resp.choices[0].message.content or "").strip()

    # 健壮解析：提取第一个 JSON 数组
    start = raw.find("[")
    end   = raw.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        boxes_pct = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return []

    if not isinstance(boxes_pct, list):
        return []

    results: list[tuple[int, int, int, int]] = []
    for b in boxes_pct:
        if not isinstance(b, dict):
            continue
        try:
            px1 = float(b.get("x1", b.get("left",   0)))
            py1 = float(b.get("y1", b.get("top",    0)))
            px2 = float(b.get("x2", b.get("right",  0)))
            py2 = float(b.get("y2", b.get("bottom", 0)))
        except (TypeError, ValueError):
            continue
        mx = max(abs(px1), abs(py1), abs(px2), abs(py2))
        # 0–1 归一化坐标（模型偶发输出小数）
        if mx <= 1.000001:
            ax1 = max(0, int(round(px1 * orig_w)))
            ax2 = max(0, int(round(px2 * orig_w)))
            ay1 = max(0, int(round(py1 * orig_h)))
            ay2 = max(0, int(round(py2 * orig_h)))
        elif mx > 100.0:
            # 误用像素坐标：直接裁剪到边界内
            ax1 = int(np.clip(px1, 0, orig_w - 1))
            ax2 = int(np.clip(px2, 0, orig_w))
            ay1 = int(np.clip(py1, 0, orig_h - 1))
            ay2 = int(np.clip(py2, 0, orig_h))
        else:
            ax1 = max(0, int(px1 / 100.0 * orig_w))
            ay1 = max(0, int(py1 / 100.0 * orig_h))
            ax2 = min(orig_w, int(px2 / 100.0 * orig_w))
            ay2 = min(orig_h, int(py2 / 100.0 * orig_h))
        if ax2 < ax1:
            ax1, ax2 = ax2, ax1
        if ay2 < ay1:
            ay1, ay2 = ay2, ay1
        if ax2 > ax1 and ay2 > ay1:
            results.append((ax1, ay1, ax2, ay2))

    return results


def _clip_bbox_to_margin_frame(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    w: int,
    h: int,
    margin_frac: float,
) -> list[tuple[int, int, int, int]]:
    """与四条边框带状区域求交；丢弃落入腺体中部的大框误检。"""
    m = max(4, int(round(min(h, w) * margin_frac)))
    strips = (
        (0, 0, w, min(m, h)),
        (0, max(0, h - m), w, h),
        (0, 0, min(m, w), h),
        (max(0, w - m), 0, w, h),
    )
    out: list[tuple[int, int, int, int]] = []
    for sx1, sy1, sx2, sy2 in strips:
        ix1 = max(x1, sx1)
        iy1 = max(y1, sy1)
        ix2 = min(x2, sx2)
        iy2 = min(y2, sy2)
        if ix2 > ix1 + 1 and iy2 > iy1 + 1:
            out.append((ix1, iy1, ix2, iy2))
    return out


def blackout_label_regions(
    gray: np.ndarray,
    bboxes: list[tuple[int, int, int, int]],
    pad: int | None = None,
    *,
    margin_frac: float | None = None,
    max_box_area_frac: float | None = None,
    max_box_side_frac: float | None = None,
    return_stats: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, int]]:
    """将坐标框区域（含 pad）涂黑；默认仅作用于靠近片缘的带状带，避免误伤腺体。

    形参为 None 时采用内置默认值，并允许环境变量覆盖该默认值（显式传入非 None 则不再读 env）。
    """
    empty_stats: dict[str, int] = {
        "bboxes_in": 0,
        "skipped_area": 0,
        "skipped_side": 0,
        "skipped_no_margin_intersection": 0,
        "painted_rects": 0,
    }
    if not bboxes:
        return (gray.copy(), empty_stats) if return_stats else gray

    pad_eff = 4 if pad is None else pad
    if pad is None:
        pad_eff = _optional_env_int("LABEL_BLACKOUT_PAD", pad_eff)

    margin_eff = 0.22 if margin_frac is None else margin_frac
    if margin_frac is None:
        margin_eff = _optional_env_float("LABEL_BLACKOUT_MARGIN_FRAC", margin_eff)

    max_area_eff = 0.052 if max_box_area_frac is None else max_box_area_frac
    if max_box_area_frac is None:
        max_area_eff = _optional_env_float("LABEL_MAX_BOX_AREA_FRAC", max_area_eff)

    max_side_eff = 0.36 if max_box_side_frac is None else max_box_side_frac
    if max_box_side_frac is None:
        max_side_eff = _optional_env_float("LABEL_MAX_BOX_SIDE_FRAC", max_side_eff)

    result = gray.copy()
    h, w = result.shape[:2]
    max_area_px = max_area_eff * float(h * w)
    max_side_w = max_side_eff * float(w)
    max_side_h = max_side_eff * float(h)

    stats = {
        "bboxes_in": len(bboxes),
        "skipped_area": 0,
        "skipped_side": 0,
        "skipped_no_margin_intersection": 0,
        "painted_rects": 0,
    }

    for x1, y1, x2, y2 in bboxes:
        bw, bh = x2 - x1, y2 - y1
        if bw * bh > max_area_px:
            stats["skipped_area"] += 1
            continue
        if bw > max_side_w or bh > max_side_h:
            stats["skipped_side"] += 1
            continue
        clips = _clip_bbox_to_margin_frame(x1, y1, x2, y2, w, h, margin_eff)
        if not clips:
            stats["skipped_no_margin_intersection"] += 1
            continue
        for bx1, by1, bx2, by2 in clips:
            rx1 = max(0, bx1 - pad_eff)
            ry1 = max(0, by1 - pad_eff)
            rx2 = min(w, bx2 + pad_eff)
            ry2 = min(h, by2 + pad_eff)
            if rx2 <= rx1 + 1 or ry2 <= ry1 + 1:
                continue
            result[ry1:ry2, rx1:rx2] = 0
            stats["painted_rects"] += 1

    if _truthy_env("LABEL_GUARD_DEBUG"):
        _LOG.info("blackout_label_regions %s", stats)

    if return_stats:
        return result, stats
    return result
