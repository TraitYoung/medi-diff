# Python 知识整理 — 从 medi-diff 项目入手

本文档面向从本项目入门 Python 的读者，按知识点分类，每个条目附带项目中的实际代码引用。

---

## 1. 模块与导入

### 1.1 `from __future__ import annotations`（PEP 563）

**项目用法：** 几乎每个 `.py` 文件第一行。

```python
from __future__ import annotations
```

**作用：** 推迟类型注解的求值，所有注解变成字符串，运行时不会被解析。
- 避免前向引用问题（类可以引用自己）
- 减少 import 开销（类型注解不会在运行时加载）
- Python 3.11+ 默认行为，3.7-3.10 需要这个 import

### 1.2 `if __name__ == "__main__"` 守卫

```python
# run_mammo_sd15.py:968-971
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, ...)
    main()
```

当脚本被 `python script.py` 直接运行时执行 `main()`，被 `import` 时不执行。

### 1.3 `sys.path.insert(0, ...)` 动态添加搜索路径

```python
# run_mammo_sd15.py:36-37
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
```

**作用：** 在 PYTHONPATH 前面插入项目根目录，让脚本能 `from scripts.core.xxx import` — 不需要把项目装成 package。

### 1.4 可选依赖的延迟导入（lazy import）

```python
# review_generated_images.py:54-113
try:
    from scipy.stats import wasserstein_distance as _scipy_wasserstein
except Exception:
    _scipy_wasserstein = None  # 降级为 None，后续判断跳过

try:
    import torch as _torch
    import piq as _piq
    _PIQ_OK = True
except Exception:
    _PIQ_OK = False
```

**模式：** 非核心依赖导入失败时设 `None` 或 `_FLAG = False`，在用到的函数里先检查再使用。避免 `ImportError` 导致整个脚本崩溃。

```python
# 使用时检查
def compute_brisque(gray):
    if not _PIQ_OK:
        return -1  # 降级返回值
    # ... 正常逻辑
```

## 2. 类型注解（Type Hints）

### 2.1 联合类型：`X | None` 代替 `Optional[X]`

```python
# pipeline_config.py:18-39
@dataclass
class GenParams:
    strength: float = 0.44
    prompt: str = (...)
```

```python
# tuning_state.py:71
def load_latest_tuning_state(report_base: Path) -> dict[str, Any] | None:
```

Python 3.10+ 支持 `X | Y`，比 `Union[X, Y]` 简洁。

### 2.2 复杂泛型

```python
# run_mammo_sd15.py:66-67
def _detect_metal_markers(gray, ...) -> list[tuple[int, int, int]]:
```

```python
# api_server.py:118
jobs: dict[str, JobRecord] = {}
```

`list[tuple[int, int, int]]` 比 `List[Tuple[int, int, int]]` 更现代（Python 3.9+ 内置容器支持泛型）。

### 2.3 `Literal` 类型

```python
# api_server.py:106
kind: Literal["generate", "review"]
```

约束参数只能取特定字符串值，IDE 能自动补全，mypy 能做穷举检查。

## 3. 数据结构

### 3.1 `dataclass` — 数据容器首选

```python
# pipeline_config.py:17-39
@dataclass
class GenParams:
    strength: float = 0.44
    guidance_scale: float = 7.5
    num_steps: int = 50
    scheduler: str = "dpm"
```

- 自动生成 `__init__`、`__repr__`、`__eq__`
- 可设默认值，支持类型注解
- 比 `NamedTuple` 更灵活（可以修改），比手写 class 更省代码

### 3.2 Pydantic `BaseModel` — API 数据校验

```python
# api_server.py:65-77
class GenerateSD15Request(BaseModel):
    num_images: int = Field(default=6, ge=1, le=50, description="生成张数")
    strength: float = Field(default=0.42, ge=0.05, le=0.80)
```

- `Field(ge=, le=)` 自动校验取值范围
- FastAPI 里自动生成 OpenAPI schema
- 比 `dataclass` 多了运行时校验和序列化能力

### 3.3 `collections.Counter` — 频次统计

```python
# review_generated_images.py:2431-2442
tag_counts = Counter()
for r in rows:
    tags_str = r.get('tags', '')
    if tags_str and tags_str != 'pass':
        for t in tags_str.split('|'):
            tag_counts[t.strip()] += 1
violation_rates = {k: v / max(1, len(rows)) for k, v in tag_counts.items()}
```

`Counter` 继承 `dict`，额外提供 `.most_common()` 等方法。

### 3.4 `frozenset` — 不可变集合用于常量查找

```python
# review_generated_images.py:1851-1857
hard_tags = frozenset({
    'BANDING', 'BLOWOUT', 'AREA_BAD', 'DEAD_DARK', 'SHAPE_ODD',
    'ANATOMY_NON_MAMMO', 'CONTOUR_FRACTURED', ...
})
# 使用: any(t in hard_tags for t in tags)
```

`frozenset` 不可变、可哈希、可作全局常量。`in` 操作 O(1)。

## 4. 函数设计模式

### 4.1 Kwarg-only 参数（`*` 分隔符）

```python
# label_guard.py:118-123
def erase_bright_border_labels(
    gray: np.ndarray,
    border_frac: float = 0.028,
    bright_pct: float = 99.0,
    *,
    mode: str = "cc",
) -> np.ndarray:
```

`*` 后面的参数必须用关键字传递，不能按位置传参。防止调用时搞混参数顺序。

### 4.2 默认参数绑定不可变对象

项目中所有默认值都是 `int`、`float`、`str`、`None`，从不使用 `[]` 或 `{}`。

```python
# 正确 ✅
def func(data: list | None = None):
    if data is None:
        data = []

# 错误 ❌ — 同一个 list 会在多次调用间共享
def func(data: list = []):
```

### 4.3 尽早 return（guard clause）

```python
# label_guard.py:82-102
for i in range(len(areas)):
    area = float(areas[i])
    if area <= 0 or area > label_max_area:
        continue  # 不符合条件就直接跳过
    cc_label = i + 1
    if cc_h < 10 or cc_w < 8:
        continue
    if max(cc_h, cc_w) / max(1, min(cc_h, cc_w)) > 7:
        continue
    # ... 真正处理逻辑
```

减少嵌套深度，逻辑扁平化。

### 4.4 字典解包 `**`

```python
# run_mammo_sd15.py:433-442
if _has_source_artifact_burden(
    marker_score=marker_score,
    bg_marker_count=bg_marker_count,
    **lesion_stats,  # dict 解包为关键字参数
):
```

也可以用于合并字典：
```python
# review_generated_images.py:1803-1807
weights = dict(DEFAULT_GROUP_WEIGHTS)
weights.update(json.loads(args.weights))  # 用户覆盖
```

## 5. NumPy 性能要点

### 5.1 向量化代替 Python 循环

```python
# label_guard.py:76-109 — 用 numpy 切片和布尔索引
nb_outside = nb_gray[nb_lbl != cc_label].flatten()
dark_frac = float(np.sum(nb_outside < dark_bg_thr)) / len(nb_outside)
```

核心原则：**遍历是 numpy 数组上做，不是 Python 循环里做**。

### 5.2 `ogrid` 生成坐标网格

```python
# run_mammo_sd15.py:92-93
yy, xx = np.ogrid[y1:y2, x1:x2]
core = ((xx - x) ** 2 + (yy - y) ** 2) <= (r + 2) ** 2
```

`ogrid` 返回 "open" 网格，内存开销远小于 `mgrid`。适用于生成二维距离场。

### 5.3 `np.indices` 生成完整坐标

```python
# review_generated_images.py:485
yy, xx = np.indices((h, w))
```

返回与图像同尺寸的坐标矩阵，用于 FFT 半径计算等。

### 5.4 就地操作 vs 新建数组

```python
# label_guard.py:112-113
result[label_regions > 0] = (result[label_regions > 0].astype(np.float32) * 0.35).astype(np.uint8)
```

布尔索引 + 赋值比遍历每个像素快 100x+。

### 5.5 数据类型转换

```python
gray.astype(np.float32)  # 转浮点准备计算
np.clip(result, 0, 255).astype(np.uint8)  # 转回 uint8 保存
```

uint8 范围 0-255，float32/64 用于数值计算。计算完记得 clip 防止溢出。

## 6. OpenCV（cv2）常用操作

### 6.1 图像读取 / 写入

```python
gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
cv2.imwrite("out.png", gray)
```

灰度图是 `(H, W) uint8` 二维数组；彩色图是 `(H, W, 3)`。

### 6.2 Otsu 自动阈值

```python
otsu_thr, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

Otsu 自动找一个最优阈值区分前景和背景，不需要手动调。

### 6.3 形态学操作

```python
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
```

- `MORPH_CLOSE`：先膨胀再腐蚀，填小黑洞
- `MORPH_OPEN`：先腐蚀再膨胀，去小白点
- `MORPH_ELLIPSE`：椭圆形结构元素，圆滑过渡

### 6.4 连通域分析

```python
n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
for i in range(1, n):
    area = int(stats[i, cv2.CC_STAT_AREA])
    x, y, w, h = stats[i, cv2.CC_STAT_LEFT:cv2.CC_STAT_TOP+1, ...]
```

`stats` 每行是 `[x, y, w, h, area]`；`labels` 是和输入同尺寸的标记图。

### 6.5 轮廓分析

```python
cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
c = max(cnts, key=cv2.contourArea)  # 最大轮廓
area = float(cv2.contourArea(c))
circularity = 4 * np.pi * area / (cv2.arcLength(c, True) ** 2)
```

### 6.6 缩放插值选择

```python
# 缩小：INTER_AREA（平均采样，最不易产生摩尔纹）
cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

# 放大：INTER_LANCZOS4（8x8 邻域 sinc 插值，最高质量）
cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
```

## 7. PyTorch 常用模式

### 7.1 设备选择

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe.to(device)
```

### 7.2 混合精度加载

```python
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    ..., torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
)
```

FP16 在 GPU 上内存减半、速度翻倍，在 CPU 上不被支持。

### 7.3 `torch.no_grad()` 推理上下文

```python
with torch.no_grad():
    latents = vae.encode(pixel_values).latent_dist.sample()
```

禁用梯度计算，节省显存，加速推理。

### 7.4 `torch.Generator` 确定性随机

```python
gen = torch.Generator(device=device).manual_seed(base_seed)
result = pipe(..., generator=gen).images[0]
```

保证同一 seed 产生完全相同的图像。

### 7.5 指数移动平均（EMA）

```python
# train_mammo_lora.py:373
loss_ema = loss.item() if loss_ema is None else 0.98 * loss_ema + 0.02 * loss.item()
```

比直接取最近 N 个 loss 的平均更节省内存和计算。

## 8. 并发与并行

### 8.1 `multiprocessing.Pool` 并行处理

```python
# build_breast_masks.py:121-125
with mp.Pool(args.workers) as pool:
    for i, r in enumerate(pool.imap_unordered(process_one, payloads, chunksize=16)):
        results.append(r)
```

- `imap_unordered`：结果谁先出来就先处理，不保证顺序，比 `map` 更快
- `chunksize=16`：每次给进程分配 16 个任务，减少 IPC 开销
- 适用场景：CPU-bound 的图像特征提取、文件批量处理

### 8.2 `threading.Thread` 后台异步任务

```python
# api_server.py:149
threading.Thread(target=_run_job, args=(job.id,), daemon=True).start()
```

- `daemon=True`：主进程退出时自动终止
- 适合 I/O-bound（subprocess 等待），不适合 CPU-bound（GIL 限制）

### 8.3 `ThreadPoolExecutor` 的替代

项目中未使用 `concurrent.futures`，但 `multiprocessing.Pool` 是 CPU-bound 场景更好的选择。

### 8.4 锁机制

```python
# api_server.py:119-148
jobs_lock = threading.Lock()
with jobs_lock:
    jobs[job.id] = job
```

多线程写共享 dict 必须加锁，否则可能数据损坏。

## 9. 文件 I/O

### 9.1 `pathlib.Path` — 现代路径操作

```python
# 常用操作
ROOT = Path(__file__).resolve().parents[2]     # 项目根目录
out_dir = output_base / f"{prefix}_{ts}"        # 路径拼接用 /
out_dir.mkdir(parents=True, exist_ok=True)       # 递归创建目录
path.is_file() / path.is_dir() / path.exists()  # 存在性检查
path.read_text(encoding="utf-8")                 # 一次性读文本
path.write_text(content, encoding="utf-8")       # 一次性写文本
list(path.glob("*.png"))                         # 通配符搜索
list(path.rglob("*.png"))                        # 递归搜索
path.suffix / path.stem / path.name              # 文件扩展名/干名/全名
```

比 `os.path` 更可读，拼接不用写 `os.path.join`。

### 9.2 JSON 读写标准写法

```python
# 写
path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

# 读
data = json.loads(path.read_text(encoding="utf-8"))
```

- `ensure_ascii=False`：中文直接存储，不转义成 `\uXXXX`
- `indent=2`：格式化输出，便于 git diff 和人工查看

### 9.3 CSV 写入

```python
# review_generated_images.py:2421-2425
with open(csv_path, 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
    w.writeheader()
    for r in rows:
        w.writerow(_row_for_csv(r))
```

- `DictWriter` 用字段名匹配，不用担心列顺序
- `newline=''` 防止 Windows 下多余空行

## 10. 错误处理

### 10.1 `try/except` 精确异常

```python
# ask_advisor.py:366-375
try:
    out = ask_advisor(user, ...)
except RuntimeError as e:
    print(str(e), file=sys.stderr)
    return 1
except urllib.error.HTTPError as e:
    print(f"HTTP {e.code}: ...", file=sys.stderr)
    return 1
except Exception as e:         # 兜底
    print(f"请求失败: {e}", file=sys.stderr)
    return 1
```

**原则：** 从具体到笼统，先 catch 特定异常，最后兜底。

### 10.2 项目里的反例（值得注意）

```python
# 过于宽泛的 Exception 捕获 — 只在"必须容错"的边界场景用
except Exception:
    pass  # 容易吞掉真正的 bug
```

项目中 `filter_source_pool` 确实有 `except Exception`，因为个别损坏的源图不能中断整个批次生成。这种"容错"需谨慎使用。

## 11. CLI 设计

### 11.1 `argparse` 参数分组

```python
# run_mammo_sd15.py:720-778
p = argparse.ArgumentParser(description="...", formatter_class=argparse.RawDescriptionHelpFormatter)
g = p.add_argument_group("Model")
g.add_argument("--base-model-local", ...)
g = p.add_argument_group("Sampling")
g.add_argument("--strength", type=float, default=GenParams.strength)
```

- `add_argument_group`：把参数按功能分组，`--help` 输出更清晰
- 默认值引用 `GenParams.strength` 而非手写 `0.44`，单一来源避免不一致

### 11.2 `BooleanOptionalAction`（Python 3.9+）

```python
p.add_argument('--auto-calibrate', action=argparse.BooleanOptionalAction, default=True)
# 自动生成 --auto-calibrate 和 --no-auto-calibrate 两个 flag
```

## 12. 上下文管理器

### 12.1 多重 with

```python
# review_generated_images.py:149-151
with tempfile.TemporaryDirectory(prefix='review_gen_') as tg, \
     tempfile.TemporaryDirectory(prefix='review_real_') as tr:
    # tg 和 tr 在 with 结束后自动删除
```

### 12.2 自定义 contextmanager

```python
# review_generated_images.py:146-159
@contextlib.contextmanager
def _temporary_symlink_dirs(gen_paths, real_paths):
    with tempfile.TemporaryDirectory(prefix='review_gen_') as tg, \
         tempfile.TemporaryDirectory(prefix='review_real_') as tr:
        for i, p in enumerate(gen_paths):
            os.symlink(p.resolve(), Path(tg) / f'g_{i:05d}{ext}')
        yield Path(tg), Path(tr)  # 把控制权交给 with 块
    # 退出后自动清理
```

`yield` 之前的代码相当于 `__enter__`，之后的相当于 `__exit__`。

## 13. logging

### 13.1 模块级 logger

```python
# run_mammo_sd15.py:39
logger = logging.getLogger(__name__)

# 使用
logger.info("Device: %s  Mode: %s", device, args.mode)
logger.warning("Skipping %d images...", n)
```

- `__name__` 让子模块的日志显示完整路径
- 格式化用 `%s` 而非 f-string — logging 只在需要输出时才做字符串格式化，省性能

### 13.2 入口点配置

```python
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    main()
```

只在入口点配置 logging，库代码不应调用 `basicConfig`。

## 14. 环境变量与配置

### 14.1 `.env` 手动解析

```python
# ask_advisor.py:45-63
def _load_dotenv() -> None:
    env_path = ROOT / ".env"
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip())
```

手写 `.env` 解析器（不用 python-dotenv），`setdefault` 不覆盖已设的环境变量。

### 14.2 `os.environ.setdefault` 模式

```python
os.environ.setdefault("HF_ENDPOINT", args.hf_endpoint)
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")
```

## 15. 测试编写

### 15.1 纯函数式测试（不使用 unittest/pytest 框架）

```python
# test_fullimage_output_size.py
def test_resize_long_side_downscale():
    wide = np.random.RandomState(42).randint(0, 256, (3000, 2000), dtype=np.uint8)
    result = resize_long_side_gray(wide, 2048)
    assert max(result.shape) == 2048, f"Expected 2048, got {max(result.shape)}"

if __name__ == "__main__":
    test_resize_long_side_downscale()
    test_resize_long_side_upscale()
    # ...
    print("All tests passed.")
```

- 用 `assert` 而非框架的 `self.assertEqual`
- `np.random.RandomState(42)` 保证每次运行相同的测试数据
- 适合小项目/脚本，大项目建议用 pytest

## 16. subprocess 调用

### 16.1 调用另一个 Python 脚本

```python
# app_gradio.py:31-45
proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
```

- `text=True`：输出是 str 而非 bytes
- `capture_output=True`：捕获 stdout/stderr
- `check=False`：不因非零返回码抛异常，手动处理 `proc.returncode`

### 16.2 找到最近生成的目录

```python
# run_generate_eval_advise.py:133
candidates = sorted(gen_base.glob(f"{tag_prefix}_*"), key=lambda p: p.stat().st_mtime)
gen_dir = candidates[-1]  # 最新修改的
```

## 17. 代码文档与注释规范

### 17.1 模块 docstring（简明）

```python
"""GenParams: single source of truth for all generation parameters.

All parameter defaults live here. CLI argparse definitions in run_mammo_sd15.py
reference these defaults rather than duplicating them.
"""
```

说明"为什么存在"而非"包含什么"。

### 17.2 函数 docstring（参数+用途）

```python
def erase_background_labels(gray: np.ndarray) -> np.ndarray:
    """Erase DICOM burn-in labels from background regions.

    Core heuristic: each bright small CC whose local neighbourhood is >60% dark
    background is classified as a label and removed.
    """
```

说明算法思路，而非逐行解释代码。

## 18. 综合要点速查

| 要点 | 说明 |
|------|------|
| `from __future__ import annotations` | 延迟注解求值，每个文件顶部 |
| `X \| None` | 比 `Optional[X]` 更现代 |
| `@dataclass` | 自动生成 `__init__`/`__repr__` |
| `pathlib.Path` | 比 `os.path` 更易读 |
| `Path.read_text/write_text` | 简化的文件读写 |
| `ensure_ascii=False` | JSON 里中文不转义 |
| 向量化 > for 循环 | NumPy 运算的核心原则 |
| `cv2.INTER_AREA` | 缩小时用 |
| `cv2.INTER_LANCZOS4` | 放大时用 |
| `torch.no_grad()` | 推理时禁用梯度 |
| `multiprocessing.Pool` | CPU 并行 |
| `threading.Thread(daemon=True)` | I/O 后台任务 |
| `try/except 具体异常 > Exception` | 错误处理优先级 |
| `logging.getLogger(__name__)` | 模块级日志 |
| `argparse.add_argument_group` | 参数分组可读 |
| `Contrast with `subprocess.run` | 调用外部脚本 |
