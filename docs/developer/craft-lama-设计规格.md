# CRAFT + LaMa 文字擦除管线设计

## 目标

全自动擦除 CBIS_CLEAN_V2 训练数据中的 DICOM 文字/标注，生成 CBIS_CLEAN_V3，并重训 LoRA v5 以根治文字幻觉。

## 背景

- CBIS_CLEAN_V2 已清除了乳腺 mask **外**的背景文字，但 mask **内**的文字标注（小字符、箭头、标记）仍然存在
- 当前基于亮度阈值的 CC 检测法无法区分高亮文字和高亮组织（灰阶重叠）
- SD1.5 本身带有文字偏见（预训练见过 DICOM 标签），用 SD 去擦文字会原地生成新文字
- 需要同时满足两个条件：检测器能识别文字形状（不只是亮度），擦除器没有文字偏见

## 管线架构

```
CBIS_CLEAN_V2 (1296 images, 1024px)
    │
    ▼
Step 1: EasyOCR CRAFT 文字检测
    - 输出: per-image bbox list + confidence
    - 策略: low threshold (宁可多检)
    │
    ▼
Step 2: Binary Mask 生成
    - bbox → mask (dilate 4px)
    - 跳过 mask 面积=0 的图像
    │
    ▼
Step 3: LaMa Inpainting
    - 主力: simple_lama (FFT+GAN, Places2)
    - 回退: OpenCV INPAINT_TELEA (mask面积 < 100px)
    │
    ▼
Step 4: 输出 CBIS_CLEAN_V3
    - 保持与 V2 相同目录结构
    - metadata_clean.csv 新增 text_bbox_count, text_area_pct 列
    - 保存检测可视化到 _debug/
    │
    ▼
Step 5: Caption 更新 (text-free)
    - 在每个 prompt 中显式注入: "no text, no labels, no annotations, no DICOM markers"
    │
    ▼
Step 6: LoRA v5 训练
    - 基于 CBIS_CLEAN_V3 + text-free captions
```

## 检测端: EasyOCR CRAFT

- EasyOCR 1.7.2 已安装，内置 CRAFT (Character Region Awareness for Text Detection)
- CRAFT 检测字符笔画的**几何特征**（笔画宽度、字符间距、空间排列），不依赖亮度
- Macdonald et al. (2024, PMID:38587767) 已验证 CRAFT+EasyOCR 在 415k 乳腺片上的效果
- 参数: `text_threshold=0.3, low_text=0.2` 低阈值扫描，宁可多检不漏检
- GPU 加速，单图 <1s

## 擦除端: LaMa

- `simple_lama` pip 包（需安装: `pip install simple-lama`）
- LaMa (Large Mask Inpainting) 基于 FFT 全局感受野 + 对抗训练
- 在 Places2（自然场景）上训练，**无任何 DICOM/医学文字知识** → 不可能生成文字
- 比 SD inpainting 快 10x+（单图 <2s）
- 回退: mask 面积 < 100px → OpenCV INPAINT_TELEA（更快且对小区域效果足够）

## 自动审核策略

全自动，零人工干预：

1. **检测**: all 1296 images，低阈值全扫
2. **擦除**: 所有检测到 bbox 的图像都擦除；无 bbox 的直接复制
3. **质量控制**: 无需人工审核。LaMa 的误擦（擦掉小块组织）概率极低，且即使发生也只是让那块变模糊，不会引入伪影。漏检的文字由 text-free caption 在训练时抑制
4. **异常处理**: 单张图检测/擦除失败 → 跳过，保留 V2 原图，记录到 error_log

## 文件结构

```
datasets/CBIS_CLEAN_V3/
├── metadata_clean.csv        # 新增列: text_bbox_count, text_area_pct, clean_method
├── _debug/                   # CRAFT 检测结果可视化
├── _logs/                    # 处理日志
│   └── error_log.jsonl
├── MLO/
│   ├── fatty/
│   ├── scattered/
│   ├── heterogeneous/
│   └── dense/
└── CC/  (同上)
```

## 关键文件

| 文件 | 作用 |
|------|------|
| `scripts/preprocessing/clean_text_craft_lama.py` | 新建，主管线脚本 |
| `scripts/preprocessing/generate_captions.py` | 修改，注入 text-free descriptions |
| `scripts/training/prepare_lora_dataset.py` | 修改，指向 CBIS_CLEAN_V3 |
| `scripts/training/train_lora_quick.py` | 修改，输出到 LoRA v5 |

## 依赖安装

```bash
pip install simple-lama
# EasyOCR 1.7.2 + OpenCV 4.13 已安装
```

## 预期效果

- 训练数据中文字区域减少 >90%
- text-free captions 在训练时强化"乳腺 = 无文字"的条件映射
- LoRA v5 生成图像的文字幻觉率预期从当前降低到接近零
- BRISQUE 应保持接近 R4 水平（17-20），不会因 LaMa 擦除引入显著降质
