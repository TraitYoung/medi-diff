#!/usr/bin/env bash
set -euo pipefail

# LoRA v5 训练入口（v3/v4 同架构 + Phase 3 数据增强）。
#
# 前置步骤：
#   python3 scripts/training/prepare_lora_dataset_v5.py
#
# 改进点（相对 v4）：
# - 数据集：CBIS_CLEAN_V3（CBIS_CLEAN CC + CBIS_CLEAN_V2 MLO 合并）
# - 训练时数据增强：RandomHorizontalFlip + RandomRotation(8°) + ColorJitter(±5%)
# - 步数：5000（v4 的 4000 + 增强补偿）
# - warmup：500（v4 的 200，适配增强后的变化）

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

DATASET_DIR="${DATASET_DIR:-datasets/CBIS_CLEAN_V3}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/lora/mammo_sd15_v5}"
BASE_MODEL_LOCAL="${BASE_MODEL_LOCAL:-hf_cache/sd15}"
BASE_MODEL="${BASE_MODEL:-runwayml/stable-diffusion-v1-5}"
MAX_STEPS="${MAX_STEPS:-5000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
RANK="${RANK:-64}"
LR="${LR:-1e-4}"
SEED="${SEED:-2026}"

if [[ ! -f "$DATASET_DIR/metadata.csv" ]]; then
  echo "[ERR] Missing $DATASET_DIR/metadata.csv. Run prepare_lora_dataset_v5.py first." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR" outputs/logs/lora_v5

CMD=(
  accelerate launch --num_cpu_threads_per_process=8
  scripts/training/train_mammo_lora.py
  --dataset-dir "$DATASET_DIR"
  --output-dir "$OUTPUT_DIR"
  --max-train-steps "$MAX_STEPS"
  --learning-rate "$LR"
  --rank "$RANK"
  --batch-size "$BATCH_SIZE"
  --resolution 512
  --gradient-accumulation 1
  --lr-scheduler cosine
  --lr-warmup-steps 500
  --mixed-precision fp16
  --save-steps 500
  --seed "$SEED"
)

if [[ -n "$BASE_MODEL_LOCAL" ]]; then
  CMD+=(--base-model-local "$BASE_MODEL_LOCAL")
else
  CMD+=(--base-model "$BASE_MODEL")
fi

echo "[Train LoRA v5] ${CMD[*]}"
"${CMD[@]}" 2>&1 | tee "outputs/logs/lora_v5/train_lora_v5_$(date +%Y%m%d_%H%M%S).log"
