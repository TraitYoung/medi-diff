#!/usr/bin/env bash
# 全自动：生成 → 评审 → Qwen 顾问 → outputs/reports/<tag>/FINAL_REPORT.md
# 默认仅用文本顾问（省 token）；需看图加：--advisor-mode both；再压缩上下文：--compact-advisor
# 生成阶段默认与 run_mammo_sd15 一致开启频域后处理；若需 raw / 省时可加：--no-postprocess
# 用法示例：
#   bash scripts/assistant/start_full_report.sh
#   bash scripts/assistant/start_full_report.sh --num-images 6 --compact-advisor

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
exec python3 scripts/assistant/run_full_report.py "$@"
