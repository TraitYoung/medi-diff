#!/usr/bin/env bash
# 强制释放本项目的 FastAPI / Gradio 端口（不依赖 start.sh 是否还在运行）
# 用法：
#   bash apps/stop.sh
#   bash apps/stop.sh --api-port 8000 --ui-port 7860

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=./port_util.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/port_util.sh"

API_PORT=8000
UI_PORT=7860

while [[ $# -gt 0 ]]; do
    case "$1" in
        --api-port) API_PORT="$2"; shift 2 ;;
        --ui-port)  UI_PORT="$2";  shift 2 ;;
        -h|--help)
            head -n 6 "$0" | tail -n +2
            exit 0
            ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

echo "[stop] 结束 Gradio 进程并清理端口（含 ${UI_PORT}/$((UI_PORT+1))/$((UI_PORT+2))，防止上次漂端口残留）…"
pkill -f "python3.*app_gradio" 2>/dev/null || true
# 仅清端口可能漏掉孤儿 uvicorn（与 start.sh 脱节时）；按命令行一并结束 FastAPI。
pkill -f "uvicorn.*api_server:app" 2>/dev/null || true
pkill -f "apps\.api_server:app" 2>/dev/null || true
sleep 0.3
port_kill_tcp "$API_PORT"
port_ensure_free "$API_PORT"
for _p in "$UI_PORT" $((UI_PORT + 1)) $((UI_PORT + 2)); do
    port_kill_tcp "$_p"
done
rm -f "$ROOT/.gradio_pid" 2>/dev/null || true
echo "[stop] 完成。可重新 bash apps/start.sh"
