#!/usr/bin/env bash
# 一键启动：FastAPI 后端 + Gradio 前端
# 用法：
#   bash apps/start.sh                     # FastAPI :8000 + Gradio :7860
#   bash apps/start.sh --api-only          # 仅 API
#   bash apps/start.sh --ui-only           # 仅 Gradio
#   bash apps/start.sh --api-port 8080 --ui-port 7861
#   bash apps/start.sh --gradio-share      # Gradio 公网 share 链接
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/port_util.sh"

API_PORT=8000
UI_PORT=7860
MODE="both"
GRADIO_SHARE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            head -n 10 "$0" | tail -n +4
            exit 0 ;;
        --api-only)  MODE="api-only"; shift ;;
        --ui-only)   MODE="ui-only";  shift ;;
        --api-port)  API_PORT="$2";   shift 2 ;;
        --ui-port)   UI_PORT="$2";    shift 2 ;;
        --gradio-share) GRADIO_SHARE=1; shift ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

# 检查依赖
python3 -c "import fastapi, gradio, uvicorn" 2>/dev/null || {
    pip install --quiet fastapi "uvicorn[standard]" gradio
}

# 释放端口
pkill -f "python3.*app_gradio" 2>/dev/null || true
pkill -f "uvicorn.*api_server:app" 2>/dev/null || true
sleep 0.2
[[ "$MODE" != "ui-only"  ]] && { port_kill_tcp "$API_PORT"; port_ensure_free "$API_PORT"; }
if [[ "$MODE" != "api-only" ]]; then
    for _gp in "$UI_PORT" $((UI_PORT + 1)) $((UI_PORT + 2)); do
        port_kill_tcp "$_gp"
    done
fi
rm -f "$ROOT/.gradio_pid"

echo "=================================="
echo " medi-diff 启动"
echo " 模式: $MODE"
[[ "$MODE" != "ui-only"  ]] && echo " FastAPI → :${API_PORT}"
[[ "$MODE" != "api-only" ]] && echo " Gradio  → :${UI_PORT}"
echo "=================================="

start_api() {
    echo "[FastAPI] :${API_PORT} ..."
    port_ensure_free "$API_PORT"
    local logf="/tmp/fastapi_${API_PORT}.log"
    python3 -m uvicorn apps.api_server:app --host 0.0.0.0 --port "$API_PORT" --log-level info \
        >>"$logf" 2>&1 &
    API_PID=$!
    echo "        PID=$API_PID  log: $logf"
}

start_ui() {
    echo "[Gradio] :${UI_PORT} ..."
    local logf="/tmp/gradio_ui_${UI_PORT}.log"
    SHARE_ARG=()
    [[ -n "$GRADIO_SHARE" ]] && SHARE_ARG=(--share)
    PYTHONUNBUFFERED=1 python3 apps/app_gradio.py --host 0.0.0.0 --port "$UI_PORT" "${SHARE_ARG[@]}" \
        >>"$logf" 2>&1 &
    UI_PID=$!
    echo "        PID=$UI_PID  log: $logf"
}

_gradio_listen_port() {
    local pid="$1"
    [[ -z "${pid:-}" ]] && return 0
    python3 "$ROOT/apps/listen_port_of_pid.py" "$pid" 2>/dev/null || true
}

print_gradio_effective_url() {
    local actual_port=""
    local i=0
    while [[ $i -lt 48 ]]; do
        actual_port=$(_gradio_listen_port "${UI_PID:-}")
        [[ -n "$actual_port" ]] && break
        sleep 0.25
        i=$((i + 1))
    done
    [[ -z "$actual_port" ]] && actual_port="$UI_PORT"
    echo ""
    echo "  Gradio → http://127.0.0.1:${actual_port}"
    echo ""
}

cleanup() {
    echo ""
    echo "[shutdown] 正在关闭..."
    _sig() { local pid="$1" sig="$2"; [[ -n "${pid:-}" ]] && kill -s "$sig" "$pid" 2>/dev/null; }
    _sig "${API_PID:-}" TERM; _sig "${UI_PID:-}" TERM
    sleep 0.8
    _sig "${API_PID:-}" KILL; _sig "${UI_PID:-}" KILL
    sleep 0.3
    [[ "$MODE" != "ui-only"  ]] && port_kill_tcp "$API_PORT"
    if [[ "$MODE" != "api-only" ]]; then
        for _gp in "${UI_PORT}" $((UI_PORT + 1)) $((UI_PORT + 2)); do
            port_kill_tcp "$_gp"
        done
    fi
    echo "[shutdown] 完成"
    exit 0
}
trap cleanup SIGINT SIGTERM

case "$MODE" in
    both)     start_api; sleep 2; start_ui ;;
    api-only) start_api ;;
    ui-only)  start_ui ;;
esac

if [[ "$MODE" != "api-only" ]]; then
    print_gradio_effective_url
fi

echo "服务已启动，Ctrl+C 退出。"

_watch_proc() {
    local name="$1" pid="$2" logf="$3"
    [[ -z "$pid" ]] && return
    while kill -0 "$pid" 2>/dev/null; do sleep 2; done
    echo ""
    echo "[warn] $name PID=$pid 已退出，查看: tail -40 $logf"
}

[[ "$MODE" != "ui-only"  ]] && _watch_proc "FastAPI" "${API_PID:-}" "/tmp/fastapi_${API_PORT}.log" &
[[ "$MODE" != "api-only" ]] && _watch_proc "Gradio"  "${UI_PID:-}"  "/tmp/gradio_ui_${UI_PORT}.log" &

set +e
wait
