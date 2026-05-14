#!/usr/bin/env bash
# 一键启动：FastAPI 后端 + Gradio 前端
# 用法：
#   bash apps/start.sh                     # FastAPI :8000 + Gradio :7860（默认 API 关热重载，稳定）
#   bash apps/start.sh --api-reload        # API 开 uvicorn --reload（仅监视 apps/，排除 app_gradio.py）
#   bash apps/start.sh --api-only          # 仅 API
#   bash apps/start.sh --ui-only           # 仅 Gradio（「生成」内含默认/调参子页，共用日志）
#   bash apps/start.sh --api-port 8080 --ui-port 7861
#   bash apps/start.sh --gradio-share      # Gradio 生成公网 share 链接（需网络）
#   bash apps/start.sh --no-reload         # 显式关闭 API 热重载（与默认相同）
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=./port_util.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/port_util.sh"

API_PORT=8000
UI_PORT=7860
MODE="both"   # both | api-only | ui-only
GRADIO_SHARE=""
# 默认关闭 uvicorn --reload：与 Gradio 同仓时，热重载易在重绑定 8000 时 Errno 98，导致 API 退出。
NO_RELOAD="1"

usage() {
    head -n 17 "$0" | tail -n +4
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)   usage; exit 0 ;;
        --api-only)  MODE="api-only"; shift ;;
        --ui-only)   MODE="ui-only";  shift ;;
        --api-port)  API_PORT="$2";   shift 2 ;;
        --ui-port)   UI_PORT="$2";    shift 2 ;;
        --gradio-share) GRADIO_SHARE=1; shift ;;
        --api-reload) NO_RELOAD=""; shift ;;
        --no-reload) NO_RELOAD=1; shift ;;
        *) echo "未知参数: $1 — 可使用: bash apps/start.sh --help"; exit 1 ;;
    esac
done

# 检查依赖
python3 -c "import fastapi, gradio, uvicorn" 2>/dev/null || {
    echo "[安装依赖] pip install fastapi uvicorn gradio"
    pip install --quiet fastapi "uvicorn[standard]" gradio
}

# ── 端口释放（启动前）─────────────────────────────────────────────────────────
# 先杀所有 app_gradio.py（防止多实例）并清 Gradio 常用漂端口 7860–7862
pkill -f "python3.*app_gradio" 2>/dev/null || true
pkill -f "uvicorn.*api_server:app" 2>/dev/null || true
pkill -f "apps\.api_server:app" 2>/dev/null || true
sleep 0.2
[[ "$MODE" != "ui-only"  ]] && { port_kill_tcp "$API_PORT"; port_ensure_free "$API_PORT"; }
if [[ "$MODE" != "api-only" ]]; then
    for _gp in "$UI_PORT" $((UI_PORT + 1)) $((UI_PORT + 2)); do
        port_kill_tcp "$_gp"
    done
fi
rm -f "$ROOT/.gradio_pid"

echo "======================================================"
echo " 乳腺钼靶扩散生成系统"
echo " 根目录: $ROOT"
echo " 模式  : $MODE"
[[ "$MODE" != "ui-only"  ]] && echo " FastAPI: http://0.0.0.0:${API_PORT}  (Swagger: /docs)"
[[ "$MODE" != "api-only" ]] && echo " Gradio : http://0.0.0.0:${UI_PORT}${GRADIO_SHARE:+  (附带 --share)}"
echo " ────────────────────────────────────────────────────"
echo " 日常浏览器（以下为默认端口；★ Gradio 请以启动后「实际监听」为准 ★）"
[[ "$MODE" != "api-only" ]] && echo "   Gradio 界面（默认）: http://127.0.0.1:${UI_PORT}"
[[ "$MODE" != "ui-only"  ]] && echo "   API 文档:            http://127.0.0.1:${API_PORT}/docs"
echo "======================================================"

# 后台函数：启动 FastAPI
start_api() {
    echo "[FastAPI] 启动中 (port $API_PORT)..."
    # 与上方清端口之间若间隔较久，再确认一次，避免 Errno 98。
    port_ensure_free "$API_PORT"
    local -a uv
    local logf="/tmp/fastapi_${API_PORT}.log"
    uv=(python3 -m uvicorn apps.api_server:app --host 0.0.0.0 --port "$API_PORT")
    if [[ -z "$NO_RELOAD" ]]; then
        uv+=(--reload)
        # 启用热重载时仅监视 apps/，并排除 app_gradio.py，减少无意义重载与端口争用（Errno 98）。
        uv+=(--reload-dir "$ROOT/apps")
        uv+=(--reload-exclude "app_gradio.py")
    fi
    uv+=(--log-level info)
    "${uv[@]}" >>"$logf" 2>&1 &
    API_PID=$!
    echo "[FastAPI] PID=$API_PID  （日志: tail -f $logf）"
}

# 后台：启动 Gradio（无缓冲日志 + $! 为真实 python 进程）
start_ui() {
    echo "[Gradio] 启动中 (port $UI_PORT)..."
    SHARE_ARG=()
    [[ -n "$GRADIO_SHARE" ]] && SHARE_ARG=(--share)
    local logf="/tmp/gradio_ui_${UI_PORT}.log"
    PYTHONUNBUFFERED=1 python3 apps/app_gradio.py --host 0.0.0.0 --port "$UI_PORT" "${SHARE_ARG[@]}" \
        >>"$logf" 2>&1 &
    UI_PID=$!
    echo "[Gradio] PID=$UI_PID  （日志: tail -50 $logf）"
}

# 用内核里该 PID 的 LISTEN 端口为准：ss（含 pid）失败时再用 lsof 按 PID 查 LISTEN
_gradio_listen_port() {
    local pid="$1"
    # listen_port_of_pid 在未 bind 时 exit 1；禁止在 set -e 的命令替换里传播该状态，否则脚本首轮就退出。
    [[ -z "${pid:-}" ]] && return 0
    python3 "$ROOT/apps/listen_port_of_pid.py" "$pid" 2>/dev/null || true
}

# 打印 Gradio 本机访问地址（优先 ss 查 PID，其次读日志）
print_gradio_effective_url() {
    local logf="/tmp/gradio_ui_${UI_PORT}.log"
    local actual_port=""
    local i=0
    # 等待进程完成 bind（最多 ~12s）
    while [[ $i -lt 48 ]]; do
        actual_port=$(_gradio_listen_port "${UI_PID:-}")
        [[ -n "$actual_port" ]] && break
        sleep 0.25
        i=$((i + 1))
    done
    # 兜底：从日志抓 Running on（无缓冲后一般很快出现）
    if [[ -z "$actual_port" ]] && [[ -f "$logf" ]]; then
        # 注意：在 set -euo pipefail 下，grep 无匹配会返回 1，导致整段脚本静默退出、
        # 后台 API/Gradio 被挂起 → 浏览器看似有页面但无法交互。管道末尾需 || true。
        actual_port=$(grep -oE 'http://(0\.0\.0\.0|127\.0\.0\.1):[0-9]+' "$logf" 2>/dev/null | tail -1 | sed -n 's/.*:\([0-9][0-9]*\).*/\1/p' || true)
        [[ -z "$actual_port" ]] && actual_port=$(grep -oE '0\.0\.0\.0:[0-9]+' "$logf" 2>/dev/null | tail -1 | sed 's/.*://' || true)
    fi
    echo ""
    if [[ -n "$actual_port" ]]; then
        echo "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓"
        echo "┃  请用浏览器打开（实际监听端口）                  ┃"
        echo "┃                                                  ┃"
        echo "┃       http://127.0.0.1:${actual_port}                      ┃"
        echo "┃                                                  ┃"
        echo "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛"
        echo ""
        if [[ -n "${UI_PID:-}" ]] && [[ -r "/proc/$UI_PID/cmdline" ]]; then
            echo "[自检] Gradio 进程命令行（应含 apps/app_gradio.py）:"
            tr '\0' ' ' < "/proc/$UI_PID/cmdline" | fold -s -w 100
            echo ""
        fi
        if [[ "$actual_port" != "$UI_PORT" ]]; then
            echo "⚠️  与默认 ${UI_PORT} 不同：旧进程可能仍占默认口。若界面是旧的，先: bash apps/stop.sh"
            echo ""
        fi
        if command -v curl >/dev/null 2>&1; then
            _code=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 3 "http://127.0.0.1:${actual_port}/" 2>/dev/null || echo "000")
            echo "[自检] curl http://127.0.0.1:${actual_port}/ → HTTP ${_code}"
            [[ "$_code" == "200" ]] || echo "     （若非 200，请看日志: tail -40 $logf）"
            echo ""
        fi
        echo "────────── AutoDL / 本机浏览器 ──────────"
        echo "  在实例里服务正常时，本机浏览器要访问必须走「映射」，不要只在自己电脑开 127.0.0.1（那会指本机，不是容器）。"
        echo "  ① AutoDL 控制台 → 自定义服务 / 开放端口 → 添加 TCP ${actual_port} → 用生成的 https:// 链接打开。"
        echo "  ② 或用 Cursor/VSCode「端口」里转发 ${actual_port}，再打开提示的 localhost 链接。"
        echo "  ③ 临时外网直链（需联网，勿传隐私）："
        echo "       bash apps/stop.sh && bash apps/start.sh --gradio-share"
        echo "     启动后在日志里会出现 *.gradio.live 或类似公网 URL。"
        echo "────────────────────────────────────────"
        echo ""
    else
        echo "[提示] 暂不解析到端口。请执行:"
        echo "       python3 \"$ROOT/apps/listen_port_of_pid.py\" ${UI_PID:-}"
        echo "   或: tail -40 $logf | grep Running"
        echo ""
    fi
}

# 捕获退出：先礼貌结束子进程 → 强杀 → 再按端口清一次（专治 uvicorn --reload / Gradio 子进程残留）
cleanup() {
    echo ""
    echo "[shutdown] 正在关闭服务..."
    _sig() {
        local pid="$1" sig="$2"
        [[ -n "${pid:-}" ]] && kill -s "$sig" "$pid" 2>/dev/null && return 0
        return 1
    }
    _sig "${API_PID:-}" TERM
    _sig "${UI_PID:-}" TERM
    sleep 0.8
    _sig "${API_PID:-}" KILL
    _sig "${UI_PID:-}" KILL
    sleep 0.3
    echo "[shutdown] 按端口再做一次清理（防止子进程/重载器残留）…"
    [[ "$MODE" != "ui-only"  ]] && port_kill_tcp "$API_PORT"
    if [[ "$MODE" != "api-only" ]]; then
        for _gp in "${UI_PORT}" $((UI_PORT + 1)) $((UI_PORT + 2)); do
            port_kill_tcp "$_gp"
        done
    fi
    echo "[shutdown] 完成。若端口仍被占，可单独执行: bash apps/stop.sh"
    exit 0
}
trap cleanup SIGINT SIGTERM

case "$MODE" in
    both)
        start_api
        sleep 2
        start_ui
        ;;
    api-only)
        start_api
        ;;
    ui-only)
        start_ui
        ;;
esac

if [[ "$MODE" != "api-only" ]]; then
    print_gradio_effective_url
    if [[ -n "${GRADIO_SHARE:-}" ]] && [[ -f "/tmp/gradio_ui_${UI_PORT}.log" ]]; then
        echo ""
        echo "[share] 正在从日志提取 Gradio 公网链接（若还未生成可稍后再 tail 日志）…"
        sleep 5
        grep -oE 'https?://[^ ]+' "/tmp/gradio_ui_${UI_PORT}.log" 2>/dev/null | tail -8 || \
            tail -15 "/tmp/gradio_ui_${UI_PORT}.log"
        echo ""
    fi
fi

echo ""
echo "服务已启动，按 Ctrl+C 退出（终端不会自动结束）。"

# 监控进程：意外退出时打印告警
_watch_proc() {
    local name="$1" pid="$2" logf="$3"
    [[ -z "$pid" ]] && return
    while kill -0 "$pid" 2>/dev/null; do
        sleep 2
    done
    echo ""
    echo "⚠️  [warn] $name 进程 PID=$pid 已退出！"
    [[ -n "$logf" ]] && echo "     查看日志: tail -40 $logf"
    echo "     重启服务: bash apps/stop.sh && bash apps/start.sh"
}

[[ "$MODE" != "ui-only"  ]] && _watch_proc "FastAPI" "${API_PID:-}" "/tmp/fastapi_${API_PORT}.log" &
[[ "$MODE" != "api-only" ]] && _watch_proc "Gradio"  "${UI_PID:-}"  "/tmp/gradio_ui_${UI_PORT}.log" &

# 关掉 set -e，防止子进程非零退出码导致整个 start.sh 退出
set +e
wait
