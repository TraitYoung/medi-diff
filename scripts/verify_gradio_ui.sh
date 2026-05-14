#!/usr/bin/env bash
# 自检：本机 Gradio 是否为本仓库当前精简界面
# 用法: bash scripts/verify_gradio_ui.sh [端口，默认 7860]
set -euo pipefail
PORT="${1:-7860}"
URL="http://127.0.0.1:${PORT}/"
echo "GET $URL"
if ! out="$(curl -fsS --max-time 5 "$URL" 2>/dev/null)"; then
    echo "失败：无法连接 $URL （进程未监听或端口错误）"
    exit 1
fi
if echo "$out" | grep -q '生成（SD1.5）'; then
    echo "OK：含 Tab「生成（SD1.5）」。"
else
    echo "警告：未发现「生成（SD1.5）」文案。"
fi
if echo "$out" | grep -Fq '"快速"'; then
    echo 'OK：模式单选含「快速」选项。'
else
    echo "警告：未发现模式选项「快速」（可能非当前版页面）。"
fi
exit 0
