#!/usr/bin/env bash
# 供 start.sh / stop.sh source：按 TCP 端口结束监听进程（含 lsof / ss 兜底）

port_kill_tcp_py() {
    local port="$1"
    python3 -c "
import subprocess, re, os, signal
port = int('${port}')
try:
    out = subprocess.check_output(['ss','-tlnp'], text=True, stderr=subprocess.DEVNULL)
except (FileNotFoundError, subprocess.CalledProcessError):
    out = ''
pids = set()
for line in out.splitlines():
    if f':{port}' in line and 'LISTEN' in line:
        for m in re.finditer(r'pid=(\d+)', line):
            pids.add(int(m.group(1)))
for pid in pids:
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
if pids:
    print('[port] ss 端口', port, '→ PID', sorted(pids), flush=True)
" 2>/dev/null || true
}

# 先 TERM 再必要时由调用方 KILL；这里直接清监听端口，对内进程树最稳
port_kill_tcp() {
    local port="$1"
    if command -v lsof >/dev/null 2>&1; then
        local p
        p=$(lsof -t -iTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)
        if [[ -n "$p" ]]; then
            echo "[port] $port ← lsof PID $p → kill -9"
            kill -9 $p 2>/dev/null || true
        fi
    else
        port_kill_tcp_py "$port"
    fi
    if command -v fuser >/dev/null 2>&1; then
        fuser -k "${port}/tcp" 2>/dev/null || true
    fi
    sleep 0.3
}

# 轮询直到 TCP $port 无 LISTEN（或超出次数）。解决「旧 uvicorn 仍占端口 → 新进程 Errno 98」。
port_ensure_free() {
    local port="$1"
    local max="${2:-30}"
    local i=0
    while [[ $i -lt "$max" ]]; do
        local busy=0
        if command -v lsof >/dev/null 2>&1; then
            [[ -n "$(lsof -t -iTCP:"$port" -sTCP:LISTEN 2>/dev/null)" ]] && busy=1
        else
            [[ $i -ge 12 ]] && return 0
        fi
        [[ "$busy" -eq 0 ]] && return 0
        port_kill_tcp "$port"
        sleep 0.25
        i=$((i + 1))
    done
    if command -v lsof >/dev/null 2>&1; then
        [[ -n "$(lsof -t -iTCP:"$port" -sTCP:LISTEN 2>/dev/null)" ]] && \
            echo "[port] 警告: TCP ${port} 上仍有 LISTEN，FastAPI/其它服务可能绑定失败（请 lsof -i :${port}）" >&2
    fi
    return 0
}
