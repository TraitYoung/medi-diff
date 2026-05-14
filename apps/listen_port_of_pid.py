#!/usr/bin/env python3
"""从 PID 解析 TCP LISTEN 端口：优先 ss → lsof → /proc/net 兜底。供 start.sh 调用。"""
from __future__ import annotations

import os
import re
import subprocess
import sys


def from_ss(pid: int) -> str | None:
    try:
        out = subprocess.check_output(["ss", "-tlnp"], text=True, stderr=subprocess.DEVNULL, timeout=5)
    except Exception:
        return None
    for line in out.splitlines():
        if f"pid={pid}," not in line and f"pid={pid})" not in line:
            continue
        m = re.search(r"0\.0\.0\.0:(\d+)|\[::\]:(\d+)|\*:(\d+)", line)
        if m:
            p = m.group(1) or m.group(2) or m.group(3)
            if p:
                return p
    return None


def from_lsof(pid: int) -> str | None:
    try:
        r = subprocess.run(
            ["lsof", "-Pan", "-p", str(pid), "-iTCP", "-sTCP:LISTEN"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    if r.returncode != 0 or not r.stdout:
        return None
    for line in r.stdout.splitlines():
        if "LISTEN" not in line:
            continue
        m = re.search(r":(\d+)\s+\(LISTEN\)", line) or re.search(r"\*:(\d+)", line)
        if m:
            return m.group(1)
    return None


def from_proc(pid: int) -> str | None:
    """通过 /proc/net/tcp 的 inode 与 /proc/<pid>/fd/ 匹配找 LISTEN 端口。"""
    try:
        proc_fd = f"/proc/{pid}/fd"
        if not os.path.isdir(proc_fd):
            return None
        # 建立该 PID 下 socket inode → fd 的映射
        sock_inodes: set[int] = set()
        for fd_name in os.listdir(proc_fd):
            fd_path = os.path.join(proc_fd, fd_name)
            try:
                target = os.readlink(fd_path)
                if target.startswith("socket:["):
                    inode = int(target[8:-1])
                    sock_inodes.add(inode)
            except (OSError, ValueError):
                continue
        if not sock_inodes:
            return None
        # 解析 /proc/net/tcp（含 v6 映射到 0.0.0.0 的情况）
        for tcp_file in ("/proc/net/tcp", "/proc/net/tcp6"):
            if not os.path.isfile(tcp_file):
                continue
            for line in open(tcp_file):
                parts = line.strip().split()
                if len(parts) < 10:
                    continue
                # 状态字段：0A = LISTEN
                if parts[3] != "0A":
                    continue
                inode = int(parts[9])
                if inode not in sock_inodes:
                    continue
                # 本地地址格式：00000000:1F90 或 0100007F:1F90 → 小端 hex port
                addr = parts[1]
                if ":" not in addr:
                    continue
                port_hex = addr.rsplit(":", 1)[1]
                port = int(port_hex, 16)
                if 1 <= port <= 65535:
                    return str(port)
    except Exception:
        pass
    return None


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(1)
    pid = int(sys.argv[1])
    for fn in (from_ss, from_lsof, from_proc):
        try:
            port = fn(pid)
            if port:
                print(port)
                sys.exit(0)
        except Exception:
            continue
    sys.exit(1)


if __name__ == "__main__":
    main()
