#!/usr/bin/env python3
"""
调用 .env 中的「顾问」大模型。

文本顾问 `ask_advisor`：
  ADVISOR_TEXT_BACKEND  auto（默认）：若配置了 DEEPSEEK_API_KEY 则用 DeepSeek，其次 GLM，否则 Qwen
                      deepseek / glm / qwen：强制指定后端
  —— DeepSeek（V4 专家模式默认开 thinking）：DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL
     / DEEPSEEK_MODEL（默认 deepseek-v4-pro；API 仅接受 pro/flash）/ DEEPSEEK_THINKING（enabled|disabled）
     / DEEPSEEK_REASONING_EFFORT（high|max，默认 max）/ DEEPSEEK_MAX_TOKENS（默认 8192）
  —— GLM（智谱 OpenAI 兼容）：GLM_API_KEY / GLM_BASE_URL（默认 https://open.bigmodel.cn/api/paas/v4）
     / GLM_MODEL（默认 glm-5.1）
  —— Qwen（DashScope 兼容）：QWEN_API_KEY / QWEN_BASE_URL / QWEN_MODEL（默认 qwen-plus）

图像顾问 `ask_advisor_vl` 仍使用 Qwen-VL：
  QWEN_API_KEY / QWEN_BASE_URL / QWEN_VL_MODEL（默认 qwen3-vl-plus）

用法：
  python3 scripts/assistant/ask_advisor.py "请用要点总结这段需求……"
  python3 scripts/assistant/ask_advisor.py -f notes.txt --system "你是技术顾问，回答用中文"
  python3 scripts/assistant/ask_advisor.py "点评这张图" --image path/to.png

不在代码里写死密钥；请勿将 .env 提交到 Git。
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# DeepSeek OpenAI 兼容接口当前仅接受 deepseek-v4-pro / deepseek-v4-flash（勿使用带 [1m] 后缀的别名）。
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-pro"

_DOTENV_LOADED = False


def _load_dotenv() -> None:
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True
    env_path = ROOT / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if val.startswith('"') and val.endswith('"'):
            val = val[1:-1].replace('\\"', '"')
        elif val.startswith("'") and val.endswith("'"):
            val = val[1:-1]
        os.environ.setdefault(key, val)


def _extract_assistant_text(payload: dict) -> str:
    """取 assistant 正文；兼容 DeepSeek thinking 场景的 content / reasoning_content。"""
    try:
        msg = payload["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"Unexpected API response: {payload!r}") from e

    def _s(x: object) -> str:
        if x is None:
            return ""
        if isinstance(x, str):
            return x.strip()
        return str(x).strip()

    content = _s(msg.get("content"))
    reasoning = _s(msg.get("reasoning_content"))
    if content:
        return content
    if reasoning:
        return reasoning
    raise RuntimeError(f"Unexpected API response: empty content: {payload!r}")


def _text_backend_raw() -> str:
    """未调用 _load_dotenv：由已在 ask_advisor 内 load 之后的逻辑使用。"""
    raw = os.environ.get("ADVISOR_TEXT_BACKEND", "auto").strip().lower()
    if raw not in {"auto", "deepseek", "glm", "qwen"}:
        return "auto"
    return raw


def resolved_text_backend() -> str:
    """返回实际使用的文本后端 deepseek | qwen（供流水线写抬头说明）。"""
    _load_dotenv()
    tb = _text_backend_raw()
    if tb == "deepseek":
        if not os.environ.get("DEEPSEEK_API_KEY", "").strip():
            raise RuntimeError("ADVISOR_TEXT_BACKEND=deepseek 但未配置 DEEPSEEK_API_KEY")
        return "deepseek"
    if tb == "glm":
        if not os.environ.get("GLM_API_KEY", "").strip():
            raise RuntimeError("ADVISOR_TEXT_BACKEND=glm 但未配置 GLM_API_KEY")
        return "glm"
    if tb == "qwen":
        return "qwen"
    # auto: DeepSeek > GLM > Qwen
    if os.environ.get("DEEPSEEK_API_KEY", "").strip():
        return "deepseek"
    if os.environ.get("GLM_API_KEY", "").strip():
        return "glm"
    return "qwen"


def text_advisor_banner_meta() -> str:
    """一行说明当前文本顾问环境与模型（不含密钥）。"""
    _load_dotenv()
    try:
        b = resolved_text_backend()
    except RuntimeError:
        b = "qwen"
    if b == "deepseek":
        model = os.environ.get("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL).strip()
        tm = os.environ.get("DEEPSEEK_THINKING", "enabled").strip().lower()
        effort = os.environ.get("DEEPSEEK_REASONING_EFFORT", "max").strip().lower()
        return (
            f"DeepSeek `{model}` · `POST .../chat/completions` · thinking=`{tm}` · "
            f"reasoning_effort=`{effort}`"
        )
    model = os.environ.get("QWEN_MODEL", "qwen-plus").strip()
    return f"DashScope（Qwen）`{model}` · `POST .../chat/completions`"


def _chat_messages(
    *,
    api_key: str,
    base_url: str,
    model: str,
    messages: list[dict],
    timeout: int,
    max_tokens: int = 6144,
    extra_body: dict | None = None,
) -> str:
    """OpenAI 兼容 chat/completions，支持多模态 messages（含 image_url）。"""
    base = base_url.rstrip("/")
    url = f"{base}/chat/completions"
    body: dict = {
        "model": model,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": max_tokens,
    }
    if extra_body:
        body.update(extra_body)
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return _extract_assistant_text(payload)


def _chat(
    *,
    api_key: str,
    base_url: str,
    model: str,
    system: str,
    user: str,
    timeout: int,
    max_tokens: int = 6144,
) -> str:
    messages: list[dict] = []
    if system.strip():
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return _chat_messages(
        api_key=api_key,
        base_url=base_url,
        model=model,
        messages=messages,
        timeout=timeout,
        max_tokens=max_tokens,
        extra_body=None,
    )


def _jpeg_data_url(path: Path, *, max_side: int = 768, jpeg_quality: int = 85) -> str:
    import cv2  # noqa: PLC0415

    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise ValueError(f"无法读取图像: {path}")
    h, w = im.shape[:2]
    m = max(h, w)
    if m > max_side:
        scale = max_side / float(m)
        nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
        im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", im, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        raise ValueError(f"JPEG 编码失败: {path}")
    b64 = base64.standard_b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def ask_advisor(
    user: str,
    system: str = "",
    *,
    timeout: int = 120,
    max_tokens: int = 6144,
) -> str:
    """供其它脚本调用：读取 .env 后发起一次 chat completion。"""
    _load_dotenv()
    backend = resolved_text_backend()
    if backend == "deepseek":
        api_key_ds = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not api_key_ds:
            raise RuntimeError("缺少环境变量 DEEPSEEK_API_KEY（请在项目根 .env 中配置）")
        base_url = os.environ.get(
            "DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"
        ).strip()
        model_ds = os.environ.get("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL).strip()
        eff = int(os.environ.get("DEEPSEEK_MAX_TOKENS", "8192").strip())
        mt = eff if max_tokens == 6144 else max_tokens
        thinking_on = (
            os.environ.get("DEEPSEEK_THINKING", "enabled").strip().lower() != "disabled"
        )
        effort = os.environ.get("DEEPSEEK_REASONING_EFFORT", "max").strip()
        extra: dict[str, object] = (
            {"thinking": {"type": "enabled", "reasoning_effort": effort}}
            if thinking_on
            else {"thinking": {"type": "disabled"}}
        )
        msgs: list[dict] = []
        if system.strip():
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": user})
        return _chat_messages(
            api_key=api_key_ds,
            base_url=base_url,
            model=model_ds,
            messages=msgs,
            timeout=timeout,
            max_tokens=mt,
            extra_body=extra,
        )

    api_key = os.environ.get("QWEN_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "缺少文本顾问密钥：请配置 QWEN_API_KEY，或设置 DEEPSEEK_API_KEY "
            "（ADVISOR_TEXT_BACKEND=auto 时优先 DeepSeek）"
        )
    base_url = os.environ.get(
        "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ).strip()
    model = os.environ.get("QWEN_MODEL", "qwen-plus").strip()
    return _chat(
        api_key=api_key,
        base_url=base_url,
        model=model,
        system=system,
        user=user,
        timeout=timeout,
        max_tokens=max_tokens,
    )


def ask_advisor_vl(
    user: str,
    system: str = "",
    *,
    image_paths: list[Path],
    timeout: int = 240,
    max_side: int = 768,
    max_images: int = 6,
    jpeg_quality: int = 85,
    max_tokens: int = 6144,
) -> str:
    """Qwen-VL（DashScope OpenAI 兼容）：在 user 文本前附加若干张本地图（JPEG base64 data URL）。"""
    _load_dotenv()
    api_key = os.environ.get("QWEN_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("缺少环境变量 QWEN_API_KEY（请在项目根 .env 中配置）")
    base_url = os.environ.get(
        "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ).strip()
    model = os.environ.get("QWEN_VL_MODEL", "qwen3-vl-plus").strip()

    paths = [p.resolve() for p in image_paths if p.is_file()][:max_images]
    if not paths:
        return ask_advisor(user, system=system, timeout=timeout, max_tokens=max_tokens)

    preamble = (
        "以下为钼靶生成灰度样例；请看接缝/过曝发糊/伪影。随后为评审指标摘要。"
    )
    content: list[dict] = [{"type": "text", "text": preamble}]
    for p in paths:
        url = _jpeg_data_url(p, max_side=max_side, jpeg_quality=jpeg_quality)
        content.append({"type": "image_url", "image_url": {"url": url}})
    content.append({"type": "text", "text": user})

    messages: list[dict] = []
    if system.strip():
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})
    return _chat_messages(
        api_key=api_key,
        base_url=base_url,
        model=model,
        messages=messages,
        timeout=timeout,
        max_tokens=max_tokens,
        extra_body=None,
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description="Ask project advisor text LLM（DeepSeek 或 DashScope Qwen · .env）"
    )
    p.add_argument("prompt", nargs="*", help="用户消息（可省略若使用 -f）")
    p.add_argument("-f", "--file", type=Path, help="从文件读取用户消息")
    p.add_argument("--system", default="", help="可选 system 提示")
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument(
        "--image",
        type=Path,
        action="append",
        default=[],
        help="附加本地图片（可重复），启用 Qwen-VL（QWEN_VL_MODEL；与文本后端无关）",
    )
    args = p.parse_args()

    if args.file:
        user = args.file.read_text(encoding="utf-8", errors="replace")
    else:
        user = " ".join(args.prompt).strip()
    if not user:
        print("请提供 prompt 或使用 -f", file=sys.stderr)
        return 2

    try:
        if args.image:
            out = ask_advisor_vl(
                user,
                system=args.system or "",
                image_paths=args.image,
                timeout=max(args.timeout, 180),
            )
        else:
            out = ask_advisor(user, system=args.system or "", timeout=args.timeout)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")
        print(f"HTTP {e.code}: {err}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"请求失败: {e}", file=sys.stderr)
        return 1

    sys.stdout.write(out)
    if not out.endswith("\n"):
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
