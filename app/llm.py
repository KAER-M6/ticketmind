import json
import time
from typing import Any

from openai import APIConnectionError, APITimeoutError, OpenAI

from app.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        if not DEEPSEEK_API_KEY:
            raise RuntimeError("DEEPSEEK_API_KEY 未设置，请在 .env 中配置")
        _client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    return _client


def chat(
    prompt: str,
    system: str | None = None,
    model: str | None = None,
    temperature: float = 0.2,
    json_mode: bool = False,
    max_tokens: int = 2000,
    examples: list[dict[str, str]] | None = None,
) -> str:
    """examples: [{"input": ..., "output": ...}] 作为 few-shot 对话示例。"""
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    if examples:
        for ex in examples:
            messages.append({"role": "user", "content": ex["input"]})
            messages.append({"role": "assistant", "content": ex["output"]})
    messages.append({"role": "user", "content": prompt})

    kwargs: dict[str, Any] = {
        "model": model or DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": 60,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    last_err: Exception | None = None
    for attempt in range(3):  # 超时/连接错误重试 2 次，指数退避
        try:
            resp = get_client().chat.completions.create(**kwargs)
            return resp.choices[0].message.content
        except (APITimeoutError, APIConnectionError) as e:
            last_err = e
            wait = 2**attempt
            print(f"[llm] 调用失败({type(e).__name__})，{wait}s 后重试 ({attempt+1}/3)")
            time.sleep(wait)
    raise last_err


def chat_json(
    prompt: str,
    system: str | None = None,
    examples: list[dict[str, str]] | None = None,
    **kwargs,
) -> dict:
    """强制返回可解析的 JSON 对象。"""
    text = chat(prompt, system=system, examples=examples, json_mode=True, **kwargs)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise
