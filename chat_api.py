"""OpenRouter client helpers (kept free of Streamlit so they are easy to test)."""

import json
import os
from collections.abc import Iterator

import requests

OPENROUTER_URL = os.environ.get("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions")

# Verified against https://openrouter.ai/api/v1/models on 2026-10-14.
MODELS = {
    "GPT-4o mini": "openai/gpt-4o-mini",
    "GPT-4o": "openai/gpt-4o",
    "Claude Sonnet 4.5": "anthropic/claude-sonnet-4.5",
    "Gemini 2.5 Flash": "google/gemini-2.5-flash",
    "Gemma 4 31B · Free": "google/gemma-4-31b-it:free",
    "Auto (free models)": "openrouter/free",
}

MAX_HISTORY_MESSAGES = 20


class ChatError(Exception):
    """User-facing error from the chat API."""


def trim_history(messages: list, limit: int = MAX_HISTORY_MESSAGES) -> list:
    return messages[-limit:] if limit > 0 else list(messages)


def _error_from_response(resp) -> ChatError:
    try:
        err_json = resp.json()
        err_msg = err_json.get("error", {}).get("message", str(err_json))
    except Exception:
        err_msg = resp.text
    hint = ""
    if resp.status_code in (401, 403):
        hint = " Check OPENROUTER_API_KEY in your Streamlit secrets."
    elif resp.status_code == 402:
        hint = " Your OpenRouter account is out of credits; try a free model."
    elif resp.status_code == 429:
        hint = " Rate limited; wait a moment and retry."
    return ChatError(f"Status {resp.status_code}: {err_msg}.{hint}")


def stream_chat(
    api_key: str,
    messages: list,
    model: str,
    system_prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    usage: dict | None = None,
) -> Iterator[str]:
    """Yield reply text chunks from OpenRouter's SSE stream.

    If `usage` is given it is filled with the final usage block (if sent).
    """
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system_prompt}, *trim_history(messages)],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        "usage": {"include": True},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://streamlit.io",
        "X-Title": "Lumen",
    }
    try:
        resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60, stream=True)
    except requests.exceptions.Timeout:
        raise ChatError("Request timed out. Try again.") from None
    except requests.exceptions.ConnectionError:
        raise ChatError("Cannot connect to OpenRouter. Check your internet connection.") from None

    if resp.status_code != 200:
        raise _error_from_response(resp)

    for raw in resp.iter_lines(decode_unicode=True):
        if not raw or not raw.startswith("data:"):
            continue  # blank lines and ": OPENROUTER PROCESSING" keep-alives
        data = raw[len("data:") :].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        if "error" in chunk:
            raise ChatError(f"Stream error: {chunk['error'].get('message', chunk['error'])}")
        if usage is not None and chunk.get("usage"):
            usage.update(chunk["usage"])
        for choice in chunk.get("choices") or []:
            text = (choice.get("delta") or {}).get("content")
            if text:
                yield text
