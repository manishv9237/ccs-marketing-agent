"""Minimal OpenAI Chat Completions client (plain `requests`, no SDK).

It adapts to newer model families automatically: if the API rejects a parameter
(for example `max_tokens` on gpt-5-style models, or `response_format`), the call is
retried with the offending parameter swapped or removed.
"""
from __future__ import annotations

import logging
import time

import requests

from .config import Settings

log = logging.getLogger("agent.llm")


class LLMError(RuntimeError):
    """Safe-to-display failure (never contains the API key)."""


def _error_message(resp: requests.Response) -> str:
    try:
        return str(resp.json().get("error", {}).get("message", ""))[:300]
    except Exception:
        return resp.text[:200]


def chat(messages: list[dict], model: str, settings: Settings, max_output_tokens: int = 6000, json_mode: bool = False,
         session: requests.Session | None = None) -> str:
    if not settings.openai_api_key:
        raise LLMError("The server has no OPENAI_API_KEY configured.")
    http = session or requests
    payload: dict = {"model": model, "messages": messages, "max_completion_tokens": max_output_tokens}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    headers = {"Authorization": f"Bearer {settings.openai_api_key}", "Content-Type": "application/json"}

    transient = 0
    grew = False
    for _ in range(8):
        try:
            resp = http.post(f"{settings.openai_base_url}/chat/completions", headers=headers, json=payload, timeout=settings.llm_timeout)
        except requests.Timeout:
            raise LLMError("The model took too long to respond. Try again or use fewer variations.")
        except requests.RequestException as exc:
            raise LLMError(f"Could not reach the model API ({exc.__class__.__name__}).")

        if resp.status_code == 200:
            choice = resp.json()["choices"][0]
            text = (choice["message"].get("content") or "").strip()
            if not text and choice.get("finish_reason") == "length" and not grew:
                payload["max_completion_tokens"] = min(payload["max_completion_tokens"] * 2, 32000)
                grew = True  # reasoning ate the whole budget: retry once with more room
                continue
            if not text:
                raise LLMError(f"Model '{model}' returned an empty answer (finish_reason={choice.get('finish_reason')}).")
            return text

        msg = _error_message(resp)
        low = msg.lower()
        if resp.status_code == 400:
            if "max_completion_tokens" in low and "max_completion_tokens" in payload:
                payload["max_tokens"] = payload.pop("max_completion_tokens")
                continue
            if "max_tokens" in low and "max_tokens" in payload:
                payload["max_completion_tokens"] = payload.pop("max_tokens")
                continue
            if "response_format" in low and "response_format" in payload:
                payload.pop("response_format")
                continue
            dropped = False
            for p in ("temperature", "reasoning_effort"):
                if p in low and p in payload:
                    payload.pop(p)
                    dropped = True
            if dropped:
                continue
            raise LLMError(f"The model rejected the request: {msg}")
        if resp.status_code == 401:
            raise LLMError("The server's OpenAI API key was rejected (401). Check OPENAI_API_KEY.")
        if resp.status_code == 404 or "model_not_found" in low:
            raise LLMError(f"Model '{model}' is not available for this API key.")
        if resp.status_code in (429, 500, 502, 503, 504) and transient < 3:
            transient += 1
            time.sleep(min(2 ** transient, 8))
            continue
        raise LLMError(f"Model API error (HTTP {resp.status_code}): {msg or 'no details'}")
    raise LLMError("The model API did not accept the request after several adjustments.")
