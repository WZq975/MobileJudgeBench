"""Shared helpers for the judge implementations: API configuration, chat-completion calls with retries,
cost accounting and message logging."""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Optional

import requests

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENAI_BASE_URL = "https://api.openai.com/v1"
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def get_api_config(provider: str, api_key: Optional[str] = None, base_url: Optional[str] = None,
                   required: bool = True) -> tuple[Optional[str], str]:
    """Resolve ``(api_key, base_url)`` for ``provider`` (``openai`` or ``openrouter``). Explicit arguments
    win; otherwise ``OPENAI_API_KEY``/``OPENAI_BASE_URL`` or ``OPENROUTER_API_KEY`` are used. A missing key
    raises ``ValueError``, or gives ``None`` when ``required`` is false."""
    if provider == "openrouter":
        key = api_key or os.getenv("OPENROUTER_API_KEY")
        url = base_url or OPENROUTER_BASE_URL
        env = "OPENROUTER_API_KEY"
    elif provider == "openai":
        key = api_key or os.getenv("OPENAI_API_KEY")
        url = base_url or os.getenv("OPENAI_BASE_URL", OPENAI_BASE_URL)
        env = "OPENAI_API_KEY"
    else:
        raise ValueError(f"unknown provider {provider!r}")
    if not key and required:
        raise ValueError(f"no API key for provider {provider!r}: pass --api_key or set {env}")
    return key or None, url


def is_gpt5(model: str) -> bool:
    return "gpt-5" in str(model).lower()


def post_chat_completion(base_url: Optional[str], api_key: Optional[str], payload: dict, timeout: int = 300,
                         retries: int = 5) -> dict:
    """POST an OpenAI-style chat completion and return the parsed JSON body.

    Rate limits and server errors are retried with exponential back-off; other client errors raise
    ``requests.HTTPError``. On OpenRouter the response carries the request's price in ``usage.cost``.
    """
    base_url = base_url or os.getenv("OPENAI_BASE_URL", OPENAI_BASE_URL)
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if "openrouter.ai" in base_url:
        payload = {**payload, "usage": {"include": True}}
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code in RETRY_STATUS:
                raise requests.exceptions.HTTPError(
                    f"{resp.status_code} {resp.reason}: {resp.text[:300]}", response=resp)
            resp.raise_for_status()
            data = resp.json()
            if not data.get("choices"):
                raise RuntimeError(f"no choices in response: {json.dumps(data.get('error') or data)[:300]}")
            if not (data["choices"][0].get("message") or {}).get("content"):
                raise RuntimeError(f"empty completion: {json.dumps(data['choices'][0])[:300]}")
            return data
        except (requests.exceptions.RequestException, RuntimeError, ValueError) as e:
            last_err = e
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status is not None and status not in RETRY_STATUS:
                raise
            if attempt < retries - 1:
                time.sleep(min(60, 2 ** attempt))
    assert last_err is not None
    raise last_err


# USD per 1M tokens (input, cached input, output) for OpenAI models; used only when the API does not
# report the cost itself (OpenRouter does, in usage.cost).
_OPENAI_PRICES = {
    "gpt-5-nano": (0.05, 0.005, 0.40),
    "gpt-5-mini": (0.25, 0.025, 2.00),
    "gpt-5": (1.25, 0.125, 10.00),
    "gpt-4.1-nano": (0.10, 0.025, 0.40),
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
    "gpt-4.1": (2.00, 0.50, 8.00),
    "gpt-4o-mini": (0.15, 0.075, 0.60),
    "gpt-4o": (2.50, 1.25, 10.00),
}


def usage_cost(usage: Optional[dict], model: str) -> Optional[float]:
    """Cost in USD of one API call: the provider-reported ``usage.cost`` when present, else the OpenAI
    price list above, else None."""
    if not usage:
        return None
    if isinstance(usage.get("cost"), (int, float)):
        return float(usage["cost"])
    name = str(model).lower().split("/")[-1]
    for key in sorted(_OPENAI_PRICES, key=len, reverse=True):
        if name.startswith(key):
            p_in, p_cached, p_out = _OPENAI_PRICES[key]
            pt = int(usage.get("prompt_tokens") or 0)
            ct = int(usage.get("completion_tokens") or 0)
            cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
            return (max(pt - cached, 0) * p_in + cached * p_cached + ct * p_out) / 1e6
    return None


def save_messages(save_dir: str, identifier: str, messages: list[dict], response_text: str, model: str,
                  extra_info: Optional[dict] = None) -> str:
    """Write one API exchange (base64 image data removed) to ``save_dir`` as JSON; returns the path."""
    os.makedirs(save_dir, exist_ok=True)
    safe_id = "".join(c if c.isalnum() or c in "-_." else "_" for c in identifier)
    if len(safe_id) > 100:
        safe_id = safe_id[:90] + "_" + hashlib.md5(identifier.encode()).hexdigest()[:8]
    now = datetime.now(timezone.utc)
    path = os.path.join(save_dir, f"{safe_id}_{now.strftime('%Y%m%d_%H%M%S_%f')}.json")

    def _clean(msg: dict) -> dict:
        if not isinstance(msg.get("content"), list):
            return msg
        out = []
        for item in msg["content"]:
            if isinstance(item, dict) and item.get("type") == "image_url" and \
                    str(item.get("image_url", {}).get("url", "")).startswith("data:"):
                out.append({"type": "image_url", "image_url": {"url": "[BASE64_IMAGE_DATA_REMOVED]",
                                                                "detail": item["image_url"].get("detail", "auto")}})
            else:
                out.append(item)
        return {**msg, "content": out}

    data: dict[str, Any] = {"timestamp": now.isoformat(), "identifier": identifier, "model": model,
                            "messages": [_clean(m) for m in messages], "response": response_text}
    if extra_info:
        data["extra_info"] = extra_info
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path
