#!/usr/bin/env python3
"""Two example judges for ``python -m mobilejudgebench.judges.evaluate --judge_method examples/custom_judge.py:<function>``.

A judge is a function ``judge(inp, **kw)`` that returns ``True``/``False`` (or a dict with ``success`` and
optionally ``reason``, ``tokens``, ``cost``, ``details``). ``inp`` is a ``mobilejudgebench.judges.inputs.JudgeInput``;
the keyword arguments carry the run's ``model``, ``api_key``, ``base_url``, ``max_output_tokens``, ``temperature``,
``save_messages_dir`` and ``message_identifier`` (absorb the ones you do not need with ``**kw``).

    # no API needed: success iff the agent declared the task complete
    python -m mobilejudgebench.judges.evaluate --root ./MobileJudgeBench \
        --judge_method examples/custom_judge.py:claimed_completion

    # one vision-model call per trajectory: the goal, the action list and the final screenshot
    python -m mobilejudgebench.judges.evaluate --root ./MobileJudgeBench \
        --judge_method examples/custom_judge.py:final_screen_llm --model gpt-5-mini --api_provider openai
"""
from __future__ import annotations

import base64
from typing import Optional

from mobilejudgebench.judges.common import is_gpt5, post_chat_completion, save_messages, usage_cost
from mobilejudgebench.judges.inputs import JudgeInput
from mobilejudgebench.judges.simple_baseline_judge import format_action


def claimed_completion(inp: JudgeInput, **_) -> bool:
    """Success iff the agent's last action declares the task complete."""
    return bool(inp.actions) and inp.actions[-1]["type"] == "task complete"


def final_screen_llm(inp: JudgeInput, model: str, api_key: Optional[str], base_url: Optional[str],
                     temperature: Optional[float] = None, max_output_tokens: Optional[int] = None,
                     save_messages_dir: Optional[str] = None, message_identifier: str = "", **_) -> dict:
    """Ask a vision model whether the final screenshot shows the goal accomplished."""
    with open(inp.image_paths[-1], "rb") as f:
        image = base64.b64encode(f.read()).decode()
    actions = "\n".join(f"{i}. {format_action(a)}" for i, a in enumerate(inp.actions, 1))
    prompt = (f"Task given to a mobile agent: {inp.goal}\n\nThe agent's actions, in order:\n{actions}\n\n"
              "The image is the final screen. Was the task completed successfully? "
              "Answer SUCCESS or FAILURE on the first line, then one sentence of justification.")
    messages = [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}", "detail": "high"}},
    ]}]
    payload: dict = {"model": model, "messages": messages}
    if temperature is not None and not is_gpt5(model):
        payload["temperature"] = temperature
    if max_output_tokens is not None:
        payload["max_completion_tokens" if is_gpt5(model) else "max_tokens"] = max_output_tokens
    data = post_chat_completion(base_url, api_key, payload)
    text = data["choices"][0]["message"]["content"].strip()
    if save_messages_dir:
        save_messages(save_messages_dir, message_identifier, messages, text, model)
    usage = data.get("usage") or {}
    return {"success": text.upper().startswith("SUCCESS"), "reason": text,
            "tokens": usage.get("total_tokens"), "cost": usage_cost(usage, model)}
