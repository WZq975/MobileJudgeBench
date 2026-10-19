"""The paper's Baseline judge.

The judge receives the task, then the trajectory step by step in order: each step shows the action,
optionally the agent's reasoning and the UI element list, and, at sampled steps, the screenshot. It
answers with a success/failure verdict, a confidence level and a short reason.

Configurable: the number of screenshots and how they are sampled (``uniform`` or ``event_based``),
whether the agent's reasoning and the UI element lists are shown (the lists at the sampled steps or
at every step), and the screenshot resolution.
"""

import os
import io
import re
import base64
from typing import Dict, List, Optional, Tuple

try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = None

from .common import is_gpt5, post_chat_completion, save_messages, usage_cost


# ============================================================================
# Prompt Templates
# ============================================================================

SYSTEM_PROMPT = """You are an expert evaluator for mobile agent task completion. Your role is to determine whether a given task has been successfully completed based on the provided trajectory evidence.

## Evaluation Guidelines:

1. **Outcome-Focused**: Judge based on whether the primary objective was achieved, not the path taken.

2. **Use All Provided Evidence**: Base your judgment on all available information — screenshots, actions, agent reasoning, and UI element lists. Do not assume information beyond what is provided.

3. **Mid-Trajectory Success**: The task may be completed in an intermediate step rather than the final one. Additional actions after completion do not invalidate success, unless they explicitly undo it.

4. **Corrective Actions**: If the agent made mistakes but corrected them and achieved the goal, that counts as success.

5. **UI State Indicators**: Pay attention to visual indicators like selected tabs, checkboxes, highlighted items, and confirmation messages.

Be balanced in your judgment - avoid being overly strict (missing true successes) or overly lenient (accepting failures).
"""

USER_PROMPT_HEADER = """## Task
{task}

## Agent Trajectory
The following shows the agent's execution trajectory in chronological order. Each step's screenshot is placed immediately after its description.
"""

USER_PROMPT_FOOTER = """
## Your Judgment
Based on the trajectory above, determine if the task was successfully completed.

Reply in the following format:
Result: <1 for success, 0 for failure>
Confidence: <high/medium/low>
Reason: <brief explanation in 1-2 sentences>
"""

# Sampled steps: text followed by the screenshot
STEP_TEMPLATE_FULL = """
### Step {step_num}{step_label}
**Action**: {action}
**Agent Reasoning**: {reasoning}
**UI Elements**: {ui_elements}
[Screenshot attached]
"""

STEP_TEMPLATE_NO_REASONING = """
### Step {step_num}{step_label}
**Action**: {action}
**UI Elements**: {ui_elements}
[Screenshot attached]
"""

STEP_TEMPLATE_NO_UI = """
### Step {step_num}{step_label}
**Action**: {action}
**Agent Reasoning**: {reasoning}
[Screenshot attached]
"""

STEP_TEMPLATE_MINIMAL = """
### Step {step_num}{step_label}
**Action**: {action}
[Screenshot attached]
"""

# The final screen (a screenshot after the last action)
FINAL_STEP_TEMPLATE = """
### Step {step_num} (Final State)
[Screenshot attached]
{optional_ui}
"""

# Steps without a screenshot
STEP_TEMPLATE_ACTION_ONLY = """
### Step {step_num}{step_label}
**Action**: {action}
"""

STEP_TEMPLATE_ACTION_REASONING_ONLY = """
### Step {step_num}{step_label}
**Action**: {action}
**Agent Reasoning**: {reasoning}
"""

STEP_TEMPLATE_ACTION_UI_ONLY = """
### Step {step_num}{step_label}
**Action**: {action}
**UI Elements**: {ui_elements}
"""

STEP_TEMPLATE_ACTION_REASONING_UI_ONLY = """
### Step {step_num}{step_label}
**Action**: {action}
**Agent Reasoning**: {reasoning}
**UI Elements**: {ui_elements}
"""


# ============================================================================
# Screenshot Sampling
# ============================================================================

def sample_uniform(indices: List[int], max_samples: int) -> List[int]:
    """Uniform sampling across trajectory.

    Algorithm:
    1. Always include: first (0), second-to-last (n-2), last (n-1)
    2. Uniformly sample remaining slots from the middle portion (1 to n-3)
    """
    n = len(indices)
    if n <= max_samples:
        return indices

    if max_samples <= 0:
        return []
    if max_samples == 1:
        return [indices[-1]]  # Just last
    if max_samples == 2:
        return [indices[0], indices[-1]]  # First and last

    # n >= 3 and max_samples >= 3
    first_idx = 0
    last_idx = n - 1
    second_to_last_idx = n - 2

    # Handle edge case where second-to-last equals first (n=2)
    if second_to_last_idx <= first_idx:
        return [indices[0], indices[-1]]

    # Start with fixed selections: first, second-to-last, last
    result = [indices[first_idx], indices[second_to_last_idx], indices[last_idx]]

    # Uniformly sample remaining from middle portion (1 to n-3)
    remaining = max_samples - 3
    if remaining > 0 and second_to_last_idx > 1:
        # Middle portion is from index 1 to second_to_last_idx - 1 (exclusive)
        middle_start = 1
        middle_end = second_to_last_idx - 1  # Last index in middle portion
        middle_range = middle_end - middle_start + 1

        if middle_range > 0:
            step = middle_range / (remaining + 1)
            for i in range(1, remaining + 1):
                idx = int(middle_start + i * step - 0.5)  # round-half-up adjustment
                idx = max(middle_start, min(idx, middle_end))  # clamp to valid range
                if indices[idx] not in result:
                    result.append(indices[idx])

    return sorted(set(result))[:max_samples]


EVENT_TIER_1 = {
    "type", "open app", "press home", "press back", "long press",
    "task complete", "task impossible", "answer", "finished",
}
EVENT_TIER_2 = {"click"}


def sample_event_based(
    n_total: int,
    actions: Optional[List[Dict]],
    max_samples: int,
) -> List[int]:
    """Event-based screenshot sampling.

    A step's 'event score' is the sum of:
      +10 if i is the first or last step (trajectory boundary)
      + 8 if action.type is in EVENT_TIER_1 (text entry, app switch, navigation, terminal)
      + 3 if action.type is in EVENT_TIER_2 (click)
      + 5 if action.type changed vs the previous step (transition / new sub-phase)
      + 0 otherwise (e.g., scroll, wait)

    Selection is greedy: at each step we pick the candidate that maximises a
    weighted blend of (a) its event score and (b) its distance from the already
    picked steps, so picks spread across the trajectory rather than clustering
    at the start. Boundaries are always included.

    If `actions` is missing or shorter than `n_total`, falls back to uniform.
    """
    if actions is None or len(actions) == 0 or max_samples >= n_total:
        return sample_uniform(list(range(n_total)), max_samples)

    scores = [0.0] * n_total
    for i in range(n_total):
        if i == 0 or i == n_total - 1:
            scores[i] += 10
        if i < len(actions):
            t = actions[i].get("type") if isinstance(actions[i], dict) else None
            if t in EVENT_TIER_1:
                scores[i] += 8
            elif t in EVENT_TIER_2:
                scores[i] += 3
            if i > 0 and i - 1 < len(actions):
                prev_t = actions[i - 1].get("type") if isinstance(actions[i - 1], dict) else None
                if t and prev_t and t != prev_t:
                    scores[i] += 5

    # Pre-seed with the two boundaries (always informative).
    picks = {0, n_total - 1}

    def min_dist_to_picks(i):
        return min(abs(i - p) for p in picks)

    while len(picks) < max_samples:
        best_i, best_key = None, None
        for i in range(n_total):
            if i in picks:
                continue
            d = min_dist_to_picks(i)
            # Combined key: prefer high score, then far distance, then mid-of-traj first.
            key = (scores[i] + 0.5 * d, d, -abs(i - (n_total - 1) / 2))
            if best_key is None or key > best_key:
                best_i, best_key = i, key
        if best_i is None:
            break
        picks.add(best_i)

    return sorted(picks)


def sample_screenshots(
    n_total: int,
    max_samples: int = 5,
    strategy: str = "uniform",
    actions: Optional[List[Dict]] = None,
) -> List[int]:
    """Indices of the screenshots to attach (0-based), ``strategy`` = "uniform" | "event_based"."""
    if n_total <= 0:
        return []

    all_indices = list(range(n_total))

    if n_total <= max_samples:
        return all_indices

    if strategy == "event_based":
        return sample_event_based(n_total, actions, max_samples)
    if strategy == "uniform":
        return sample_uniform(all_indices, max_samples)
    raise ValueError(f"unknown sampling strategy: {strategy}")


# ============================================================================
# Image Utilities
# ============================================================================

def encode_image_base64(path: str) -> str:
    """Encode an image file to base64 string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def resize_image_to_base64(
    image_path: str,
    max_long_edge: int,
    jpeg_quality: int = 85,
) -> Tuple[str, str]:
    """Resize an image so its longest edge is at most ``max_long_edge`` pixels
    (aspect ratio preserved; smaller images are kept as they are), then encode as JPEG base64.

    Returns:
        Tuple of (base64_string, mime_type).

    Raises:
        RuntimeError: If PIL/Pillow is not installed.
    """
    if PILImage is None:
        raise RuntimeError(
            "Pillow is required for image resizing. Install with: pip install Pillow"
        )

    with PILImage.open(image_path) as img:
        # Convert to RGB if needed (e.g. RGBA PNGs, palette images)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        w, h = img.size
        long_edge = max(w, h)
        if long_edge > max_long_edge:
            scale = max_long_edge / long_edge
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = img.resize((new_w, new_h), PILImage.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=jpeg_quality)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return b64, "image/jpeg"


def build_image_content(
    image_path: str,
    max_long_edge: Optional[int] = None,
    jpeg_quality: int = 85,
) -> Optional[Dict]:
    """Image content block for the API call: the file as is (PNG/JPEG), or resized to
    ``max_long_edge`` pixels on the longest side and re-encoded as JPEG."""
    if not os.path.isfile(image_path):
        return None

    if max_long_edge is not None:
        b64, mime = resize_image_to_base64(image_path, max_long_edge, jpeg_quality)
    else:
        ext = os.path.splitext(image_path)[1].lower()
        mime = "image/png" if ext == ".png" else "image/jpeg"
        b64 = encode_image_base64(image_path)

    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{mime};base64,{b64}",
            "detail": "high",
        },
    }


# ============================================================================
# Trajectory Formatting
# ============================================================================

def format_action(action: Optional[Dict]) -> str:
    """Format an action dict into readable text."""
    if not action:
        return "[No action]"

    action_type = action.get("type", "unknown")
    detail = action.get("detail", "")
    detail_type = action.get("detail_type", "")

    if detail_type == "coordinates" and isinstance(detail, (list, tuple)):
        return f"{action_type} at ({detail[0]}, {detail[1]})"
    elif detail:
        return f"{action_type}: {detail}"
    else:
        return str(action_type)


def format_ui_elements(ui_elements: Optional[str]) -> str:
    """The UI element text block of a step, or ``[Not available]``."""
    if not ui_elements:
        return "[Not available]"
    if ui_elements.strip() in ("", "[No UI elements]", "None"):
        return "[Not available]"
    return ui_elements


def format_reasoning(reasoning: Optional[str]) -> str:
    """The agent's reasoning at a step, or ``[Not available]``."""
    if not reasoning:
        return "[Not available]"

    reasoning = str(reasoning).strip()
    if not reasoning or reasoning.lower() in ("none", "null", ""):
        return "[Not available]"
    return reasoning


def build_trajectory_content(
    image_paths: List[str],
    actions: Optional[List[Dict]] = None,
    ui_elements_list: Optional[List[str]] = None,
    reasonings: Optional[List[str]] = None,
    sampled_indices: Optional[List[int]] = None,
    include_ui_elements: bool = True,
    include_reasoning: bool = True,
    ui_elements_all_steps: bool = False,
    image_max_long_edge: Optional[int] = None,
) -> List[Dict]:
    """The trajectory as interleaved text and image blocks.

    Every step is listed with its action (and the agent's reasoning when ``include_reasoning``).
    Sampled steps also show the UI element list (when ``include_ui_elements``) and are followed by
    their screenshot; the other steps show the UI element list only when ``ui_elements_all_steps``.

    Args:
        image_paths: List of all screenshot paths
        actions: List of action dicts (len = n_screenshots - 1 typically)
        ui_elements_list: UI element text block for each step
        reasonings: List of agent reasoning strings for each step
        sampled_indices: Which indices get screenshots (None = all)
        include_ui_elements: Whether to include UI elements in prompt
        include_reasoning: Whether to include agent reasoning in prompt
        ui_elements_all_steps: If True, attach UI elements at ALL steps (not just sampled).
            Only effective when include_ui_elements is True.
        image_max_long_edge: If set, resize screenshots to this max long edge

    Returns:
        List of content items (text and image_url dicts) in interleaved order
    """
    n_images = len(image_paths)
    if n_images == 0:
        return [{"type": "text", "text": "[No trajectory data]"}]

    # Default to all indices if not specified
    if sampled_indices is None:
        sampled_indices = list(range(n_images))

    sampled_set = set(sampled_indices)

    # Ensure lists have correct length with None padding
    actions = actions or []
    ui_elements_list = ui_elements_list or []
    reasonings = reasonings or []

    content_items: List[Dict] = []

    for step_idx in range(n_images):
        is_final = (step_idx == n_images - 1)
        is_first = (step_idx == 0)
        is_sampled = (step_idx in sampled_set)

        # Get data for this step (with bounds checking)
        action = actions[step_idx] if step_idx < len(actions) else None
        ui_elem = ui_elements_list[step_idx] if step_idx < len(ui_elements_list) else None
        reasoning = reasonings[step_idx] if step_idx < len(reasonings) else None

        # Build step label
        if is_first and is_final:
            step_label = " (Initial & Final State)"
        elif is_first:
            step_label = " (Initial State)"
        elif is_final:
            step_label = " (Final State)"
        else:
            step_label = ""

        has_action = action is not None
        has_ui = include_ui_elements and ui_elem is not None
        has_reasoning = include_reasoning and reasoning is not None

        if is_sampled:
            # Sampled step: include screenshot + UI + reasoning + action
            if is_final and not has_action:
                optional_ui = ""
                if has_ui:
                    optional_ui = f"**UI Elements**: {format_ui_elements(ui_elem)}"
                step_text = FINAL_STEP_TEMPLATE.format(
                    step_num=step_idx + 1,
                    optional_ui=optional_ui,
                )
            elif has_reasoning and has_ui:
                step_text = STEP_TEMPLATE_FULL.format(
                    step_num=step_idx + 1,
                    step_label=step_label,
                    action=format_action(action),
                    reasoning=format_reasoning(reasoning),
                    ui_elements=format_ui_elements(ui_elem),
                )
            elif has_reasoning and not has_ui:
                step_text = STEP_TEMPLATE_NO_UI.format(
                    step_num=step_idx + 1,
                    step_label=step_label,
                    action=format_action(action),
                    reasoning=format_reasoning(reasoning),
                )
            elif has_ui and not has_reasoning:
                step_text = STEP_TEMPLATE_NO_REASONING.format(
                    step_num=step_idx + 1,
                    step_label=step_label,
                    action=format_action(action),
                    ui_elements=format_ui_elements(ui_elem),
                )
            else:
                step_text = STEP_TEMPLATE_MINIMAL.format(
                    step_num=step_idx + 1,
                    step_label=step_label,
                    action=format_action(action),
                )

            # Add step text then immediately its screenshot (interleaved)
            content_items.append({"type": "text", "text": step_text})

            img_content = build_image_content(
                image_paths[step_idx],
                max_long_edge=image_max_long_edge,
            )
            if img_content:
                content_items.append(img_content)
        else:
            # Non-sampled step: action (+ reasoning if enabled) (+ UI if ui_elements_all_steps), no screenshot
            if not has_action:
                continue  # Skip steps with no action and no screenshot

            # UI elements at non-sampled steps only when ui_elements_all_steps is enabled
            show_ui = has_ui and ui_elements_all_steps

            if has_reasoning and show_ui:
                step_text = STEP_TEMPLATE_ACTION_REASONING_UI_ONLY.format(
                    step_num=step_idx + 1,
                    step_label=step_label,
                    action=format_action(action),
                    reasoning=format_reasoning(reasoning),
                    ui_elements=format_ui_elements(ui_elem),
                )
            elif has_reasoning:
                step_text = STEP_TEMPLATE_ACTION_REASONING_ONLY.format(
                    step_num=step_idx + 1,
                    step_label=step_label,
                    action=format_action(action),
                    reasoning=format_reasoning(reasoning),
                )
            elif show_ui:
                step_text = STEP_TEMPLATE_ACTION_UI_ONLY.format(
                    step_num=step_idx + 1,
                    step_label=step_label,
                    action=format_action(action),
                    ui_elements=format_ui_elements(ui_elem),
                )
            else:
                step_text = STEP_TEMPLATE_ACTION_ONLY.format(
                    step_num=step_idx + 1,
                    step_label=step_label,
                    action=format_action(action),
                )

            content_items.append({"type": "text", "text": step_text})

    return content_items


# ============================================================================
# Response Parsing
# ============================================================================

def parse_response(response_text: str) -> Tuple[int, str, str]:
    """
    Parse the judge response.

    Returns:
        Tuple of (result, confidence, reason)
    """
    text = response_text.strip()

    # Parse result
    result_match = re.search(r'Result\s*:\s*(\d)', text, re.IGNORECASE)
    if result_match:
        result = int(result_match.group(1))
    else:
        # Fallback: look for yes/no or success/failure
        text_lower = text.lower()
        if any(w in text_lower for w in ["success", "yes", "completed", "achieved"]):
            result = 1
        else:
            result = 0

    # Parse confidence
    conf_match = re.search(r'Confidence\s*:\s*(high|medium|low)', text, re.IGNORECASE)
    confidence = conf_match.group(1).lower() if conf_match else "medium"

    # Parse reason
    reason_match = re.search(r'Reason\s*:\s*(.+?)(?:\n|$)', text, re.IGNORECASE | re.DOTALL)
    if reason_match:
        reason = reason_match.group(1).strip()
    else:
        # Use full response as reason if parsing fails
        reason = text[:500] if len(text) > 500 else text

    return result, confidence, reason


# ============================================================================
# Main Judge Function
# ============================================================================

def call_simple_baseline_judge(
    goal: str,
    image_paths: List[str],
    actions: Optional[List[Dict]] = None,
    ui_elements_list: Optional[List[str]] = None,
    reasonings: Optional[List[str]] = None,
    model: str = "gpt-5-mini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    max_samples: int = 5,
    sampling_strategy: str = "uniform",
    include_ui_elements: bool = True,
    include_reasoning: bool = True,
    ui_elements_all_steps: bool = False,
    image_max_long_edge: Optional[int] = None,
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    save_messages_dir: Optional[str] = None,
    message_identifier: Optional[str] = None,
) -> Tuple[int, str, Optional[int], Optional[float], Dict]:
    """
    Call the simple baseline judge to evaluate task success.

    Args:
        goal: Task description/goal
        image_paths: List of screenshot paths (chronological order)
        actions: Optional list of action dicts for each step
        ui_elements_list: Optional list of UI element text blocks for each step
        reasonings: Optional list of agent reasoning strings for each step
        model: Model name for API call
        api_key: API key (or from OPENAI_API_KEY env)
        base_url: API base URL (or from OPENAI_BASE_URL env)
        max_samples: Maximum number of screenshots to include
        sampling_strategy: "uniform" or "event_based"
        include_ui_elements: Whether to include UI elements in prompt
        include_reasoning: Whether to include agent reasoning in prompt
        ui_elements_all_steps: If True (and include_ui_elements is True), attach
            UI element lists at ALL steps, not just sampled ones.
        image_max_long_edge: If set, resize screenshots so the longest edge
            does not exceed this many pixels (aspect ratio preserved, JPEG).
            Requires Pillow.  None means no resizing.
        temperature: Sampling temperature (None = 0, or the model default for GPT-5 models)
        max_output_tokens: Max output tokens (None = model default)
        save_messages_dir: Optional directory to save messages for debugging
        message_identifier: Optional identifier for saved messages

    Returns:
        Tuple of (result, reason, total_tokens, api_cost, details)
        - result: 1 for success, 0 for failure
        - reason: Explanation string
        - total_tokens: Total tokens used
        - api_cost: Estimated API cost
        - details: Dict with confidence, sampled_indices, etc.
    """
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    if not image_paths:
        return 0, "No screenshots provided", None, None, {"error": "no_images"}

    if not goal:
        return 0, "No task description provided", None, None, {"error": "no_goal"}

    # Sample screenshots
    n_images = len(image_paths)
    sampled_indices = sample_screenshots(n_images, max_samples, sampling_strategy, actions=actions)

    # Build trajectory content (interleaved text + images)
    trajectory_content = build_trajectory_content(
        image_paths=image_paths,
        actions=actions,
        ui_elements_list=ui_elements_list,
        reasonings=reasonings,
        sampled_indices=sampled_indices,
        include_ui_elements=include_ui_elements,
        include_reasoning=include_reasoning,
        ui_elements_all_steps=ui_elements_all_steps,
        image_max_long_edge=image_max_long_edge,
    )

    # Build user content: header (task + intro) -> interleaved trajectory -> footer (judgment instruction)
    user_content: List[Dict] = []
    user_content.append({"type": "text", "text": USER_PROMPT_HEADER.format(task=goal)})
    user_content.extend(trajectory_content)
    user_content.append({"type": "text", "text": USER_PROMPT_FOOTER})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    # Build API payload
    gpt5 = is_gpt5(model)
    payload = {"model": model, "messages": messages}
    if temperature is not None:
        payload["temperature"] = temperature
    elif not gpt5:
        payload["temperature"] = 0
    if max_output_tokens is not None:
        payload["max_completion_tokens" if gpt5 else "max_tokens"] = max_output_tokens

    data = post_chat_completion(base_url, api_key, payload)

    # Parse response
    response_text = data["choices"][0]["message"]["content"]
    result, confidence, reason = parse_response(response_text)

    # Usage
    usage = data.get("usage", {}) or {}
    total_tokens = usage.get("total_tokens")
    api_cost = usage_cost(usage, model)

    # Build details
    details = {
        "confidence": confidence,
        "sampled_indices": sampled_indices,
        "n_total_screenshots": n_images,
        "n_sampled_screenshots": len(sampled_indices),
        "include_ui_elements": include_ui_elements,
        "include_reasoning": include_reasoning,
        "ui_elements_all_steps": ui_elements_all_steps,
        "image_max_long_edge": image_max_long_edge,
        "sampling_strategy": sampling_strategy,
        "full_response": response_text,
    }

    # Save messages if directory provided
    if save_messages_dir:
        save_messages(
            save_dir=save_messages_dir,
            identifier=message_identifier or "simple_baseline_eval",
            messages=messages,
            response_text=response_text,
            model=model,
            extra_info={
                "judge_method": "simple_baseline",
                "goal": goal,
                "result": result,
                "confidence": confidence,
                "total_tokens": total_tokens,
                "sampled_indices": sampled_indices,
            },
        )

    return result, reason, total_tokens, api_cost, details
