"""Turn a release-format trajectory into the inputs the judge consumes.

The judge takes the representation used by the paper's pipeline:

* ``image_paths``: the screenshots in time order (one per step, plus the final screen when available);
* ``actions``: one dict per step, ``{"type", "detail_type", "detail"[, "reasoning"]}``, which the judge
  renders as e.g. ``click at (540, 1200)``, ``type: hello`` or ``task complete: Task completed.``;
* ``observations``: one text block per screenshot listing the accessibility-tree elements, or
  ``[No UI elements]`` when no element list is available;
* ``reasonings``: the agent's stated reasoning per step (None when absent);
* ``trajectory``: the loaded ``Trajectory`` itself, for judges that want the raw actions, the UI element
  lists or the images.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ..loader import Trajectory, load_trajectory

NO_UI = "[No UI elements]"
MAX_OBSERVATION_ELEMENTS = 50


def legacy_action(action: dict[str, Any]) -> dict[str, Any]:
    """Map a release ``action`` to the judges' ``{type, detail_type, detail}`` form."""
    t = str(action.get("type", "unknown"))

    def point(label: str) -> dict[str, Any]:
        if action.get("x") is not None and action.get("y") is not None:
            return {"type": label, "detail_type": "coordinates", "detail": [int(action["x"]), int(action["y"])]}
        return {"type": label, "detail_type": "string", "detail": "index not found"}

    def text(label: str, detail: str) -> dict[str, Any]:
        return {"type": label, "detail_type": "string", "detail": detail}

    if t == "click":
        return point("click")
    if t == "long_press":
        return point("long press")
    if t == "double_tap":
        return point("double tap")
    if t == "input_text":
        return text("type", str(action.get("text", "")))
    if t == "answer":
        return text("answer", str(action.get("text", "")))
    if t == "open_app":
        return text("open app", str(action.get("app_name", "")))
    if t == "keyboard_enter":
        return text("press enter", "Press the enter key.")
    if t == "navigate_home":
        return text("press home", "Return to home page.")
    if t == "navigate_back":
        return text("press back", "Back to the previous page.")
    if t == "wait":
        return text("wait", "Wait.")
    if t == "scroll":
        direction = str(action.get("direction", "")).lower()
        if action.get("touch_xy") and action.get("lift_xy"):
            (sx, sy), (ex, ey) = action["touch_xy"], action["lift_xy"]
            return text("scroll", f"Scroll {direction} from ({sx},{sy}) to ({ex},{ey}).")
        return text("scroll", f"Scroll {direction}.")
    if t == "swipe":
        direction = str(action.get("direction", "")).lower()
        return text("swipe", f"Swipe {direction}." if direction else "")
    if t == "status":
        status = str(action.get("goal_status", "")).lower()
        message = str(action["text"]) if "text" in action else None  # UI-TARS passes a completion message
        if status == "complete":
            return text("task complete", "Task completed." if message is None else message)
        if status in ("infeasible", "impossible"):
            return text("task impossible", "Task impossible." if message is None else message)
        return text("status", status)
    return text(t, "")


def format_ui_observation(elements: Optional[list[dict[str, Any]]]) -> str:
    """One line per element (at most 50): class, resource name, interactivity, state and text."""
    if not elements:
        return NO_UI
    lines = []
    for i, e in enumerate(elements[:MAX_OBSERVATION_ELEMENTS]):
        if e is None:
            continue
        parts = []
        class_name = e.get("class_name") or ""
        if class_name:
            parts.append(str(class_name).split(".")[-1] if "." in str(class_name) else str(class_name))
        resource = e.get("resource_id") or e.get("resource_name") or ""
        if resource:
            if isinstance(resource, str) and ":id/" in resource:
                resource = resource.split(":id/")[-1]
            parts.append(str(resource))
        interactions = [name for key, name in (("is_clickable", "click"), ("is_long_clickable", "long-click"),
                                               ("is_editable", "editable"), ("is_scrollable", "scroll")) if e.get(key)]
        if interactions:
            parts.append(";".join(interactions))
        states = [name for key, name in (("is_selected", "selected"), ("is_checked", "checked")) if e.get(key)]
        if states:
            parts.append("; " + " ".join(states))
        display = str(e.get("text")).strip() if e.get("text") else ""
        content_desc = e.get("content_description") or ""
        hint = e.get("hint_text") or ""
        if content_desc and str(content_desc).strip() != display:
            display = f"{display} {content_desc}".strip()
        if hint and str(hint).strip() not in display:
            display = f"{display} ({hint})".strip()
        if display:
            parts.append(display)
        if parts:
            lines.append(f"[{i}] {' ;'.join(parts)}")
    if len(elements) > MAX_OBSERVATION_ELEMENTS:
        lines.append(f"... ({len(elements) - MAX_OBSERVATION_ELEMENTS} more elements)")
    return "\n".join(lines) if lines else NO_UI


@dataclass
class JudgeInput:
    trajectory_id: str
    benchmark: str
    agent: str
    task_id: str
    goal: str
    success: bool                   # human label
    n_steps: int
    image_paths: list[str]          # time order; len = n_steps or n_steps + 1
    actions: list[dict[str, Any]]   # one per step
    observations: list[str]         # one per screenshot
    reasonings: list[Optional[str]]  # one per step
    trajectory: Trajectory = field(repr=False)  # raw actions, UI element lists, load_images()

    @property
    def has_ui(self) -> bool:
        return any(o != NO_UI for o in self.observations)


def build_judge_input(traj: Trajectory) -> JudgeInput:
    actions = []
    for s in traj.steps:
        a = legacy_action(s.action)
        if s.reasoning:
            a["reasoning"] = s.reasoning
        actions.append(a)
    n_shots = len(traj.screenshot_paths)
    if traj.ui_elements is not None:
        observations = [format_ui_observation(e) for e in traj.ui_elements[:n_shots]]
        observations += [NO_UI] * (n_shots - len(observations))
    else:
        observations = [NO_UI] * n_shots
    return JudgeInput(
        trajectory_id=traj.trajectory_id, benchmark=traj.benchmark, agent=traj.agent, task_id=traj.task_id,
        goal=traj.goal, success=traj.success, n_steps=traj.n_steps, image_paths=list(traj.screenshot_paths),
        actions=actions, observations=observations, reasonings=[s.reasoning or None for s in traj.steps],
        trajectory=traj,
    )


def load_judge_input(root: str, trajectory_id: str) -> JudgeInput:
    return build_judge_input(load_trajectory(root, trajectory_id, with_ui_elements=True))
