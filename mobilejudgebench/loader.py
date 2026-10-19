"""Minimal, dependency-light loader for the MobileJudgeBench release format.

Dataset layout (see docs/FORMAT.md)::

    <root>/
      annotations.csv                       one row per trajectory (label + metadata)
      tasks.csv                             one row per task
      trajectories/<benchmark>/<agent>/<task_id>/
          trajectory.json
          ui_elements.json                  (absent for UI-TARS trajectories)
          screenshots/step_000.png ...      one per step, plus the final screen when available

Example::

    from mobilejudgebench import load_annotations, load_trajectory
    rows = load_annotations("/path/to/MobileJudgeBench")
    traj = load_trajectory("/path/to/MobileJudgeBench", rows[0]["trajectory_id"])
    print(traj.goal, traj.success, traj.n_steps)
    for step in traj.steps:
        print(step.index, step.action, step.screenshot_path)
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from typing import Any, Iterator


# ---------------------------------------------------------------------------
# Tabular files
# ---------------------------------------------------------------------------
def _read_csv(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_annotations(root: str) -> list[dict[str, Any]]:
    """Rows of ``annotations.csv`` with numeric columns converted to int/bool. ``root`` is the release
    root or the path of an ``annotations.csv``."""
    rows = _read_csv(root if os.path.isfile(root) else os.path.join(root, "annotations.csv"))
    for r in rows:
        r["success"] = bool(int(r["success"]))
        for k in ("n_steps", "n_screenshots"):
            r[k] = int(r[k])
    return rows


def load_tasks(root: str) -> list[dict[str, str]]:
    return _read_csv(os.path.join(root, "tasks.csv"))


# ---------------------------------------------------------------------------
# Trajectories
# ---------------------------------------------------------------------------
@dataclass
class Step:
    index: int
    screenshot_path: str | None
    action: dict[str, Any]
    reasoning: str
    ui_elements: list[dict[str, Any]] | None = None


@dataclass
class Trajectory:
    trajectory_id: str
    benchmark: str
    agent: str
    task_id: str
    task_category: str | None
    app: str | None
    goal: str
    success: bool
    n_steps: int
    steps: list[Step]
    screenshot_paths: list[str]  # time order (judge input): one per step, plus the final screen when available
    ui_elements: list[list[dict[str, Any]] | None] | None  # aligned with screenshot_paths; None if not loaded
    directory: str
    raw: dict[str, Any] = field(repr=False)

    @property
    def n_screenshots(self) -> int:
        return len(self.screenshot_paths)

    @property
    def final_screenshot_path(self) -> str | None:
        """The last screenshot in time order (the final state)."""
        return self.screenshot_paths[-1] if self.screenshot_paths else None

    @property
    def final_ui_elements(self) -> list[dict[str, Any]] | None:
        """Elements of the last screenshot (None when not available or not loaded)."""
        return self.ui_elements[-1] if self.ui_elements else None

    def load_images(self):
        """Return the screenshots as PIL images (requires Pillow)."""
        from PIL import Image

        return [Image.open(p) for p in self.screenshot_paths]


def trajectory_dir(root: str, trajectory_id: str) -> str:
    return os.path.join(root, "trajectories", *trajectory_id.split("/"))


def list_trajectory_dirs(root: str, benchmark: str | None = None, agent: str | None = None) -> list[str]:
    """Sorted trajectory ids found under ``<root>/trajectories`` (optionally filtered)."""
    base = os.path.join(root, "trajectories")
    out = []
    for b in sorted(os.listdir(base)):
        if benchmark and b != benchmark:
            continue
        for a in sorted(os.listdir(os.path.join(base, b))):
            if agent and a != agent:
                continue
            for t in sorted(os.listdir(os.path.join(base, b, a))):
                if os.path.isfile(os.path.join(base, b, a, t, "trajectory.json")):
                    out.append(f"{b}/{a}/{t}")
    return out


def load_trajectory(root: str, trajectory_id: str, with_ui_elements: bool = True) -> Trajectory:
    d = trajectory_dir(root, trajectory_id)
    with open(os.path.join(d, "trajectory.json"), encoding="utf-8") as f:
        t = json.load(f)
    ui = None
    if with_ui_elements and os.path.isfile(os.path.join(d, "ui_elements.json")):
        with open(os.path.join(d, "ui_elements.json"), encoding="utf-8") as f:
            ui = json.load(f)["elements"]
    steps = []
    for s in t["steps"]:
        steps.append(Step(
            index=s["step"],
            screenshot_path=os.path.join(d, s["screenshot"]) if s.get("screenshot") else None,
            action=s["action"],
            reasoning=s.get("reasoning", ""),
            ui_elements=(ui[s["step"]] if ui else None),
        ))
    return Trajectory(
        trajectory_id=t["trajectory_id"], benchmark=t["benchmark"], agent=t["agent"], task_id=t["task_id"],
        task_category=t.get("task_category"), app=t.get("app"), goal=t["goal"], success=bool(t["human_success"]),
        n_steps=t["n_steps"], steps=steps,
        screenshot_paths=[os.path.join(d, p) for p in t["screenshots"] if p],
        ui_elements=ui, directory=d, raw=t,
    )


def iter_trajectories(root: str, benchmark: str | None = None, agent: str | None = None,
                      with_ui_elements: bool = False) -> Iterator[Trajectory]:
    for tid in list_trajectory_dirs(root, benchmark, agent):
        yield load_trajectory(root, tid, with_ui_elements=with_ui_elements)
