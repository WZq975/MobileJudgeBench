#!/usr/bin/env python3
"""Convert raw MobileJudgeBench trajectory pickles (``*.pkl.gz``) into the public release format.

The raw pickles are produced by the unified AndroidWorld-based runner used to collect
MobileJudgeBench (M3A, T3A and UI-TARS agents). Each pickle is a dict with ``goal``,
``task_template``, ``agent_name``, ``episode_data`` (a dict of per-step lists holding raw
numpy screenshots, accessibility-tree elements, the agent's raw output, ...) and a few
run-level fields.

For every trajectory listed in the ground-truth CSV this script writes::

    <out_root>/trajectories/<benchmark>/<agent>/<task_id>/
        trajectory.json          goal, label, per-step actions / reasoning / screenshot refs
        ui_elements.json         accessibility-tree elements per screenshot (absent for UI-TARS)
        screenshots/step_000.png screen *before* step 0 (one per step: step_<i>.png)
        screenshots/step_<n>.png final screen, when available

and writes one JSON diagnostics record per trajectory to ``--records_out`` (build log).

This script is provided for provenance; end users should download the released data.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import gzip
import io
import json
import os
import pickle
import re
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Naming maps: raw collection names -> release identifiers
# ---------------------------------------------------------------------------
BENCHMARKS = {  # raw dir -> (release id, display name)
    "a3": ("a3", "A3"),
    "android-lab": ("android_lab", "AndroidLab"),
    "android_world": ("android_world", "AndroidWorld"),
    "arena": ("android_arena", "AndroidArena"),
    "bmoca": ("bmoca", "B-MoCA"),
    "spa": ("spa_bench", "SPA-Bench"),
}
AGENTS = {  # raw agent tag -> (release id, model name, agent framework, model id used for the agent)
    "gpt-5-mini": ("gpt-5-mini", "GPT-5-mini", "M3A", "gpt-5-mini-2025-08-07"),
    "qwen": ("qwen2.5-vl-72b", "Qwen2.5-VL-72B", "M3A", "Qwen2.5-VL-72B-Instruct"),
    "llama": ("llama-3.3-70b", "Llama-3.3-70B", "T3A", "Llama-3.3-70B-Instruct"),
    "ui-tars": ("ui-tars-72b", "UI-TARS-72B", "UI-TARS", "UI-TARS-72B-SFT"),
}
ANDROID_WORLD_APPS = sorted(
    ["Clock", "Contacts", "Expense", "Files", "Markor", "Notes", "OpenApp", "OsmAnd", "Recipe", "Retro"],
    key=lambda x: -len(x),
)
# Release app identifiers: one lower-case name per app, shared across benchmarks. The raw task ids
# spell the same app differently (AndroidWorld "Clock", SPA-Bench "clock"), and AndroidWorld names
# task families rather than apps ("Expense" tasks run in Pro Expense, "Notes" in Joplin, "Recipe" in
# Broccoli, "Retro" in Retro Music; the released "OpenApp" tasks open Contacts).
APP_IDS = {  # (release benchmark id, app name parsed from the task id) -> release app id
    ("android_world", "Clock"): "clock", ("android_world", "Contacts"): "contacts",
    ("android_world", "Expense"): "pro_expense", ("android_world", "Files"): "files",
    ("android_world", "Markor"): "markor", ("android_world", "Notes"): "joplin",
    ("android_world", "OpenApp"): "contacts", ("android_world", "OsmAnd"): "osmand",
    ("android_world", "Recipe"): "broccoli", ("android_world", "Retro"): "retro_music",
    ("a3", "CNN"): "cnn", ("a3", "Gmail"): "gmail", ("a3", "GooglePlayBooks"): "google_play_books",
    ("a3", "GoogleTasks"): "google_tasks", ("a3", "YTmusic"): "youtube_music",
    ("spa_bench", "booking.com"): "booking", ("spa_bench", "calendar"): "google_calendar",
    ("android_arena", "calendar"): "google_calendar", ("android_arena", "weather"): "the_weather_channel",
    ("android_lab", "calendar"): "google_calendar",
    ("android_lab", "setting"): "settings", ("android_lab", "pimusic"): "pi_music_player",
}
CROSS_APP_APPS = {  # (release benchmark id, task id) -> the apps the instruction involves, in order
    ("spa_bench", "General_Tool_2_ENG"): ["settings", "clock"],
    ("spa_bench", "Information_Management_2_ENG"): ["chrome", "spotify"],
    ("spa_bench", "Media_Entertainment_2_ENG"): ["youtube", "clock"],
    ("spa_bench", "Multi_Apps_4_ENG"): ["expedia", "evernote", "accuweather"],
    ("spa_bench", "Social_Sharing_1_ENG"): ["bbc", "gmail"],
    ("spa_bench", "Web_Shopping_2_ENG"): ["instagram", "airbnb"],
    ("android_arena", "cross_app_13"): ["contacts", "google_maps"],
    ("android_arena", "cross_app_14"): ["contacts", "the_weather_channel"],
    ("android_arena", "cross_app_15"): ["google_maps", "messages"],
}
# AndroidWorld UIElement fields that carry information in this dataset (is_focused, is_visible,
# resource_id and tooltip are constant or always null in every collected element and are not released).
UI_ELEMENT_FIELDS = [
    "text", "content_description", "class_name", "bbox_pixels", "hint_text",
    "is_checked", "is_checkable", "is_clickable", "is_editable", "is_enabled",
    "is_focusable", "is_long_clickable", "is_scrollable", "is_selected",
    "package_name", "resource_name",
]
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{8,}\d)(?!\d)")


# ---------------------------------------------------------------------------
# Safe unpickling: the pickles reference classes from the collection runner
# (android_world.*, requests.Response, ...). We only need their state.
# ---------------------------------------------------------------------------
class _State:
    """Generic placeholder that keeps the pickled ``__dict__`` of an unknown class."""

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        self.__dict__ = state if isinstance(state, dict) else {}


class _SafeUnpickler(pickle.Unpickler):
    _ALLOWED_PREFIXES = ("numpy", "datetime", "builtins", "collections", "_codecs", "copyreg")

    def find_class(self, module, name):
        if module.startswith(self._ALLOWED_PREFIXES):
            return super().find_class(module, name)
        return _State


def load_raw(path: str) -> dict:
    with gzip.open(path, "rb") as f:
        data = _SafeUnpickler(io.BytesIO(f.read())).load()
    if isinstance(data, list) and len(data) == 1:
        data = data[0]
    if not isinstance(data, dict):
        raise ValueError(f"unexpected top-level object {type(data).__name__}")
    return data


def _state(obj: Any) -> dict | None:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    return getattr(obj, "__dict__", None)


def normalize_steps(episode: Any) -> list[dict]:
    """episode_data is a dict of equal-length lists (one per field); turn it into per-step dicts."""
    if isinstance(episode, list):
        steps = [s for s in episode if isinstance(s, dict)]
    elif isinstance(episode, dict):
        n = max((len(v) for v in episode.values() if isinstance(v, (list, tuple))), default=0)
        steps = []
        for i in range(n):
            step = {}
            for k, v in episode.items():
                step[k] = v[i] if isinstance(v, (list, tuple)) and i < len(v) else (None if isinstance(v, (list, tuple)) else v)
            steps.append(step)
    else:
        steps = []
    steps.sort(key=lambda s: (s.get("step_number") if isinstance(s.get("step_number"), int) else 0))
    return steps


def first_array(*vals):
    for v in vals:
        if isinstance(v, np.ndarray):
            return v
    return None


# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------
def extract_reasoning(agent_output: str) -> str:
    """'Reason: ...\\n\\nAction: {...}' (M3A/T3A) or 'Thought: ...\\nAction: ...' (UI-TARS)."""
    if not isinstance(agent_output, str):
        return ""
    text = agent_output.strip()
    if "Thought:" in text:
        m = re.match(r"Thought:\s*(.*?)(?:\nAction:|\Z)", text, re.DOTALL)
        if m:
            return m.group(1).strip()
    if "Reason:" in text:
        m = re.match(r"Reason:\s*(.*?)(?:\n\s*\nAction:|\n\s*Action:|\Z)", text, re.DOTALL)
        if m:
            return m.group(1).strip()
    return ""


def parse_action_json_from_text(agent_output: str) -> dict | None:
    """Extract the JSON action that follows 'Action:' in an M3A/T3A output."""
    if not isinstance(agent_output, str):
        return None
    m = re.search(r"Action:\s*(\{[\s\S]*?\})\s*$", agent_output) or re.search(r"(\{[\s\S]*\})", agent_output)
    if not m:
        return None
    s = m.group(1)
    for cand in (s, s.replace("'", '"')):
        try:
            d = json.loads(cand)
            return d if isinstance(d, dict) else None
        except Exception:
            continue
    return None


def element_center(elements: list | None, idx: Any) -> tuple[int, int] | None:
    try:
        idx = int(idx)
    except Exception:
        return None
    if not isinstance(elements, (list, tuple)) or not (0 <= idx < len(elements)):
        return None
    st = _state(elements[idx]) or {}
    bb = _state(st.get("bbox_pixels")) or _state(st.get("bbox"))
    if not bb:
        return None
    try:
        return int((bb["x_min"] + bb["x_max"]) / 2), int((bb["y_min"] + bb["y_max"]) / 2)
    except Exception:
        return None


ACTION_KEYS = ("index", "x", "y", "text", "direction", "touch_xy", "lift_xy", "app_name", "goal_status")


def unify_action(raw: dict | None, elements: list | None) -> tuple[dict, list[str]]:
    """Release action: the runner's JSONAction restricted to the documented fields (ACTION_KEYS), null
    fields dropped, and index-based targets resolved to pixel coordinates (centre of the referenced
    accessibility-tree element). Returns (action, keys dropped because they are outside the schema,
    e.g. from malformed model output)."""
    if not raw:
        return {"type": "unknown"}, []
    a: dict[str, Any] = {"type": str(raw.get("action_type") or "unknown")}
    dropped = []
    for k, v in raw.items():
        if k == "action_type" or v is None:
            continue
        if k not in ACTION_KEYS:
            dropped.append(k)
            continue
        if isinstance(v, tuple):
            v = list(v)
        a[k] = v
    if a.get("x") is None and "index" in a:
        c = element_center(elements, a["index"])
        if c:
            a["x"], a["y"] = c
    return a, dropped


# ---------------------------------------------------------------------------
# Naming helpers
# ---------------------------------------------------------------------------
def parse_source_path(rel: str) -> dict:
    """'spa/spa_gpt-5-mini_single/airbnb_1_0.pkl.gz' -> ids."""
    raw_bench, raw_sub, fname = rel.split("/")
    if raw_bench not in BENCHMARKS:
        raise ValueError(f"unknown benchmark dir {raw_bench}")
    rest = raw_sub[len(raw_bench) + 1:]
    spa_split = None
    if raw_bench == "spa":
        rest, spa_split = rest.rsplit("_", 1)
    if rest not in AGENTS:
        raise ValueError(f"unknown agent tag {rest} in {rel}")
    stem = fname[: -len(".pkl.gz")]
    m = re.match(r"^(.*)_(\d+)$", stem)
    task_id, instance = (m.group(1), int(m.group(2))) if m else (stem, 0)
    bench_id, bench_name = BENCHMARKS[raw_bench]
    agent_id, agent_model, framework, api_model = AGENTS[rest]
    app, category = extract_app(raw_bench, task_id, spa_split)
    return dict(benchmark=bench_id, benchmark_name=bench_name, agent=agent_id, agent_model=agent_model,
                agent_framework=framework, api_model=api_model, task_id=task_id, instance=instance, app=app,
                task_category=category, raw_benchmark=raw_bench, raw_agent_tag=rest)


def extract_app(raw_bench: str, task_id: str, spa_split: str | None) -> tuple[str, str]:
    """Release app id(s) and task category of a task.

    The app name is parsed from the task id per benchmark naming convention and mapped to the
    release identifier (``APP_IDS``). SPA-Bench cross-app tasks (``General_Tool_2_ENG``) and
    AndroidArena ``cross_app_*`` tasks span several apps: ``app`` lists them separated by ``;``
    (``CROSS_APP_APPS``) and ``task_category`` is ``cross_app``.
    """
    bench = BENCHMARKS[raw_bench][0]
    if raw_bench == "android_world":
        raw_app = next((p for p in ANDROID_WORLD_APPS if task_id.startswith(p)), None)
        if raw_app is None:
            m = re.match(r"([A-Z][a-z]+)", task_id)
            raw_app = m.group(1) if m else task_id
    elif raw_bench == "a3":
        parts = task_id.split("_task")
        raw_app = parts[0] if len(parts) > 1 else task_id
    elif raw_bench == "spa" and spa_split == "cross":
        raw_app = "cross_app"
    else:
        parts = task_id.split("_")
        idx = len(parts) - 1
        while idx > 0 and parts[idx].isdigit():
            idx -= 1
        raw_app = "_".join(parts[: idx + 1])
    if raw_app == "cross_app":
        return ";".join(CROSS_APP_APPS[(bench, task_id)]), "cross_app"
    app = APP_IDS.get((bench, raw_app), raw_app)
    if not re.fullmatch(r"[a-z0-9_]+", app):
        raise ValueError(f"unmapped app name {raw_app!r} for {bench}/{task_id}")
    return app, "single_app"


# ---------------------------------------------------------------------------
# Per-trajectory conversion
# ---------------------------------------------------------------------------
def ui_element_to_dict(el: Any) -> dict:
    st = _state(el) or {}
    out = {}
    for k in UI_ELEMENT_FIELDS:
        v = st.get(k)
        if k == "bbox_pixels":
            bb = _state(v)
            v = {kk: bb.get(kk) for kk in ("x_min", "y_min", "x_max", "y_max")} if bb else None
        elif isinstance(v, (np.integer,)):
            v = int(v)
        elif isinstance(v, (np.floating,)):
            v = float(v)
        elif isinstance(v, (np.bool_,)):
            v = bool(v)
        out[k] = v
    return out


def save_png(arr: np.ndarray, path: str, compress_level: int) -> int:
    img = Image.fromarray(np.ascontiguousarray(arr))
    img.save(path, format="PNG", compress_level=compress_level)
    return os.path.getsize(path)


def scan_pii(texts: list[str]) -> dict:
    emails, phones = set(), set()
    for t in texts:
        if not t:
            continue
        emails.update(EMAIL_RE.findall(t))
        phones.update(p.strip() for p in PHONE_RE.findall(t))
    return {"emails": sorted(emails)[:20], "n_emails": len(emails), "phones": sorted(phones)[:20], "n_phones": len(phones)}


def write_json_atomic(path: str, obj: Any, **kw) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, **kw)
    os.replace(tmp, path)


def convert_one(args: tuple) -> dict:
    rel, raw_root, out_root, human_success, compress_level, overwrite, skip_images = args
    t0 = time.time()
    ids = parse_source_path(rel)
    traj_id = f"{ids['benchmark']}/{ids['agent']}/{ids['task_id']}"
    out_dir = os.path.join(out_root, "trajectories", *traj_id.split("/"))
    shot_dir = os.path.join(out_dir, "screenshots")
    if os.path.isfile(os.path.join(out_dir, "trajectory.json")) and not overwrite:
        return {"trajectory_id": traj_id, "source_file": rel, "skipped": True}
    os.makedirs(shot_dir, exist_ok=True)

    data = load_raw(os.path.join(raw_root, rel))
    steps = normalize_steps(data.get("episode_data"))
    raw_agent = str(data.get("agent_name") or "")
    goal = str(data.get("goal") or "").strip()
    task_template = data.get("task_template")
    screen = _state(data.get("screen_config")) or {}

    step_records, ui_steps, texts = [], [], [goal]
    dropped_action_keys: set[str] = set()
    action_sources: set[str] = set()
    n_missing_shots = n_elements = 0
    bytes_written = 0
    has_ui = False
    for i, step in enumerate(steps):
        shot = first_array(step.get("clean_screenshot"), step.get("before_screenshot"), step.get("raw_screenshot"))
        shot_rel = None
        if shot is not None:
            shot_rel = f"screenshots/step_{i:03d}.png"
            if not skip_images:
                bytes_written += save_png(shot, os.path.join(out_dir, shot_rel), compress_level)
        else:
            n_missing_shots += 1

        elements = step.get("before_ui_elements")
        if elements is None:
            elements = step.get("before_element_list")
        if isinstance(elements, (list, tuple)):
            has_ui = True
            el_dicts = [ui_element_to_dict(e) for e in elements]
            n_elements += len(el_dicts)
            texts.extend(str(e[k]) for e in el_dicts for k in ("text", "content_description", "hint_text") if e[k])
        else:
            el_dicts = None
        ui_steps.append(el_dicts)

        agent_output = step.get("action_output")
        agent_output = agent_output if isinstance(agent_output, str) else (None if agent_output is None else str(agent_output))
        raw_json = _state(step.get("action_output_json"))
        action_source = "runner_json"
        if raw_json is None and agent_output:
            raw_json = parse_action_json_from_text(agent_output)
            action_source = "parsed_from_agent_output" if raw_json else "none"
        if raw_json is not None:
            raw_json = {k: (list(v) if isinstance(v, tuple) else v) for k, v in raw_json.items()}
        reasoning = step.get("action_reason")
        if not isinstance(reasoning, str) or not reasoning.strip():
            reasoning = extract_reasoning(agent_output or "")
        summary = step.get("summary")
        summary = summary if isinstance(summary, str) else None
        texts.extend([agent_output or "", reasoning or "", summary or ""])
        action, dropped = unify_action(raw_json, elements if isinstance(elements, (list, tuple)) else None)
        dropped_action_keys.update(dropped)
        action_sources.add(action_source)
        # Released per step: what the agent observed (screenshot, elements), did (action) and why
        # (reasoning).
        step_records.append({
            "step": i,
            "screenshot": shot_rel,
            "action": action,
            "reasoning": reasoning or "",
        })

    # Final screen after the last step, when the raw data holds one: released as screenshot index
    # n_steps, so a trajectory has n_steps or n_steps + 1 screenshots and the last is the final state.
    final_rel, final_som, final_elements = None, False, None
    if steps:
        last = steps[-1]
        fin = first_array(last.get("after_screenshot"))
        if fin is None:
            fin = first_array(last.get("after_screenshot_with_som"))
            final_som = fin is not None
        if fin is not None:
            final_rel = f"screenshots/step_{len(steps):03d}.png"
            if not skip_images:
                bytes_written += save_png(fin, os.path.join(out_dir, final_rel), compress_level)
            fe = last.get("after_element_list")
            if fe is None:
                fe = last.get("after_ui_elements")
            if isinstance(fe, (list, tuple)):
                final_elements = [ui_element_to_dict(e) for e in fe]
    screenshots = [s["screenshot"] for s in step_records] + ([final_rel] if final_rel else [])
    if has_ui and final_rel:
        ui_steps.append(final_elements)  # None when no element list is available

    is_successful = data.get("is_successful")
    try:
        is_successful = None if is_successful is None or (isinstance(is_successful, float) and np.isnan(is_successful)) else float(is_successful)
    except Exception:
        is_successful = None
    finish = data.get("finish_dtime")
    finish = finish.isoformat() if isinstance(finish, _dt.datetime) else (str(finish) if finish else None)
    exc = data.get("exception_info")

    traj = {
        "trajectory_id": traj_id,
        "benchmark": ids["benchmark"],
        "agent": ids["agent"],
        "task_id": ids["task_id"],
        "task_category": ids["task_category"],
        "app": ids["app"],
        "goal": goal,
        "human_success": human_success,
        "n_steps": len(steps),
        "n_screenshots": len([p for p in screenshots if p]),
        "screenshots": screenshots,
        "steps": step_records,
    }
    write_json_atomic(os.path.join(out_dir, "trajectory.json"), traj, indent=2)
    if has_ui:
        write_json_atomic(os.path.join(out_dir, "ui_elements.json"),
                          {"trajectory_id": traj_id, "elements": ui_steps},
                          separators=(",", ":"))

    rec = {
        "trajectory_id": traj_id, "source_file": rel, **{k: ids[k] for k in ("benchmark", "agent", "task_id", "app", "task_category", "raw_agent_tag")},
        "task_template": task_template, "template_matches_filename": (task_template == ids["task_id"]),
        "goal": goal, "human_success": human_success, "env_is_successful_raw": is_successful,
        "n_steps": len(steps), "n_screenshots": traj["n_screenshots"], "has_post_action_screenshot": final_rel is not None,
        "post_action_screenshot_has_som_marks": final_som, "has_post_action_elements": final_elements is not None,
        "has_ui_elements": has_ui, "n_ui_elements": n_elements, "n_missing_screenshots": n_missing_shots,
        "runner_agent_name": raw_agent, "api_model": ids["api_model"], "instance_id": data.get("instance_id"),
        "seed": data.get("seed"), "run_time_seconds": data.get("run_time"), "finish_time": finish,
        "screen": {"width": screen.get("width"), "height": screen.get("height"), "orientation": screen.get("orientation")},
        "exception_info": None if exc is None else str(exc),
        "action_sources": sorted(action_sources), "dropped_action_keys": sorted(dropped_action_keys),
        "action_types": sorted({s["action"]["type"] for s in step_records}),
        "episode_keys": sorted(steps[0].keys()) if steps else [],
        "pii": scan_pii(texts), "bytes_written": bytes_written, "seconds": round(time.time() - t0, 1),
    }
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw_root", required=True, help="Directory holding <benchmark>/<benchmark>_<agent>/<task>_<i>.pkl.gz")
    ap.add_argument("--annotations_csv", required=True, help="Ground-truth CSV with 'trajectory' (raw relative path) and 'final_success' (yes/no)")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--records_out", required=True, help="JSONL with one conversion record per trajectory")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--compress_level", type=int, default=9)
    ap.add_argument("--only", default=None, help="regex on the raw relative path (pilot runs)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--skip_images", action="store_true", help="rewrite the JSON files only (screenshots must already exist)")
    args = ap.parse_args()

    with open(args.annotations_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    jobs = []
    for r in rows:
        rel = r["trajectory"].strip()
        if args.only and not re.search(args.only, rel):
            continue
        label = r["final_success"].strip().lower()
        if label not in ("yes", "no"):
            raise ValueError(f"bad final_success {label!r} for {rel}")
        jobs.append((rel, args.raw_root, args.out_root, label == "yes", args.compress_level, args.overwrite, args.skip_images))
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"{len(jobs)} trajectories to convert -> {args.out_root}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.records_out)), exist_ok=True)

    n_ok = n_err = 0
    t0 = time.time()
    with open(args.records_out, "w", encoding="utf-8") as fout, ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(convert_one, j): j[0] for j in jobs}
        for k, fut in enumerate(as_completed(futs), 1):
            rel = futs[fut]
            try:
                rec = fut.result()
                n_ok += 1
            except Exception as e:  # keep going; report at the end
                rec = {"source_file": rel, "error": repr(e), "traceback": traceback.format_exc()}
                n_err += 1
                print(f"[ERROR] {rel}: {e!r}", flush=True)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            if k % 25 == 0 or k == len(jobs):
                print(f"[{k}/{len(jobs)}] ok={n_ok} err={n_err} elapsed={time.time()-t0:.0f}s", flush=True)
    print(f"done: ok={n_ok} err={n_err} in {time.time()-t0:.0f}s", flush=True)
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
