#!/usr/bin/env python3
"""Validate the structure and internal consistency of a MobileJudgeBench release directory.

Checks that every trajectory in annotations.csv has a directory whose trajectory.json, screenshots
and ui_elements.json agree with the CSV, optionally decodes every PNG and verifies SHA-256 sums.
Exit status is non-zero when any problem is found.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

TRAJ_KEYS = {"trajectory_id", "benchmark", "agent", "task_id", "task_category", "app", "goal", "human_success",
             "n_steps", "n_screenshots", "screenshots", "steps"}
STEP_KEYS = {"step", "screenshot", "action", "reasoning"}
ACTION_KEYS = {"type", "index", "x", "y", "text", "direction", "touch_xy", "lift_xy", "app_name", "goal_status"}
ELEMENT_KEYS = {"text", "content_description", "class_name", "bbox_pixels", "hint_text", "is_checked", "is_checkable",
                "is_clickable", "is_editable", "is_enabled", "is_focusable", "is_long_clickable", "is_scrollable",
                "is_selected", "package_name", "resource_name"}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_trajectory(root: str, row: dict, decode: bool) -> list[str]:
    probs = []
    d = os.path.join(root, "trajectories", *row["trajectory_id"].split("/"))
    tj_path = os.path.join(d, "trajectory.json")
    if not os.path.isfile(tj_path):
        return [f"{row['trajectory_id']}: missing trajectory.json"]
    with open(tj_path, encoding="utf-8") as f:
        t = json.load(f)
    tid = row["trajectory_id"]
    if t["trajectory_id"] != tid:
        probs.append(f"{tid}: id mismatch {t['trajectory_id']}")
    if set(t) != TRAJ_KEYS:
        probs.append(f"{tid}: unexpected top-level keys {sorted(set(t) ^ TRAJ_KEYS)}")
    if int(bool(t["human_success"])) != int(row["success"]):
        probs.append(f"{tid}: label mismatch json={t['human_success']} csv={row['success']}")
    if t["n_steps"] != len(t["steps"]) or t["n_steps"] != int(row["n_steps"]):
        probs.append(f"{tid}: n_steps mismatch")
    if t["goal"] != row["goal"]:
        probs.append(f"{tid}: goal mismatch")
    shots = t["screenshots"]
    if len(shots) not in (t["n_steps"], t["n_steps"] + 1):
        probs.append(f"{tid}: {len(shots)} screenshots for {t['n_steps']} steps")
    if shots[: t["n_steps"]] != [s.get("screenshot") for s in t["steps"]]:
        probs.append(f"{tid}: screenshots[:n_steps] differs from steps[].screenshot")
    if any(p != f"screenshots/step_{i:03d}.png" for i, p in enumerate(shots) if p):
        probs.append(f"{tid}: screenshot names are not step_<index>.png")
    n_shots = len([p for p in shots if p])
    if t["n_screenshots"] != n_shots or n_shots != int(row["n_screenshots"]):
        probs.append(f"{tid}: n_screenshots mismatch")
    expected = {p for p in shots if p}
    present = {f"screenshots/{f}" for f in os.listdir(os.path.join(d, "screenshots"))} if os.path.isdir(os.path.join(d, "screenshots")) else set()
    if expected != present:
        probs.append(f"{tid}: screenshot files differ (missing={sorted(expected - present)[:3]}, extra={sorted(present - expected)[:3]})")
    for i, s in enumerate(t["steps"]):
        if s["step"] != i:
            probs.append(f"{tid}: step index {s['step']} at position {i}")
        if set(s) != STEP_KEYS:
            probs.append(f"{tid}: step {i} has unexpected keys {sorted(set(s) ^ STEP_KEYS)}")
        if not isinstance(s.get("action"), dict) or "type" not in s["action"]:
            probs.append(f"{tid}: step {i} has no action.type")
        elif set(s["action"]) - ACTION_KEYS:
            probs.append(f"{tid}: step {i} action has undocumented keys {sorted(set(s['action']) - ACTION_KEYS)}")
    ui_path = os.path.join(d, "ui_elements.json")
    if t["agent"] != "ui-tars-72b":  # M3A and T3A trajectories carry accessibility-tree elements
        if not os.path.isfile(ui_path):
            probs.append(f"{tid}: ui_elements.json missing")
        else:
            with open(ui_path, encoding="utf-8") as f:
                u = json.load(f)
            if u.get("trajectory_id") != tid:
                probs.append(f"{tid}: ui_elements.json id mismatch")
            if len(u["elements"]) != len(shots):
                probs.append(f"{tid}: ui_elements has {len(u['elements'])} entries for {len(shots)} screenshots")
            bad_el = next((e for lst in u["elements"] if lst for e in lst if set(e) != ELEMENT_KEYS), None)
            if bad_el is not None:
                probs.append(f"{tid}: element keys {sorted(set(bad_el) ^ ELEMENT_KEYS)} differ from the schema")
    elif os.path.isfile(ui_path):
        probs.append(f"{tid}: unexpected ui_elements.json")
    if decode:
        from PIL import Image
        for rel in sorted(expected):
            try:
                with Image.open(os.path.join(d, rel)) as im:
                    im.verify()
            except Exception as e:
                probs.append(f"{tid}: cannot decode {rel}: {e!r}")
    return probs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--expect_trajectories", type=int, default=None)
    ap.add_argument("--decode_images", action="store_true", help="decode every PNG (slower)")
    ap.add_argument("--checksums", default=None, help="SHA256SUMS-style file (relative paths) to verify")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--subset", action="store_true", help="only verify trajectories whose directory exists (partial downloads)")
    args = ap.parse_args()
    root = args.root

    with open(os.path.join(root, "annotations.csv"), newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ids = [r["trajectory_id"] for r in rows]
    problems = []
    if len(set(ids)) != len(ids):
        problems.append("duplicate trajectory ids in annotations.csv")
    on_disk = set()
    base = os.path.join(root, "trajectories")
    for b in os.listdir(base):
        for a in os.listdir(os.path.join(base, b)):
            for t in os.listdir(os.path.join(base, b, a)):
                on_disk.add(f"{b}/{a}/{t}")
    extra = on_disk - set(ids)
    if extra:
        problems.append(f"{len(extra)} trajectory dirs not in annotations.csv, e.g. {sorted(extra)[:3]}")
    if args.expect_trajectories is not None and len(rows) != args.expect_trajectories:
        problems.append(f"annotations.csv has {len(rows)} rows, expected {args.expect_trajectories}")
    if args.subset:
        present = [r for r in rows if r["trajectory_id"] in on_disk]
        print(f"subset mode: {len(present)} of {len(rows)} trajectories present on disk")
        if not present:
            problems.append("no trajectory directories found")
        rows = present

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for probs in ex.map(lambda r: check_trajectory(root, r, args.decode_images), rows):
            problems.extend(probs)

    if args.checksums:
        bad = 0
        with open(args.checksums, encoding="utf-8") as f:
            entries = [l.rstrip("\n").split("  ", 1) for l in f if l.strip()]
        if args.subset:  # partial download: only the trajectories present and the root files that exist
            keep = {r["trajectory_id"] for r in rows}
            entries = [e for e in entries
                       if (e[1].startswith("trajectories/") and "/".join(e[1].split("/")[1:4]) in keep)
                       or ("/" not in e[1] and os.path.isfile(os.path.join(root, e[1])))]
        def _chk(e):
            h, rel = e
            p = os.path.join(root, rel)
            if not os.path.isfile(p):
                return f"missing file: {rel}"
            return None if sha256(p) == h else f"checksum mismatch: {rel}"
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for r in ex.map(_chk, entries):
                if r:
                    bad += 1
                    if bad <= 5:
                        problems.append(r)
        if bad:
            problems.append(f"{bad} checksum problems in total")
        print(f"checksums verified: {len(entries) - bad}/{len(entries)} ok")

    per_b = collections.Counter(r["benchmark"] for r in rows)
    per_a = collections.Counter(r["agent"] for r in rows)
    n_succ = sum(int(r["success"]) for r in rows)
    print(f"trajectories={len(rows)} tasks={len({(r['benchmark'], r['task_id']) for r in rows})} success={n_succ} failure={len(rows) - n_succ}")
    print("per benchmark:", dict(per_b))
    print("per agent:", dict(per_a))
    if problems:
        print(f"\n{len(problems)} PROBLEMS:")
        for p in problems[:50]:
            print("  -", p)
        sys.exit(1)
    print("OK: release is consistent")


if __name__ == "__main__":
    main()
