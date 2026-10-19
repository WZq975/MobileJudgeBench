#!/usr/bin/env python3
"""Build the tabular index files of a MobileJudgeBench release from the trajectory directories.

Reads every ``trajectories/*/*/*/trajectory.json`` under ``--root`` and writes, into ``--root``
(and optionally ``--copy_to``):

    annotations.csv   one row per trajectory
    tasks.csv         one row per task

and prints the dataset statistics (per benchmark as a Markdown table).
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import shutil

ANN_COLS = ["trajectory_id", "benchmark", "agent", "task_id", "task_category", "app", "goal", "success",
            "n_steps", "n_screenshots"]
TASK_COLS = ["benchmark", "task_id", "task_category", "app", "goal"]
BENCH_ORDER = ["spa_bench", "android_world", "a3", "android_arena", "bmoca", "android_lab"]
BENCH_NAMES = {"spa_bench": "SPA-Bench", "android_world": "AndroidWorld", "a3": "A3", "android_arena": "AndroidArena",
               "bmoca": "B-MoCA", "android_lab": "AndroidLab"}
AGENT_ORDER = ["gpt-5-mini", "qwen2.5-vl-72b", "llama-3.3-70b", "ui-tars-72b"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--copy_to", default=None, help="also copy the CSV outputs into this directory (e.g. the repo's data/)")
    args = ap.parse_args()
    root = args.root

    rows = []
    base = os.path.join(root, "trajectories")
    for b in sorted(os.listdir(base)):
        for a in sorted(os.listdir(os.path.join(base, b))):
            for t in sorted(os.listdir(os.path.join(base, b, a))):
                p = os.path.join(base, b, a, t, "trajectory.json")
                if not os.path.isfile(p):
                    continue
                with open(p, encoding="utf-8") as f:
                    tj = json.load(f)
                tid = tj["trajectory_id"]
                assert tid == f"{b}/{a}/{t}", (tid, b, a, t)
                rows.append({
                    "trajectory_id": tid, "benchmark": b, "agent": a, "task_id": t,
                    "task_category": tj.get("task_category") or "", "app": tj.get("app") or "", "goal": tj["goal"],
                    "success": int(bool(tj["human_success"])),
                    "n_steps": tj["n_steps"],
                    "n_screenshots": tj["n_screenshots"],
                })
    rows.sort(key=lambda r: (BENCH_ORDER.index(r["benchmark"]) if r["benchmark"] in BENCH_ORDER else 99,
                             AGENT_ORDER.index(r["agent"]) if r["agent"] in AGENT_ORDER else 99, r["task_id"]))

    tasks = collections.OrderedDict()
    for r in rows:
        k = (r["benchmark"], r["task_id"])
        if k not in tasks:  # goal of the task's first trajectory (per-trajectory goals are in annotations.csv)
            tasks[k] = {"benchmark": r["benchmark"], "task_id": r["task_id"], "task_category": r["task_category"],
                        "app": r["app"], "goal": r["goal"]}

    # statistics
    stats = {"n_trajectories": len(rows), "n_tasks": len(tasks), "n_success": sum(r["success"] for r in rows),
             "n_failure": sum(1 - r["success"] for r in rows), "n_steps_total": sum(r["n_steps"] for r in rows),
             "n_screenshots": sum(r["n_screenshots"] for r in rows),
             "n_apps": len({a for r in rows for a in r["app"].split(";") if a}),
             "per_benchmark": {}, "per_agent": {}}
    for b in BENCH_ORDER:
        br = [r for r in rows if r["benchmark"] == b]
        if not br:
            continue
        stats["per_benchmark"][b] = {
            "name": BENCH_NAMES.get(b, b), "tasks": len({r["task_id"] for r in br}), "trajectories": len(br),
            "apps": len({a for r in br for a in r["app"].split(";") if a}), "success": sum(r["success"] for r in br),
            "success_rate": round(sum(r["success"] for r in br) / len(br), 3),
            "mean_steps": round(sum(r["n_steps"] for r in br) / len(br), 1),
        }
    for a in AGENT_ORDER:
        ar = [r for r in rows if r["agent"] == a]
        if ar:
            stats["per_agent"][a] = {"trajectories": len(ar), "success": sum(r["success"] for r in ar),
                                     "success_rate": round(sum(r["success"] for r in ar) / len(ar), 3),
                                     "mean_steps": round(sum(r["n_steps"] for r in ar) / len(ar), 1)}

    outs = {"annotations.csv": (ANN_COLS, rows), "tasks.csv": (TASK_COLS, list(tasks.values()))}
    for name, (cols, data) in outs.items():
        with open(os.path.join(root, name), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(data)
    if args.copy_to:
        os.makedirs(args.copy_to, exist_ok=True)
        for name in ("annotations.csv", "tasks.csv"):
            shutil.copy2(os.path.join(root, name), os.path.join(args.copy_to, name))

    print(f"trajectories={stats['n_trajectories']} tasks={stats['n_tasks']} success={stats['n_success']} "
          f"failure={stats['n_failure']} steps={stats['n_steps_total']} screenshots={stats['n_screenshots']} "
          f"apps={stats['n_apps']}")
    print("| Benchmark | Tasks | Traj. | Apps | Success rate | Mean steps |")
    print("|---|---|---|---|---|---|")
    for b, s in stats["per_benchmark"].items():
        print(f"| {s['name']} | {s['tasks']} | {s['trajectories']} | {s['apps']} | {s['success_rate']:.1%} | {s['mean_steps']} |")


if __name__ == "__main__":
    main()
