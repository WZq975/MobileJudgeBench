#!/usr/bin/env python3
"""Evaluate the Baseline LLM judge on MobileJudgeBench.

    # the paper's configuration (48 screenshots at 600 px, no UI elements, no reasoning)
    python -m mobilejudgebench.judges.evaluate --root ./MobileJudgeBench \
        --model gpt-5-mini --api_provider openai \
        --baseline_max_samples 48 --baseline_no_ui --baseline_no_reasoning --baseline_resize 600

    # any OpenAI-compatible endpoint; subsets; parallel requests
    python -m mobilejudgebench.judges.evaluate --root ./MobileJudgeBench \
        --model qwen/qwen2.5-vl-72b-instruct --api_provider openrouter --benchmark android_lab --max_workers 4

Writes ``<out_dir>/<method>_<model>_<timestamp>.json`` (per-trajectory predictions and summary
metrics) and a ``.csv`` with the predictions. Metrics compare the judge's binary verdict with the
human label (positive class = successful trajectory).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Optional

import requests

from ..loader import load_annotations
from . import METHODS
from .common import get_api_config
from .inputs import JudgeInput, load_judge_input
from .metrics import print_summary, summarize
from .simple_baseline_judge import call_simple_baseline_judge


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="release root (contains annotations.csv and trajectories/)")
    ap.add_argument("--judge_method", choices=METHODS, default="simple_baseline")
    ap.add_argument("--model", default="gpt-5-mini")
    ap.add_argument("--api_provider", choices=["openai", "openrouter"], default="openai",
                    help="openai: OPENAI_API_KEY (+ optional OPENAI_BASE_URL for any compatible server); "
                         "openrouter: OPENROUTER_API_KEY")
    ap.add_argument("--api_key", default=None)
    ap.add_argument("--base_url", default=None, help="override the provider's endpoint, e.g. a local vLLM server")
    ap.add_argument("--benchmark", action="append", default=None, help="filter (repeatable)")
    ap.add_argument("--agent", action="append", default=None, help="filter (repeatable)")
    ap.add_argument("--trajectory_id", action="append", default=None, help="evaluate only these ids (repeatable)")
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N selected trajectories")
    ap.add_argument("--max_workers", type=int, default=4, help="parallel API requests")
    ap.add_argument("--out_dir", default="judge_results")
    ap.add_argument("--out_json", default=None, help="results file (default: <out_dir>/<method>_<model>_<timestamp>.json)")
    ap.add_argument("--resume", action="store_true", help="skip trajectories already completed in --out_json")
    ap.add_argument("--save_messages_dir", default=None, help="save every prompt/response (without image data) here")
    ap.add_argument("--max_output_tokens", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None,
                    help="sampling temperature (default: 0; GPT-5 models use their default)")
    ap.add_argument("--baseline_max_samples", type=int, default=5, help="screenshots attached per trajectory")
    ap.add_argument("--baseline_sampling", choices=["uniform", "event_based"], default="uniform")
    ap.add_argument("--baseline_no_ui", action="store_true", help="omit the UI element lists")
    ap.add_argument("--baseline_no_reasoning", action="store_true", help="omit the agent's reasoning")
    ap.add_argument("--baseline_ui_all_steps", action="store_true", help="UI element lists at every step, not only at the sampled ones")
    ap.add_argument("--baseline_resize", type=int, default=None, metavar="MAX_PX", help="resize screenshots to this longest edge (JPEG)")
    return ap


def judge_one(inp: JudgeInput, args: argparse.Namespace, api_key: str, base_url: str) -> tuple[int, str, Optional[int], Optional[float], Optional[dict]]:
    ident = inp.trajectory_id.replace("/", "_")
    common = dict(model=args.model, api_key=api_key, base_url=base_url, max_output_tokens=args.max_output_tokens,
                  save_messages_dir=args.save_messages_dir, message_identifier=ident)
    include_ui, include_reasoning = not args.baseline_no_ui, not args.baseline_no_reasoning
    return call_simple_baseline_judge(
        goal=inp.goal, image_paths=inp.image_paths, actions=inp.actions,
        ui_elements_list=inp.observations if (include_ui and inp.has_ui) else None,
        reasonings=inp.reasonings if include_reasoning else None,
        max_samples=args.baseline_max_samples, sampling_strategy=args.baseline_sampling,
        include_ui_elements=include_ui, include_reasoning=include_reasoning,
        ui_elements_all_steps=args.baseline_ui_all_steps, image_max_long_edge=args.baseline_resize,
        temperature=args.temperature, **common)


def _model_tag(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "-", model.split("/")[-1]).strip("-")


def run_config(args: argparse.Namespace) -> dict:
    keys = ["judge_method", "model", "api_provider", "base_url", "max_output_tokens", "temperature"]
    keys += sorted(k for k in vars(args) if k.startswith("baseline_"))
    return {k: getattr(args, k) for k in keys}


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    root = os.path.abspath(args.root)
    api_key, base_url = get_api_config(args.api_provider, args.api_key, args.base_url)

    rows = load_annotations(root)
    if args.benchmark:
        rows = [r for r in rows if r["benchmark"] in args.benchmark]
    if args.agent:
        rows = [r for r in rows if r["agent"] in args.agent]
    if args.trajectory_id:
        wanted = set(args.trajectory_id)
        rows = [r for r in rows if r["trajectory_id"] in wanted]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        sys.exit("no trajectories selected")

    os.makedirs(args.out_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_json = args.out_json or os.path.join(args.out_dir, f"{args.judge_method}_{_model_tag(args.model)}_{ts}.json")
    out_csv = os.path.splitext(out_json)[0] + ".csv"

    results: list[dict] = []
    done: set[str] = set()
    if args.resume and os.path.isfile(out_json):
        with open(out_json, encoding="utf-8") as f:
            prev = json.load(f).get("results", [])
        results = [r for r in prev if r.get("judge_result") is not None]
        done = {r["trajectory_id"] for r in results}
        print(f"resuming {out_json}: {len(results)} completed, {len(prev) - len(results)} failed entries will be retried")
    todo = [r for r in rows if r["trajectory_id"] not in done]
    print(f"{len(todo)} trajectories to evaluate ({args.judge_method}, {args.model} via {args.api_provider}) -> {out_json}", flush=True)
    config = run_config(args)

    def evaluate(i: int, row: dict) -> dict:
        tid = row["trajectory_id"]
        t0 = time.perf_counter()
        base = {"trajectory_id": tid, "benchmark": row["benchmark"], "agent": row["agent"], "task_id": row["task_id"],
                "goal": row["goal"], "annotation_success": int(row["success"]), "n_steps": row["n_steps"],
                "n_screenshots": row["n_screenshots"], "judge_method": args.judge_method}
        try:
            inp = load_judge_input(root, tid)
            res, reason, tokens, cost, details = judge_one(inp, args, api_key, base_url)
            dur = time.perf_counter() - t0
            print(f"[{i}/{len(todo)}] {tid}: judge={res} label={base['annotation_success']} "
                  f"correct={int(res == base['annotation_success'])} {dur:.1f}s", flush=True)
            return {**base, "judge_result": res, "judge_reason": reason, "judge_tokens": tokens, "judge_api_cost": cost,
                    "correct": int(res == base["annotation_success"]), "duration_seconds": dur, "details": details}
        except Exception as e:  # noqa: BLE001 - record and continue
            dur = time.perf_counter() - t0
            body = None
            if isinstance(e, requests.exceptions.HTTPError) and e.response is not None:
                body = e.response.text[:1000]
            print(f"[{i}/{len(todo)}] {tid}: ERROR {e!r} {dur:.1f}s", flush=True)
            return {**base, "error": repr(e), "error_response": body, "duration_seconds": dur}

    def write_outputs() -> dict:
        summary = summarize(results)
        summary.update(config=config, timestamp=datetime.now(timezone.utc).isoformat(),
                       duration_seconds=time.perf_counter() - t_start)
        ordered = sorted(results, key=lambda r: order.get(r["trajectory_id"], 10**9))
        tmp = out_json + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "results": ordered}, f, ensure_ascii=False, indent=1)
        os.replace(tmp, out_json)
        cols = ["trajectory_id", "benchmark", "agent", "task_id", "judge_result", "annotation_success", "correct",
                "judge_tokens", "judge_api_cost", "duration_seconds", "error", "judge_reason"]
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(ordered)
        return summary

    order = {r["trajectory_id"]: i for i, r in enumerate(rows)}
    t_start = time.perf_counter()
    ex = ThreadPoolExecutor(max_workers=max(1, args.max_workers))
    futures = [ex.submit(evaluate, i, row) for i, row in enumerate(todo, 1)]
    collected: set = set()
    try:
        for k, fut in enumerate(as_completed(futures), 1):
            results.append(fut.result())
            collected.add(fut)
            if k % 20 == 0:
                write_outputs()
        ex.shutdown(wait=True)
    except KeyboardInterrupt:
        print("interrupted: dropping queued trajectories, waiting for in-flight requests", flush=True)
        ex.shutdown(wait=True, cancel_futures=True)
        for fut in futures:
            if fut.done() and not fut.cancelled() and fut not in collected:
                results.append(fut.result())
    summary = write_outputs()
    print(f"\nsaved {out_json} and {out_csv}")
    print(f"evaluated={summary['evaluated']} errors={summary['errors']} "
          f"cost=${sum(r.get('judge_api_cost') or 0 for r in results):.4f} in {summary['duration_seconds']:.0f}s")
    print_summary(summary)


if __name__ == "__main__":
    main()
