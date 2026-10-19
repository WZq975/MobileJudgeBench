#!/usr/bin/env python3
"""Evaluate a judge on MobileJudgeBench: the paper's Baseline judge, or your own.

    # the paper's Baseline judge in the paper's configuration (48 screenshots at 600 px, no UI elements, no reasoning)
    python -m mobilejudgebench.judges.evaluate --root ./MobileJudgeBench \
        --model gpt-5-mini --api_provider openai \
        --baseline_max_samples 48 --baseline_no_ui --baseline_no_reasoning --baseline_resize 600

    # any OpenAI-compatible endpoint; subsets; parallel requests
    python -m mobilejudgebench.judges.evaluate --root ./MobileJudgeBench \
        --model qwen/qwen2.5-vl-72b-instruct --api_provider openrouter --benchmark android_lab --max_workers 4

    # your own judge: a function judge(inp, **kw) -> bool in a file or a module (see docs/JUDGES.md)
    python -m mobilejudgebench.judges.evaluate --root ./MobileJudgeBench --judge_method my_judge.py:judge

Writes ``<out_dir>/<method>_<model>_<timestamp>.json`` (per-trajectory verdicts and summary metrics)
and a ``.csv`` with the verdicts. Metrics compare the judge's binary verdict with the human label
(positive class = successful trajectory).
"""
from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import requests

from ..loader import load_annotations, trajectory_dir
from . import METHODS
from .common import get_api_config
from .inputs import JudgeInput, load_judge_input
from .metrics import print_summary, summarize
from .simple_baseline_judge import call_simple_baseline_judge

# (success, reason, tokens, cost in USD, details) -- what every judge call yields
Verdict = "tuple[int, str, Optional[int], Optional[float], Optional[dict]]"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="release root (contains annotations.csv and trajectories/)")
    ap.add_argument("--judge_method", default="simple_baseline",
                    help="'simple_baseline' (the paper's Baseline judge) or your own judge as FILE.py:FUNCTION "
                         "or MODULE:FUNCTION (see docs/JUDGES.md)")
    ap.add_argument("--model", default=None,
                    help="judge backbone (Baseline judge default: gpt-5-mini; a custom judge receives it as model=)")
    ap.add_argument("--api_provider", choices=["openai", "openrouter"], default="openai",
                    help="openai: OPENAI_API_KEY (+ optional OPENAI_BASE_URL for any compatible server); "
                         "openrouter: OPENROUTER_API_KEY")
    ap.add_argument("--api_key", default=None)
    ap.add_argument("--base_url", default=None, help="override the provider's endpoint, e.g. a local vLLM server")
    ap.add_argument("--benchmark", action="append", default=None, help="filter (repeatable)")
    ap.add_argument("--agent", action="append", default=None, help="filter (repeatable)")
    ap.add_argument("--trajectory_id", action="append", default=None, help="evaluate only these ids (repeatable)")
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N selected trajectories")
    ap.add_argument("--max_workers", type=int, default=4, help="parallel judge calls")
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


def _is_file_spec(target: str) -> bool:
    return target.endswith(".py") or os.path.sep in target or (os.path.altsep is not None and os.path.altsep in target)


def load_custom_judge(spec: str) -> Callable[..., Any]:
    """Resolve ``FILE.py:FUNCTION`` or ``MODULE:FUNCTION`` to the judge function."""
    if ":" not in spec:
        raise ValueError(f"unknown judge {spec!r}: use {', '.join(METHODS)}, FILE.py:FUNCTION or MODULE:FUNCTION")
    target, func = spec.rsplit(":", 1)
    if _is_file_spec(target):
        if not os.path.isfile(target):
            raise ValueError(f"no such file: {target!r}")
        name = "mjb_custom_judge_" + re.sub(r"\W+", "_", os.path.splitext(os.path.basename(target))[0])
        module_spec = importlib.util.spec_from_file_location(name, target)
        if module_spec is None or module_spec.loader is None:
            raise ValueError(f"cannot load {target!r}")
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[name] = module
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    fn = getattr(module, func, None)
    if not callable(fn):
        raise ValueError(f"{target!r} has no function {func!r}")
    return fn


def method_tag(spec: str) -> str:
    """Name of the judge in output file names: ``simple_baseline`` or ``<file or module>.<function>``."""
    if spec in METHODS:
        return spec
    target, func = spec.rsplit(":", 1)
    if _is_file_spec(target):
        target = os.path.splitext(os.path.basename(target))[0]
    return re.sub(r"[^A-Za-z0-9._-]+", "-", f"{target}.{func}").strip("-")


def _binary(value: Any) -> int:
    if isinstance(value, str) or value is None:
        raise TypeError(f"the judge returned {value!r}; return True/False or a dict with a boolean 'success'")
    try:
        v = int(value)
    except (TypeError, ValueError):
        raise TypeError(f"the judge returned {type(value).__name__}; return True/False or a dict with a boolean 'success'") from None
    if v not in (0, 1):
        raise ValueError(f"the judge returned {value!r}; return True/False or a dict with a boolean 'success'")
    return v


def normalize_verdict(ret: Any) -> "Verdict":
    """A custom judge returns ``True``/``False`` (or 1/0), or a dict with ``success`` and optionally
    ``reason``, ``tokens``, ``cost`` (USD) and ``details``."""
    if isinstance(ret, dict):
        if "success" not in ret:
            raise ValueError("the judge returned a dict without a 'success' key")
        return (_binary(ret["success"]), str(ret.get("reason") or ""), ret.get("tokens"), ret.get("cost"),
                ret.get("details"))
    return _binary(ret), "", None, None, None


def judge_one(inp: JudgeInput, args: argparse.Namespace, judge_fn: Optional[Callable[..., Any]],
              api_key: Optional[str], base_url: Optional[str]) -> "Verdict":
    ident = inp.trajectory_id.replace("/", "_")
    common = dict(model=args.model, api_key=api_key, base_url=base_url, max_output_tokens=args.max_output_tokens,
                  temperature=args.temperature, save_messages_dir=args.save_messages_dir, message_identifier=ident)
    if judge_fn is not None:
        return normalize_verdict(judge_fn(inp, **common))
    include_ui, include_reasoning = not args.baseline_no_ui, not args.baseline_no_reasoning
    return call_simple_baseline_judge(
        goal=inp.goal, image_paths=inp.image_paths, actions=inp.actions,
        ui_elements_list=inp.observations if (include_ui and inp.has_ui) else None,
        reasonings=inp.reasonings if include_reasoning else None,
        max_samples=args.baseline_max_samples, sampling_strategy=args.baseline_sampling,
        include_ui_elements=include_ui, include_reasoning=include_reasoning,
        ui_elements_all_steps=args.baseline_ui_all_steps, image_max_long_edge=args.baseline_resize, **common)


def _model_tag(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "-", model.split("/")[-1]).strip("-")


def run_config(args: argparse.Namespace) -> dict:
    keys = ["judge_method", "model", "api_provider", "base_url", "max_output_tokens", "temperature"]
    if args.judge_method in METHODS:
        keys += sorted(k for k in vars(args) if k.startswith("baseline_"))
    return {k: getattr(args, k) for k in keys}


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    root = os.path.abspath(args.root)
    custom = args.judge_method not in METHODS
    try:
        judge_fn = load_custom_judge(args.judge_method) if custom else None
    except (ValueError, ImportError) as e:
        sys.exit(f"--judge_method: {e}")
    if not custom and not args.model:
        args.model = "gpt-5-mini"
    try:  # a custom judge may call no API at all, so the key is optional for it
        api_key, base_url = get_api_config(args.api_provider, args.api_key, args.base_url, required=not custom)
    except ValueError as e:
        sys.exit(str(e))

    rows = load_annotations(root)
    if args.benchmark:
        rows = [r for r in rows if r["benchmark"] in args.benchmark]
    if args.agent:
        rows = [r for r in rows if r["agent"] in args.agent]
    if args.trajectory_id:
        wanted = set(args.trajectory_id)
        rows = [r for r in rows if r["trajectory_id"] in wanted]
    present = [r for r in rows if os.path.isdir(trajectory_dir(root, r["trajectory_id"]))]
    if len(present) < len(rows):
        print(f"{len(rows) - len(present)} of the {len(rows)} selected trajectories are not under {root} "
              "(partial download) and are skipped", flush=True)
        rows = present
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        sys.exit("no trajectories selected")

    os.makedirs(args.out_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    stem = method_tag(args.judge_method) + (f"_{_model_tag(args.model)}" if args.model else "")
    out_json = args.out_json or os.path.join(args.out_dir, f"{stem}_{ts}.json")
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
    via = f", {args.model} via {args.api_provider}" if args.model else ""
    print(f"{len(todo)} trajectories to evaluate ({args.judge_method}{via}) -> {out_json}", flush=True)
    config = run_config(args)

    def evaluate(i: int, row: dict) -> dict:
        tid = row["trajectory_id"]
        t0 = time.perf_counter()
        base = {"trajectory_id": tid, "benchmark": row["benchmark"], "agent": row["agent"], "task_id": row["task_id"],
                "goal": row["goal"], "annotation_success": int(row["success"]), "n_steps": row["n_steps"],
                "n_screenshots": row["n_screenshots"], "judge_method": args.judge_method}
        try:
            inp = load_judge_input(root, tid)
            res, reason, tokens, cost, details = judge_one(inp, args, judge_fn, api_key, base_url)
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
        print("interrupted: dropping queued trajectories, waiting for in-flight calls", flush=True)
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
