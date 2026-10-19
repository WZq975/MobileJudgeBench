#!/usr/bin/env python3
"""Metrics of judge verdicts against the human labels.

    # a results file written by evaluate
    python -m mobilejudgebench.judges.compute_metrics judge_results/simple_baseline_gpt-5-mini_20260101_000000.json

    # verdicts produced elsewhere: a CSV or JSON file with one prediction per trajectory
    python -m mobilejudgebench.judges.compute_metrics --predictions my_verdicts.csv --root ./MobileJudgeBench

A predictions CSV needs a ``trajectory_id`` column and a ``prediction`` column (``judge_result``,
``verdict`` and ``pred`` are accepted too) with values 1/0, true/false, yes/no or success/failure.
JSON works as a list of objects with the same keys or as an object ``{trajectory_id: prediction}``.
The labels come from ``<root>/annotations.csv`` or ``--annotations FILE``.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any

from ..loader import load_annotations
from .metrics import print_summary, summarize

PREDICTION_COLUMNS = ("prediction", "judge_result", "verdict", "pred")
TRUE_VALUES = {"1", "true", "yes", "success", "successful"}
FALSE_VALUES = {"0", "false", "no", "failure", "fail", "failed", "unsuccessful"}


def parse_prediction(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and value in (0, 1):
        return int(value)
    s = str(value).strip().lower()
    if s in TRUE_VALUES:
        return 1
    if s in FALSE_VALUES:
        return 0
    raise ValueError(f"cannot read the prediction {value!r} (use 1/0, true/false, yes/no or success/failure)")


def read_predictions(path: str) -> dict[str, int]:
    """``{trajectory_id: 0/1}`` from a CSV or JSON predictions file; entries without a value are skipped."""
    if path.lower().endswith(".json"):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("results"), list):  # a results file written by evaluate
            data = data["results"]
        rows = [{"trajectory_id": k, "prediction": v} for k, v in data.items()] if isinstance(data, dict) else list(data)
    else:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path}: no entries")
    column = next((c for c in PREDICTION_COLUMNS if any(isinstance(r, dict) and c in r for r in rows)), None)
    if column is None:
        raise ValueError(f"{path}: no prediction column (one of {', '.join(PREDICTION_COLUMNS)})")
    out: dict[str, int] = {}
    for i, r in enumerate(rows, 1):
        if not isinstance(r, dict) or not r.get("trajectory_id"):
            raise ValueError(f"{path}: entry {i} has no trajectory_id")
        tid = str(r["trajectory_id"])
        if tid in out:
            raise ValueError(f"{path}: trajectory_id {tid!r} appears twice")
        value = r.get(column)
        if value is None or str(value).strip() == "":
            continue
        out[tid] = parse_prediction(value)
    return out


def join_predictions(predictions_path: str, annotations_path: str) -> list[dict]:
    """One row per labelled trajectory: label, metadata and the prediction (None when absent)."""
    preds = read_predictions(predictions_path)
    labels = load_annotations(annotations_path)
    unknown = sorted(set(preds) - {r["trajectory_id"] for r in labels})
    if unknown:
        raise ValueError(f"{len(unknown)} trajectory ids in {predictions_path} are not in annotations.csv, "
                         f"e.g. {', '.join(unknown[:3])}")
    rows = [{"trajectory_id": r["trajectory_id"], "benchmark": r["benchmark"], "agent": r["agent"],
             "task_id": r["task_id"], "judge_result": preds.get(r["trajectory_id"]),
             "annotation_success": int(r["success"])} for r in labels]
    missing = sum(1 for r in rows if r["judge_result"] is None)
    if missing:
        print(f"warning: {missing} of {len(rows)} trajectories have no prediction and are left out of the metrics",
              file=sys.stderr)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_json", nargs="*", help="results file(s) written by evaluate")
    ap.add_argument("--predictions", default=None, help="CSV or JSON file of verdicts produced elsewhere (see above)")
    ap.add_argument("--root", default=None, help="release root whose annotations.csv provides the labels for --predictions")
    ap.add_argument("--annotations", default=None, help="annotations.csv to use instead of <root>/annotations.csv")
    ap.add_argument("--benchmark", action="append", default=None, help="restrict to these benchmarks (repeatable)")
    ap.add_argument("--agent", action="append", default=None, help="restrict to these agents (repeatable)")
    ap.add_argument("--out_json", default=None, help="also write the metrics (and the scored rows) to this file")
    args = ap.parse_args()
    if not args.results_json and not args.predictions:
        ap.error("give a results file or --predictions")
    if args.out_json and len(args.results_json) + bool(args.predictions) > 1:
        ap.error("--out_json takes a single input")

    runs = []
    for path in args.results_json:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        config = data.get("summary", {}).get("config", {})
        runs.append((path, config.get("model") or config.get("judge_method") or "?", data["results"]))
    if args.predictions:
        labels = args.annotations or (os.path.join(args.root, "annotations.csv") if args.root else None)
        if not labels:
            ap.error("--predictions needs --root or --annotations for the labels")
        try:
            runs.append((args.predictions, "predictions", join_predictions(args.predictions, labels)))
        except (ValueError, OSError) as e:
            sys.exit(str(e))

    for path, label, results in runs:
        if args.benchmark:
            results = [r for r in results if r.get("benchmark") in args.benchmark]
        if args.agent:
            results = [r for r in results if r.get("agent") in args.agent]
        summary = summarize(results)
        print(f"\n=== {path} ({label})")
        print_summary(summary)
        if args.out_json:
            with open(args.out_json, "w", encoding="utf-8") as f:
                json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=1)
            print(f"\nsaved {args.out_json}")


if __name__ == "__main__":
    main()
