#!/usr/bin/env python3
"""Recompute the summary metrics of a results file written by ``evaluate``.

    python -m mobilejudgebench.judges.compute_metrics judge_results/simple_baseline_gpt-5-mini_20260101_000000.json
"""
from __future__ import annotations

import argparse
import json

from .metrics import print_summary, summarize


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_json", nargs="+")
    ap.add_argument("--benchmark", action="append", default=None, help="restrict to these benchmarks (repeatable)")
    ap.add_argument("--agent", action="append", default=None, help="restrict to these agents (repeatable)")
    args = ap.parse_args()
    for path in args.results_json:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        results = data["results"]
        if args.benchmark:
            results = [r for r in results if r.get("benchmark") in args.benchmark]
        if args.agent:
            results = [r for r in results if r.get("agent") in args.agent]
        print(f"\n=== {path} ({data.get('summary', {}).get('config', {}).get('model', '?')})")
        print_summary(summarize(results))


if __name__ == "__main__":
    main()
