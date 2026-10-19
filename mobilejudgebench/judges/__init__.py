"""Judge evaluation on MobileJudgeBench: the paper's Baseline LLM-as-a-judge method, the pipeline that
runs it (or your own judge function) over the release, and the metrics.

    python -m mobilejudgebench.judges.evaluate --root ./MobileJudgeBench \
        --model gpt-5-mini --api_provider openai
    python -m mobilejudgebench.judges.evaluate --root ./MobileJudgeBench --judge_method my_judge.py:judge

See docs/JUDGES.md.
"""
METHODS = ["simple_baseline"]
