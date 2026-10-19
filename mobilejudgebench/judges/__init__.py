"""The Baseline LLM-as-a-judge method from the MobileJudgeBench paper, running on the release format.

    python -m mobilejudgebench.judges.evaluate --root ./MobileJudgeBench \
        --model gpt-5-mini --api_provider openai

See docs/JUDGES.md.
"""
METHODS = ["simple_baseline"]
