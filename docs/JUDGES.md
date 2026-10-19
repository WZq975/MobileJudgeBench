# Evaluating judges

`mobilejudgebench.judges` runs a judge over the release and scores its verdicts against the human
labels: the Baseline LLM-as-a-judge method from [the paper](https://arxiv.org/abs/2608.11434), or
your own. It needs `requests` and `Pillow` (`pip install -e ".[judges]"`); a judge that calls an API
needs a key, `OPENAI_API_KEY` (any OpenAI-compatible server via `OPENAI_BASE_URL` or `--base_url`) or
`OPENROUTER_API_KEY`.

Every run writes `judge_results/<judge>_<model>_<timestamp>.json` (per-trajectory verdicts and the
summary metrics: accuracy, precision, recall, F1 and balanced accuracy, overall and per benchmark and
agent; positive class = successful trajectory) and a `.csv` with the verdicts. `--resume` continues an
interrupted run and retries its failed trajectories; `--benchmark`, `--agent`, `--trajectory_id`
(repeatable) and `--limit` select a subset; `--max_workers` (default 4) judges trajectories in parallel.
`python -m mobilejudgebench.judges.compute_metrics <results.json>` recomputes the metrics of a run,
optionally for a subset (`--benchmark`, `--agent`).

## The Baseline judge

```bash
# the paper's configuration: 48 screenshots at 600 px, no UI elements, no reasoning
python -m mobilejudgebench.judges.evaluate --root ./MobileJudgeBench \
    --model gpt-5-mini --api_provider openai \
    --baseline_max_samples 48 --baseline_no_ui --baseline_no_reasoning --baseline_resize 600

# another provider, a subset, parallel requests
python -m mobilejudgebench.judges.evaluate --root ./MobileJudgeBench \
    --model qwen/qwen2.5-vl-72b-instruct --api_provider openrouter --benchmark android_world --max_workers 4
```

The judge receives the task goal and then the trajectory step by step: every step shows its action,
optionally the agent's reasoning, and at the sampled steps the UI element list (optional) followed by
the screenshot. It answers with a success/failure verdict, a confidence level and a short reason.

| Option (default) | Effect |
|---|---|
| `--baseline_max_samples 5` | screenshots attached per trajectory (all of them when the trajectory has fewer) |
| `--baseline_sampling uniform` | which steps get a screenshot: `uniform` keeps the first, second-to-last and last screens and spreads the rest evenly; `event_based` prefers steps with text entry, app switches, navigation, long presses and terminal actions, and changes of action type, spread over the trajectory |
| `--baseline_no_ui` | omit the UI element lists |
| `--baseline_no_reasoning` | omit the agent's reasoning |
| `--baseline_ui_all_steps` | UI element lists at every step, not only at the sampled ones |
| `--baseline_resize MAX_PX` | resize the screenshots to this longest edge (JPEG); original resolution when not set |

Common options: `--root`, `--model`, `--api_provider {openai,openrouter}`, `--api_key`, `--base_url`,
`--benchmark`/`--agent`/`--trajectory_id` (repeatable filters), `--limit`, `--max_workers`,
`--out_dir`/`--out_json`, `--resume`, `--save_messages_dir` (every prompt and response, without
image data), `--max_output_tokens`, `--temperature` (default 0; GPT-5 models use their default).

## Your own judge

The pipeline runs any judge that is a Python function taking a `JudgeInput` and returning a verdict.
Put it in a file or an importable module and pass it as `--judge_method FILE.py:FUNCTION` (or
`MODULE:FUNCTION`):

```python
# my_judge.py
from mobilejudgebench.judges.inputs import JudgeInput

def judge(inp: JudgeInput, **kw) -> bool:
    ...                     # look at inp.goal, inp.image_paths, inp.actions, ...
    return success          # True = the agent completed the task
```

```bash
python -m mobilejudgebench.judges.evaluate --root ./MobileJudgeBench --judge_method my_judge.py:judge \
    --model gpt-5-mini --api_provider openai     # --model/--api_provider only if the judge calls an API
```

`inp` describes one trajectory:

| Field | Content |
|---|---|
| `goal` | the task instruction |
| `image_paths` | the screenshots in time order: one before each step, plus the final screen when available |
| `actions` | one dict per step, `{"type", "detail_type", "detail"[, "reasoning"]}`; `format_action` in `simple_baseline_judge` renders one as text, e.g. `click at (540, 1200)` (table below) |
| `observations` | one UI-element text block per screenshot, `[No UI elements]` where none exists (UI-TARS trajectories); `has_ui` tells whether any exists |
| `reasonings` | the agent's stated reasoning per step (`None` when absent) |
| `trajectory` | the loaded `mobilejudgebench.loader.Trajectory`: raw per-step action dicts, UI element lists, `load_images()` |
| `trajectory_id`, `benchmark`, `agent`, `task_id`, `n_steps` | identifiers and size |
| `success` | the human label, kept for inspection; a judge must not read it |

The keyword arguments are the run's `model`, `api_key`, `base_url`, `max_output_tokens`,
`temperature`, `save_messages_dir` and `message_identifier`; take the ones you need and absorb the
rest with `**kw`. `api_key` and `base_url` are resolved from `--api_provider`, `--api_key` and
`--base_url` as for the Baseline judge, and are `None` when no key is configured, so a judge that
calls no API needs none. `mobilejudgebench.judges.common` offers `post_chat_completion` (an
OpenAI-style chat completion with retries), `usage_cost` and `save_messages`.

Return `True`/`False` (or 1/0), or a dict `{"success": bool, "reason": str, "tokens": int,
"cost": float, "details": ...}` with everything but `success` optional, to have the reason, token
count, cost in USD and any details recorded per trajectory. An exception inside the judge is recorded
as an error for that trajectory and the run continues (`--resume` retries the errors). Judges run in
parallel threads; use `--max_workers 1` if yours is not thread-safe. The results file is
`judge_results/<file>.<function>_<model>_<timestamp>.json`.

[`examples/custom_judge.py`](../examples/custom_judge.py) has two complete judges:
`claimed_completion` (success iff the agent's last action declares the task complete; no API needed) and
`final_screen_llm` (one vision-model call with the goal, the action list and the final screenshot):

```bash
python -m mobilejudgebench.judges.evaluate --root ./MobileJudgeBench \
    --judge_method examples/custom_judge.py:claimed_completion
python -m mobilejudgebench.judges.evaluate --root ./MobileJudgeBench \
    --judge_method examples/custom_judge.py:final_screen_llm --model gpt-5-mini --api_provider openai
```

## Verdicts produced elsewhere

A judge that runs outside this pipeline is scored from a file of its verdicts:

```bash
python -m mobilejudgebench.judges.compute_metrics --predictions my_verdicts.csv --root ./MobileJudgeBench
```

The CSV needs a `trajectory_id` column and a `prediction` column (`judge_result`, `verdict` and `pred`
are accepted too) with values `1`/`0`, `true`/`false`, `yes`/`no` or `success`/`failure`. A JSON file
works as a list of objects with the same keys or as an object `{trajectory_id: prediction}`. The labels
come from `annotations.csv` (`--root`, or `--annotations FILE`). Trajectories without a prediction are
reported and left out of the metrics; `--out_json` saves the metrics and the scored rows.

## How a trajectory is presented

`mobilejudgebench.judges.inputs` turns a loaded trajectory into the judge's input: the screenshots
in time order, one action line per step, one UI-element text block per screenshot and the agent's
reasoning per step.

| `action.type` | text shown to the judge |
|---|---|
| `click`, `long_press`, `double_tap` | `click at (x, y)` (`click: index not found` when the request had no resolvable target) |
| `input_text`, `answer`, `open_app` | `type: <text>`, `answer: <text>`, `open app: <app_name>` |
| `scroll` | `scroll: Scroll <direction>.` (UI-TARS: `... from (x,y) to (x,y).`) |
| `navigate_home`, `navigate_back`, `keyboard_enter`, `wait` | `press home: Return to home page.`, `press back: Back to the previous page.`, `press enter: Press the enter key.`, `wait: Wait.` |
| `status` | `task complete: Task completed.` / `task impossible: Task impossible.` (UI-TARS: its own completion message) |
| `unknown`, `swipe` | `unknown`, `swipe` |

A UI-element block lists at most 50 elements, one per line: class, resource name, interactivity
(`click;long-click;editable;scroll`), state (`selected`, `checked`) and text; it is
`[No UI elements]` where no element list is available.
