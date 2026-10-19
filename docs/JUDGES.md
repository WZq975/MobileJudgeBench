# Running the Baseline judge

`mobilejudgebench.judges` contains the Baseline LLM-as-a-judge method from
[the paper](https://arxiv.org/abs/2608.11434), running on the release format. It needs `requests` and
`Pillow` (`pip install -e ".[judges]"`) and an API key: `OPENAI_API_KEY` (any OpenAI-compatible server
via `OPENAI_BASE_URL` or `--base_url`) or `OPENROUTER_API_KEY`.

```bash
# the paper's configuration: 48 screenshots at 600 px, no UI elements, no reasoning
python -m mobilejudgebench.judges.evaluate --root ./MobileJudgeBench \
    --model gpt-5-mini --api_provider openai \
    --baseline_max_samples 48 --baseline_no_ui --baseline_no_reasoning --baseline_resize 600

# another provider, a subset, parallel requests
python -m mobilejudgebench.judges.evaluate --root ./MobileJudgeBench \
    --model qwen/qwen2.5-vl-72b-instruct --api_provider openrouter --benchmark android_world --max_workers 4
```

Each run writes `judge_results/simple_baseline_<model>_<timestamp>.json` (per-trajectory verdicts and
summary metrics) and a `.csv` with the verdicts; `--resume` continues an interrupted run and retries
its failed trajectories. `python -m mobilejudgebench.judges.compute_metrics <results.json>` recomputes
the metrics, optionally for a subset (`--benchmark`, `--agent`).

## The judge

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
