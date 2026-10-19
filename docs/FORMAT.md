# MobileJudgeBench data format

This document describes every file in a MobileJudgeBench release. All identifiers are
lower-case ASCII, paths are relative to the release root, and all JSON is UTF-8.

```
<root>/
├── README.md, LICENSE (code: MIT), DATA_LICENSE (data: CC BY 4.0), VERSION
├── annotations.csv                 one row per trajectory: human label + metadata (931 rows)
├── tasks.csv                       one row per task (289 rows)
└── trajectories/<benchmark>/<agent>/<task_id>/
    ├── trajectory.json             goal, label, per-step actions, reasoning, screenshot references
    ├── ui_elements.json            accessibility-tree elements per screenshot (absent for UI-TARS)
    └── screenshots/
        ├── step_000.png            screen *before* step 0 (one per step, 1080x2400 PNG)
        ├── ...
        └── step_<n_steps>.png      final screen, when available
```

## Identifiers

| Field | Values |
|---|---|
| `benchmark` | `spa_bench`, `android_world`, `a3`, `android_arena`, `bmoca`, `android_lab` |
| `agent` | `gpt-5-mini` (M3A, `gpt-5-mini-2025-08-07`), `qwen2.5-vl-72b` (M3A, `Qwen2.5-VL-72B-Instruct`), `llama-3.3-70b` (T3A, `Llama-3.3-70B-Instruct`), `ui-tars-72b` (UI-TARS, `UI-TARS-72B-SFT`) |
| `task_id` | the source benchmark's task name, e.g. `ClockStopWatchRunning`, `airbnb_1`, `CNN_task3` |
| `trajectory_id` | `<benchmark>/<agent>/<task_id>`, also the directory path under `trajectories/` |

Every task was attempted once by each agent, so a task has at most 4 trajectories. Agents are
the [AndroidWorld](https://github.com/google-research/android_world) reference agents M3A
(multimodal, set-of-mark screenshots + accessibility tree) and T3A (text-only, accessibility
tree), and the native UI-TARS agent (screenshot-only, pixel-coordinate actions).

## `annotations.csv`

| Column | Description |
|---|---|
| `trajectory_id`, `benchmark`, `agent`, `task_id` | identifiers (see above) |
| `task_category` | `single_app` or `cross_app` (SPA-Bench cross-app tasks and AndroidArena `cross_app_*` tasks) |
| `app` | the app the task runs in, one lower-case identifier per app shared across benchmarks (`clock`, `google_maps`, `markor`); cross-app tasks list their apps separated by `;` (`settings;clock`) |
| `goal` | natural-language task instruction given to the agent |
| `success` | **human label**: `1` = task completed successfully, `0` = not completed |
| `n_steps` | number of agent steps |
| `n_screenshots` | number of screenshots (`n_steps` or `n_steps + 1`, see "Screenshots") |

## `tasks.csv`

`benchmark`, `task_id`, `task_category`, `app`, `goal`: one row per task. `goal` is the instruction
of the task's first trajectory; one task (`android_world/MarkorCreateFolder`) has a per-run folder
name in its instruction, so per-trajectory goals are in `annotations.csv` and `trajectory.json`.

## `trajectory.json`

```jsonc
{
  "trajectory_id": "android_world/gpt-5-mini/ClockStopWatchRunning",
  "benchmark": "android_world", "agent": "gpt-5-mini",
  "task_id": "ClockStopWatchRunning", "task_category": "single_app", "app": "clock",
  "goal": "Run the stopwatch.",
  "human_success": true,             // human label
  "n_steps": 4,
  "n_screenshots": 5,
  "screenshots": ["screenshots/step_000.png", "...", "screenshots/step_004.png"],  // time order, see "Screenshots"
  "steps": [
    {
      "step": 0,
      "screenshot": "screenshots/step_000.png",      // what the agent observed before acting
      "action": {"type": "click", "index": 11, "x": 540, "y": 1200},   // what it did, see "Actions"
      "reasoning": "..."                              // the agent's stated reasoning for this step
    }
  ]
}
```

Every screenshot is 1080x2400 pixels (portrait); `x`/`y` and element bounding boxes use that
pixel grid with the origin at the top-left corner.

### Actions

`action` is the action the agent requested, as parsed by the runner (AndroidWorld `JSONAction`),
restricted to the fields below with null fields removed. For index-based actions the target is
resolved to pixel coordinates (`x`, `y` = centre of element `index` in `ui_elements.json`). A
request the runner could not execute (an out-of-range or malformed `index`, a `swipe` without
`direction`, ...) is kept as the agent produced it; it has no resolved `x`/`y` and did not change
the screen. Action types:

| `type` | extra fields | notes |
|---|---|---|
| `click`, `long_press`, `double_tap` | `x`, `y` (+ `index` for M3A/T3A) | |
| `input_text` | `text` (+ `index`/`x`/`y` for the target field) | |
| `scroll` | `direction` (`up`/`down`/`left`/`right`), optional `index`, `touch_xy`/`lift_xy` | UI-TARS scrolls carry start/end points |
| `swipe` | `direction` | |
| `open_app` | `app_name` | |
| `navigate_home`, `navigate_back`, `keyboard_enter`, `wait` | | |
| `answer` | `text` | agent's answer for question tasks |
| `status` | `goal_status` (`complete` / `infeasible`), optional `text` | agent declares the episode finished |
| `unknown` | | no action could be parsed from the model output; the runner executes nothing for such a step (43 steps in 18 trajectories, mostly Qwen2.5-VL) |

### Screenshots

* `screenshots` lists the screens in time order. `step_<i>.png` is the raw screen (no overlays)
  before step *i*, i.e. what the agent observed when choosing action *i*. M3A agents saw a
  set-of-mark version of it (numbered boxes drawn from `ui_elements.json`), which can be
  regenerated from the element bounding boxes; T3A agents saw only the element list.
* `step_<n_steps>.png`, when available, is the final screen after the last step. A trajectory
  therefore has `n_steps` or `n_steps + 1` screenshots, and the last entry is always the final
  state.
* The `screenshots` list is exactly the image sequence used as judge input in the paper.

## `ui_elements.json`

```jsonc
{"trajectory_id": "...",
 "elements": [ [ {element}, ... ],   // elements visible in screenshots[0] (list index = `action.index`)
               ... ] }               // one entry per screenshot; null when no element list is available
```

Each element (AndroidWorld `UIElement`, accessibility-tree node):
`text`, `content_description`, `hint_text`, `class_name`, `resource_name`, `package_name`,
`bbox_pixels` (`x_min`, `y_min`, `x_max`, `y_max`), and the state flags `is_clickable`,
`is_long_clickable`, `is_editable`, `is_scrollable`, `is_focusable`, `is_selected`, `is_checked`,
`is_checkable`, `is_enabled`. UI-TARS trajectories have no element lists (the agent does not use
the accessibility tree), so the file is absent for them.

## Provenance

* Trajectories were collected in October-November 2025 on Android emulators (Pixel-class device,
  1080x2400) with a unified runner built on the AndroidWorld codebase; apps, task definitions and
  initial states come from the six source benchmarks.
* Labels: 9 trained graduate-student annotators labelled the trajectories (2-4 per trajectory)
  with a dedicated annotation tool (video playback, step-by-step screenshot inspection and a
  structured form); `success` is the final label. The annotation protocol and agreement
  statistics are described in [the paper](https://arxiv.org/abs/2608.11434).
* Privacy: all runs used test accounts and seeded content; no real user data appears in
  screenshots, UI trees, actions or reasoning.
