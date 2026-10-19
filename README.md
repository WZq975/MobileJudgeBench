# MobileJudgeBench

**931 human-annotated mobile-agent trajectories for evaluating LLM-as-judge methods.**

[![arXiv](https://img.shields.io/badge/arXiv-2608.11434-b31b1b.svg)](https://arxiv.org/abs/2608.11434)
[![Paper on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Paper-blue)](https://huggingface.co/papers/2608.11434)
[![Dataset on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow)](https://huggingface.co/datasets/WesleyWzq/MobileJudgeBench)

MobileJudgeBench is the benchmark introduced in [*Benchmarking LLM Judges for Mobile Agent
Evaluation*](https://arxiv.org/abs/2608.11434). Mobile-agent benchmarks and training increasingly rely on LLM judges to decide whether an agent
completed a task; this dataset lets you measure how reliable such judges are. Every trajectory is
a full record of one agent attempting one task on an Android emulator (screenshots, accessibility
trees, actions, the agent's reasoning) together with a human success/failure label.

| | |
|---|---|
| Trajectories | 931 (492 success / 439 failure) |
| Tasks | 289 tasks over 59 apps, from SPA-Bench, AndroidWorld, A3, AndroidArena, B-MoCA and AndroidLab |
| Agents | GPT-5-mini and Qwen2.5-VL-72B (M3A), Llama-3.3-70B (T3A), UI-TARS-72B |
| Steps | 11,502 agent steps, 11,911 screenshots (1080x2400 PNG) |
| Labels | binary task-success label from human annotation (9 trained annotators, 2-4 per trajectory) |
| License | data CC BY 4.0, code MIT |

## Download

The trajectory data is 4.6 GiB (PNG screenshots), distributed as 24 tar archives (one per
benchmark x agent, each under 1 GiB) listed in `manifest.json` with SHA-256 sums. The archives are
hosted on the Hugging Face Hub at
[huggingface.co/datasets/WesleyWzq/MobileJudgeBench](https://huggingface.co/datasets/WesleyWzq/MobileJudgeBench).
The index files (`annotations.csv`, `tasks.csv`) also live in this repository under [`data/`](data/).

```bash
git clone https://github.com/WZq975/MobileJudgeBench.git
cd MobileJudgeBench

# everything (~4.6 GiB)
python -m mobilejudgebench.download --dest ./MobileJudgeBench

# subsets
python -m mobilejudgebench.download --dest ./MJB --benchmark spa_bench --benchmark a3
python -m mobilejudgebench.download --dest ./MJB --agent ui-tars-72b
python -m mobilejudgebench.download --dest ./MJB --metadata_only
```

After downloading, `python scripts/verify_release.py --root ./MobileJudgeBench --expect_trajectories 931`
checks that everything is complete and consistent.

## Quick start

```python
from mobilejudgebench import load_annotations, load_trajectory, iter_trajectories

root = "./MobileJudgeBench"
rows = load_annotations(root)                       # list of dicts, one per trajectory
traj = load_trajectory(root, rows[0]["trajectory_id"])
print(traj.goal, traj.success, traj.n_steps)
for step in traj.steps:
    print(step.index, step.action, step.screenshot_path)
    # step.reasoning, step.ui_elements (None for UI-TARS)
images = traj.load_images()                         # PIL images in time order (the judge input)

for traj in iter_trajectories(root, benchmark="android_world", agent="gpt-5-mini"):
    ...
```

`examples/quickstart.py --root ./MobileJudgeBench` prints dataset statistics and renders one
trajectory as a contact sheet. The loader has no third-party dependency (Pillow only for images).

## Evaluating a judge

The paper's Baseline judge runs on the release through `mobilejudgebench.judges` (needs `requests`
and `Pillow`, and `OPENAI_API_KEY` or `OPENROUTER_API_KEY`). With the paper's configuration:

```bash
python -m mobilejudgebench.judges.evaluate --root ./MobileJudgeBench \
    --model gpt-5-mini --api_provider openai \
    --baseline_max_samples 48 --baseline_no_ui --baseline_no_reasoning --baseline_resize 600
```

It writes the per-trajectory verdicts and the metrics (accuracy, precision, recall, F1 and balanced
accuracy against the human labels; positive class = successful trajectory) to `judge_results/`.

The same pipeline evaluates your own judge: write a function `judge(inp, **kw) -> bool` and pass it
as `--judge_method my_judge.py:judge`. Verdicts produced outside the pipeline are scored with
`python -m mobilejudgebench.judges.compute_metrics --predictions my_verdicts.csv --root ./MobileJudgeBench`.
What a judge receives, the options and two complete example judges: [docs/JUDGES.md](docs/JUDGES.md).

## Data format

```
trajectories/<benchmark>/<agent>/<task_id>/
├── trajectory.json      goal, human label, per-step action and reasoning
├── ui_elements.json     accessibility-tree elements for each screenshot (M3A and T3A agents)
└── screenshots/
    ├── step_000.png     screen before step 0 (one per step)
    ├── ...
    └── step_<n>.png     final screen, when available
```

Benchmark ids: `spa_bench`, `android_world`, `a3`, `android_arena`, `bmoca`, `android_lab`.
Agent ids: `gpt-5-mini`, `qwen2.5-vl-72b`, `llama-3.3-70b`, `ui-tars-72b`. `annotations.csv`
holds one row per trajectory (`success` = human label, plus task, app and step-count metadata);
`tasks.csv` lists the 289 tasks. Field-level documentation and provenance notes:
[docs/FORMAT.md](docs/FORMAT.md).

## Repository layout

```
data/                      annotations.csv, tasks.csv
docs/FORMAT.md             full schema documentation
docs/JUDGES.md             evaluating judges: the Baseline judge, your own judge, verdict files
mobilejudgebench/          loader (loader.py), downloader (download.py), judge pipeline and Baseline judge (judges/)
scripts/convert_raw_to_release.py   provenance: raw collection pickles -> this format
scripts/build_metadata.py           rebuilds the CSV index files from trajectory directories
scripts/verify_release.py           structural / checksum validation of a downloaded copy
scripts/package_release.py          builds the release archives + checksums
examples/quickstart.py     load the data, print statistics, render a contact sheet
examples/custom_judge.py   two example judges for the evaluation pipeline
```

## License and attribution

The trajectories, annotations and documentation are released under
[CC BY 4.0](DATA_LICENSE); the code under the [MIT License](LICENSE).

The tasks and app configurations originate from the source benchmarks, which we gratefully
acknowledge: [AndroidWorld](https://github.com/google-research/android_world) (Apache-2.0),
[B-MoCA](https://github.com/gimme1dollar/b-moca) (Apache-2.0),
[SPA-Bench](https://github.com/ai-agents-2030/SPA-Bench) (MIT),
[A3](https://github.com/YuxiangChai/AITK) (Apache-2.0), [AndroidLab](https://github.com/THUDM/Android-Lab)
(MIT) and [AndroidArena](https://github.com/AndroidArenaAgent/AndroidArena) (task definitions
used for research evaluation with attribution). Agents: AndroidWorld M3A/T3A (Apache-2.0) and
[UI-TARS](https://github.com/bytedance/UI-TARS) (Apache-2.0).

## Citation

```bibtex
@misc{wang2026benchmarkingllmjudgesmobile,
  title         = {Benchmarking LLM Judges for Mobile Agent Evaluation},
  author        = {Ziqiang Wang and Li Gu and Zhixiang Chi and Zhi Liu and Seyed Mehdi Ayyoubzadeh and Yuanhao Yu and Yang Wang},
  year          = {2026},
  eprint        = {2608.11434},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2608.11434}
}
```

## Changelog

* **v1.0.0** - initial public release: 931 trajectories, 289 tasks, human success labels.
