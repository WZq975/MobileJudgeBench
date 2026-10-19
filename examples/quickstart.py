#!/usr/bin/env python3
"""Load MobileJudgeBench, print a few statistics, and render one trajectory as a contact sheet."""
import argparse
import collections
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mobilejudgebench import load_annotations, load_trajectory  # noqa: E402
from mobilejudgebench.loader import trajectory_dir  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True, help="release root (contains annotations.csv and trajectories/)")
ap.add_argument("--trajectory_id", default=None, help="default: the first downloaded trajectory in annotations.csv")
ap.add_argument("--out", default="contact_sheet.png")
args = ap.parse_args()

rows = load_annotations(args.root)
present = [r for r in rows if os.path.isdir(trajectory_dir(args.root, r["trajectory_id"]))]
if not present:
    sys.exit(f"no trajectory directories under {args.root}/trajectories")
print(f"{len(rows)} trajectories in annotations.csv ({sum(r['success'] for r in rows) / len(rows):.1%} successful), "
      f"{len(present)} downloaded:")
for b, c in sorted(collections.Counter(r["benchmark"] for r in present).items()):
    print(f"  {b:14s} {c:4d}")

tid = args.trajectory_id or present[0]["trajectory_id"]
t = load_trajectory(args.root, tid)
print(f"\n{t.trajectory_id}\n  goal: {t.goal}\n  human label: {'success' if t.success else 'failure'}; "
      f"{t.n_steps} steps; {len(t.screenshot_paths)} screenshots; ui elements: {t.steps[0].ui_elements is not None}")
for s in t.steps:
    print(f"  step {s.index:2d}: {s.action}  | {s.reasoning[:70]!r}")

try:
    from PIL import Image
except ImportError:
    sys.exit("install Pillow to render the contact sheet")
imgs = t.load_images()
thumb_w = 216
thumbs = [im.resize((thumb_w, int(im.height * thumb_w / im.width))) for im in imgs]
cols = min(8, len(thumbs))
rows_n = (len(thumbs) + cols - 1) // cols
h = max(im.height for im in thumbs)
sheet = Image.new("RGB", (cols * thumb_w, rows_n * h), "white")
for i, im in enumerate(thumbs):
    sheet.paste(im, ((i % cols) * thumb_w, (i // cols) * h))
sheet.save(args.out)
print(f"\ncontact sheet with {len(thumbs)} screenshots -> {args.out}")
