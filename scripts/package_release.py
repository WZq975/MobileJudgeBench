#!/usr/bin/env python3
"""Package a MobileJudgeBench release directory into downloadable archives.

Produces, in ``--out``:

    MobileJudgeBench-<ver>-metadata.tar.gz           README, FORMAT.md, licenses, annotations/tasks CSVs
    MobileJudgeBench-<ver>-<benchmark>-<agent>[-partN].tar   trajectory directories (uncompressed tar;
                                                             PNGs do not compress; parts stay below --max_part_gb)
    SHA256SUMS                                       sha256 of every archive
    manifest.json                                    archive -> trajectory ids, sizes, hashes
    CHECKSUMS_files.sha256                           sha256 of every file inside the release (for verify_release.py)

Archives extract relative to the release root (``trajectories/<benchmark>/<agent>/<task_id>/...``).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tarfile
from collections import OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor

METADATA_FILES = ["README.md", "LICENSE", "DATA_LICENSE", "annotations.csv", "tasks.csv", "VERSION", "FORMAT.md"]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dir_size(path: str) -> int:
    total = 0
    for dp, _, fns in os.walk(path):
        for fn in fns:
            total += os.path.getsize(os.path.join(dp, fn))
    return total


def _hash_rel(args):
    root, rel = args
    return sha256(os.path.join(root, rel)), rel


def make_tar(job):
    root, out_path, rel_dirs = job
    with tarfile.open(out_path, "w") as tf:
        for rel in rel_dirs:
            tf.add(os.path.join(root, rel), arcname=rel, recursive=True)
    return out_path, sha256(out_path), os.path.getsize(out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", default=None, help="default: contents of <root>/VERSION")
    ap.add_argument("--max_part_gb", type=float, default=1.9, help="GitHub release assets must be < 2 GiB")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip_file_checksums", action="store_true")
    args = ap.parse_args()
    root, out = os.path.abspath(args.root), os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    version = args.version or open(os.path.join(root, "VERSION")).read().strip()
    prefix = f"MobileJudgeBench-v{version}"
    max_bytes = int(args.max_part_gb * (1 << 30))

    with open(os.path.join(root, "annotations.csv"), newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    groups = defaultdict(list)
    for r in rows:
        groups[(r["benchmark"], r["agent"])].append(r["trajectory_id"])

    jobs, manifest = [], OrderedDict()
    for (b, a), tids in groups.items():
        parts, cur, cur_size = [], [], 0
        for tid in sorted(tids):
            sz = dir_size(os.path.join(root, "trajectories", *tid.split("/")))
            if cur and cur_size + sz > max_bytes:
                parts.append(cur)
                cur, cur_size = [], 0
            cur.append(tid)
            cur_size += sz
        if cur:
            parts.append(cur)
        for i, part in enumerate(parts, 1):
            name = f"{prefix}-{b}-{a}" + (f"-part{i}" if len(parts) > 1 else "") + ".tar"
            jobs.append((root, os.path.join(out, name), [f"trajectories/{t}" for t in part]))
            manifest[name] = {"benchmark": b, "agent": a, "trajectory_ids": part}

    meta_name = f"{prefix}-metadata.tar.gz"
    with tarfile.open(os.path.join(out, meta_name), "w:gz") as tf:
        for fn in METADATA_FILES:
            p = os.path.join(root, fn)
            if os.path.isfile(p):
                tf.add(p, arcname=fn)
    manifest[meta_name] = {"files": [fn for fn in METADATA_FILES if os.path.isfile(os.path.join(root, fn))]}
    manifest[meta_name].update(sha256=sha256(os.path.join(out, meta_name)), bytes=os.path.getsize(os.path.join(out, meta_name)))

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for path, h, size in ex.map(make_tar, jobs):
            name = os.path.basename(path)
            manifest[name].update(sha256=h, bytes=size)
            print(f"{name}: {size / 2**30:.2f} GiB, {len(manifest[name].get('trajectory_ids', []))} trajectories")

    with open(os.path.join(out, "SHA256SUMS"), "w") as f:
        for name, m in manifest.items():
            f.write(f"{m['sha256']}  {name}\n")
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump({"version": version, "archives": manifest}, f, indent=1)

    if not args.skip_file_checksums:
        rels = []
        for dp, _, fns in os.walk(root):
            for fn in fns:
                rels.append(os.path.relpath(os.path.join(dp, fn), root))
        rels.sort()
        with ProcessPoolExecutor(max_workers=args.workers) as ex, open(os.path.join(out, "CHECKSUMS_files.sha256"), "w") as f:
            for h, rel in ex.map(_hash_rel, [(root, r) for r in rels], chunksize=64):
                f.write(f"{h}  {rel}\n")
        print(f"hashed {len(rels)} files")
    print("done ->", out)


if __name__ == "__main__":
    main()
