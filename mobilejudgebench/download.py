#!/usr/bin/env python3
"""Download and extract MobileJudgeBench.

    python -m mobilejudgebench.download --dest ./MobileJudgeBench                 # everything
    python -m mobilejudgebench.download --dest ./MJB --benchmark spa_bench        # one benchmark
    python -m mobilejudgebench.download --dest ./MJB --agent ui-tars-72b          # one agent
    python -m mobilejudgebench.download --dest ./MJB --metadata_only              # CSVs + docs only

Sources: ``--source hf`` (the Hugging Face Hub dataset repository; default), ``--source github``
(GitHub release assets) or ``--source local --local_dir DIR`` (archives already downloaded, e.g.
from a mirror). All carry the same archives, listed in ``manifest.json`` and checked against
``SHA256SUMS`` before extraction. No third-party package is needed; set ``HF_TOKEN`` to download
from a private Hugging Face repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import urllib.error
import urllib.parse
import urllib.request

GITHUB_REPO = "WZq975/MobileJudgeBench"
HF_DATASET_ID = "WesleyWzq/MobileJudgeBench"
DEFAULT_TAG = "v1.0.0"
USER_AGENT = "mobilejudgebench-download"


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class _AuthRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Follow redirects but do not forward the Authorization header to another host (e.g. a CDN)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlsplit(newurl).netloc != urllib.parse.urlsplit(req.full_url).netloc:
            new.remove_header("Authorization")
        return new


_opener = urllib.request.build_opener(_AuthRedirectHandler)


def _request(url: str, accept: str | None = None) -> urllib.request.Request:
    headers = {"User-Agent": USER_AGENT}
    if accept:
        headers["Accept"] = accept
    if os.environ.get("HF_TOKEN") and urllib.parse.urlsplit(url).netloc == "huggingface.co":
        headers["Authorization"] = "Bearer " + os.environ["HF_TOKEN"]
    return urllib.request.Request(url, headers=headers)


def _fetch(url: str, dest: str) -> None:
    tmp = dest + ".part"
    with _opener.open(_request(url)) as resp, open(tmp, "wb") as f:
        total = resp.headers.get("Content-Length")
        done = shown = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total and done - shown >= (64 << 20):
                shown = done
                print(f"\r  {os.path.basename(dest)}: {done / 2**20:.0f}/{int(total) / 2**20:.0f} MiB", end="", flush=True)
    print(f"\r  {os.path.basename(dest)}: {done / 2**20:.0f} MiB downloaded")
    os.replace(tmp, dest)


def github_asset_urls(tag: str) -> dict[str, str]:
    url = f"https://api.github.com/repos/{GITHUB_REPO}/releases/tags/{tag}"
    with _opener.open(_request(url, accept="application/vnd.github+json")) as resp:
        rel = json.load(resp)
    return {a["name"]: a["browser_download_url"] for a in rel["assets"]}


def hf_asset_url(name: str, revision: str) -> str:
    return f"https://huggingface.co/datasets/{HF_DATASET_ID}/resolve/{urllib.parse.quote(revision)}/{urllib.parse.quote(name)}"


def local_asset_urls(local_dir: str) -> dict[str, str]:
    return {f: "file://" + os.path.abspath(os.path.join(local_dir, f)) for f in os.listdir(local_dir)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", required=True, help="release root to create (archives extract into it)")
    ap.add_argument("--source", choices=["hf", "github", "local"], default="hf")
    ap.add_argument("--local_dir", default=None, help="directory holding the archives (with --source local)")
    ap.add_argument("--tag", default=DEFAULT_TAG, help="release tag (Hugging Face revision or GitHub release; the default tag maps to the main branch on Hugging Face)")
    ap.add_argument("--benchmark", action="append", default=None, help="filter (repeatable)")
    ap.add_argument("--agent", action="append", default=None, help="filter (repeatable)")
    ap.add_argument("--metadata_only", action="store_true")
    ap.add_argument("--keep_archives", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.dest, exist_ok=True)
    cache = os.path.join(args.dest, "_downloads")
    os.makedirs(cache, exist_ok=True)
    if args.source == "github":
        table = github_asset_urls(args.tag)
        url_for = table.__getitem__
    elif args.source == "hf":
        revision = "main" if args.tag == DEFAULT_TAG else args.tag
        url_for = lambda name: hf_asset_url(name, revision)  # noqa: E731
    else:
        if not args.local_dir:
            sys.exit("--source local requires --local_dir")
        table = local_asset_urls(args.local_dir)
        url_for = table.__getitem__
    for name in ("manifest.json", "SHA256SUMS"):
        try:
            _fetch(url_for(name), os.path.join(cache, name))
        except (KeyError, urllib.error.URLError) as e:
            sys.exit(f"cannot fetch {name} from source {args.source} ({args.tag}): {e}")
    manifest = json.load(open(os.path.join(cache, "manifest.json")))["archives"]
    sums = dict(line.split("  ", 1)[::-1] for line in open(os.path.join(cache, "SHA256SUMS")).read().splitlines() if line.strip())

    wanted = []
    for name, m in manifest.items():
        if "trajectory_ids" not in m:
            wanted.append(name)  # metadata archive
            continue
        if args.metadata_only:
            continue
        if args.benchmark and m["benchmark"] not in args.benchmark:
            continue
        if args.agent and m["agent"] not in args.agent:
            continue
        wanted.append(name)
    print(f"{len(wanted)} archives to download, {sum(manifest[n]['bytes'] for n in wanted) / 2**30:.2f} GiB")

    for name in wanted:
        path = os.path.join(cache, name)
        if not (os.path.isfile(path) and sha256(path) == sums[name]):
            _fetch(url_for(name), path)
            if sha256(path) != sums[name]:
                sys.exit(f"checksum mismatch for {name}")
        with tarfile.open(path, "r:*") as tf:
            tf.extractall(args.dest, filter="data") if hasattr(tarfile, "data_filter") else tf.extractall(args.dest)
        print(f"  extracted {name}")
        if not args.keep_archives:
            os.remove(path)
    if not args.keep_archives:
        shutil.rmtree(cache, ignore_errors=True)
    print("done ->", args.dest)


if __name__ == "__main__":
    main()
