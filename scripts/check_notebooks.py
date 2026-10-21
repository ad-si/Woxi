#!/usr/bin/env python3
"""Run `woxi-studio --check` over a directory of notebooks and classify them.

Usage:
  uv run scripts/check_notebooks.py NOTEBOOK_DIR RESULTS.jsonl
      [--jobs N] [--timeout SECONDS] [--binary PATH] [--only FILE...]

Results are appended to RESULTS.jsonl, one object per notebook, so an
interrupted run resumes where it stopped. `--rerun STATUS` re-checks the
notebooks whose last recorded status is STATUS. `--summary` prints the
status counts and the most frequent warnings without running anything.
"""

import argparse
import collections
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

MAX_RSS_BYTES = 4 * 1024**3


def rss_bytes(pid):
    try:
        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True
        ).stdout.strip()
        return int(out) * 1024 if out else 0
    except ValueError:
        return 0


def check(binary, path, timeout):
    start = time.monotonic()
    proc = subprocess.Popen(
        [binary, "--check", path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        cwd="/tmp",
    )
    status = None
    deadline = start + timeout
    out_chunks, err_chunks = [], []
    readers = [
        threading.Thread(target=lambda: out_chunks.append(proc.stdout.read())),
        threading.Thread(target=lambda: err_chunks.append(proc.stderr.read())),
    ]
    for t in readers:
        t.start()
    while proc.poll() is None:
        if time.monotonic() > deadline:
            status = "timeout"
        elif rss_bytes(proc.pid) > MAX_RSS_BYTES:
            status = "memory"
        if status:
            os.killpg(proc.pid, signal.SIGKILL)
            break
        time.sleep(0.5)
    proc.wait()
    for t in readers:
        t.join()
    out, err = out_chunks[0], err_chunks[0]
    code = proc.returncode
    elapsed = time.monotonic() - start
    out = out.decode("utf-8", "replace")
    err = err.decode("utf-8", "replace")
    if status:
        pass
    elif code == 0:
        status = "ok"
    elif code == 1:
        status = "fail"
    elif code == 3:
        status = "parse"
    else:
        status = "crash"
    return {
        "file": os.path.basename(path),
        "status": status,
        "code": code,
        "seconds": round(elapsed, 2),
        "stdout": out[-4000:],
        "stderr": err[-4000:],
    }


def load_results(path):
    results = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    results[r["file"]] = r
    return results


def effective_status(r):
    """A clean render still fails when loading printed messages to stderr."""
    if r["status"] == "ok" and r["stderr"].strip():
        return "stderr"
    return r["status"]


def normalize(line):
    line = re.sub(r"\d+(\.\d+)?", "N", line)
    return line[:120]


def summary(results):
    results = {k: r for k, r in results.items() if not k.startswith("._")}
    counts = collections.Counter(effective_status(r) for r in results.values())
    print("status:", dict(counts.most_common()), "total:", len(results))
    unimpl = collections.Counter()
    msgs = collections.Counter()
    for r in results.values():
        seen_u, seen_m = set(), set()
        text = r["stdout"] + "\n" + r["stderr"]
        for line in text.splitlines():
            m = re.match(r"(?:warning: )?(.*) (?:is|are) built-in Wolfram", line)
            if m:
                for call in re.findall(r"(?:^|, )([A-Z$][A-Za-z0-9`$]*)\[", m.group(1)):
                    seen_u.add(call)
                continue
            m = re.match(r"(?:warning: )?([A-Za-z$`]+::[A-Za-z]+)", line)
            if m:
                seen_m.add(m.group(1))
                continue
            for kind in ("widget N: error", "widget N: empty", "parse error",
                         "panicked", "widgets: "):
                if kind in normalize(line):
                    seen_m.add(kind if kind != "widgets: " else None)
        if r["status"] in ("timeout", "crash", "memory"):
            seen_m.add(r["status"])
        unimpl.update(seen_u)
        msgs.update(x for x in seen_m if x)
    print("\nunimplemented functions (notebooks):")
    for k, v in unimpl.most_common(60):
        print(f"  {v:5d} {k}")
    print("\nmessages / failures (notebooks):")
    for k, v in msgs.most_common(60):
        print(f"  {v:5d} {k}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook_dir")
    ap.add_argument("results")
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--timeout", type=float, default=60)
    ap.add_argument("--binary", default="target/studio/woxi-studio")
    ap.add_argument("--rerun", action="append", default=[])
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()

    results = load_results(args.results)
    if args.summary:
        summary(results)
        return

    binary = os.path.abspath(args.binary)
    files = args.only or sorted(
        f
        for f in os.listdir(args.notebook_dir)
        # `._*` files are macOS AppleDouble metadata, not notebooks.
        if f.endswith(".nb") and not f.startswith("._")
    )
    todo = [
        f
        for f in files
        if args.only
        or f not in results
        or results[f]["status"] in args.rerun
    ]
    print(f"{len(todo)} notebooks to check", file=sys.stderr)
    done = 0
    with open(args.results, "a") as out, ThreadPoolExecutor(args.jobs) as pool:
        futures = [
            pool.submit(
                check, binary, os.path.join(args.notebook_dir, f), args.timeout
            )
            for f in todo
        ]
        for fut in as_completed(futures):
            r = fut.result()
            out.write(json.dumps(r) + "\n")
            out.flush()
            results[r["file"]] = r
            done += 1
            if done % 100 == 0:
                counts = collections.Counter(x["status"] for x in results.values())
                print(f"{done}/{len(todo)} {dict(counts)}", file=sys.stderr)
    summary(results)


if __name__ == "__main__":
    main()
