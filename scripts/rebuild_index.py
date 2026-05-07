# /// script
# requires-python = ">=3.11"
# ///
"""Rebuild a category index from existing per-function pages.

Reads the directory `<parent>.with_suffix("")/` and lists every .md file
in it as a bullet link. Preserves any preamble (text before the first
bullet/heading-2) in the existing index file.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def rebuild(index_path: Path) -> int:
    base_dir = index_path.with_suffix("")
    if not base_dir.is_dir():
        print(f"skip (no dir): {index_path}", file=sys.stderr)
        return 0

    text = index_path.read_text()
    lines = text.splitlines(keepends=True)

    # Preamble = lines until first bullet (`- [`...) or `## ` heading
    preamble: list[str] = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("- [") or stripped.startswith("## "):
            break
        preamble.append(line)
    preamble_text = "".join(preamble).rstrip()

    # Collect function pages alphabetically.
    funcs = sorted(p.stem for p in base_dir.glob("*.md"))
    bullets = "\n".join(
        f"- [`{name}`]({base_dir.name}/{name}.md)" for name in funcs
    )

    new_content = preamble_text + "\n\n" + bullets + "\n"
    index_path.write_text(new_content)
    print(f"{index_path}: {len(funcs)} entries")
    return len(funcs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", type=Path)
    args = ap.parse_args()
    total = 0
    for p in args.paths:
        total += rebuild(p)
    print(f"total: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
