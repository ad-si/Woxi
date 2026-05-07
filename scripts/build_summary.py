# /// script
# requires-python = ">=3.11"
# ///
"""Sync SUMMARY.md so every per-function page is listed under its category.

For any entry whose target file has a sibling directory of per-function
pages (e.g. `math/bits.md` ↔ `math/bits/`), the function pages are
inserted as nested children below the category entry. Files already
referenced are skipped so each page appears exactly once. Idempotent —
running it again after function pages are added/removed updates the tree.
"""

from __future__ import annotations

import re
from pathlib import Path

CLI_DIR = Path("tests/cli")
SUMMARY = CLI_DIR / "SUMMARY.md"

ENTRY_RE = re.compile(r"^(\s*)- \[([^\]]+)\]\(([^)]+)\)\s*$")


def main() -> int:
    text = SUMMARY.read_text()
    # Targets already mentioned at any level in SUMMARY.md — never duplicate.
    referenced: set[str] = set()
    for line in text.splitlines():
        m = ENTRY_RE.match(line)
        if m:
            referenced.add(m.group(3))

    out: list[str] = []
    for line in text.splitlines():
        out.append(line)
        m = ENTRY_RE.match(line)
        if not m:
            continue
        indent, _label, target = m.group(1), m.group(2), m.group(3)
        if not target.endswith(".md"):
            continue
        target_path = CLI_DIR / target
        if not target_path.is_file():
            continue
        sibling_dir = target_path.with_suffix("")
        if not sibling_dir.is_dir():
            continue
        children = sorted(sibling_dir.glob("*.md"))
        child_indent = indent + "  "
        for child in children:
            rel = child.relative_to(CLI_DIR).as_posix()
            if rel in referenced:
                continue
            # Skip children that are themselves topical parents
            # (i.e. have their own sibling subdirectory).
            if child.with_suffix("").is_dir():
                continue
            label = f"`{child.stem}`"
            out.append(f"{child_indent}- [{label}]({rel})")
    new_text = "\n".join(out) + "\n"
    SUMMARY.write_text(new_text)
    print(f"wrote {SUMMARY} ({len(out)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
