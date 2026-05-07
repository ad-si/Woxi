# /// script
# requires-python = ">=3.11"
# ///
"""Split mdbook function docs into per-function pages.

For each input markdown file, every `## `FuncName`` section is extracted
into its own page at `<dir>/<basename>/<FuncName>.md`. The original file
is rewritten as an index page: any preamble before the first function
section is preserved, and the function sections are replaced by a
bulleted list of links to the new per-function pages. Subheaders like
`# Section` that group function blocks become `## Section` headers in
the index so the grouping is preserved.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Matches any `## `FuncName`...` header — the first backticked identifier
# on a `## ` line is treated as the function name. Trailing content (e.g.
# description after a dash, operator-form parenthetical, `and OtherName`)
# is preserved in the per-function page heading.
FUNC_HEADER_RE = re.compile(
    r"^## `([A-Za-z][A-Za-z0-9]*)`",
)
# Used to extract every backticked identifier from a multi-name header line.
NAME_RE = re.compile(r"`([A-Za-z][A-Za-z0-9]*)`")
# Matches a top-level `# Section` header (not a function header).
SECTION_HEADER_RE = re.compile(r"^# (?!`)(.+?)\s*$", re.MULTILINE)


def split_file(src: Path, dry_run: bool = False) -> int:
    text = src.read_text()
    lines = text.splitlines(keepends=True)

    # Walk lines, grouping into "blocks":
    #   * preamble (everything before first `## `Func`` or `# Section`)
    #   * `# Section` markers
    #   * `## `Func`` blocks (until next `## `Func``, `# Section`, or EOF)
    blocks: list[dict] = []
    i = 0
    preamble: list[str] = []
    while i < len(lines):
        line = lines[i]
        m_func = FUNC_HEADER_RE.match(line.rstrip("\n"))
        m_section = SECTION_HEADER_RE.match(line.rstrip("\n"))
        if m_func:
            names = NAME_RE.findall(line)
            body_lines = [line]
            i += 1
            while i < len(lines):
                nxt = lines[i].rstrip("\n")
                if FUNC_HEADER_RE.match(nxt) or SECTION_HEADER_RE.match(nxt):
                    break
                body_lines.append(lines[i])
                i += 1
            body = "".join(body_lines)
            for name in names:
                # For multi-name headers, give each name its own page with
                # the same body but a single-name `## `Name`` heading.
                if len(names) > 1:
                    rewritten = f"## `{name}`\n" + "".join(body_lines[1:])
                    blocks.append({"kind": "func", "name": name, "body": rewritten})
                else:
                    blocks.append({"kind": "func", "name": name, "body": body})
        elif m_section and blocks:
            # Only treat `# Section` as a divider if we already saw a function
            # block — otherwise it is the page title and belongs in preamble.
            blocks.append({"kind": "section", "title": m_section.group(1)})
            i += 1
        else:
            if not blocks:
                preamble.append(line)
            else:
                # `# Section` before any function (e.g., page title) — keep
                # it as part of preamble. We only get here if SECTION matched
                # but blocks is empty.
                preamble.append(line)
            i += 1

    funcs = [b for b in blocks if b["kind"] == "func"]
    if not funcs:
        return 0

    base_dir = src.with_suffix("")  # e.g. tests/cli/math/bits/
    rel_dir = base_dir.name        # "bits"

    # Build per-function pages.
    func_files: list[tuple[str, str]] = []
    for b in funcs:
        name = b["name"]
        body = b["body"]
        # Promote the `## `Func`` header to a `# `Func`` h1 for the standalone
        # page; trim trailing whitespace.
        body = body.replace(f"## `{name}`", f"# `{name}`", 1)
        body = body.rstrip() + "\n"
        func_files.append((name, body))

    # Build index page.
    index_parts: list[str] = ["".join(preamble).rstrip() + "\n", ""]
    current_section: str | None = None
    section_links: dict[str | None, list[str]] = {None: []}
    section_order: list[str | None] = [None]
    for b in blocks:
        if b["kind"] == "section":
            current_section = b["title"]
            if current_section not in section_links:
                section_links[current_section] = []
                section_order.append(current_section)
        else:
            link = f"- [`{b['name']}`]({rel_dir}/{b['name']}.md)"
            section_links[current_section].append(link)

    # Deduplicate links per section, preserving first-seen order.
    rendered_sections: list[str] = []
    seen: set[str] = set()
    for sec in section_order:
        links = []
        for link in section_links[sec]:
            if link not in seen:
                seen.add(link)
                links.append(link)
        if not links:
            continue
        if sec is None:
            rendered_sections.append("\n".join(links))
        else:
            rendered_sections.append(f"## {sec}\n\n" + "\n".join(links))
    index_body = "\n\n".join(rendered_sections).rstrip() + "\n"
    index_content = "".join(preamble).rstrip() + "\n\n" + index_body

    if dry_run:
        print(f"[dry-run] {src}: {len(funcs)} functions")
        for name, _ in func_files:
            print(f"  -> {base_dir}/{name}.md")
        return len(funcs)

    base_dir.mkdir(parents=True, exist_ok=True)
    for name, body in func_files:
        (base_dir / f"{name}.md").write_text(body)
    src.write_text(index_content)
    return len(funcs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    total = 0
    for p in args.paths:
        if not p.exists():
            print(f"skip (missing): {p}", file=sys.stderr)
            continue
        n = split_file(p, dry_run=args.dry_run)
        print(f"{p}: split {n} functions")
        total += n
    print(f"total: {total} functions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
