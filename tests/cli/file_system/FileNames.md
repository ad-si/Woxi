# `FileNames`

Returns a list of file names matching a pattern in the current directory.

```scrut
$ wo 'ListQ[FileNames["*"]]'
True
```

The pattern is a string pattern, in which `*` stands for any sequence
of characters:

```scrut
$ mkdir -p "$TMPDIR/wildcard" && cd "$TMPDIR/wildcard" &&
> touch report.txt notes.txt readme.md && wo 'FileNames["*.txt"]'
{notes.txt, report.txt}
```

A list of patterns matches file names matching any of them:

```scrut
$ mkdir -p "$TMPDIR/list" && cd "$TMPDIR/list" &&
> touch report.txt notes.txt readme.md &&
> wo 'FileNames[{"*.md", "notes.txt"}]'
{notes.txt, readme.md}
```

Patterns joined with `|` mean the same thing:

```scrut
$ mkdir -p "$TMPDIR/alternatives" && cd "$TMPDIR/alternatives" &&
> touch report.txt notes.txt &&
> wo 'FileNames["report.txt" | "missing.txt"]'
{report.txt}
```
