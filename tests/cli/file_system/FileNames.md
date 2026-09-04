# `FileNames`

Returns a list of file names matching a pattern in the current directory.

```scrut
$ wo 'ListQ[FileNames["*"]]'
True
```

The first argument is a string pattern, in which `*` stands for any
sequence of characters.
Set up a `notes` directory to search:

```scrut
$ wo 'CreateDirectory["notes"]; CreateFile["notes/report.txt"]; CreateFile["notes/notes.txt"]; CreateFile["notes/readme.md"]; FileNames["*.txt", "notes"]'
{notes/notes.txt, notes/report.txt}
```

A list of patterns matches the file names matching any of them:

```scrut
$ wo 'FileNames[{"*.md", "notes.txt"}, "notes"]'
{notes/notes.txt, notes/readme.md}
```

Patterns joined with `|` mean the same thing:

```scrut
$ wo 'FileNames["report.txt" | "missing.txt", "notes"]'
{notes/report.txt}
```

A third argument sets how many directory levels to include.
Set up a `docs` directory with a `report.txt` on three levels:

```scrut
$ wo 'CreateDirectory["docs/inner/deeper"]; CreateFile["docs/report.txt"]; CreateFile["docs/inner/report.txt"]; CreateFile["docs/inner/deeper/report.txt"]; FileNames["*", "docs"]'
{docs/inner, docs/report.txt}
```

The default, `1`, only looks at `docs` itself,
so the copies further down are not reported:

```scrut
$ wo 'FileNames["report.txt", "docs"]'
{docs/report.txt}
```

With `2`, the immediate subdirectories are searched as well:

```scrut
$ wo 'FileNames["report.txt", "docs", 2]'
{docs/inner/report.txt, docs/report.txt}
```

`Infinity` descends without a limit:

```scrut
$ wo 'FileNames["report.txt", "docs", Infinity]'
{docs/inner/deeper/report.txt, docs/inner/report.txt, docs/report.txt}
```

A level in braces restricts the search to exactly that level:

```scrut
$ wo 'FileNames["report.txt", "docs", {2}]'
{docs/inner/report.txt}
```

Braces hold a single level only — there is no range form,
so a two-element list is not a level specification and stays unevaluated:

```scrut
$ wo 'FileNames["report.txt", "docs", {2, 3}]'
FileNames[report.txt, docs, {2, 3}]
```

The current directory is the one `SetDirectory` last set,
not the one the script was started from.

```scrut
$ wo 'dir = CreateDirectory[]; Export[FileNameJoin[{dir, "hello.txt"}], "x"]; SetDirectory[dir]; FileNames[]'
{hello.txt}
```
