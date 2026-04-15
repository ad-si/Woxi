# File System & I/O

This page documents Woxi's file-system and I/O functions.
For many of these the example output contains non-deterministic paths,
so the expected lines use scrut's `(regex)` annotation.


## `Directory`

Returns the current working directory.

```scrut
$ wo 'Directory[]'
/\S+ (regex)
```

```scrut
$ wo 'StringQ[Directory[]]'
True
```


## `CreateFile`

Creates an empty file on disk, returning the resolved path.
With no argument, an auto-generated name in the temporary
directory is used.

```scrut
$ wo 'CreateFile[]'
/\S+ (regex)
```

```scrut
$ wo 'CreateFile["_delete_me_"]'
.+_delete_me_ (regex)
```

```scrut
$ wo 'a = CreateFile[]; a'
/\S+ (regex)
```


## `ImportString`

Parses a string as a supported format, such as CSV.

```scrut
$ wo 'ImportString["1,2,3\n4,5,6", "CSV"]'
{{1, 2, 3}, {4, 5, 6}}
```

### Options

- **`"CharacterEncoding"`** — encoding of the input string (default `"UTF8"`).
- **`"Numeric"`** — if `True`, numeric fields are converted to numbers.
- **`"HeaderLines"`** — number of header lines to skip (format-dependent).
- **`"Delimiter"`** — CSV/TSV field delimiter.


## `ExportString`

The inverse of `ImportString`: serialises an expression in a given format
and returns the result as a string.
Wolfram supports many named backends (`"CSV"`, `"JSON"`, `"PNG"`, etc.);
Woxi currently implements a subset.


## `FileNameJoin`

Joins path components using the platform path separator.

```scrut
$ wo 'FileNameJoin[{"a", "b", "c"}]'
a/b/c
```


## `FileNameSplit`

Splits a path into its components.

```scrut
$ wo 'FileNameSplit["a/b/c.txt"]'
{a, b, c.txt}
```


## `FileNameDrop`

Drops the last component of a path — the directory only.

```scrut
$ wo 'FileNameDrop["a/b/c.txt"]'
a/b
```


## `DirectoryName`

Returns the directory portion of a path, with a trailing slash.

```scrut
$ wo 'DirectoryName["/path/to/file.txt"]'
/path/to/
```


## `FileBaseName`

Returns the filename without its extension.

```scrut
$ wo 'FileBaseName["/path/to/file.txt"]'
file
```


## `FileExtension`

Returns the filename extension (without the leading dot).

```scrut
$ wo 'FileExtension["/path/to/file.txt"]'
txt
```


## `ExpandFileName`

Returns the absolute form of a path.

```scrut
$ wo 'StringTake[ExpandFileName["test.txt"], -8]'
test.txt
```


## `Import` / `Export`

`Import[path, format]` reads a file and parses it, while
`Export[path, expr, format]` writes an expression to a file.
Both route through the same format back-ends as `ImportString` and
`ExportString`, and accept the same options.

Common recognised formats include `"CSV"`, `"TSV"`, `"JSON"`, `"Text"`,
`"Lines"`, `"PNG"`, `"SVG"`, and `"String"`.

### Common options

- **`"CharacterEncoding"`** — text encoding (default `"UTF8"`).
- **`"HeaderLines"`** — number of leading lines to treat as header.
- **`"Delimiter"`** — separator character for CSV/TSV.
- **`"Numeric"`** — if `True`, numeric-looking fields are parsed as numbers.
- **`"IncludeWindowsLineBreaks"`** — for text-based writers.
