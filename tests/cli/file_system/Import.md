# `Import`

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
