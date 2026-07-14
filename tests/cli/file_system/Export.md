# `Export`

`Import[path, format]` reads a file and parses it, while
`Export[path, expr, format]` writes an expression to a file.
Both route through the same format back-ends as `ImportString` and
`ExportString`, and accept the same options.

Common recognised formats include `"CSV"`, `"TSV"`, `"JSON"`, `"Text"`,
`"Lines"`, `"PNG"`, `"SVG"`, and `"String"`.

### Exporting an image to SVG

Exporting an `Image` to an `.svg` file wraps the raster in a base64-encoded
PNG `<image>` element, keeping the file a valid SVG.  The file opens with the
XML declaration, matching wolframscript.

```scrut
$ wo 'Export["green.svg", Image[ConstantArray[{0, 1, 0.5}, {2, 2}]]]; StringTake[ReadString["green.svg"], 5]'
<?xml
```

### Common options

- **`"CharacterEncoding"`** — text encoding (default `"UTF8"`).
- **`"HeaderLines"`** — number of leading lines to treat as header.
- **`"Delimiter"`** — separator character for CSV/TSV.
- **`"Numeric"`** — if `True`, numeric-looking fields are parsed as numbers.
- **`"IncludeWindowsLineBreaks"`** — for text-based writers.
