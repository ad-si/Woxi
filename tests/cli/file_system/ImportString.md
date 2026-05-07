# `ImportString`

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
