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

An empty field has no value, which is reported as `Missing["NotAvailable"]`:

```scrut
$ wo 'ImportString["1,,3", "CSV"]'
{{1, Missing[NotAvailable], 3}}
```

`"Numeric" -> False` keeps every field as the string it was written as:

```scrut
$ wo 'ToString[ImportString["a,b\n1,2", "CSV", "Numeric" -> False], InputForm]'
{{"a", "b"}, {"1", "2"}}
```

An empty document has no rows to read, so the tabular formats fail:

```scrut {output_stream: combined}
$ wo 'ImportString["", "CSV"]'

Import::fmterr: Cannot import data as CSV format.
$Failed
```

`"List"` reads one field per line, typed the way a CSV field is:

```scrut
$ wo 'ToString[ImportString["1\nabc", "List"], InputForm]'
{1, "abc"}
```

A JSON number written without a decimal point stays exact:

```scrut
$ wo 'ToString[ImportString["{\"a\": 1e3}", "JSON"], InputForm]'
{"a" -> 1000}
```

```scrut
$ wo 'ToString[ImportString["{\"a\": 1e-3}", "JSON"], InputForm]'
{"a" -> 1/1000}
```

`"XML"` reads a document into symbolic XML — `XMLObject["Document"]` holding
the prolog, the root `XMLElement[tag, {attributes}, {children}]` and the
epilog. Whitespace between elements is layout rather than text, and comments
carry no content:

```scrut
$ wo 'ToString[ImportString["<a x=\"1\"><b>2</b></a>", "XML"], InputForm]'
XMLObject["Document"][{}, XMLElement["a", {"x" -> "1"}, {XMLElement["b", {}, {"2"}]}], {}]
```

```scrut {output_stream: combined}
$ wo 'ImportString["not xml", "XML"]'

Import::nfprserr: invalid document structure at line: 1 character: 1 in input string.
$Failed
```

A prefixed name carries its namespace URI, written `{uri, local}`, and the
`xmlns` declarations become attributes of their own namespace:

```scrut
$ wo 'ToString[ImportString["<ns:a xmlns:ns=\"u\"><ns:b>t</ns:b></ns:a>", "XML"], InputForm]'
XMLObject["Document"][{}, XMLElement[{"u", "a"}, {{"http://www.w3.org/2000/xmlns/", "ns"} -> "u"}, {XMLElement[{"u", "b"}, {}, {"t"}]}], {}]
```

A `{format, element}` spec reads one part of the document:

```scrut
$ wo 'ToString[ImportString["<a><b>t1</b><b>t2</b></a>", {"XML", "Tags"}], InputForm]'
{"a", "b"}
```

```scrut
$ wo 'ImportString["<a><b>t1</b><b>t2</b></a>", {"XML", "Plaintext"}]'
t1
t2
```

```scrut {output_stream: combined}
$ wo 'ImportString["<a>1</a>", {"XML", "Bogus"}]'

Import::noelem: The Import element "Bogus" is not present when importing as XML.
$Failed
```

The tabular formats take an element as well:

```scrut
$ wo 'ToString[ImportString["1,2\n3,4", {"CSV", "Data"}], InputForm]'
{{1, 2}, {3, 4}}
```
