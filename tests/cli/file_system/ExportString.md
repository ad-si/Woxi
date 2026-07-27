# `ExportString`

The inverse of `ImportString`: serialises an expression in a given format
and returns the result as a string.
Wolfram supports many named backends (`"CSV"`, `"JSON"`, `"PNG"`, etc.);
Woxi currently implements a subset.

Only machine numbers are written bare; every other value is quoted, and a
compound one that CSV cannot represent becomes a `-Head-` placeholder:

```scrut
$ wo 'StringTrim[ExportString[{{a, 1, True, x + y}}, "CSV"]]'
"a",1,true,"-Plus-"
```

`"String"` gives the expression's own text:

```scrut
$ wo 'ExportString[{{1, 2}}, "String"]'
{{1, 2}}
```

A value JSON cannot represent fails the export rather than being approximated:

```scrut {output_stream: combined}
$ wo 'ExportString[<|"a" -> Pi|>, "JSON"]'

Export::jsonstrictencoding: Expression Pi cannot be exported as JSON.
$Failed
```

Symbolic XML is written back out as markup:

```scrut
$ wo 'ExportString[XMLElement["a", {"x" -> "1"}, {"t"}], "XML"]'
<a x="1">t</a>
```
