# `TraditionalForm`

Displays expressions in traditional mathematical notation.

```scrut
$ wo 'TraditionalForm[1 + 2]'
TraditionalForm[3]
```

The 2D form it asks for shows up in the boxes it makes — a factorial, for
one, is written postfix:

```scrut
$ wo 'ToBoxes[TraditionalForm[n!]]'
TagBox[FormBox[RowBox[{n, !}], TraditionalForm], TraditionalForm, Editable -> True]
```

A special function carries its order as a subscript and any further index
as a superscript — `LegendreP[n, x]` is written `Pₙ(x)` and
`LegendreP[n, m, x]` is written `Pₙᵐ(x)`. Woxi writes that layout out as
`SubscriptBox`/`SubsuperscriptBox`; wolframscript instead defers it to a
named front-end template (`TemplateBox[{n, x}, "LegendreP"]`), so only the
`TagBox`/`FormBox` wrapper around it is common to both:

```scrut
$ wo 'Head[ToBoxes[TraditionalForm[LegendreP[n, x]]]]'
TagBox
```

`Row` is a display wrapper, so it just joins its parts — again inline in
Woxi and as a `RowDefault` template in wolframscript:

```scrut
$ wo 'Head[ToBoxes[TraditionalForm[Row[{2, x, t}]]]]'
TagBox
```

Written out as text, though, a `Row` does become that template — the
box-syntax escape `InputForm` produces names it rather than spelling the
layout out:

```scrut
$ wo 'ToString[InputForm[TraditionalForm[Row[{"a", 1}]]]]'
DisplayForm[FormBox[TemplateBox[{"a", 1}, RowDefault], TraditionalForm]]
```

A separator rides along inside the template, which loses the plural of its
name when the separator is not a string:

```scrut
$ wo 'ToString[InputForm[TraditionalForm[Row[{"a", "b"}, ", "]]]]'
DisplayForm[FormBox[TemplateBox[{, , ", ", "a", "b"}, RowWithSeparators], TraditionalForm]]
```

```scrut
$ wo 'ToString[InputForm[TraditionalForm[Row[{1, 2}, x]]]]'
DisplayForm[FormBox[TemplateBox[{x, 1, 2}, RowWithSeparator], TraditionalForm]]
```

Naming the template is what lets the text read back as the `Row` it was
typeset from, rather than as the run of parts it draws:

```scrut
$ wo 'Head[ToExpression[ToString[InputForm[TraditionalForm[Row[{"a", 1}]]]]]]'
Row
```

```scrut
$ wo 'ToExpression[ToString[InputForm[TraditionalForm[Row[{"a", 1}]]]]][[1]]'
{a, 1}
```
