# `TraditionalForm`

Displays expressions in traditional mathematical notation.

```scrut
$ wo 'TraditionalForm[1 + 2]'
TraditionalForm[3]
```

The 2D form it asks for shows up in the boxes it makes: a factorial is
written postfix, and a special function carries its order as a subscript
and any further index as a superscript.

```scrut
$ wo 'ToBoxes[TraditionalForm[n!]]'
TagBox[FormBox[RowBox[{n, !}], TraditionalForm], TraditionalForm, Editable -> True]
```

```scrut
$ wo 'ToBoxes[TraditionalForm[LegendreP[n, x]]]'
TagBox[FormBox[RowBox[{SubscriptBox[P, n], (, x, )}], TraditionalForm], TraditionalForm, Editable -> True]
```

```scrut
$ wo 'ToBoxes[TraditionalForm[LegendreP[n, m, x]]]'
TagBox[FormBox[RowBox[{SubsuperscriptBox[P, n, m], (, x, )}], TraditionalForm], TraditionalForm, Editable -> True]
```

`Row` is a display wrapper, so it just joins its parts:

```scrut
$ wo 'ToBoxes[TraditionalForm[Row[{2, x, t}]]]'
TagBox[FormBox[RowBox[{2, x, t}], TraditionalForm], TraditionalForm, Editable -> True]
```
