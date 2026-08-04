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
