# `ToString`

Converts an expression to a string.

```scrut
$ wo 'ToString[123]'
123
```

```scrut
$ wo 'ToString[{1, 2, 3}]'
{1, 2, 3}
```

```scrut
$ wo 'ToString[1 + 2]'
3
```

`NumberForm` renders a number to a given number of significant figures.

```scrut
$ wo 'ToString[NumberForm[3.14159, 3]]'
3.14
```

With a `{n, f}` specification it shows exactly `f` digits after the decimal
point (zero-padded).

```scrut
$ wo 'ToString[NumberForm[3.0, {5, 2}]]'
3.00
```

A second argument names the form to render in. `TraditionalForm` typesets
into boxes and hands them back in the box-syntax escape notation, which
displays as the typeset expression: a known function takes its roman name
and round brackets.

```scrut
$ wo 'ToString[Sin[x], TraditionalForm]'
DisplayForm[FormBox[RowBox[{sin, (, x, )}], TraditionalForm]]
```

`HoldForm` keeps a symbol from evaluating while still displaying it, so a
label can name a function and apply it without computing anything. The boxes
keep a `TagBox` recording that it was held; the tag draws nothing.

```scrut
$ wo 'ToString[HoldForm[g][HoldForm[x]], TraditionalForm]'
DisplayForm[FormBox[RowBox[{TagBox[g, HoldForm], (, TagBox[x, HoldForm], )}], TraditionalForm]]
```
