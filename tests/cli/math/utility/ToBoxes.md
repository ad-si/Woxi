# `ToBoxes`

Converts an expression to its box form representation for typesetting.

```scrut
$ wo 'ToBoxes[x]'
x
```

A transpose is set through a postfix superscript template, its operand
bracketed when it is an operator expression:

```scrut
$ wo 'ToBoxes[Transpose[a + b]]'
TemplateBox[{RowBox[{(, RowBox[{a, +, b}], )}]}, Transpose, SyntaxForm -> SuperscriptBox]
```

A conjugate keeps its function form in StandardForm and takes the
template only in TraditionalForm:

```scrut
$ wo 'ToBoxes[Conjugate[a + b]]'
RowBox[{Conjugate, [, RowBox[{a, +, b}], ]}]
```
