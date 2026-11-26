# `ToBoxes`

Converts an expression to its box form representation for typesetting.

```scrut
$ wo 'ToBoxes[x]'
x
```

A conjugate is set as a postfix superscript mark, its operand bracketed
when it is an operator expression.

```scrut
$ wo 'ToBoxes[Conjugate[a + b]]'
SuperscriptBox[RowBox[{(, RowBox[{a, +, b}], )}], *]
```
