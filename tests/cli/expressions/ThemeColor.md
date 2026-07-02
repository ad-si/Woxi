# `ThemeColor`

Represents a named color of the current notebook theme, such as
`"Foreground"`, `"Background"`, `"Accent1"` … `"Accent9"`, or the syntax
highlighting colors `"Syntax1"` … `"Syntax8"` and
`"SyntaxError1"` … `"SyntaxError6"`.

The kernel does not resolve theme colors — they stay symbolic and are
resolved by the front end when rendering (picking the light or dark variant
depending on the current appearance):

```scrut
$ wo 'ThemeColor["Accent1"]'
ThemeColor[Accent1]
```

```scrut
$ wo 'Head[ThemeColor["Foreground"]]'
ThemeColor
```

Since the kernel keeps it symbolic, it is not a color object:

```scrut
$ wo 'ColorQ[ThemeColor["Accent1"]]'
False
```
