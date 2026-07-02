# `SystemColor`

Represents a named color provided by the windowing system theme, such as
`"Window"`, `"WindowText"`, or `"Highlight"`.

The kernel does not resolve system colors — they stay symbolic and are
resolved by the front end when rendering:

```scrut
$ wo 'SystemColor["Window"]'
SystemColor[Window]
```

```scrut
$ wo 'Head[SystemColor["Highlight"]]'
SystemColor
```

Since the kernel keeps it symbolic, it is not a color object:

```scrut
$ wo 'ColorQ[SystemColor["Window"]]'
False
```
