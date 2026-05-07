# `TrueQ`

Returns `True` only when the argument is literally `True`,
`False` for everything else (including non-Boolean expressions).

```scrut
$ wo 'TrueQ[True]'
True
```

```scrut
$ wo 'TrueQ[1]'
False
```
