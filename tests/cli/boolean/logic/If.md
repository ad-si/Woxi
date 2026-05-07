# `If`

Conditional operation.

```scrut
$ wo 'If[True, 1]'
1
```

```scrut
$ wo 'If[False, 1]'
Null
```

```scrut
$ wo 'If[True, 1, 0]'
1
```

```scrut
$ wo 'If[False, 1, 0]'
0
```

```scrut
$ wo 'If["x", 1, 0, 2]'
2
```

```scrut
$ wo 'If[True, 1, 0, 2, 3]'

If::argb: If called with 5 arguments; between 2 and 4 arguments are expected.
.* (regex*)
If[True, 1, 0, 2, 3]
```
