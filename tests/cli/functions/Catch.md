# `Catch`

Non-local exit — `Throw[val]` unwinds until the innermost `Catch`.

```scrut
$ wo 'Catch[Throw[42]]'
42
```

```scrut
$ wo 'Catch[Do[If[i == 3, Throw[i]], {i, 1, 5}]]'
3
```
