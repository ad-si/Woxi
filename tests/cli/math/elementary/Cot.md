# `Cot`

Returns the cotangent.

```scrut
$ wo 'Cot[0]'
ComplexInfinity
```

`Cot` collapses its own inverse: `Cot[ArcCot[x]] == x`.

```scrut
$ wo 'Cot[ArcCot[x]]'
x
```
