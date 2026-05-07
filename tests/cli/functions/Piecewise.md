# `Piecewise`

Defines a piecewise-defined expression from a list of
`{value, condition}` pairs.

```scrut
$ wo 'Piecewise[{{1, x > 0}}] /. x -> 1'
1
```
