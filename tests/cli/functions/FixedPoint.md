# `FixedPoint`

Applies a function repeatedly until the result stops changing,
or until `n` iterations have occurred.

```scrut
$ wo 'FixedPoint[# + 1 &, 0, 3]'
3
```
