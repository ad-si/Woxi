# `Simplify`

Applies basic simplification rules.

```scrut
$ wo 'Simplify[(x^2 - 1)/(x - 1)]'
1 + x
```

A standalone integer log multiple folds when that is simpler.

```scrut
$ wo 'Simplify[2*Log[2]]'
Log[4]
```
