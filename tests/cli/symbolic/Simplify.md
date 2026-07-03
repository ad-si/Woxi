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

Integer-base logs in a sum merge into a single log.

```scrut
$ wo 'Simplify[Log[2] + Log[3]]'
Log[6]
```

```scrut
$ wo 'Simplify[20*Log[2] + 20*Log[3]]'
20*Log[6]
```
