# `StringInsert`

Inserts a string at a given 1-based position.

```scrut
$ wo 'StringInsert["abcd", "X", 2]'
aXbcd
```

Valid positions run from `1` to `n+1` (and `-1` to `-(n+1)`). An
out-of-range position leaves the call unevaluated:

```scrut
$ wo 'StringInsert["abc", "X", 5]'
StringInsert[abc, X, 5]
```
