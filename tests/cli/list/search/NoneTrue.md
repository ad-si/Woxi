# `NoneTrue`

Checks if no element satisfies a predicate.

```scrut
$ wo 'NoneTrue[{1, 3, 5}, EvenQ]'
True
```

```scrut
$ wo 'NoneTrue[{1, 2, 3}, EvenQ]'
False
```

```scrut
$ wo 'NoneTrue[{1, 2, 3}, # > 5 &]'
True
```
