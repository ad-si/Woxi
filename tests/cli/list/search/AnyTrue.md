# `AnyTrue`

Checks if any element satisfies a predicate.

```scrut
$ wo 'AnyTrue[{1, 2, 3, 4}, EvenQ]'
True
```

```scrut
$ wo 'AnyTrue[{1, 3, 5}, EvenQ]'
False
```

```scrut
$ wo 'AnyTrue[{1, 2, 3}, # > 2 &]'
True
```
