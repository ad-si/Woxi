# `FoldWhile`

Folds a function over a list while a predicate holds, returning the first
accumulator value for which the predicate fails (or the final value if it
never fails).

```scrut
$ wo 'FoldWhile[Times, 2, {2, 3, 4, 5}, # < 100 &]'
240
```

The initial value is tested first, so a value that already fails is returned
without any folding:

```scrut
$ wo 'FoldWhile[Times, 2, {2, 3, 4, 5}, # < 1 &]'
2
```

The three-argument form takes the initial value from the head of the list:

```scrut
$ wo 'FoldWhile[Plus, {1, 2, 3, 4, 5}, # < 6 &]'
6
```

An `m` argument supplies the last `m` results to the test:

```scrut
$ wo 'FoldWhile[Times, 1, {2, 2, 3, 3, 4}, Unequal, 2]'
144
```

A trailing `n` continues folding `n` extra times past the failing value
(or steps back with a negative value):

```scrut
$ wo 'FoldWhile[Plus, 0, {1, 2, 3, 4, 5}, # < 6 &, 1, 1]'
10
```
