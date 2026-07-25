# `FoldWhileList`

Like `FoldList`, but stops once the predicate returns `False`.

```scrut
$ wo 'FoldWhileList[Plus, 0, {1, 2, 3, 4}, # < 5 &]'
{0, 1, 3, 6}
```

A fifth argument gives the test the last `m` results (or `All` of them),
and a sixth shifts where the fold stops — negative to stop earlier:

```scrut
$ wo 'FoldWhileList[#1 + #2 &, 0, {1, 2, 3, 4}, # < 4 &, 1, -1]'
{0, 1, 3}
```

The operator form folds a list, taking its first element as the start:

```scrut
$ wo 'FoldWhileList[#1 + #2 &, # < 4 &][{1, 2, 3, 4}]'
{1, 3, 6}
```
