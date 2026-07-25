# `MapThread`

Applies a function to corresponding elements in several lists.

```scrut
$ wo 'MapThread[Plus, {{1, 2}, {3, 4}}]'
{4, 6}
```

An element that is not a list has nothing to thread over.

```scrut
$ wo 'MapThread[f, {g}]'

MapThread::mptd: Object g at position {2, 1} in MapThread[f, {g}] has only 0 of required 1 dimensions.
MapThread[f, {g}]
```
