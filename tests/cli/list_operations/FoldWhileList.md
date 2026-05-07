# `FoldWhileList`

Like `FoldList`, but stops once the predicate returns `False`.

```scrut
$ wo 'FoldWhileList[Plus, 0, {1, 2, 3, 4}, # < 5 &]'
{0, 1, 3, 6}
```
