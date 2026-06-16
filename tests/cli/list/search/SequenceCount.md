# `SequenceCount`

Counts non-overlapping occurrences of a sub-sequence inside a list.

```scrut
$ wo 'SequenceCount[{1, 2, 3, 1, 2, 3}, {1, 2}]'
2
```

The sub-sequence elements may be patterns; `{__Symbol}` counts maximal runs
of consecutive symbols.

```scrut
$ wo 'SequenceCount[{1, 2, a, b, 3, c, d, 4, e, f, g}, {__Symbol}]'
3
```
