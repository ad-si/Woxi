# `SequenceSplit`

Splits a list into segments separated by the (non-overlapping, left-to-right)
subsequences that match a pattern. The separators are dropped, as are empty
segments — except that when the pattern matches nothing the whole list is
returned as a single segment.

```scrut
$ wo 'SequenceSplit[{1, 0, 2, 3, 0, 4}, {0}]'
{{1}, {2, 3}, {4}}
```

```scrut
$ wo 'SequenceSplit[{0, 1, 0}, {0}]'
{{1}}
```

```scrut
$ wo 'SequenceSplit[{1, 2, 3}, {5}]'
{{1, 2, 3}}
```

The separator can be a pattern:

```scrut
$ wo 'SequenceSplit[{1, 2, 3, 4}, {x_ /; EvenQ[x]}]'
{{1}, {3}}
```

```scrut
$ wo 'SequenceSplit[{1, 2, 3, 4, 5, 6}, {a_, b_} /; a + b == 7]'
{{1, 2}, {5, 6}}
```
