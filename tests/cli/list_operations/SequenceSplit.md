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

A rule keeps its right-hand side where the separator was:

```scrut
$ wo 'SequenceSplit[{x, x, a, b, y, a, c, z}, {a, e_} :> {e}]'
{{x, x}, {b}, {y}, {c}, {z}}
```

Each rule in a list gets its own replacement:

```scrut
$ wo 'SequenceSplit[{1, 2, 1, 2, 3}, {{1, 2} -> {a}, {3} -> {b}}]'
{{a}, {a}, {b}}
```

A third argument caps the number of sublists, the last of which holds the
unsplit remainder:

```scrut
$ wo 'SequenceSplit[{x, x, a, b, y, a, c, z}, {a, _}, 2]'
{{x, x}, {y, a, c, z}}
```
