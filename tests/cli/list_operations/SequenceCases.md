# `SequenceCases`

Find matching subsequences in a list.

```scrut
$ wo 'SequenceCases[x, y]'

SequenceCases::list: List expected at position 1 in SequenceCases[x, y].
SequenceCases[x, y]
```

The `Overlaps` option controls how much of the list a match may share with its
neighbours. The default keeps matches disjoint, `Overlaps -> True` reports the
longest match at every start position, and `Overlaps -> All` reports *every*
match at every start position:

```scrut
$ wo 'SequenceCases[{1, 2, 3}, {__}]'
{{1, 2, 3}}
```

```scrut
$ wo 'SequenceCases[{1, 2, 3}, {__}, Overlaps -> True]'
{{1, 2, 3}, {2, 3}, {3}}
```

```scrut
$ wo 'SequenceCases[{1, 2, 3}, {__}, Overlaps -> All]'
{{1, 2, 3}, {1, 2}, {1}, {2, 3}, {2}, {3}}
```

The `name : pattern :> body` binding form lets the right-hand side reference
the matched sub-list:

```scrut
$ wo 'SequenceCases[{1/2, 1/3, 1/16}, l : {_, 1 ...} :> Length[l]]'
{1, 1, 1}
```
