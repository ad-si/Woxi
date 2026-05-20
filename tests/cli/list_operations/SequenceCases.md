# `SequenceCases`

Find matching subsequences in a list.

```scrut
$ wo 'SequenceCases[x, y]'

SequenceCases::list: List expected at position 1 in SequenceCases[x, y].
SequenceCases[x, y]
```

The `name : pattern :> body` binding form lets the right-hand side reference
the matched sub-list:

```scrut
$ wo 'SequenceCases[{1/2, 1/3, 1/16}, l : {_, 1 ...} :> Length[l]]'
{1, 1, 1}
```
