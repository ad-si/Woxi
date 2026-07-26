# `OrderlessPatternSequence`

`OrderlessPatternSequence[p1, …, pk]` matches the block of `k` arguments at
its position in any order:

```scrut
$ wo '{MatchQ[{1, 2}, {OrderlessPatternSequence[2, 1]}], MatchQ[{1, 2, 3}, {_, OrderlessPatternSequence[3, 2]}]}'
{True, True}
```

The arguments it takes stay contiguous, so a match that would have to skip
one fails:

```scrut
$ wo 'MatchQ[{1, 2, 3}, {OrderlessPatternSequence[3, 1], _}]'
False
```

Names bind in the order the patterns are written:

```scrut
$ wo '{3, 1, 2} /. {OrderlessPatternSequence[1, x_], ___} :> x'
3
```
