# `FoldPairList`

Like `FoldList`, but the combining function returns a pair
`{value_to_output, new_state}`.

```scrut
$ wo 'FoldPairList[{#1, #1 + #2} &, 0, {1, 2, 3, 4}]'
{0, 1, 3, 6}
```

A fourth argument is applied to the whole `{value, new_state}` pair before
it is emitted.

```scrut
$ wo 'FoldPairList[QuotientRemainder, 498, {100, 50, 20, 5, 1}, Identity]'
{{4, 98}, {1, 48}, {2, 8}, {1, 3}, {3, 0}}
```
