# `FoldPairList`

Like `FoldList`, but the combining function returns a pair
`{value_to_output, new_state}`.

```scrut
$ wo 'FoldPairList[{#1, #1 + #2} &, 0, {1, 2, 3, 4}]'
{0, 1, 3, 6}
```
