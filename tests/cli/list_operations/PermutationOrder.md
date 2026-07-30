# `PermutationOrder`

Order of a permutation.

```scrut
$ wo 'PermutationOrder[{1, 2, 3}]'
1
```

A permutation list has to be a rearrangement of `1` through `n`, so a repeated
entry names no permutation:

```scrut
$ wo 'PermutationOrder[{1, 1}]'

PermutationOrder::permlist: Invalid permutation list {1, 1}.
PermutationOrder[{1, 1}]
```

```scrut
$ wo 'PermutationOrder[{2, 3, 1}]'
3
```
