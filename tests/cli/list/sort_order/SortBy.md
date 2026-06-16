# `SortBy`

Sorts elements of a list according to a function.

```scrut
$ wo 'SortBy[{3, 1, 2}, # &]'
{1, 2, 3}
```

A list of functions sorts by each criterion in turn.

```scrut
$ wo 'SortBy[{1, 2, 3, 4}, {Mod[#, 2] &, # &}]'
{2, 4, 1, 3}
```
