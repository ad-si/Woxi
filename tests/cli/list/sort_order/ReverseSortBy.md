# `ReverseSortBy`

Sorts a list in descending order by a key function.

```scrut
$ wo 'ReverseSortBy[{1, -3, 2, -4}, Abs]'
{-4, -3, 2, 1}
```

It is `Reverse[SortBy[…]]`, so equal keys are reversed too, and a list of
functions gives a multi-criteria sort.

```scrut
$ wo 'ReverseSortBy[{1, 2, 3, 4}, {Mod[#, 2] &, # &}]'
{3, 1, 4, 2}
```
