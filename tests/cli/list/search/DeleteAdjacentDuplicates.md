# `DeleteAdjacentDuplicates`

Removes consecutive duplicate elements.

```scrut
$ wo 'DeleteAdjacentDuplicates[{1, 1, 2, 2, 3, 1, 1}]'
{1, 2, 3, 1}
```

A test decides what continues a run, and an association is grouped by its
values.

```scrut
$ wo 'DeleteAdjacentDuplicates[{1, 2, 4, 7}, Abs[#1 - #2] < 3 &]'
{1, 7}
```

```scrut
$ wo 'DeleteAdjacentDuplicates[Association[a -> 1, b -> 1, c -> 2, d -> 2, e -> 1]]'
<|a -> 1, c -> 2, e -> 1|>
```
