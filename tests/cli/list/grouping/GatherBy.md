# `GatherBy`

Groups elements by applying a function, maintaining order of first appearance.

```scrut
$ wo 'GatherBy[{1, 2, 3, 4, 5}, EvenQ]'
{{1, 3, 5}, {2, 4}}
```

```scrut
$ wo 'GatherBy[{-2, -1, 0, 1, 2}, Sign]'
{{-2, -1}, {0}, {1, 2}}
```
