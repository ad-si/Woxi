# `Total`

Sums the elements of a list.

```scrut
$ wo 'Total[{1, 2, 3}]'
6
```

A level-0 spec leaves the expression untouched:

```scrut
$ wo 'Total[{{1, 2}, {3, 4}}, {0}]'
{{1, 2}, {3, 4}}
```

An empty list totals to `0` at every positive level:

```scrut
$ wo 'Total[{}, {2}]'
0
```
