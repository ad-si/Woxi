# `CountDistinct`

Counts the number of distinct elements in a list.

```scrut
$ wo 'CountDistinct[{1, 2, 2, 3, 3, 3}]'
3
```

A second argument supplies a sameness test.

```scrut
$ wo 'CountDistinct[{1, 2, 4}, Abs[#1 - #2] < 2 &]'
2
```
