# `ArrayDepth`

Returns the depth of a nested list.

```scrut
$ wo 'ArrayDepth[{1, 2, 3}]'
1
```

```scrut
$ wo 'ArrayDepth[{{1, 2}, {3, 4}}]'
2
```

```scrut
$ wo 'ArrayDepth[{{{1}}}]'
3
```

```scrut
$ wo 'ArrayDepth[5]'
0
```
