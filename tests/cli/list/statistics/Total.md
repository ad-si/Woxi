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

A trailing option is not a level spec, so it leaves the total alone:

```scrut
$ wo 'Total[{1, 2, 3}, Method -> Automatic]'
6
```

By default only `List` is descended through, so another head is left as it is:

```scrut
$ wo 'Total[f[1, 2, 3]]'
Total[f[1, 2, 3]]
```

`AllowedHeads -> All` descends through any head:

```scrut
$ wo 'Total[f[1, 2, 3], AllowedHeads -> All]'
6
```

```scrut
$ wo 'Total[{1, f[2, 3]}, 2, AllowedHeads -> All]'
6
```

Only as deep as the level asks, and an exact level keeps the head above it:

```scrut
$ wo 'Total[f[1, g[2, 3]], AllowedHeads -> All]'
1 + g[2, 3]
```

```scrut
$ wo 'Total[f[1, g[2, 3]], {2}, AllowedHeads -> All]'
f[1, 5]
```

Rationals and complex numbers stay atoms however permissive the heads are:

```scrut
$ wo 'Total[{1/2, 1/3}, AllowedHeads -> All]'
5/6
```
