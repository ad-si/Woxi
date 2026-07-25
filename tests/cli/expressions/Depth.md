# `Depth`

Returns the maximum depth (+1) of the expression tree.

```scrut
$ wo 'Depth[{{1, 2}, {3, 4}}]'
3
```

An expression with no parts is still one level deeper than an atom.

```scrut
$ wo 'Depth[{}]'
2
```
