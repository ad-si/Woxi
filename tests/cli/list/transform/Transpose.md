# `Transpose`

Transposes a matrix (list of lists).

```scrut
$ wo 'Transpose[{{1, 2}, {3, 4}}]'
{{1, 3}, {2, 4}}
```

```scrut
$ wo 'Transpose[{{a, b, c}, {d, e, f}}]'
{{a, d}, {b, e}, {c, f}}
```

```scrut
$ wo 'Transpose[{{1, 2, 3}}]'
{{1}, {2}, {3}}
```

A permutation specifies the destination level for each level. A partial
permutation leaves the remaining levels in place, so `{1}` is the identity.

```scrut
$ wo 'Transpose[{{1, 2}, {3, 4}}, {1}]'
{{1, 2}, {3, 4}}
```

```scrut
$ wo 'Transpose[{{1, 2, 3}, {4, 5, 6}}, {2, 1}]'
{{1, 4}, {2, 5}, {3, 6}}
```

Two levels sent to the same destination collapse onto a diagonal.

```scrut
$ wo 'Transpose[{{1, 2}, {3, 4}}, {1, 1}]'
{1, 4}
```
