# `Split`

Splits list at boundaries where consecutive elements differ.

```scrut
$ wo 'Split[{1, 1, 2, 2, 3}]'
{{1, 1}, {2, 2}, {3}}
```

```scrut
$ wo 'Split[{a, a, b, c, c, c}]'
{{a, a}, {b}, {c, c, c}}
```
