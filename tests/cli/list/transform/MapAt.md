# `MapAt`

Applies a function to the element at a given position.

```scrut
$ wo 'MapAt[f, {a, b, c}, 2]'
{a, f[b], c}
```

Any level of the position may be `All` or a span, selecting a whole row or
column:

```scrut
$ wo 'MapAt[f, {{a, b, c}, {d, e, f}}, {All, 2}]'
{{a, f[b], c}, {d, f[e], f}}
```

```scrut
$ wo 'MapAt[F, {a, b, c, d, e}, {Span[1, -1, 2]}]'
{F[a], b, F[c], d, F[e]}
```
