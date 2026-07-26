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

Position 0 is the head, and a longer path descends into a compound one:

```scrut
$ wo '{MapAt[f, g[x], {0}], MapAt[f, g[x][y], {0}], MapAt[f, g[x][y], {0, 1}]}'
{f[g][x], f[g[x]][y], g[f[x]][y]}
```

Both sides of a rule are addressable:

```scrut
$ wo '{MapAt[f, a -> b, {1}], MapAt[f, a -> b, {2}]}'
{f[a] -> b, a -> f[b]}
```

The result comes back evaluated, so an operator-form head prints as the
operator:

```scrut
$ wo '{MapAt[f, a == b, {1}], MapAt[f, a + b, {1}]}'
{f[a] == b, b + f[a]}
```
