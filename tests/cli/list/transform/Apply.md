# `Apply`

Applies a function to list elements as arguments.

```scrut
$ wo 'Apply[Plus, {1, 2, 3}]'
6
```

```scrut
$ wo 'Apply[Times, {2, 3, 4}]'
24
```

A compound head is a head like any other, so it is what gets replaced:

```scrut
$ wo '{Apply[f, g[x][y]], Apply[f, g[x][y][z]]}'
{f[y], f[z]}
```

Rules and comparison chains have heads too, which a level specification
reaches:

```scrut
$ wo '{Apply[f, a -> b, {0}], Apply[f, a == b, {0}]}'
{f[a, b], f[a, b]}
```
