# `Thread`

Threads a function over corresponding list elements.

```scrut
$ wo 'Thread[f[{a, b}, {x, y}]]'
{f[a, x], f[b, y]}
```

```scrut
$ wo 'Thread[Plus[{1, 2}, {3, 4}]]'
{4, 6}
```

```scrut
$ wo 'Thread[g[{a, b, c}, {1, 2, 3}]]'
{g[a, 1], g[b, 2], g[c, 3]}
```
