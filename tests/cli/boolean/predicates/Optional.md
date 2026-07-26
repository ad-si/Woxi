# `Optional`

Represents a pattern with an optional default value.

`Optional[p, default]` is the explicit form of `p : default`:

```scrut
$ wo 'f[a] /. f[x_, Optional[y_, 2]] -> {x, y}'
{a, 2}
```

The arguments that are present fill the slots from the left, so the optionals
falling back on their defaults are the rightmost ones:

```scrut
$ wo 'g[1] /. g[x_ : 0, y_ : 0] -> {x, y}'
{1, 0}
```
