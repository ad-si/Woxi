# `NestList`

Like [`Nest`](Nest.md), but returns a list of intermediate results.

```scrut
$ wo 'NestList[f, x, 3]'
{x, f[x], f[f[x]], f[f[f[x]]]}
```
