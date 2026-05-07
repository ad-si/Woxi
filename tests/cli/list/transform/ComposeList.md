# `ComposeList`

Returns the list `{x, f[x], g[f[x]], h[g[f[x]]], …}` for a list of functions.

```scrut
$ wo 'ComposeList[{f, g, h}, x]'
{x, f[x], g[f[x]], h[g[f[x]]]}
```
