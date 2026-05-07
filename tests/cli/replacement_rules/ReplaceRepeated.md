# `ReplaceRepeated` (`//.`)

Repeatedly applies transformation rules until no more changes occur.

```scrut
$ wo 'f[f[f[2]]] //. f[x_] -> x'
2
```

```scrut
$ wo 'f[f[f[f[2]]]] //. f[2] -> 2'
2
```

```scrut
$ wo 'f[f[f[2]]] //. f[x_] -> x + 1'
5
```
