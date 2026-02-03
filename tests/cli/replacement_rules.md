# Replacement rules tests

## `ReplaceAll` (`/.`)

Replaces parts of an expression according to rules.

```scrut
$ wo '{a, b} /. a -> x'
{x, b}
```


## `ReplaceRepeated` (`//.`)

Repeatedly applies transformation rules until no more changes occur.

```scrut
$ wo 'ReplaceRepeated[f[2] -> 2][f[f[f[f[2]]]]]'
2
```

```scrut
$ wo 'f[f[f[f[2]]]] //. f[2] -> 2'
2
```
