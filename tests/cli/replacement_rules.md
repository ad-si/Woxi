# Replacement rules tests

## `ReplaceAll` (`/.`)

Replaces parts of an expression according to rules.

```todo
$ wo '{a, b} /. a -> x'
{x, b}
```


## `ReplaceRepeated` (`//.`)

Repeatedly applies transformation rules until no more changes occur.

```todo
$ wo 'ReplaceRepeated[f[2] -> 2][f[f[f[f[2]]]]]'
2
```

```todo
$ wo 'f[f[f[f[2]]]] //. f[2] -> 2'
2
```
