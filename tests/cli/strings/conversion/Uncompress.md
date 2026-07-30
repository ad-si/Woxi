# `Uncompress`

Inverse of `Compress` — decodes a base-64–encoded compressed
expression back to its original form.

```scrut
$ wo 'Uncompress[Compress["hello"]]'
hello
```

The decompressed expression evaluates like any other:

```scrut
$ wo 'Uncompress[Compress[1 + 1]]'
2
```
