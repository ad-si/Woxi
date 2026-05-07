# `Do`

`Do[expr, {iter}]` iterates `expr` for each value of the iterator.
The result is always `Null`.

```scrut
$ wo 'Do[Print[k], {k, 1, 3}]'
1
2
3
Null
```
