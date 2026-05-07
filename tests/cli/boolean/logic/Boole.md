# `Boole`

Maps `True -> 1` and `False -> 0`, threading over lists.

```scrut
$ wo 'Boole[True]'
1
```

```scrut
$ wo 'Boole[False]'
0
```

```scrut
$ wo 'Boole[{True, False, True}]'
{1, 0, 1}
```
