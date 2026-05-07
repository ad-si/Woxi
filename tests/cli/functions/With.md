# `With`

`With` substitutes constant values into the body expression.

```scrut
$ wo 'With[{x = 5}, x + 1]'
6
```

```scrut
$ wo 'With[{x = 2, y = 3}, x + y]'
5
```

```scrut
$ wo 'With[{l = Length[{1,2,3}]}, l + 1]'
4
```
