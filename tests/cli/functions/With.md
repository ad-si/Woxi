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

Several variable specifications may be given.
Each one is scoped inside the ones before it,
so a later one can build on an earlier one:

```scrut
$ wo 'With[{x = 5}, {y = x + 1}, y^2]'
36
```

```scrut
$ wo 'With[{x = 5}, {x = x + 1}, x^2]'
36
```
