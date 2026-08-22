# `Compile`

Compiles a function for numerical evaluation.

```scrut
$ wo 'sqr = Compile[{x}, x x]; sqr[2]'
4.
```

An undeclared argument defaults to `_Real`, which is why the result above is
inexact. A declared `_Integer` argument binds an integer:

```scrut
$ wo 'Compile[{{n, _Integer, 0}}, Nest[# + 1 &, 0, n]][4]'
4
```

A signature that includes a real works in machine reals throughout, so exact
numbers from the body come back inexact:

```scrut
$ wo 'Compile[{{x, _Real, 0}}, Clip[x, {-4, 4}]][-5.]'
-4.
```

A rank-`n` `_Real` argument is converted element by element:

```scrut
$ wo 'Compile[{{g, _Real, 2}}, g][{{1, 1/2}, {3, 4}}]'
{{1., 0.5}, {3., 4.}}
```

Trailing option rules such as `RuntimeAttributes` and `RuntimeOptions` are
accepted and don't change the result:

```scrut
$ wo 'Compile[{{x, _Real}}, x^2, RuntimeAttributes -> {Listable}, RuntimeOptions -> "Speed"][3.]'
9.
```
