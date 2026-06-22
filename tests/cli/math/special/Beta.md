# `Beta`

Euler beta function.

```scrut
$ wo 'Beta[1, 1]'
1
```

`Beta[a, b]` is rational whenever one argument is an integer, including
half-integer and other rational partners:

```scrut
$ wo 'Beta[1/2, 3]'
16/15
```

The three-argument form is the incomplete beta function `Beta[z, a, b]`. It
auto-evaluates to a closed form only when both `a` and `b` are integers:

```scrut
$ wo 'Beta[2, 2, 3]'
2/3
```

For an exact non-integer parameter it stays symbolic, but any inexact argument
forces numeric evaluation:

```scrut
$ wo 'Beta[2, 1/2, 3]'
Beta[2, 1/2, 3]
```

```scrut
$ wo 'N[Beta[2, 1/2, 3]]'
1.3199326582148885
```
