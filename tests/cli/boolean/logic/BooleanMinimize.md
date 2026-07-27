# `BooleanMinimize`

Find the minimal sum-of-products Boolean expression.

```scrut
$ wo 'BooleanMinimize[True]'
True
```

The terms are written in descending order of their literal pattern, reading
the variables left to right: a positive literal ranks above a negative one,
and a variable that does not appear at all ranks below both.

```scrut
$ wo 'BooleanMinimize[(a && b) || (!a && !b)]'
(a && b) || ( !a &&  !b)
```

```scrut
$ wo 'BooleanMinimize[(a && b) || (b && c) || (a && c)]'
(a && b) || (a && c) || (b && c)
```

```scrut
$ wo 'BooleanMinimize[Xor[a, b, c]]'
(a && b && c) || (a &&  !b &&  !c) || ( !a && b &&  !c) || ( !a &&  !b && c)
```
