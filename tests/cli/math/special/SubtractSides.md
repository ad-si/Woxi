# `SubtractSides`

Subtract an expression from both sides of an equation.

```scrut
$ wo 'SubtractSides[x + 3 == 5, 3]'
x == 2
```

With one argument, the right-hand side is subtracted from both sides.

```scrut
$ wo 'SubtractSides[a + b == c]'
a + b - c == 0
```

When the second argument is itself an equation, the matching sides are paired.

```scrut
$ wo 'SubtractSides[a == b, c == d]'
a - c == b - d
```
