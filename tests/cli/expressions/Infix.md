# `Infix`

Displays the operands of an expression with an operator between them.

```scrut
$ wo 'Infix[{a, b, c}, "+"]'
a+b+c
```

The operands keep their own order, so `Infix` can print a sum with its
terms arranged differently from the canonical order.

```scrut
$ wo 'Infix[Reverse[List @@ (1 + x + x y)], "+"]'
x*y+x+1
```

`Infix` itself stays unevaluated.

```scrut
$ wo 'Head[Infix[{a, b}, "+"]]'
Infix
```
