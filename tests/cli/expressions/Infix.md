# `Infix`

Displays the operands of an expression with an operator between them.
The printed display keeps the call, as wolframscript's script mode does;
`ToString` lays it out.

```scrut
$ wo 'Infix[{a, b, c}, "+"]'
Infix[{a, b, c}, +]
```

```scrut
$ wo 'ToString[Infix[{a, b, c}, "+"]]'
a+b+c
```

Without an operator the head is set between tildes:

```scrut
$ wo 'ToString[Infix[f[a, b, c]]]'
a ~f~ b ~f~ c
```

The operands keep their own order, so `Infix` can print a sum with its
terms arranged differently from the canonical order. An operand that
binds no tighter than `Infix`'s precedence (`630` by default) is
bracketed.

```scrut
$ wo 'ToString[Infix[Reverse[List @@ (1 + x + x y)], "+"]]'
(x y)+x+1
```

`Infix` itself stays unevaluated.

```scrut
$ wo 'Head[Infix[{a, b}, "+"]]'
Infix
```
