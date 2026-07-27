# `Definition`

Shows the rules and attributes attached to a symbol.

```scrut
$ wo 'Definition[Sin]'
Attributes[Sin] = {Listable, NumericFunction, Protected}
```

The call stays held — its head is `Definition` and its one part is the
symbol — so only the display is the definition text.

```scrut
$ wo 'g[x_] := x^2; Head[Definition[g]]'
Definition
```

```scrut
$ wo 'g[x_] := x^2; Definition[g][[1]]'
g
```
