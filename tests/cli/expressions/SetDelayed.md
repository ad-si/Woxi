# `SetDelayed`

Defines a delayed assignment that evaluates the RHS each time.

```scrut
$ wo 'SetDelayed[h[x_], x^3]; h[4]'
64
```

The right-hand side is a template: a `Function` parameter that is itself a
pattern variable is a slot the caller fills, so the symbol passed in really
does become the pure function's argument.

```scrut
$ wo 'q[f_, s_] := Function[s, f]; q[s^2, s][4]'
16
```

A parameter the caller did *not* supply still binds, and is renamed when an
incoming value would otherwise be captured by it.

```scrut
$ wo 'h[a_] := Function[y, a + y]; h[y]'
Function[y$, y + y$]
```

A list on the left-hand side destructures the argument, and it keeps
destructuring through elements that are themselves calls: names bound inside
`foo[…]` are as available to the body as those bound by a nested list.

```scrut
$ wo 'pair[{foo[a_, b_]}] := a + b; pair[{foo[3, 4]}]'
7
```

```scrut
$ wo 'edge[{f : foo[{lo_, hi_}, ___]}] := {f, lo, hi}; edge[{foo[{0, 1}, extra]}]'
{foo[{0, 1}, extra], 0, 1}
```

An argument sequence in such an element leaves the positions around it
readable: earlier arguments count from the front and later ones from the back.

```scrut
$ wo 'last[{foo[__, z_]}] := z; last[{foo[1, 2, 3]}]'
3
```
