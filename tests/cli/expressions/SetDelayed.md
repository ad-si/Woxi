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
