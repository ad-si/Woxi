# `C`

The default generated parameter of `DSolve`, `RSolve`, `Reduce`, and `Solve`.
Solutions with free constants are expressed in terms of `C[1]`, `C[2]`, …

```scrut
$ wo "DSolve[y'[x] == y[x], y[x], x]"
{{y[x] -> E^x*C[1]}}
```

Since `C` is a built-in symbol it is `Protected`,
so it cannot be used as a variable.

```scrut
$ wo 'Attributes[C]'
{NHoldAll, Protected, ReadProtected}
```

```scrut
$ wo 'C = 12'

Set::wrsym: Symbol C is Protected.
12
```

Use `Unprotect` to override the protection:

```scrut
$ wo 'Unprotect[C]; C = 12; C'
12
```
