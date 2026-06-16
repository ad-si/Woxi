# `RSolve`

Solve recurrence equations.

The general solution of a first-order linear recurrence anchors its constant at
`n = 1`, so the root carries the exponent `n - 1`.

```scrut
$ wo 'RSolve[a[n] == 2 a[n-1], a[n], n]'
{{a[n] -> 2^(-1 + n)*C[1]}}
```

With an initial condition the constant is determined.

```scrut
$ wo 'RSolve[{a[n + 1] == 2 a[n], a[0] == 1}, a, n]'
{{a -> Function[{n}, 2^n]}}
```

A repeated characteristic root contributes an extra factor of `n` per
multiplicity.

```scrut
$ wo 'RSolve[a[n] == 4 a[n-1] - 4 a[n-2], a[n], n]'
{{a[n] -> 2^n*C[1] + 2^n*n*C[2]}}
```
