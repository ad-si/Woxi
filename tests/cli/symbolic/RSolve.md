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

An initial condition at a non-zero index shifts the exponent accordingly.

```scrut
$ wo 'RSolve[{a[n] == 2 a[n-1], a[2] == 5}, a[n], n]'
{{a[n] -> 5*2^(-2 + n)}}
```

A repeated characteristic root contributes an extra factor of `n` per
multiplicity.

```scrut
$ wo 'RSolve[a[n] == 4 a[n-1] - 4 a[n-2], a[n], n]'
{{a[n] -> 2^n*C[1] + 2^n*n*C[2]}}
```

The golden-ratio recurrence has irrational characteristic roots, so its
solutions are expressed in the `Fibonacci`/`LucasL` basis.

```scrut
$ wo 'RSolve[{a[n] == a[n-1] + a[n-2], a[0] == 0, a[1] == 1}, a[n], n]'
{{a[n] -> Fibonacci[n]}}
```

```scrut
$ wo 'RSolve[a[n] == a[n-1] + a[n-2], a[n], n]'
{{a[n] -> C[1]*Fibonacci[n] + C[2]*LucasL[n]}}
```

The nonlinear logistic map `a[n+1] == 4 a[n] (1 - a[n])` (the fully chaotic
case) has the closed form `(1 - Cos[2^n*ArcCos[1 - 2 c]])/2` for the initial
condition `a[0] == c`.

```scrut
$ wo 'RSolve[{x[n + 1] == 4 x[n] (1 - x[n]), x[0] == 1/10}, x, n]'
{{x -> Function[{n}, (1 - Cos[2^n*ArcCos[4/5]])/2]}}
```

Without an initial condition the phase stays an arbitrary constant.

```scrut
$ wo 'RSolve[{x[n + 1] == 4 x[n] (1 - x[n])}, x, n]'
{{x -> Function[{n}, 1/2 - Cos[2^n*C[1]]/2]}}
```
