# Symbolic Computing

```scrut
$ wo 'cow + 5'
5 + cow
```

```scrut
$ wo 'cow + 5 + 10'
15 + cow
```

```scrut
$ wo 'moo = cow + 5'
5 + cow
```

```scrut
$ wo 'D[x^n, x]'
n*x^(-1 + n)
```

```scrut
$ wo 'Integrate[x^2 + Sin[x], x]'
x^3/3 - Cos[x]
```


## Limits

```scrut
$ wo 'Limit[Sin[x]/x, x -> 0]'
1
```


## Series

```scrut
$ wo 'Series[Exp[x], {x, 0, 3}]'
SeriesData[x, 0, {1, 1, 1/2, 1/6}, 0, 4, 1]
```


## Apart

```scrut
$ wo 'Apart[1/(x^2 - 1)]'
1/(2*(-1 + x)) - 1/(2*(1 + x))
```


## Together

```scrut
$ wo 'Together[1/x + 1/y]'
(x + y)/(x*y)
```


## Cancel

```scrut
$ wo 'Cancel[(x^2 - 1)/(x - 1)]'
1 + x
```


## Collect

```scrut
$ wo 'Collect[x*y + x*z, x]'
x*(y + z)
```


## ExpandAll

```scrut
$ wo 'ExpandAll[x*(x + 1)^2]'
x + 2*x^2 + x^3
```


## `D`

Symbolic differentiation.
`D[expr, x]` differentiates once, `D[expr, {x, n}]` differentiates `n` times,
`D[expr, x, y]` takes mixed partial derivatives.

```scrut
$ wo 'D[Sin[x], x]'
Cos[x]
```

```scrut
$ wo 'D[x^2 + 3 x + 5, x]'
3 + 2*x
```

```scrut
$ wo 'D[x^2 y, {x, 2}]'
2*y
```


## `Integrate`

Symbolic integration.
`Integrate[expr, x]` gives the indefinite integral;
`Integrate[expr, {x, a, b}]` gives the definite integral.

```scrut
$ wo 'Integrate[x, x]'
x^2/2
```

```scrut
$ wo 'Integrate[x^2, {x, 0, 1}]'
1/3
```

```scrut
$ wo 'Integrate[Sin[x], {x, 0, Pi}]'
2
```


## `Sum`

Symbolic summation.

```scrut
$ wo 'Sum[k, {k, 1, 10}]'
55
```

```scrut
$ wo 'Sum[k^2, {k, 1, n}]'
(n*(1 + n)*(1 + 2*n))/6
```

```scrut
$ wo 'Sum[1/k^2, {k, 1, Infinity}]'
Pi^2/6
```


## `Product`

Symbolic product.

```scrut
$ wo 'Product[k, {k, 1, 5}]'
120
```

```scrut
$ wo 'Product[k^2, {k, 1, 3}]'
36
```


## `Simplify`

Applies basic simplification rules.

```scrut
$ wo 'Simplify[(x^2 - 1)/(x - 1)]'
1 + x
```


## `FullSimplify`

Applies a much larger set of simplifying transformations
(including trig and special-function identities).

```scrut
$ wo 'FullSimplify[Sin[x]^2 + Cos[x]^2]'
1
```


## `Solve`

Symbolic equation solver.

```scrut
$ wo 'Solve[x^2 == 4, x]'
{{x -> -2}, {x -> 2}}
```

```scrut
$ wo 'Solve[{x + y == 3, x - y == 1}, {x, y}]'
{{x -> 2, y -> 1}}
```


## `NSolve`

Numeric equation solver.

```scrut
$ wo 'NSolve[x^2 - 2 == 0, x]'
{{x -> -1.4142135623730951}, {x -> 1.414213562373095}}
```


## `Reduce`

Simplifies a logical condition, e.g. a polynomial equation, to an
equivalent form describing all solutions.

```scrut
$ wo 'Reduce[x^2 == 4, x]'
x == -2 || x == 2
```


## `SeriesCoefficient`

Returns a specific coefficient from a Taylor series.

```scrut
$ wo 'SeriesCoefficient[Sin[x], {x, 0, 3}]'
-1/6
```


## `FindRoot`

Numeric root finder.

```scrut
$ wo 'FindRoot[Cos[x] == x, {x, 1}]'
{x -> 0.7390851332151607}
```


## `DSolve`

Solves ordinary differential equations symbolically.

```scrut
$ wo "DSolve[y'[x] == y[x], y[x], x]"
{{y[x] -> E^x*C[1]}}
```


## `Limit` (more examples)

```scrut
$ wo 'Limit[(1 + 1/n)^n, n -> Infinity]'
E
```

```scrut
$ wo 'Limit[(Sin[x] - x)/x^3, x -> 0]'
-1/6
```


## `Grad`

Gradient of a scalar field.

```scrut
$ wo 'Grad[x^2 y, {x, y}]'
{2*x*y, x^2}
```


## `Div`

Divergence of a vector field.

```scrut
$ wo 'Div[{x y, y z, z x}, {x, y, z}]'
x + y + z
```


## `Curl`

Curl of a vector field.

```scrut
$ wo 'Curl[{y, -x, 0}, {x, y, z}]'
{0, 0, -2}
```


## `Laplacian`

Laplacian of a scalar field.

```scrut
$ wo 'Laplacian[x^2 + y^2, {x, y}]'
4
```


## `FourierTransform`

Symbolic Fourier transform.

```scrut
$ wo 'FourierTransform[Exp[-x^2], x, w]'
1/(Sqrt[2]*E^(w^2/4))
```


## `LaplaceTransform`

Symbolic Laplace transform.

```scrut
$ wo 'LaplaceTransform[Exp[-a t], t, s]'
(a + s)^(-1)
```


## `InverseLaplaceTransform`

Symbolic inverse Laplace transform.

```scrut
$ wo 'InverseLaplaceTransform[1/(s^2 + 1), s, t]'
Sin[t]
```
