# `DSolve`

Solves ordinary differential equations symbolically.

```scrut
$ wo "DSolve[y'[x] == y[x], y[x], x]"
{{y[x] -> E^x*C[1]}}
```

A separable equation is nonlinear in `y`, so it needs an initial condition to
pin the constant of integration:

```scrut
$ wo "DSolve[{y'[t] == -t y[t]^2, y[0] == 1}, y, t]"
{{y -> Function[{t}, 2/(2 + t^2)]}}
```
