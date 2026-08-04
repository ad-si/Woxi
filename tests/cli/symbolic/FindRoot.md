# `FindRoot`

Numeric root finder.

```scrut
$ wo 'FindRoot[Cos[x] == x, {x, 1}]'
{x -> 0.7390851332151607}
```

The Newton iteration is damped, so a badly scaled function whose first step
lands far past the root still converges:

```scrut
$ wo 'FindRoot[Exp[x] - 1000, {x, 1}]'
{x -> 6.907755278982137}
```

`MaxIterations` caps the number of steps. Stopping early reports it and hands
back the point reached:

```scrut
$ wo 'FindRoot[x^2 - 2, {x, 1}, MaxIterations -> 2]'

FindRoot::cvmit: Failed to converge to the requested accuracy or precision within 2 iterations.
{x -> 1.4166666666666667}
```

```scrut
$ wo 'FindRoot[x^2 - 2, {x, 1}, MaxIterations -> 3]'

FindRoot::cvmit: Failed to converge to the requested accuracy or precision within 3 iterations.
{x -> 1.4142156862745099}
```

It has to be a positive integer, `Infinity` or `Automatic`:

```scrut
$ wo 'FindRoot[x^2 - 2, {x, 1}, MaxIterations -> 0]'

FindRoot::ioppfa: The value of the option MaxIterations -> 0 should be a positive integer, Infinity or Automatic\. (regex)
.* (regex*)
FindRoot[x^2 - 2, {x, 1}, MaxIterations -> 0]
```

A vanishing derivative reports the singular Jacobian and gives back the point
the iteration stalled at:

```scrut
$ wo 'FindRoot[x^2 + 1, {x, 1}]'

FindRoot::jsing: Encountered a singular Jacobian at the point {x} = {0.}. Try perturbing the initial point(s).
{x -> 0.}
```

The derivative is taken symbolically where that works.  A non-smooth function
has none — differentiating `Max` leaves `Derivative[1, 0][Max][…]` standing —
so the iteration falls back to a difference quotient and still converges:

```scrut
$ wo 'FindRoot[Max[x, 2 x] - 6, {x, 1}]'
{x -> 3.}
```
