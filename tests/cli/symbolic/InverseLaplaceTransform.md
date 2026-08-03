# `InverseLaplaceTransform`

Symbolic inverse Laplace transform.

```scrut
$ wo 'InverseLaplaceTransform[1/(s^2 + 1), s, t]'
Sin[t]
```

A square root of `s^2 + a^2` in the denominator inverts to a Bessel function.

```scrut
$ wo 'InverseLaplaceTransform[1/Sqrt[s^2 + a^2], s, t]'
BesselJ[0, a*t]
```

The third argument names the point the inverse transform is taken at, so it
may be an expression or a number rather than a plain symbol.

```scrut
$ wo 'InverseLaplaceTransform[1/(s + 1), s, 2 u]'
E^(-2*u)
```

```scrut
$ wo 'InverseLaplaceTransform[1/Sqrt[s^2 + 1], s, 1.5]'
0.5118276717359183
```
