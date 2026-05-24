# `RealAbs`

Returns the absolute value assuming real input.

```scrut
$ wo 'RealAbs[x]'
RealAbs[x]
```

`RealAbs` has the closed-form antiderivative `(x*RealAbs[x])/2`, so
indefinite integrals collapse cleanly:

```scrut
$ wo 'Integrate[RealAbs[x], x]'
(x*RealAbs[x])/2
```

Definite integrals fall out from the same antiderivative:

```scrut
$ wo 'Integrate[RealAbs[x], {x, -1, 1}]'
1
```
