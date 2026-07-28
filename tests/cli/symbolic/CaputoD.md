# `CaputoD`

`CaputoD[f, {x, α}]` gives the Caputo fractional differintegral of `f`. For a
power it is a single Gamma ratio, `Gamma[p + 1]/Gamma[p - α + 1] x^(p - α)`,
and the operator is linear, so a polynomial goes term by term.

```scrut
$ wo 'CaputoD[t^2, {t, 1/2}]'
(8*t^(3/2))/(3*Sqrt[Pi])
```

What sets Caputo apart from Riemann–Liouville is the constant: the function is
differentiated `⌈α⌉` times before the rest of the order is integrated away, so
a constant vanishes under any positive order.

```scrut
$ wo 'CaputoD[1, {t, 1/2}]'
0
```

A whole order is the ordinary derivative, and a negative one integrates:

```scrut
$ wo 'CaputoD[t^2, {t, 2}]'
2
```

```scrut
$ wo 'CaputoD[t^2, {t, -1}]'
t^3/3
```

The order may be symbolic:

```scrut
$ wo 'CaputoD[t^2, {t, alpha}]'
(2*t^(2 - alpha))/Gamma[3 - alpha]
```
