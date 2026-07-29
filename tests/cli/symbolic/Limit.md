# `Limit`

Computes the limit of an expression as a variable approaches a value.

```scrut
$ wo 'Limit[(1 + 1/n)^n, n -> Infinity]'
E
```

```scrut
$ wo 'Limit[(Sin[x] - x)/x^3, x -> 0]'
-1/6
```

The limit at infinity of a quotient of power sums is decided from the leading
exponents, so terms with fractional powers are handled exactly rather than by
sampling:

```scrut
$ wo 'Limit[x/(x + Sqrt[x]), x -> Infinity]'
1
```

```scrut
$ wo 'Limit[(3 x^(3/2) + x)/(2 x^(3/2) - 1), x -> Infinity]'
3/2
```

Symbolic coefficients carry through:

```scrut
$ wo 'Limit[(a x + b)/(c x + d), x -> Infinity]'
a/c
```

At the origin it is the *smallest* exponent that dominates, so quotients with
fractional powers resolve there too:

```scrut
$ wo 'Limit[(x + Sqrt[x])/(x - Sqrt[x]), x -> 0]'
-1
```

A two-sided limit only diverges when both sides agree, which for `1/x^k` means
an even integer `k`:

```scrut
$ wo 'Limit[1/x^2, x -> 0]'
Infinity
```

```scrut
$ wo 'Limit[1/x^3, x -> 0]'
Indeterminate
```

Naming a direction gives the one-sided answer:

```scrut
$ wo 'Limit[1/x, x -> 0, Direction -> "FromAbove"]'
Infinity
```
