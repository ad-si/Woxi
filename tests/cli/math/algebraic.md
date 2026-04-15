# Algebraic Manipulation

Expand, factor, simplify, and rewrite expressions.

## `Expand`

Multiplies out products and powers.

```scrut
$ wo 'Expand[(x + 1)^2]'
1 + 2*x + x^2
```


## `Factor`

Factors a polynomial.

```scrut
$ wo 'Factor[x^2 - 1]'
(-1 + x)*(1 + x)
```


## `Collect`

Groups terms sharing a common factor.

```scrut
$ wo 'Collect[a x + b x + c, x]'
c + (a + b)*x
```


## `Cancel`

Cancels common factors in a rational expression.

```scrut
$ wo 'Cancel[(x^2 - 1)/(x - 1)]'
1 + x
```


## `Apart`

Partial-fraction decomposition.

```scrut
$ wo 'Apart[1/((x - 1) (x + 1)), x]'
1/(2*(-1 + x)) - 1/(2*(1 + x))
```


## `Together`

Combines a sum of rational expressions over a common denominator.

```scrut
$ wo 'Together[1/x + 1/y]'
(x + y)/(x*y)
```


## `TrigExpand`

Expands trigonometric identities.

```scrut
$ wo 'TrigExpand[Sin[2 x]]'
2*Cos[x]*Sin[x]
```


## `TrigReduce`

Reduces a trig expression using sum-to-product identities.

```scrut
$ wo 'TrigReduce[Sin[x]^2]'
(1 - Cos[2*x])/2
```


## `TrigToExp`

Rewrites trig functions in terms of the complex exponential.

```scrut
$ wo 'TrigToExp[Cos[x]]'
1/(2*E^(I*x)) + E^(I*x)/2
```


## `ExpToTrig`

The inverse of `TrigToExp`.

```scrut
$ wo 'ExpToTrig[E^(I x)]'
Cos[x] + I*Sin[x]
```

