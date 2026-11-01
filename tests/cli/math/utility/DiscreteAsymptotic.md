# `DiscreteAsymptotic`

compute leading asymptotic term of a discrete sequence.

```scrut
$ wo 'DiscreteAsymptotic[n, n -> Infinity]'
n
```

Rational and exponential sequences keep only their dominant term:

```scrut
$ wo 'DiscreteAsymptotic[(5n^4/(3n^3 + 1)) 2^n, n -> Infinity]'
(5*2^n*n)/3
```

```scrut
$ wo 'DiscreteAsymptotic[4^n/(1 + 2^n), n -> Infinity]'
2^n
```

Hyperbolic and special sequences:

```scrut
$ wo 'DiscreteAsymptotic[Sinh[n], n -> Infinity]'
E^n/2
```

```scrut
$ wo 'DiscreteAsymptotic[Fibonacci[n], n -> Infinity]'
GoldenRatio^n/Sqrt[5]
```

More terms of Stirling's series with `SeriesTermGoal`:

```scrut
$ wo 'DiscreteAsymptotic[n!, n -> Infinity, SeriesTermGoal -> 3]'
(n^(1/2 + n)*Sqrt[2*Pi]*(1 + 1/(288*n^2) + 1/(12*n)))/E^n
```

The coefficients of a generating function, from its nearest pole:

```scrut
$ wo 'DiscreteAsymptotic[SeriesCoefficient[1/(2 - E^z), {z, 0, n}], n -> Infinity]'
Log[2]^(-1 - n)/2
```
