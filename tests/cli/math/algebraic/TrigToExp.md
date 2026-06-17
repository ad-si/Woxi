# `TrigToExp`

Rewrites trig functions in terms of the complex exponential.

```scrut
$ wo 'TrigToExp[Cos[x]]'
1/(2*E^(I*x)) + E^(I*x)/2
```

The reciprocal functions are supported as well:

```scrut
$ wo 'TrigToExp[Sec[x]]'
2/(E^(-I*x) + E^(I*x))
```

Hyperbolic functions are rewritten in terms of the real exponential:

```scrut
$ wo 'TrigToExp[Tanh[x]]'
-(1/(E^x*(E^(-x) + E^x))) + E^x/(E^(-x) + E^x)
```

Inverse functions are rewritten in terms of logarithms:

```scrut
$ wo 'TrigToExp[ArcTan[x]]'
I/2*Log[1 - I*x] - I/2*Log[1 + I*x]
```

```scrut
$ wo 'TrigToExp[ArcSinh[x]]'
Log[x + Sqrt[1 + x^2]]
```
