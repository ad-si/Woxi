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
