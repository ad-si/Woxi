# `NIntegrate`

Numerically integrates an expression over a range.

```scrut
$ wo 'NIntegrate[x^2, {x, 0, 1}]'
0\.333333333333333[0-9] (regex)
```

An integrand that blows up at an endpoint is still integrated accurately — the
nodes crowd towards the endpoints and the endpoint itself is never sampled:

```scrut
$ wo 'NIntegrate[1/Sqrt[x], {x, 0, 1}]'
2\.(0+\d*)? (regex)
```

```scrut
$ wo 'NIntegrate[Log[x]/Sqrt[x], {x, 0, 1}]'
-4\.(0+\d*)? (regex)
```

```scrut
$ wo 'NIntegrate[Sqrt[Tan[x]], {x, 0, Pi/2}]'
2\.2214414[0-9]+ (regex)
```

Breaking the interval at an interior singularity works the same way:

```scrut
$ wo 'NIntegrate[1/Abs[Sqrt[x]], {x, -1, 0, 1}]'
4\.(0+\d*)? (regex)
```
