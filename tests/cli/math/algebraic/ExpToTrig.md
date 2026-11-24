# `ExpToTrig`

The inverse of `TrigToExp`.

```scrut
$ wo 'ExpToTrig[E^(I x)]'
Cos[x] + I*Sin[x]
```

Conjugate exponentials combine into a single trigonometric function:

```scrut
$ wo 'ExpToTrig[a E^(I x) + a E^(-I x)]'
2*a*Cos[x]
```
