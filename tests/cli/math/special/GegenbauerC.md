# `GegenbauerC`

Gegenbauer (ultraspherical) polynomial C_n^lambda(x).

```scrut
$ wo 'GegenbauerC[0, 2, x]'
1
```

The two-argument form `GegenbauerC[n, x]` is the renormalized polynomial
`(2/n) ChebyshevT[n, x]`.

```scrut
$ wo 'GegenbauerC[2, x]'
-1 + 2*x^2
```

```scrut
$ wo 'GegenbauerC[3, x]'
(2*(-3*x + 4*x^3))/3
```
