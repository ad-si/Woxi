# Special Functions

Gamma, Beta, Zeta, Bessel, orthogonal polynomials, and elliptic integrals.

## `Gamma`

Gamma function. For positive integers, Gamma[n] = (n-1)!.

```scrut
$ wo 'Gamma[1]'
1
```

```scrut
$ wo 'Gamma[5]'
24
```

```scrut
$ wo 'Gamma[6]'
120
```

```scrut
$ wo 'Gamma[0]'
ComplexInfinity
```

```scrut
$ wo 'Gamma[-1]'
ComplexInfinity
```

```scrut
$ wo 'Gamma[x]'
Gamma[x]
```


## `LogGamma`

Natural logarithm of `Gamma`.

```scrut
$ wo 'LogGamma[5]'
Log[24]
```


## `Zeta`

Riemann zeta function.

```scrut
$ wo 'Zeta[2]'
Pi^2/6
```


## `PolyLog`

Polylogarithm.

```scrut
$ wo 'PolyLog[2, 0]'
0
```


## `Erf`

The error function.

```scrut
$ wo 'Erf[0]'
0
```


## `Erfc`

The complementary error function, `Erfc[x] == 1 - Erf[x]`.

```scrut
$ wo 'Erfc[0]'
1
```


## `BesselJ`

Bessel function of the first kind.

```scrut
$ wo 'BesselJ[0, 0]'
1
```


## `BesselI`

Modified Bessel function of the first kind.

```scrut
$ wo 'BesselI[0, 0]'
1
```


## `LegendreP`

Legendre polynomial.

```scrut
$ wo 'LegendreP[2, x]'
(-1 + 3*x^2)/2
```


## `ChebyshevT`

Chebyshev polynomial of the first kind.

```scrut
$ wo 'ChebyshevT[3, x]'
-3*x + 4*x^3
```


## `HermiteH`

Hermite polynomial.

```scrut
$ wo 'HermiteH[3, x]'
-12*x + 8*x^3
```


## `LaguerreL`

Laguerre polynomial.

```scrut
$ wo 'LaguerreL[2, x]'
(2 - 4*x + x^2)/2
```


## `EllipticK`

Complete elliptic integral of the first kind.

```scrut
$ wo 'EllipticK[0]'
Pi/2
```


## `EllipticE`

Complete elliptic integral of the second kind.

```scrut
$ wo 'EllipticE[0]'
Pi/2
```


