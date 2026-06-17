# `ArithmeticGeometricMean`

Arithmetic-geometric mean of two numbers.

```scrut
$ wo 'ArithmeticGeometricMean[0, 5]'
0
```

Machine-real arguments are evaluated numerically:

```scrut
$ wo 'ArithmeticGeometricMean[1.8, 1.2]'
1.4848082617417828
```

Being `Orderless`, it canonicalizes the order of symbolic arguments:

```scrut
$ wo 'ArithmeticGeometricMean[24, 6]'
ArithmeticGeometricMean[6, 24]
```
