# Polynomials

Polynomial arithmetic and analysis.

## `PolynomialQuotient`

Polynomial long division — returns the quotient.

```scrut
$ wo 'PolynomialQuotient[x^2 + x + 1, x + 1, x]'
x
```


## `PolynomialRemainder`

Remainder of polynomial long division.

```scrut
$ wo 'PolynomialRemainder[x^2 + x + 1, x + 1, x]'
1
```


## `PolynomialGCD`

Greatest common divisor of two polynomials.

```scrut
$ wo 'PolynomialGCD[x^2 - 1, x^2 + 2 x + 1]'
1 + x
```


## `Resultant`

Resultant of two polynomials in a given variable.

```scrut
$ wo 'Resultant[x^2 - 1, x + 1, x]'
0
```


## `Discriminant`

Discriminant of a polynomial.

```scrut
$ wo 'Discriminant[x^2 + b x + c, x]'
b^2 - 4*c
```


## `Coefficient`

Extracts the coefficient of a given power.

```scrut
$ wo 'Coefficient[x^2 + 3 x + 4, x]'
3
```


## `CoefficientList`

Returns the list of all coefficients of a polynomial.

```scrut
$ wo 'CoefficientList[1 + 2 x + 3 x^2, x]'
{1, 2, 3}
```


## `Exponent`

Highest power of a variable in a polynomial.

```scrut
$ wo 'Exponent[x^2 + 3 x + 4, x]'
2
```


