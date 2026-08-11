# `Resultant`

Resultant of two polynomials in a given variable.

```scrut
$ wo 'Resultant[x^2 - 1, x + 1, x]'
0
```

The resultant of two polynomials in more than one variable is a polynomial
in the variables that are left, reported multiplied out:

```scrut
$ wo 'Resultant[x^2 + y^2 - 1, (x - 1)^2 + (y - 1)^2 - 1, y]'
-8*x + 8*x^2
```
