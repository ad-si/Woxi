# `MatrixMinimalPolynomial`

The monic polynomial of least degree in the given variable that annihilates a
square matrix.

```scrut
$ wo 'MatrixMinimalPolynomial[{{2, 1}, {0, 2}}, x]'
4 - 4*x + x^2
```

A scalar multiple of the identity has a degree-1 minimal polynomial even though
its characteristic polynomial is degree 2.

```scrut
$ wo 'MatrixMinimalPolynomial[{{2, 0}, {0, 2}}, x]'
-2 + x
```

Matrices with symbolic entries return the expanded polynomial.

```scrut
$ wo 'MatrixMinimalPolynomial[{{a, b}, {c, d}}, x]'
-(b*c) + a*d - a*x - d*x + x^2
```
