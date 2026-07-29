# `CompanionMatrix`

The companion matrix of a monic polynomial, given by its coefficient vector
`{c0, …, c[n-1]}`:

```scrut
$ wo 'CompanionMatrix[{5, 2, 3, 1}]'
{{0, 0, 0, -5}, {1, 0, 0, -2}, {0, 1, 0, -3}, {0, 0, 1, -1}}
```

An explicit polynomial in a named variable works too, and is divided through by
its leading coefficient first:

```scrut
$ wo 'CompanionMatrix[5 + 2*x + 3*x^2 + x^3, x]'
{{0, 0, -5}, {1, 0, -2}, {0, 1, -3}}
```

```scrut
$ wo 'CompanionMatrix[2*x^2 + 3*x + 1, x]'
{{0, -1/2}, {1, -3/2}}
```

Its characteristic polynomial recovers the coefficients:

```scrut
$ wo 'CharacteristicPolynomial[CompanionMatrix[{2, 3, 1}], x]'
-2 - 3*x - x^2 - x^3
```

The last argument says where the negated coefficients go. `Right` is the
default; `Bottom` is its transpose, `Left` turns it through half a turn, and
`Top` is the transpose of `Left`:

```scrut
$ wo 'CompanionMatrix[{2, 4, 6}, Bottom]'
{{0, 1, 0}, {0, 0, 1}, {-2, -4, -6}}
```

```scrut
$ wo 'CompanionMatrix[{2, 4, 6}, Left]'
{{-6, 1, 0}, {-4, 0, 1}, {-2, 0, 0}}
```

```scrut
$ wo 'CompanionMatrix[x^3 + 2*x^2 + 3*x + 4, x, Top]'
{{-2, -3, -4}, {1, 0, 0}, {0, 1, 0}}
```

Anything else in the placement slot is refused:

```scrut
$ wo 'CompanionMatrix[{5, 2, 3, 1}, foo]'

CompanionMatrix::plspecc: Specification foo for placement of coefficients must be Top, Bottom, Left or Right.
CompanionMatrix[{5, 2, 3, 1}, foo]
```

So is a first argument that is neither a coefficient vector nor a polynomial of
degree at least one:

```scrut
$ wo 'CompanionMatrix[Sin[x], x]'

CompanionMatrix::clorpoly: Argument Sin[x] is neither a non-empty list of coefficients nor an explicit polynomial in a given variable.
CompanionMatrix[Sin[x], x]
```
