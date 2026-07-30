# `FromDigits`

Constructs an integer from its digits.

```scrut
$ wo 'FromDigits[{1, 2, 3, 4, 5}]'
12345
```

With base 2 (binary):

```scrut
$ wo 'FromDigits[{1, 1, 1, 1, 1, 1, 1, 1}, 2]'
255
```

The digit list is a polynomial in the base, so any base works — base 1 sums the
digits and base 0 leaves the last one:

```scrut
$ wo 'FromDigits[{1, 2, 3}, 1]'
6
```

```scrut
$ wo 'FromDigits[{1, 2}, 0]'
2
```

A rational, a real, or a symbol works the same way:

```scrut
$ wo 'FromDigits[{1, 2}, 1/2]'
5/2
```

```scrut
$ wo 'FromDigits[{1, 2, 3}, x]'
3 + 2*x + x^2
```
