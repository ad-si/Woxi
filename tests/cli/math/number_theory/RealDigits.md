# `RealDigits`

Extracts the decimal digits and exponent of a real number.

```scrut
$ wo 'RealDigits[Pi, 10, 20]'
{{3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4}, 1}
```

Extract just the digit list with `[[1]]`:

```scrut
$ wo 'RealDigits[Pi, 10, 10][[1]]'
{3, 1, 4, 1, 5, 9, 2, 6, 5, 3}
```

Works with rationals:

```scrut
$ wo 'RealDigits[1/7, 10, 12]'
{{1, 4, 2, 8, 5, 7, 1, 4, 2, 8, 5, 7}, 0}
```
