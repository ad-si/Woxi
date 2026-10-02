# `Together`

Combines a sum of rational expressions over a common denominator.

```scrut
$ wo 'Together[1/x + 1/y]'
(x + y)/(x*y)
```


With `Modulus -> p` the result is computed over the integers mod `p`:

```scrut
$ wo 'Together[(2 + 7*x)/(2*x), Modulus -> 7]'
x^(-1)
```
