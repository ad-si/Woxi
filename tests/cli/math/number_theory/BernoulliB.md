# `BernoulliB`

Bernoulli number.

```scrut
$ wo 'BernoulliB[4]'
-1/30
```

With a second argument it is the Bernoulli *polynomial* `B_n(x)`.

```scrut
$ wo 'BernoulliB[2, x]'
1/6 - x + x^2
```

```scrut
$ wo 'BernoulliB[3, z]'
z/2 - (3*z^2)/2 + z^3
```

`N` numericizes the coefficients but leaves the exponents exact, so the
result is still a polynomial that `Solve` can find the roots of.

```scrut
$ wo 'N[BernoulliB[3, z]]'
0.5*z - 1.5*z^2 + z^3
```

```scrut
$ wo 'Solve[N[BernoulliB[3, z]] == 0, z]'
{{z -> 0.}, {z -> 0.5}, {z -> 1.}}
```
