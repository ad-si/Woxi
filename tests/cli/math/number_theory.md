# Number Theory and Combinatorics

Prime numbers, integer decompositions, and combinatorial sequences.

## `Prime`

Returns the nth prime number.

```scrut
$ wo 'Prime[5]'
11
```


## `RealDigits`

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


## `IntegerDigits`

Returns the list of digits of an integer.

```scrut
$ wo 'IntegerDigits[12345]'
{1, 2, 3, 4, 5}
```

With base 2 (binary):

```scrut
$ wo 'IntegerDigits[255, 2]'
{1, 1, 1, 1, 1, 1, 1, 1}
```


## `FromDigits`

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


## `FactorInteger`

Returns the prime factorization as {prime, exponent} pairs.

```scrut
$ wo 'FactorInteger[60]'
{{2, 2}, {3, 1}, {5, 1}}
```

```scrut
$ wo 'FactorInteger[100]'
{{2, 2}, {5, 2}}
```

```scrut
$ wo 'FactorInteger[17]'
{{17, 1}}
```

```scrut
$ wo 'FactorInteger[2^128 - 1]'
{{3, 1}, {5, 1}, {17, 1}, {257, 1}, {641, 1}, {65537, 1}, {274177, 1}, {6700417, 1}, {67280421310721, 1}}
```


## `Divisors`

Returns all divisors of an integer.

```scrut
$ wo 'Divisors[12]'
{1, 2, 3, 4, 6, 12}
```

```scrut
$ wo 'Divisors[1]'
{1}
```

```scrut
$ wo 'Divisors[17]'
{1, 17}
```


## `DivisorSigma`

Returns the sum of the k-th powers of divisors.

```scrut
$ wo 'DivisorSigma[0, 12]'
6
```

```scrut
$ wo 'DivisorSigma[1, 12]'
28
```

```scrut
$ wo 'DivisorSigma[1, 6]'
12
```


## `MoebiusMu`

Returns the Möbius function value.

```scrut
$ wo 'MoebiusMu[1]'
1
```

```scrut
$ wo 'MoebiusMu[2]'
-1
```

```scrut
$ wo 'MoebiusMu[6]'
1
```

```scrut
$ wo 'MoebiusMu[4]'
0
```

```scrut
$ wo 'MoebiusMu[30]'
-1
```


## `EulerPhi`

Returns Euler's totient function (count of integers up to n that are coprime to n).

```scrut
$ wo 'EulerPhi[1]'
1
```

```scrut
$ wo 'EulerPhi[10]'
4
```

```scrut
$ wo 'EulerPhi[12]'
4
```

```scrut
$ wo 'EulerPhi[7]'
6
```


## `CoprimeQ`

Tests if two integers are coprime.

```scrut
$ wo 'CoprimeQ[3, 5]'
True
```

```scrut
$ wo 'CoprimeQ[6, 9]'
False
```

```scrut
$ wo 'CoprimeQ[14, 15]'
True
```


## `Binomial`

Binomial coefficient.

```scrut
$ wo 'Binomial[5, 2]'
10
```

```scrut
$ wo 'Binomial[10, 3]'
120
```


## `Multinomial`

Multinomial coefficient — `Multinomial[n1, n2, …]`.

```scrut
$ wo 'Multinomial[2, 3]'
10
```


## `Fibonacci`

Fibonacci number.

```scrut
$ wo 'Fibonacci[10]'
55
```


## `LucasL`

Lucas number.

```scrut
$ wo 'LucasL[10]'
123
```


## `BernoulliB`

Bernoulli number.

```scrut
$ wo 'BernoulliB[4]'
-1/30
```


## `CatalanNumber`

The `n`-th Catalan number.

```scrut
$ wo 'CatalanNumber[5]'
42
```


## `HarmonicNumber`

The `n`-th harmonic number `Sum[1/k, {k, 1, n}]`.

```scrut
$ wo 'HarmonicNumber[4]'
25/12
```


## `NextPrime`

Returns the next prime after a given number.

```scrut
$ wo 'NextPrime[10]'
11
```


## `PrimePi`

Number of primes less than or equal to `n`.

```scrut
$ wo 'PrimePi[100]'
25
```


## `IntegerExponent`

The largest exponent `k` such that `b^k` divides `n`.

```scrut
$ wo 'IntegerExponent[24, 2]'
3
```


## `IntegerPart`

The integer part of a real number (truncates toward zero).

```scrut
$ wo 'IntegerPart[3.7]'
3
```


## `FractionalPart`

The fractional part of a real number.

```scrut
$ wo 'FractionalPart[3.7]'
0.7000000000000002
```


## `Quotient`

Integer quotient.

```scrut
$ wo 'Quotient[10, 3]'
3
```


## `QuotientRemainder`

Returns `{Quotient[n, m], Mod[n, m]}`.

```scrut
$ wo 'QuotientRemainder[10, 3]'
{3, 1}
```


## `PowerMod`

Fast modular exponentiation — `PowerMod[b, e, m] == Mod[b^e, m]`.

```scrut
$ wo 'PowerMod[2, 10, 100]'
24
```


## `ExtendedGCD`

Returns `{g, {s, t}}` with `g == GCD[a, b] == s a + t b`.

```scrut
$ wo 'ExtendedGCD[12, 18]'
{6, {-1, 1}}
```


