# Basic Math Tests

## `Plus`

Add numbers.

```scrut
$ wo 'Plus[1, 2]'
3
```

```scrut
$ wo 'Plus[1, 2, 3]'
6
```

```scrut
$ wo 'Plus[-1, -2]'
-3
```


### `+`

```scrut
$ wo '1+2'
3
```

```scrut
$ wo '8 + (-5)'
3
```


### List arithmetic (threading)

Arithmetic operations are automatically threaded over lists:

```scrut
$ wo '{1, 2, 3} + 2'
{3, 4, 5}
```

```scrut
$ wo '2 + {1, 2, 3}'
{3, 4, 5}
```

```scrut
$ wo '{1, 2, 3} + {4, 5, 6}'
{5, 7, 9}
```

```scrut
$ wo '{1, 2, 3} - 1'
{0, 1, 2}
```

```scrut
$ wo '{1, 2, 3} * 2'
{2, 4, 6}
```

```scrut
$ wo '{2, 4, 6} / 2'
{1, 2, 3}
```

```scrut
$ wo '{1, 2, 3} ^ 2'
{1, 4, 9}
```


## `Minus`

Make numbers negative.

```scrut
$ wo 'Minus[5]'
-5
```

```scrut {output_stream: combined}
$ wo 'Minus[5, 2]'

Minus::argx: Minus called with 2 arguments; 1 argument is expected.
5 − 2
```


## `Subtract`

Subtract numbers.

```scrut
$ wo 'Subtract[5, 2]'
3
```


### `-`

```scrut
$ wo '5-2'
3
```


## `Times`

Multiply numbers.

```scrut
$ wo 'Times[2, 3]'
6
```

```scrut
$ wo 'Times[2, 3, 4]'
24
```

```scrut
$ wo 'Times[-2, 3]'
-6
```


### `*`

```scrut
$ wo '2*2'
4
```

```scrut
$ wo '2 * (-2)'
-4
```


## `Divide`

Divide numbers.

```scrut
$ wo 'Divide[6, 2]'
3
```

```scrut {output_stream: combined}
$ wo 'Divide[6, 2, 3]'

Divide::argrx: Divide called with 3 arguments; 2 arguments are expected.
Divide[6, 2, 3]
```


## `Sign`

Returns the sign of a number.

```scrut
$ wo 'Sign[5]'
1
```

```scrut
$ wo 'Sign[0]'
0
```

```scrut
$ wo 'Sign[-7]'
-1
```


## `Prime`

Returns the nth prime number.

```scrut
$ wo 'Prime[5]'
11
```


## `Abs`

Returns the absolute value of a number.

```scrut
$ wo 'Abs[-5]'
5
```

```scrut
$ wo 'Abs[5]'
5
```

```scrut
$ wo 'Abs[0]'
0
```


## `Floor`

Rounds down to the nearest integer.

```scrut
$ wo 'Floor[3.7]'
3
```

```scrut
$ wo 'Floor[-3.7]'
-4
```

```scrut
$ wo 'Floor[3.2]'
3
```

```scrut
$ wo 'Floor[0]'
0
```

```scrut
$ wo 'Floor[-0]'
0
```

```scrut
$ wo 'Floor[0.5]'
0
```

```scrut
$ wo 'Floor[-0.5]'
-1
```


## `Ceiling`

Rounds up to the nearest integer.

```scrut
$ wo 'Ceiling[3.2]'
4
```

```scrut
$ wo 'Ceiling[-3.2]'
-3
```

```scrut
$ wo 'Ceiling[3.7]'
4
```

```scrut
$ wo 'Ceiling[0]'
0
```

```scrut
$ wo 'Ceiling[-0]'
0
```

```scrut
$ wo 'Ceiling[0.5]'
1
```

```scrut
$ wo 'Ceiling[-0.5]'
0
```


## `Round`

Rounds to the nearest integer.

```scrut
$ wo 'Round[3.5]'
4
```

```scrut
$ wo 'Round[3.4]'
3
```

```scrut
$ wo 'Round[-3.5]'
-4
```

```scrut
$ wo 'Round[-3.4]'
-3
```

```scrut
$ wo 'Round[0.5]'
0
```

```scrut
$ wo 'Round[-0.5]'
0
```

```scrut
$ wo 'Round[0]'
0
```

```scrut
$ wo 'Round[-0]'
0
```


## `Sqrt`

Returns the square root of a number.

```scrut
$ wo 'Sqrt[16]'
4
```

```scrut
$ wo 'Sqrt[0]'
0
```


## `Mod`

Returns the remainder when dividing the first argument by the second.

```scrut
$ wo 'Mod[10, 3]'
1
```

```scrut
$ wo 'Mod[7, 4]'
3
```

```scrut
$ wo 'Mod[15, 5]'
0
```

```scrut
$ wo 'Mod[8, 3]'
2
```

```scrut
$ wo 'Mod[-1, 3]'
2
```

```scrut
$ wo 'Mod[-5, 3]'
1
```

```scrut
$ wo 'Mod[10, -3]'
-2
```

```scrut
$ wo 'Mod[7.5, 2]'
1.5
```

```scrut
$ wo 'Mod[0, 5]'
0
```


## `Max`

Returns the maximum value from a set of arguments or a list.

### Multiple arguments

```scrut
$ wo 'Max[1, 5, 3]'
5
```

```scrut
$ wo 'Max[1, 2]'
2
```

```scrut
$ wo 'Max[-5, -2, -8]'
-2
```

```scrut
$ wo 'Max[3.14, 2.71, 3.5]'
3.5
```

```scrut
$ wo 'Max[1, 2.5, 3]'
3
```

### Single list argument

```scrut
$ wo 'Max[{1, 5, 3}]'
5
```

```scrut
$ wo 'Max[{-10, -5, -20}]'
-5
```

```scrut
$ wo 'Max[{3.14, 2.71, 3.5}]'
3.5
```

### Single value

```scrut
$ wo 'Max[42]'
42
```

```scrut
$ wo 'Max[-7]'
-7
```

### Empty list

```scrut
$ wo 'Max[{}]'
-Infinity
```

### With expressions

```scrut
$ wo 'Max[2 + 3, 4 * 2, 10 - 1]'
9
```

```scrut
$ wo 'Max[{1 + 1, 2 * 2, 3 - 1}]'
4
```


## `Min`

Returns the minimum value from a set of arguments or a list.

### Multiple arguments

```scrut
$ wo 'Min[1, 5, 3]'
1
```

```scrut
$ wo 'Min[1, 2]'
1
```

```scrut
$ wo 'Min[-5, -2, -8]'
-8
```

```scrut
$ wo 'Min[3.14, 2.71, 3.5]'
2.71
```

```scrut
$ wo 'Min[1, 2.5, 3]'
1
```

### Single list argument

```scrut
$ wo 'Min[{1, 5, 3}]'
1
```

```scrut
$ wo 'Min[{-10, -5, -20}]'
-20
```

```scrut
$ wo 'Min[{3.14, 2.71, 3.5}]'
2.71
```

### Single value

```scrut
$ wo 'Min[42]'
42
```

```scrut
$ wo 'Min[-7]'
-7
```

### Empty list

```scrut
$ wo 'Min[{}]'
Infinity
```

### With expressions

```scrut
$ wo 'Min[2 + 3, 4 * 2, 10 - 1]'
5
```

```scrut
$ wo 'Min[{1 + 1, 2 * 2, 3 - 1}]'
2
```


## `Sin`

Returns the sine of an angle in radians.

```scrut
$ wo 'Sin[Pi/2]'
1
```


## `NumberQ`

Returns `True` if expr is a number, and `False` otherwise.

```scrut
$ wo 'NumberQ[2]'
True
```


## `MemberQ`

Checks if an element is in a list.

```scrut
$ wo 'MemberQ[{1, 2}, 2]'
True
```

```scrut
$ wo 'MemberQ[{1, 2}, 3]'
False
```


## `RandomInteger`

### `RandomInteger[]`

Randomly gives 0 or 1.

```scrut
$ wo 'MemberQ[{0, 1}, RandomInteger[]]'
True
```


### `RandomInteger[{1, 6}]`

Randomly gives a number between 1 and 6.

```scrut
$ wo 'MemberQ[{1, 2, 3, 4, 5, 6}, RandomInteger[{1, 6}]]'
True
```


### `RandomInteger[{1, 6}, 50]`

Randomly gives 50 numbers between 1 and 6.

```scrut
$ wo 'AllTrue[RandomInteger[{1, 6}, 50], 1 <= # <= 6 &]'
True
```


## `Power`

### `Power[2, 3]`

2 raised to the power of 3 equals 8.

```scrut
$ wo 'Power[2, 3]'
8
```


### `Power[5, 0]`

Any number raised to the power of 0 equals 1.

```scrut
$ wo 'Power[5, 0]'
1
```


### `0^0`

0 raised to 0 is Indeterminate.

```scrut {output_stream: combined}
$ wo '0^0'

                                        0
Power::indet: Indeterminate expression 0  encountered.
Indeterminate
```

```scrut {output_stream: combined}
$ wo 'Power[0, 0]'

                                        0
Power::indet: Indeterminate expression 0  encountered.
Indeterminate
```

```scrut {output_stream: combined}
$ wo '0.0^0'

                                         0
Power::indet: Indeterminate expression 0.  encountered.
Indeterminate
```


### `Power[2, -1]`

2 raised to the power of -1 equals 0.5 (1/2).

```scrut
$ wo 'Power[2, -1]'
1/2
```


### `Power[4, 0.5]`

4 raised to the power of 0.5 equals 2 (square root).

```scrut
$ wo 'Power[4, 0.5]'
2.
```


### `Power[10, 2]`

10 raised to the power of 2 equals 100.

```scrut
$ wo 'Power[10, 2]'
100
```


### `Power[-2, 3]`

-2 raised to the power of 3 equals -8.

```scrut
$ wo 'Power[-2, 3]'
-8
```


### `Power[-2, 2]`

-2 raised to the power of 2 equals 4.

```scrut
$ wo 'Power[-2, 2]'
4
```


### `Power[27, 1/3]`

27 raised to the power of 1/3 equals approximately 3 (cube root).

```scrut
$ wo 'Power[27, 1/3]'
3
```


### `Power[1.5, 2.5]`

1.5 raised to the power of 2.5 equals approximately 2.756.

```scrut
$ wo 'Power[1.5, 2.5]'
2.7556759606310752
```


## `Factorial`

### `Factorial[0]`

The factorial of 0 is 1 by definition.

```scrut
$ wo 'Factorial[0]'
1
```


### `Factorial[1]`

The factorial of 1 is 1.

```scrut
$ wo 'Factorial[1]'
1
```


### `Factorial[5]`

The factorial of 5 is 120 (5! = 5 × 4 × 3 × 2 × 1).

```scrut
$ wo 'Factorial[5]'
120
```


### `Factorial[10]`

The factorial of 10 is 3628800.

```scrut
$ wo 'Factorial[10]'
3628800
```


### `Factorial[3]`

The factorial of 3 is 6 (3! = 3 × 2 × 1).

```scrut
$ wo 'Factorial[3]'
6
```


### `Factorial[7]`

The factorial of 7 is 5040.

```scrut
$ wo 'Factorial[7]'
5040
```


### `Factorial[12]`

The factorial of 12 is 479001600.

```scrut
$ wo 'Factorial[12]'
479001600
```


### `Factorial[2]`

The factorial of 2 is 2.

```scrut
$ wo 'Factorial[2]'
2
```


### `Factorial[4]`

The factorial of 4 is 24 (4! = 4 × 3 × 2 × 1).

```scrut
$ wo 'Factorial[4]'
24
```


## `GCD`

### `GCD[12, 8]`

The GCD of 12 and 8 is 4.

```scrut
$ wo 'GCD[12, 8]'
4
```


### `GCD[48, 18]`

The GCD of 48 and 18 is 6.

```scrut
$ wo 'GCD[48, 18]'
6
```


### `GCD[100, 50]`

The GCD of 100 and 50 is 50.

```scrut
$ wo 'GCD[100, 50]'
50
```


### `GCD[17, 19]`

The GCD of 17 and 19 is 1 (coprime numbers).

```scrut
$ wo 'GCD[17, 19]'
1
```


### `GCD[0, 5]`

The GCD of 0 and any number n is |n|.

```scrut
$ wo 'GCD[0, 5]'
5
```


### `GCD[15, 25, 35]`

The GCD of 15, 25, and 35 is 5.

```scrut
$ wo 'GCD[15, 25, 35]'
5
```


### `GCD[24, 36, 60]`

The GCD of 24, 36, and 60 is 12.

```scrut
$ wo 'GCD[24, 36, 60]'
12
```


### `GCD[-12, 8]`

The GCD works with negative numbers (GCD of -12 and 8 is 4).

```scrut
$ wo 'GCD[-12, 8]'
4
```


### `GCD[21, 14]`

The GCD of 21 and 14 is 7.

```scrut
$ wo 'GCD[21, 14]'
7
```


## `Surd`

Real-valued nth root.

```scrut
$ wo 'Surd[8, 3]'
2
```

```scrut
$ wo 'Surd[27, 3]'
3
```

```scrut
$ wo 'Surd[16, 4]'
2
```

```scrut
$ wo 'Surd[-8, 3]'
-2
```

```scrut
$ wo 'Surd[x, 3]'
Surd[x, 3]
```


## `Gamma`

Gamma function. For positive integers, Gamma[n] = (n-1)!.

```scrut
$ wo 'Gamma[1]'
1
```

```scrut
$ wo 'Gamma[5]'
24
```

```scrut
$ wo 'Gamma[6]'
120
```

```scrut
$ wo 'Gamma[0]'
ComplexInfinity
```

```scrut
$ wo 'Gamma[-1]'
ComplexInfinity
```

```scrut
$ wo 'Gamma[x]'
Gamma[x]
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


## `NumericQ`

Tests if an expression has a numeric value.

```scrut
$ wo 'NumericQ[42]'
True
```

```scrut
$ wo 'NumericQ["hello"]'
False
```

```scrut
$ wo 'NumericQ[3.14]'
True
```


## `Re`

Returns the real part of a number.
For real numbers, returns the number itself.

```scrut
$ wo 'Re[5]'
5
```

```scrut
$ wo 'Re[3.14]'
3.14
```


## `Im`

Returns the imaginary part of a number.
For real numbers, returns 0.

```scrut
$ wo 'Im[5]'
0
```

```scrut
$ wo 'Im[3.14]'
0
```


## `Conjugate`

Returns the complex conjugate.
For real numbers, returns the number itself.

```scrut
$ wo 'Conjugate[5]'
5
```


## `Rationalize`

Converts a decimal number to a rational approximation.

```scrut
$ wo 'Rationalize[0.5]'
1/2
```

```scrut
$ wo 'Rationalize[0.333333]'
0.333333
```

```scrut
$ wo 'Rationalize[0.25]'
1/4
```

```scrut
$ wo 'Rationalize[3]'
3
```

With a tolerance argument, finds a rational within that tolerance:

```scrut
$ wo 'Rationalize[0.333333, 0.0001]'
1/3
```

```scrut
$ wo 'Rationalize[0.333333, 0.00001]'
1/3
```

Numbers with up to 5 decimal places are rationalized:

```scrut
$ wo 'Rationalize[0.33333]'
33333/100000
```


## `Arg`

Returns the argument (phase angle) of a complex number in radians.

```scrut
$ wo 'Arg[1]'
0
```

```scrut
$ wo 'Arg[-1]'
Pi
```

```scrut
$ wo 'Arg[0]'
0
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


## Linear Algebra

### `Dot`

```scrut
$ wo 'Dot[{1, 2, 3}, {4, 5, 6}]'
32
```

```scrut
$ wo 'Dot[{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}]'
{{19, 22}, {43, 50}}
```

### `Det`

```scrut
$ wo 'Det[{{1, 2}, {3, 4}}]'
-2
```

### `Inverse`

```scrut
$ wo 'Inverse[{{1, 2}, {3, 4}}]'
{{-2, 1}, {3/2, -1/2}}
```

### `Tr`

```scrut
$ wo 'Tr[{{1, 2}, {3, 4}}]'
5
```

### `IdentityMatrix`

```scrut
$ wo 'IdentityMatrix[3]'
{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}
```

### `DiagonalMatrix`

```scrut
$ wo 'DiagonalMatrix[{1, 2, 3}]'
{{1, 0, 0}, {0, 2, 0}, {0, 0, 3}}
```

### `Cross`

```scrut
$ wo 'Cross[{1, 2, 3}, {4, 5, 6}]'
{-3, 6, -3}
```


## Utility Functions

### `Unitize`

```scrut
$ wo 'Unitize[{0, 1, -3, 0, 5}]'
{0, 1, 1, 0, 1}
```

### `Ramp`

```scrut
$ wo 'Ramp[{-2, -1, 0, 1, 2}]'
{0, 0, 0, 1, 2}
```

### `KroneckerDelta`

```scrut
$ wo 'KroneckerDelta[1, 1]'
1
```

```scrut
$ wo 'KroneckerDelta[1, 2]'
0
```

### `UnitStep`

```scrut
$ wo 'UnitStep[{-1, 0, 1}]'
{0, 1, 1}
```

### `Reap` / `Sow`

```scrut
$ wo 'Reap[Sow[1]; Sow[2]; 42]'
{42, {{1, 2}}}
```

```scrut
$ wo 'Reap[42]'
{42, {}}
```
