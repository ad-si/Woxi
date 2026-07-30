# `Solve`

Symbolic equation solver.

```scrut
$ wo 'Solve[x^2 == 4, x]'
{{x -> -2}, {x -> 2}}
```

```scrut
$ wo 'Solve[{x + y == 3, x - y == 1}, {x, y}]'
{{x -> 2, y -> 1}}
```

A negative leading coefficient still gives simplified, correctly ordered roots.

```scrut
$ wo 'Solve[2 - x^2 == 0, x]'
{{x -> -Sqrt[2]}, {x -> Sqrt[2]}}
```

```scrut
$ wo 'Solve[6 - x - x^2 == 0, x]'
{{x -> -3}, {x -> 2}}
```

`x^n == c` is solved in radicals, the roots being the real root times the
`n`-th roots of unity:

```scrut
$ wo 'Solve[x^3 == 8, x]'
{{x -> 2}, {x -> -2*(-1)^(1/3)}, {x -> 2*(-1)^(2/3)}}
```

```scrut
$ wo 'Solve[x^3 == 2, x]'
{{x -> -(-2)^(1/3)}, {x -> 2^(1/3)}, {x -> (-1)^(2/3)*2^(1/3)}}
```

For an odd degree and a negative right-hand side the generating root is the
real one:

```scrut
$ wo 'Solve[x^3 == -8, x]'
{{x -> -2}, {x -> 2*(-1)^(1/3)}, {x -> -2*(-1)^(2/3)}}
```

`NSolve` stays on the numeric root finder, so a conjugate pair agrees to the
last bit:

```scrut
$ wo 'NSolve[x^3 == 8, x]'
{{x -> -1. - 1.7320508075688772*I}, {x -> -1. + 1.7320508075688772*I}, {x -> 2.}}
```

Inverting `Abs` splits the equation into a positive and a negative branch, and
over the complexes that leaves out the rest of the circle — `Solve` says so:

```scrut {output_stream: combined}
$ wo 'Solve[Abs[x] == 2, x]'

Solve::ifun: Inverse functions are being used by Solve, so some solutions may not be found; use Reduce for complete solution information.
{{x -> -2}, {x -> 2}}
```

Restricting the domain to the reals, or narrowing the split with a constraint,
loses nothing and so reports nothing:

```scrut {output_stream: combined}
$ wo 'Solve[Abs[x] == 2, x, Reals]'
{{x -> -2}, {x -> 2}}
```

```scrut {output_stream: combined}
$ wo 'Solve[{Abs[x] == 2, x > 0}, x]'
{{x -> 2}}
```

A list of constraints is the same system as the conjunction of them:

```scrut
$ wo 'Solve[{Sin[x] == 0, 0 < x < 7}, x]'
{{x -> Pi}, {x -> 2*Pi}}
```

`Modulus -> n` solves over the integers modulo `n`, so the answers are residues
rather than radicals:

```scrut
$ wo 'Solve[x^2 == 2, x, Modulus -> 7]'
{{x -> 3}, {x -> 4}}
```

```scrut
$ wo 'Solve[x^3 == 1, x, Modulus -> 7]'
{{x -> 1}, {x -> 2}, {x -> 4}}
```

There may be none, and the modulus need not be prime:

```scrut
$ wo 'Solve[x^2 == 3, x, Modulus -> 7]'
{}
```

```scrut
$ wo 'Solve[2 x == 4, x, Modulus -> 6]'
{{x -> 2}, {x -> 5}}
```

`MaxRoots -> n` keeps only the first `n` solutions:

```scrut
$ wo 'Solve[x^4 == 1, x, MaxRoots -> 3]'
{{x -> -1}, {x -> -I}, {x -> I}}
```

It has to be a positive integer, `Infinity` or `Automatic`:

```scrut
$ wo 'Solve[x^3 == 1, x, MaxRoots -> 0]'

Solve::maxrts: The value 0 of the MaxRoots option is not a positive integer, Infinity or Automatic.
Solve[x^3 == 1, x, MaxRoots -> 0]
```

Machine-precision coefficients are solved numerically instead of in
radicals or as `Root` objects:

```scrut
$ wo 'Solve[x^3 == 8., x]'
{{x -> -1. - 1.7320508075688772*I}, {x -> -1. + 1.7320508075688772*I}, {x -> 2.}}
```

```scrut
$ wo 'Round[x /. Solve[x^3 + 1.5 x^2 - 3.2 x + 4.7 == 0, x], 1/10^6]'
{-19079/6250, 2426/3125 - (120997*I)/125000, 2426/3125 + (120997*I)/125000}
```

Every root is reported with its multiplicity:

```scrut
$ wo 'Solve[x^3 - 4 x^2 == 0, x]'
{{x -> 0}, {x -> 0}, {x -> 4}}
```
