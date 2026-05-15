# `WeierstrassPPrime`

Derivative of the Weierstrass elliptic function. wolframscript prints a
`Power::infy` warning in 2D form when the argument is 0 (the `-3` exponent
floats above the line as a superscript); Woxi emits the same warning.

```scrut
$ wo 'WeierstrassPPrime[0, {1, 2}]'

                                  -3
Power::infy: Infinite expression 0   encountered.
ComplexInfinity
```
