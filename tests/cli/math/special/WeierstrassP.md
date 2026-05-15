# `WeierstrassP`

Weierstrass elliptic function. wolframscript prints a `Power::infy` warning
in 2D form (the `-2` exponent floats above the line as a superscript) when
the argument is 0; Woxi emits the same warning.

```scrut
$ wo 'WeierstrassP[0, {1, 1}]'

                                  -2
Power::infy: Infinite expression 0   encountered.
ComplexInfinity
```
