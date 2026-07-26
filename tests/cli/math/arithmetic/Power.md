# `Power`

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
.* (regex*)
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

A base outside the unit circle diverges — to `Infinity` only when it is a
positive real — while one inside it vanishes:

```scrut
$ wo '{2^Infinity, (-2)^Infinity, (-1/2)^Infinity}'
{Infinity, ComplexInfinity, 0}
```

Any base on the unit circle never settles:

```scrut {output_stream: combined}
$ wo '(-1)^Infinity'

                                              Infinity
Infinity::indet: Indeterminate expression (-1)         encountered.
Indeterminate
```

An infinite base raises its direction, so `Sqrt[-Infinity]` points along the
imaginary axis:

```scrut
$ wo '{(-Infinity)^(1/2), DirectedInfinity[I]^2, Infinity^Infinity}'
{DirectedInfinity[I], -Infinity, ComplexInfinity}
```
