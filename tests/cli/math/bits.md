# Bit Operations

## `BitAnd`

Bitwise AND.

```scrut
$ wo 'BitAnd[12, 10]'
8
```


## `BitOr`

Bitwise OR.

```scrut
$ wo 'BitOr[12, 10]'
14
```


## `BitXor`

Bitwise XOR.

```scrut
$ wo 'BitXor[12, 10]'
6
```


## `BitNot`

Bitwise NOT — `BitNot[x] == -x - 1` for integers.

```scrut
$ wo 'BitNot[5]'
-6
```


## `BitShiftLeft`

Shift an integer left by `n` bits.

```scrut
$ wo 'BitShiftLeft[1, 4]'
16
```


## `BitShiftRight`

Shift an integer right by `n` bits (arithmetic shift).

```scrut
$ wo 'BitShiftRight[16, 2]'
4
```


