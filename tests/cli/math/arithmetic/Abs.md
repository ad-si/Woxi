# `Abs`

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

The modulus of a complex number is reduced, pulling perfect-square factors out
of the radical.

```scrut
$ wo 'Abs[2 + 2 I]'
2*Sqrt[2]
```
