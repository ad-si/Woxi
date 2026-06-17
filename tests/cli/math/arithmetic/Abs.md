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

For a power with a strictly-positive real base, `Abs[b^z]` reduces to
`b^Re[z]`, so an imaginary exponent leaves only the unit modulus.

```scrut
$ wo 'Abs[Exp[2 I]]'
1
```

```scrut
$ wo 'Abs[Exp[2 + 3 I]]'
E^2
```
