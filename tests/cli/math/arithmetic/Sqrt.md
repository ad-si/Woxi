# `Sqrt`

Returns the square root of a number.

```scrut
$ wo 'Sqrt[16]'
4
```

```scrut
$ wo 'Sqrt[0]'
0
```

A square factor comes out of the radical even when it is far too large to
find by trial division — `1150069001035` is `100003^2 * 115`:

```scrut
$ wo 'Sqrt[1150069001035]'
100003*Sqrt[115]
```

```scrut
$ wo 'Sqrt[1150069001035]/2'
(100003*Sqrt[115])/2
```

A radicand with no repeated prime factor stays whole — `15485599740329` is
`999983 * 15485863`:

```scrut
$ wo 'Sqrt[15485599740329]'
Sqrt[15485599740329]
```

`\[Sqrt]` is a prefix operator binding to the next factor, with `^`, `!`
and `[[…]]` binding tighter than the radical:

```scrut
$ wo '\[Sqrt]x y'
Sqrt[x]*y
```

```scrut
$ wo '\[Sqrt]x^2'
Sqrt[x^2]
```

```scrut
$ wo '\[Sqrt]2 3'
3*Sqrt[2]
```
