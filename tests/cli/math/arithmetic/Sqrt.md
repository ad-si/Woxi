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
