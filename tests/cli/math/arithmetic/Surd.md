# `Surd`

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

A machine-precision argument stays inexact:

```scrut
$ wo 'Surd[1., 3]'
1.
```

The degree has to be an integer:

```scrut
$ wo 'Surd[8, 1/2]'

Surd::int: Integer expected at position 2 in Surd[8, 1/2].
Surd[8, 1/2]
```

`\[CubeRoot]` is the prefix operator for the real-valued cube root, so it
is negative for a negative argument:

```scrut
$ wo '\[CubeRoot](-8)'
-2
```

```scrut
$ wo 'Head[\[CubeRoot]y]'
Surd
```
