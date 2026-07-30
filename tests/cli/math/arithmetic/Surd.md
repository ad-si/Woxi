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
