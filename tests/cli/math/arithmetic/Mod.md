# `Mod`

Returns the remainder when dividing the first argument by the second.

```scrut
$ wo 'Mod[10, 3]'
1
```

```scrut
$ wo 'Mod[7, 4]'
3
```

```scrut
$ wo 'Mod[15, 5]'
0
```

```scrut
$ wo 'Mod[8, 3]'
2
```

```scrut
$ wo 'Mod[-1, 3]'
2
```

```scrut
$ wo 'Mod[-5, 3]'
1
```

```scrut
$ wo 'Mod[10, -3]'
-2
```

```scrut
$ wo 'Mod[7.5, 2]'
1.5
```

```scrut
$ wo 'Mod[0, 5]'
0
```

For Gaussian integers the remainder uses `m - n*Round[m/n]`.

```scrut
$ wo 'Mod[7 + 3 I, 2]'
-1 - I
```
