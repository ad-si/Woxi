# `Reverse`

Returns a list with elements in reverse order.

```scrut
$ wo 'Reverse[{1, 2, 3}]'
{3, 2, 1}
```

```scrut
$ wo 'Reverse[{7, 2, 4}]'
{4, 2, 7}
```

```scrut
$ wo 'Reverse[{1}]'
{1}
```

```scrut
$ wo 'Reverse[{}]'
{}
```

```scrut
$ wo 'Reverse[{a, b, c, d}]'
{d, c, b, a}
```

```scrut
$ wo 'Reverse[{-5, 3.14, 0, 42}]'
{42, 0, 3.14, -5}
```

Any compound expression has its arguments reversed, including rules.

```scrut
$ wo 'Reverse[x :> y]'
y :> x
```

```scrut
$ wo 'Reverse[f[a, b, c]]'
f[c, b, a]
```
