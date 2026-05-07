# `Join`

Concatenates multiple lists together.

```scrut
$ wo 'Join[{a, b}, {c, d}]'
{a, b, c, d}
```

```scrut
$ wo 'Join[{1, 2}, {3, 4}, {5, 6}]'
{1, 2, 3, 4, 5, 6}
```

```scrut
$ wo 'Join[{1}]'
{1}
```

```scrut
$ wo 'Join[{}]'
{}
```

```scrut
$ wo 'Join[{}, {1, 2}]'
{1, 2}
```

```scrut
$ wo 'Join[{1, 2}, {}]'
{1, 2}
```

```scrut
$ wo 'Join[{a}, {b}, {c}, {d}]'
{a, b, c, d}
```

```scrut
$ wo 'Join[{1, 2}, {3.14, 2.71}]'
{1, 2, 3.14, 2.71}
```
