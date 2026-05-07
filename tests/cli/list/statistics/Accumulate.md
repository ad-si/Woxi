# `Accumulate`

Returns the cumulative sums of a list.

```scrut
$ wo 'Accumulate[{1, 2, 3}]'
{1, 3, 6}
```

```scrut
$ wo 'Accumulate[{1, 2, 3, 4, 5}]'
{1, 3, 6, 10, 15}
```

```scrut
$ wo 'Accumulate[{5}]'
{5}
```

```scrut
$ wo 'Accumulate[{}]'
{}
```

```scrut
$ wo 'Accumulate[{-1, 2, -3}]'
{-1, 1, -2}
```

```scrut
$ wo 'Accumulate[{10, -5, 3}]'
{10, 5, 8}
```

```scrut
$ wo 'Accumulate[{1.5, 2.5}]'
{1.5, 4.}
```

```scrut
$ wo 'Accumulate[{0, 0, 1}]'
{0, 0, 1}
```

```scrut
$ wo 'Accumulate[{-5, -10, -15}]'
{-5, -15, -30}
```
