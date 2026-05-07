# `Differences`

Returns successive differences between consecutive elements.

```scrut
$ wo 'Differences[{1, 3, 6, 10}]'
{2, 3, 4}
```

```scrut
$ wo 'Differences[{1, 2, 3, 4, 5}]'
{1, 1, 1, 1}
```

```scrut
$ wo 'Differences[{10, 5, 3}]'
{-5, -2}
```

```scrut
$ wo 'Differences[{5}]'
{}
```

```scrut
$ wo 'Differences[{}]'
{}
```

```scrut
$ wo 'Differences[{0, 1, 0, 1}]'
{1, -1, 1}
```

```scrut
$ wo 'Differences[{1.5, 3, 5.5}]'
{1.5, 2.5}
```

```scrut
$ wo 'Differences[{-5, -3, 0, 5}]'
{2, 3, 5}
```

```scrut
$ wo 'Differences[{100, 90, 80}]'
{-10, -10}
```
