# `Intersection`

Returns the sorted intersection of lists.

```scrut
$ wo 'Intersection[{1, 2, 3, 4}, {2, 4, 6}]'
{2, 4}
```

```scrut
$ wo 'Intersection[{a, b, c}, {b, c, d}]'
{b, c}
```

```scrut
$ wo 'Intersection[{1, 2, 3}, {4, 5, 6}]'
{}
```

```scrut
$ wo 'Intersection[{1, 2, 3}, {1, 2, 3}]'
{1, 2, 3}
```

```scrut
$ wo 'Intersection[{1, 2, 3}, {2, 3}, {3}]'
{3}
```
