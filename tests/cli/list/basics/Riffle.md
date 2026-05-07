# `Riffle`

Inserts an element between each pair of elements.

```scrut
$ wo 'Riffle[{a, b, c}, x]'
{a, x, b, x, c}
```

```scrut
$ wo 'Riffle[{1, 2, 3, 4}, 0]'
{1, 0, 2, 0, 3, 0, 4}
```

```scrut
$ wo 'Riffle[{a}, x]'
{a}
```

```scrut
$ wo 'Riffle[{}, x]'
{}
```

Element-wise interleaving with a second list:

```scrut
$ wo 'Riffle[{a, b, c, d}, {1, 2, 3, 4}]'
{a, 1, b, 2, c, 3, d, 4}
```

```scrut
$ wo 'Riffle[{a, b, c}, {1, 2}]'
{a, 1, b, 2, c}
```
