# `Level`

Returns the sub-expressions at a given level.

```scrut
$ wo 'Level[{{a, b}, {c, d}}, {1}]'
{{a, b}, {c, d}}
```

```scrut
$ wo 'Level[{{a, b}, {c, d}}, {2}]'
{a, b, c, d}
```

An object with a compound head that is nevertheless an atom — a `Tree`,
a `SparseArray`, a `ByteArray`, a `NumericArray` or a `Dataset` — has no
levels inside it. The object itself sits at level `0` counted from the top
and at level `-1` counted from the bottom:

```scrut
$ wo 'Level[Tree[1, {}], {-1}]'
{Tree[1, {}]}
```

```scrut
$ wo 'Level[SparseArray[{1, 2}], {1}]'
{}
```

```scrut
$ wo 'Head[First[Level[SparseArray[{1, 2}], {0}]]]'
SparseArray
```

`All` is a level specification meaning every level, the same as
`{0, Infinity}`:

```scrut
$ wo 'Level[{{1, 2}, {3, {4, 5}}}, All]'
{1, 2, {1, 2}, 3, 4, 5, {4, 5}, {3, {4, 5}}, {{1, 2}, {3, {4, 5}}}}
```
