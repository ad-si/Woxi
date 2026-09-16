# `CubeRoot`

Returns the real-valued cube root.

```scrut
$ wo 'CubeRoot[8]'
2
```

It threads over a list.

```scrut
$ wo 'CubeRoot[{8, 27}]'
{2, 3}
```

`CubeRoot[x]` is just a spelling of `Surd[x, 3]`, so the two agree exactly,
including on a machine-precision argument:

```scrut
$ wo 'CubeRoot[7.] == Surd[7., 3] == 7.^(1/3)'
True
```

```scrut
$ wo 'CubeRoot[7.]'
1.912931182772389
```

A non-real argument is reported under the head it was written with:

```scrut
$ wo 'CubeRoot[I]'

CubeRoot::preal: The parameter I should be real valued.
CubeRoot[I]
```
