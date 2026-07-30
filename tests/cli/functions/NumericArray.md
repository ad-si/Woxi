# `NumericArray`

Typed numeric array with auto-detected element type.

```scrut
$ wo 'Last[NumericArray[{1, 2, 3}]]'
3
```

`Dimensions` and `ArrayDepth` report on the array it holds, not on the
wrapper:

```scrut
$ wo 'Dimensions[NumericArray[{{1, 2}, {3, 4}}]]'
{2, 2}
```

```scrut
$ wo 'ArrayDepth[NumericArray[{{1, 2}, {3, 4}}]]'
2
```

`RawArray` is the legacy spelling, with the arguments the other way round:

```scrut
$ wo 'RawArray["UnsignedInteger8", {{1, 2}, {3, 4}}] === NumericArray[{{1, 2}, {3, 4}}, "UnsignedInteger8"]'
True
```
