# `InverseFunction`

Represents the inverse of a function.

```scrut
$ wo 'InverseFunction[Sin]'
ArcSin
```

A geometric transform inverts by inverting its homogeneous matrix,
which covers every transform constructor:

```scrut
$ wo 'InverseFunction[TranslationTransform[{1, 2}]]'
TransformationFunction[{{1, 0, -1}, {0, 1, -2}, {0, 0, 1}}]
```

```scrut
$ wo 'InverseFunction[RotationTransform[Pi/2]][{0, 1}]'
{1, 0}
```
