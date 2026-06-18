# `TransformationMatrix`

Gives the homogeneous matrix of a transformation function.

```scrut
$ wo 'TransformationMatrix[RotationTransform[Pi/2]]'
{{0, -1, 0}, {1, 0, 0}, {0, 0, 1}}
```

```scrut
$ wo 'TransformationMatrix[TranslationTransform[{1, 2}]]'
{{1, 0, 1}, {0, 1, 2}, {0, 0, 1}}
```
