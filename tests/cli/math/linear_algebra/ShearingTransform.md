# `ShearingTransform`

Represent a shear transformation by angle in the direction of the first vector,
normal to the second vector.

```scrut
$ wo 'ShearingTransform[Pi/4, {1, 0}, {0, 1}]'
TransformationFunction[{{1, 1, 0}, {0, 1, 0}, {0, 0, 1}}]
```

Only the component of the direction perpendicular to the normal drives the shear.

```scrut
$ wo 'ShearingTransform[Pi/6, {3, 4}, {0, 1}]'
TransformationFunction[{{1, 1/Sqrt[3], 0}, {0, 1, 0}, {0, 0, 1}}]
```
