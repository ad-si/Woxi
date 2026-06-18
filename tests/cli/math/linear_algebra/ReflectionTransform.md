# `ReflectionTransform`

Represents a reflection in the hyperplane through the origin perpendicular to a
vector.

```scrut
$ wo 'ReflectionTransform[{1, 0}]'
TransformationFunction[{{-1, 0, 0}, {0, 1, 0}, {0, 0, 1}}]
```

A second argument reflects in the hyperplane through that point.

```scrut
$ wo 'ReflectionTransform[{1, 0}, {2, 3}]'
TransformationFunction[{{-1, 0, 4}, {0, 1, 0}, {0, 0, 1}}]
```

Applying the transformation reflects a point.

```scrut
$ wo 'ReflectionTransform[{0, 1}][{4, 5}]'
{4, -5}
```
