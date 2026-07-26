# `ScalingTransform`

Create a scaling transformation matrix.

```scrut
$ wo 'ScalingTransform[{2}]'
TransformationFunction[{{2, 0}, {0, 1}}]
```

A scalar factor scales along a direction vector, leaving the perpendicular
directions alone:

```scrut
$ wo 'ScalingTransform[2, {1, 1}]'
TransformationFunction[{{3/2, 1/2, 0}, {1/2, 3/2, 0}, {0, 0, 1}}]
```

A third argument centres the scaling at that point:

```scrut
$ wo 'ScalingTransform[2, {0, 1}, {3, 4}]'
TransformationFunction[{{1, 0, 0}, {0, 2, -4}, {0, 0, 1}}]
```
