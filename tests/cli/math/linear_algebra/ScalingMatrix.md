# `ScalingMatrix`

Gives the matrix that scales by a factor along the direction of a vector.

```scrut
$ wo 'ScalingMatrix[2, {1, 0}]'
{{2, 0}, {0, 1}}
```

Scaling along a diagonal direction mixes the axes.

```scrut
$ wo 'ScalingMatrix[2, {1, 1}]'
{{3/2, 1/2}, {1/2, 3/2}}
```

The single-argument list form gives a diagonal scaling matrix.

```scrut
$ wo 'ScalingMatrix[{2, 3}]'
{{2, 0}, {0, 3}}
```
