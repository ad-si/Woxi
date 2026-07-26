# `ListInterpolation`

Creates interpolation from uniformly spaced data.

The coordinates of a 2-D grid may be given either as separate arguments or as
one list, and an exact point over an exact grid interpolates exactly:

```scrut
$ wo '{ListInterpolation[{{1, 2}, {3, 4}}][{3/2, 3/2}], ListInterpolation[{{1, 2}, {3, 4}}][3/2, 3/2]}'
{5/2, 5/2}
```

```scrut
$ wo 'ListInterpolation[{{1, 2, 3}, {4, 5, 6}, {7, 8, 10}}][{3/2, 5/2}]'
253/64
```
