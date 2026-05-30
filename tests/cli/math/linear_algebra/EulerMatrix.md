# `EulerMatrix`

Build a rotation matrix from a list of Euler angles. The default axis
sequence is `{3, 2, 3}` (ZYZ); an explicit axis sequence can be given as a
second argument.

```scrut
$ wo 'EulerMatrix[{0, 0, 0}]'
{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}
```

```scrut
$ wo 'EulerMatrix[{Pi/2, 0, 0}]'
{{0, -1, 0}, {1, 0, 0}, {0, 0, 1}}
```

```scrut
$ wo 'EulerMatrix[{Pi, 0, 0}]'
{{-1, 0, 0}, {0, -1, 0}, {0, 0, 1}}
```
