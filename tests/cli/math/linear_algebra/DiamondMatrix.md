# `DiamondMatrix`

Creates a diamond-shaped matrix of ones.

```scrut
$ wo 'DiamondMatrix[2]'
{{0, 0, 1, 0, 0}, {0, 1, 1, 1, 0}, {1, 1, 1, 1, 1}, {0, 1, 1, 1, 0}, {0, 0, 1, 0, 0}}
```

A second argument centres the diamond in a grid of zeros that wide:

```scrut
$ wo 'DiamondMatrix[1, 5]'
{{0, 0, 0, 0, 0}, {0, 0, 1, 0, 0}, {0, 1, 1, 1, 0}, {0, 0, 1, 0, 0}, {0, 0, 0, 0, 0}}
```
