# `BoxMatrix`

Creates a matrix of ones with given radius.

```scrut
$ wo 'BoxMatrix[0]'
{{1}}
```

A second argument centres the box in a grid of zeros that wide:

```scrut
$ wo 'BoxMatrix[1, 5]'
{{0, 0, 0, 0, 0}, {0, 1, 1, 1, 0}, {0, 1, 1, 1, 0}, {0, 1, 1, 1, 0}, {0, 0, 0, 0, 0}}
```

An even width has no centre cell, so the box straddles the two middle ones:

```scrut
$ wo 'BoxMatrix[0, 4]'
{{0, 0, 0, 0}, {0, 1, 1, 0}, {0, 1, 1, 0}, {0, 0, 0, 0}}
```
