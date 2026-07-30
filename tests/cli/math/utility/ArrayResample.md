# `ArrayResample`

Resample an array to a different size.

```scrut
$ wo 'ArrayResample[{1, 2, 3}, 5]'
{1, 3/2, 2, 5/2, 3}
```

Every axis is resampled, so a 2x2 array taken to 3 gains interpolated columns
as well as rows:

```scrut
$ wo 'ArrayResample[{{1, 2}, {3, 4}}, 3]'
{{1, 3/2, 2}, {2, 5/2, 3}, {3, 7/2, 4}}
```

A list gives each axis its own count:

```scrut
$ wo 'ArrayResample[{1, 2, 3, 4, 5}, {2}]'
{1, 5}
```

A count that is not a positive integer names no array:

```scrut
$ wo 'ArrayResample[{1, 2, 3, 4, 5}, 0]'

ArrayResample::nodim: Invalid dimension specification 0.
ArrayResample[{1, 2, 3, 4, 5}, 0]
```
