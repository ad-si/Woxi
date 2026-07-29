# `FindPeaks`

Gives the positions and values of the local maxima of a list.

```scrut
$ wo 'FindPeaks[{1, 3, 5, 6, 6, 4, 3, 2, 4, 7, 3, 2, 4, 2, 2, 1}]'
{{9/2, 6}, {10, 7}, {13, 4}}
```

A flat plateau inside the list is reported at the mean of its positions, so a
two-wide one has a half-integer centre:

```scrut
$ wo 'FindPeaks[{1, 3, 3, 1}]'
{{5/2, 3}}
```

A plateau that runs into a list boundary is reported at that boundary index
instead, and only counts at all when it is at most two elements long:

```scrut
$ wo 'FindPeaks[{3, 3, 1}]'
{{1, 3}}
```

```scrut
$ wo 'FindPeaks[{3, 3, 3, 1}]'
{}
```

The third argument is a minimum sharpness — the drops to both neighbours added
up and divided by the plateau width:

```scrut
$ wo 'FindPeaks[{1, 3, 5, 6, 6, 4, 3, 2, 4, 7, 3, 2, 4, 2, 2, 1}, 0, 2]'
{{10, 7}, {13, 4}}
```

```scrut
$ wo 'FindPeaks[{1, 3, 5, 6, 6, 4, 3, 2, 4, 7, 3, 2, 4, 2, 2, 1}, 0, 4.5]'
{{10, 7}}
```

The fourth is a minimum peak value:

```scrut
$ wo 'FindPeaks[{1, 3, 5, 6, 6, 4, 3, 2, 4, 7, 3, 2, 4, 2, 2, 1}, 0, 0, 4.1]'
{{9/2, 6}, {10, 7}}
```

Each parameter is checked:

```scrut
$ wo 'FindPeaks[{1, 2, 3, 2, 1}, -1]'

FindPeaks::scale: The scale -1 at position 2 should be a non-negative real number.
FindPeaks[{1, 2, 3, 2, 1}, -1]
```
